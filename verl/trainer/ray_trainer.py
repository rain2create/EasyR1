# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)

## 按这个顺序来读
# 1. Role (枚举定义)
# 2. ResourcePoolManager (资源管理)
# 3. RayPPOTrainer.__init__ (初始化配置)
# 4. RayPPOTrainer.init_workers (初始化Worker)
# 5. RayPPOTrainer.fit (主训练循环) ⭐核心
# 6. RayPPOTrainer._make_batch_data (数据生成)
# 7. apply_kl_penalty + compute_advantage (后处理)
# 8. _validate / _save_checkpoint (辅助功能)




class Role(IntEnum):
    """
    【角色定义】定义RL训练系统中各个组件的角色标识，用于资源分配和Worker映射
    
    角色说明:
        Actor: 策略网络，负责生成答案（不直接用于HybridEngine模式）
        Rollout: 推理引擎（vLLM），负责快速生成序列（不直接用于HybridEngine模式）  
        ActorRollout: Actor+Rollout合并（HybridEngine早期模式，已废弃）  
        Critic: 价值网络，用于PPO+GAE算法估计状态价值（GRPO不需要）
        RefPolicy: 参考策略，用于计算KL散度（HybridEngine模式下与Actor合并）
        RewardModel: 奖励模型，如果用神经网络模型打分（当前主要用reward_function）
        ActorRolloutRef: 【当前默认】Actor+Rollout+RefPolicy三合一，节省显存  
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    【资源池管理器】管理Ray分布式训练所需的GPU资源池
    
    属性:
        resource_pool_spec: 资源池规格，如 {"global_pool": [8, 8]} 表示2个节点，每节点8卡  一个节点代表一台机器，一台机器有多少卡 这里能显示
        mapping: 角色到资源池的映射，如 {Role.ActorRolloutRef: "global_pool"}
        resource_pool_dict: 创建后的RayResourcePool实例字典
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """【创建资源池】为分布式训练创建Ray资源池
        
        说明:
            max_colocate_count=1 表示每个资源池内只创建一个WorkerGroup，
            所有角色（Actor/Rollout/Ref）共享同一个WorkerGroup（节省显存）  这里就是ActorRolloutRef三合一的实现方式
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """【获取资源池】根据角色获取对应的Ray资源池"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """【获取GPU总数】计算整个集群需要的GPU数量"""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """【检查资源】确保Ray集群中有足够的GPU资源"""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """
    【应用KL惩罚】在token-level奖励上应用KL散度惩罚，防止策略偏离参考模型太远
    
    输入数据流:
        data.batch["token_level_scores"]: (batch_size * n, response_length) - 奖励函数给出的原始分数
        data.batch["old_log_probs"]: (batch_size * n, response_length) - 当前策略的对数概率
        data.batch["ref_log_probs"]: (batch_size * n, response_length) - 参考模型的对数概率
        data.batch["response_mask"]: (batch_size * n, response_length) - response位置的mask
    
    输出数据流:
        data.batch["token_level_rewards"]: (batch_size * n, response_length) - KL惩罚后的奖励
    
    计算公式:
        token_level_rewards = token_level_scores - kl_coef * KL(current_policy || ref_policy)
    """
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """
    【计算优势估计】根据配置的算法（GRPO/GAE等）计算优势值和回报
    
    输入数据流:
        data.batch["token_level_rewards"]: (batch_size * n, response_length) - KL惩罚后的奖励
        data.batch["response_mask"]: (batch_size * n, response_length) - response位置mask
        data.non_tensor_batch["uid"]: (batch_size * n,) - 样本唯一ID（用于GRPO分组）
        data.batch["values"]: (batch_size * n, response_length) - Critic预测的价值（GAE需要）
        data.batch["reward_baselines"]: (batch_size * n,) - ReMax算法的基线奖励
    
    输出数据流:
        data.batch["advantages"]: (batch_size * n, response_length) - 优势估计值
        data.batch["returns"]: (batch_size * n, response_length) - 回报值
    
    注意:
        - GRPO算法：按uid分组，组内标准化
        - GAE算法：需要values，用TD-λ计算
        - ReMax算法：需要reward_baselines
    """
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    【PPO/GRPO训练器 - 核心类】基于Ray分布式框架的RL训练器
    
    说明:
        - 运行在driver进程（单个CPU/GPU节点）上
        - 通过RPC调用WorkerGroup的计算函数来构建PPO数据流
        - 轻量级的优势计算在driver进程上完成
        - 重计算（log_prob生成等）在Worker上分布式执行
    
    关键属性:
        use_reference_policy: 是否使用参考模型计算KL（True则创建Ref Worker）
        use_critic: 是否使用Critic网络（GAE需要，GRPO不需要）
        hybrid_engine: 是否使用HybridEngine（Actor/Rollout/Ref合并，节省显存）
        kl_ctrl: KL控制器（自适应或固定）
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        """
        【初始化训练器】配置校验和关键属性设置
        
        关键校验:
            1. GRPO/RLOO需要rollout.n > 1（每个问题多个回答才能组内比较）
            2. batch_size必须能被global_batch_size整除
            3. 如果用Critic，需要校验critic的batch配置
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # 【KL控制】根据配置决定是否使用参考模型和KL惩罚
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        # 【Critic判断】GAE算法需要Critic网络，GRPO等不需要
        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")


        ## Rollout Batch Size	多少个不同的问题（prompts）	从 dataloader 取多少条数据
        # Global Batch Size	    多少个生成的回答（responses）用于训练	rollout_batch_size × n

        #      data:
        #         rollout_batch_size: 512    # 512个不同的问题
        #      worker:
            #       rollout:
            #           n: 5                     # 每个问题生成5个回答
            #       actor:
            #           global_batch_size: 128   # 训练时batch size（会被自动调整）



        # 【配置校验】确保batch size能被正确分割
        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        # 【GRPO/RLOO校验】这些算法需要n>1才能进行组内比较
        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        # 【计算总训练步数】
        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """
        【初始化Worker集群】创建Ray资源池和WorkerGroup
        
        流程:
            1. 创建资源池（根据resource_pool_spec）
            2. 创建各角色的RayClassWithInitArgs（ActorRolloutRef, Critic, RM等）
            3. 使用create_colocated_worker_cls合并同资源池内的角色（节省显存）
            4. 调用spawn()启动所有Worker
            5. 初始化各Worker的模型（init_model）
        
        注意:
            - ActorRolloutRef在HybridEngine模式下三合一（节省显存）
            - Critic仅在GAE算法时创建
            - RewardModel仅在use_reward_model=True时创建
            - Rollout最后初始化，以便vLLM更好估计KV Cache内存
        """
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 【创建ActorRolloutRef Worker】HybridEngine模式，三合一节省显存
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # 【创建Critic Worker】仅GAE算法需要
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # 【创建RewardModel Worker】如果用神经网络模型打分
        if self.use_reward_model:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # 【初始化WorkerGroup】
        # NOTE: 如果想为不同角色使用不同资源池（不同并行度），不应使用create_colocated_worker_cls
        # 而是直接传递不同资源池给不同worker groups
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31
            self.wg_dicts.append(wg_dict)

        # 初始化Critic模型
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        # 初始化RewardModel
        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # 【最后初始化ActorRolloutRef】以便vLLM更好估计KV Cache内存
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        """
        【保存检查点】保存模型权重和训练状态
        
        保存路径结构:
            {save_checkpoint_path}/global_step_{global_step}/
                ├── actor/          # Actor模型权重
                ├── critic/         # Critic模型权重（如果use_critic）
                └── dataloader.pt   # DataLoader状态（断点续训用）
        
        同时更新最佳模型记录（checkpoint_tracker.json）
        """
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        """
        【加载检查点】恢复模型权重和训练状态（支持断点续训）
        
        加载优先级:
            1. 如果配置了load_checkpoint_path，直接加载
            2. 否则如果find_last_checkpoint=True，自动找最新的检查点
            3. 否则从头开始训练
        
        恢复内容:
            - Actor模型权重
            - Critic模型权重（如果use_critic）
            - DataLoader状态（训练位置）
            - 最佳验证分数记录
        """
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        """
        【验证流程】在验证集上评估模型性能   rl都train完后，拿train完的模型 在val上评估一下，生成回答，看看reward_score和一些指标的表现
        
        数据流:
            1. 从val_dataloader取数据
               - input_ids: (val_batch_size, prompt_length)
            2. 调用generate_sequences生成回答
               - responses: (val_batch_size * n, response_length)
            3. 调用reward_function计算奖励
               - reward_tensor: (val_batch_size * n, response_length)
            4. 收集并汇总metrics
        
        注意:
            - 使用val_override_config（通常temperature=0.6, top_p=0.95, n=1）
            - 会记录部分生成样本到日志（用于可视化）
        
        返回:
            dict包含val/reward_score和各reward组件分数
        """
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """
        【序列长度平衡】重新排序batch，使得每个DP rank处理的token数相近（负载均衡）
        
        输入:
            batch.batch["attention_mask"]: (batch_size * n, seq_len) - 注意力mask
        
        流程:
            1. 计算每个样本的有效token数（根据attention_mask）
               - global_seqlen_lst: list of length (batch_size * n,)
            2. 使用贪心算法将样本划分到world_size个分区，使各分区token总数相近
            3. 按照分区顺序重排batch
        
        注意:
            - 这会打乱batch内的数据顺序
            - GRPO/RLOO等依赖group的算法需要注意：重排后同一group的样本是否还在同一分区
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        """
        【生成训练批次 - 核心数据流】Rollout阶段，生成模型回答并组装训练数据
        
        数据流详细说明:
            
            Step 1: 从DataLoader取原始数据
                batch_dict包含:
                    - input_ids: (micro_batch_size, prompt_length)
                    - attention_mask: (micro_batch_size, prompt_length)
                    - multi_modal_data: 多模态数据（图片/视频）
                    - ground_truth: 标准答案（用于奖励计算）
            
            Step 2: 构造生成批次（pop出生成所需keys）
                gen_batch包含:
                    - input_ids: (micro_batc h_size, prompt_length)
                    - attention_mask: (micro_batch_size, prompt_length)
                    - position_ids: (micro_batch_size, prompt_length)
                    - multi_modal_data: 多模态数据
            
            Step 3: 调用vLLM生成回答（在Worker上分布式执行）
                gen_batch_output = generate_sequences(gen_batch)
                包含:
                    - responses: (micro_batch_size * n, response_length) 生成的回答
                    - old_log_probs: (micro_batch_size * n, response_length) 生成时的logprob
                    - response_mask: (micro_batch_size * n, response_length) 有效位置mask
            
            Step 4: 扩展原始batch以匹配n个回答
                new_batch = new_batch.repeat(repeat_times=n, interleave=True)
                结果: 每个原始问题重复n次，用于和n个回答对齐
                如: [q1, q2] -> [q1, q1, q1, q2, q2, q2] (n=3)
            
            Step 5: 合并生成结果
                new_batch = new_batch.union(gen_batch_output)
                最终shape:
                    - prompts: (micro_batch_size * n, prompt_length)
                    - responses: (micro_batch_size * n, response_length)
                    - old_log_probs: (micro_batch_size * n, response_length)
                    - uid: (micro_batch_size * n,) 每个样本唯一ID，用于GRPO分组
            
            Step 6 (可选): Online Filtering（DAPO算法）
                - 计算奖励分数
                - 过滤掉分数太高或太低的样本（只保留中等难度的）
            
            Step 7: 累积batch直到达到rollout_batch_size
                循环直到 current_batch_size >= rollout_batch_size
        
        返回:
            DataProto，包含完整的训练数据（prompts, responses, old_log_probs, uid等）
        """
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            # 【重要】为每个样本生成唯一ID，用于GRPO/RLOO的组识别
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # 【构造生成批次】pop出generation所需的keys
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # 【生成回答】调用vLLM生成，分布式执行
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            # 【ReMax特殊处理】需要额外的贪婪解码基线
            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0  # 贪婪解码
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)  # (batch_size,)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # 【重复对齐】将原始batch重复n次，与n个生成的回答对齐
            # interleave=True确保 [q1, q2] + [a1_1, a1_2, a1_3, a2_1, a2_2, a2_3] 正确配对
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # 【Online Filtering】DAPO算法，过滤极端样本
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            # 【累积batch】
            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        【主训练循环 - PPO/GRPO核心流程】
        
        说明:
            - Driver进程通过RPC调用WorkerGroup的计算函数来构建PPO数据流
            - 轻量级的优势计算在driver进程上完成
            - 重计算（log_prob生成等）在Worker上分布式执行
        
        完整数据流（单步）:
                         ## 这里设置了一张卡一个worker，分卡计算就是每张卡一个worker并行算，统一计算就是只用一张卡算
        【计算模式说明】  ## data计算，分卡是自动把batch分成多份，每份丢到一个worker上算(这里八张卡八个worker)，最后合并结果；统一计算就是把batch丢到driver上算（只用一张卡或者cpu算）
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  🖥️ = 统一计算（Driver进程，单点执行）                                          │
        │  🌐 = 分卡计算（Worker分布式，多卡并行）                                        │
        └─────────────────────────────────────────────────────────────────────────────┘
        
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  Step 1: 【生成阶段 - Rollout】🌐 分卡计算                                      │   ## 分micro batch做生成，因为vllm会产生大的kv cache 大batch容易oom
        │  ─────────────────────────────────────────────────────────────────────────  │
        │  调用: _make_batch_data() → actor_rollout_ref_wg.generate_sequences()        │
        │  执行位置: 所有Worker（vLLM分布式生成）                                         │
        │  输入: 原始数据 (input_ids, attention_mask, images)                           │
        │        Shape: (rollout_batch_size, prompt_length)                           │
        │  输出: gen_batch_output                                                      │
        │        - responses:     (rollout_batch_size * n, response_length)           │
        │        - old_log_probs: (rollout_batch_size * n, response_length)           │
        │        - response_mask: (rollout_batch_size * n, response_length)           │
        └─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  Step 2: 【序列长度平衡】🖥️ 统一计算                                           │
        │  ─────────────────────────────────────────────────────────────────────────  │
        │  调用: _balance_batch()                                                      │
        │  执行位置: Driver进程（单点计算，无梯度）                                       │
        │  功能: 重新排序batch，使每个DP rank处理的token数相近                           │
        │  注意: 这会打乱batch内顺序，GRPO分组依赖uid识别                                 │
        └─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
        ┌─────────────────────────────────────────────────────────────────────────────┐ 
        │  Step 3: 【计算奖励】🌐 分卡计算                                               │   ## 一次性batch全部生成
        │  ─────────────────────────────────────────────────────────────────────────  │
        │  调用: reward_fn.compute_reward.remote(batch)                                │
        │  执行位置: Reward Worker（可能多卡并行）                                        │
        │  输入: batch包含responses和ground_truth                                      │
        │  输出: reward_tensor                                                         │
        │        Shape: (rollout_batch_size * n, response_length)                     │
        │        通常是sparse reward（只在answer位置有值）                              │
        └─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  Step 4: 【重计算log probs】🌐 分卡计算                                        │  ## 分micro batch做生成
        │  ─────────────────────────────────────────────────────────────────────────   │
        │  调用: actor_rollout_ref_wg.compute_log_probs(batch)                         │
        │  执行位置: Actor Worker（FSDP分布式）                                           │
        │  输入: batch包含prompts + responses                                          │
        │  输出: old_log_probs（当前策略的logprob，用于PPO裁剪）                        │
        │        Shape: (rollout_batch_size * n, response_length)                     │
        └─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  Step 5: 【计算参考模型log probs】🌐 分卡计算（如果需要KL）                     │  分micro batch做生成
        │  ─────────────────────────────────────────────────────────────────────────  │
        │  调用: actor_rollout_ref_wg.compute_ref_log_probs(batch)                     │
        │  执行位置: Ref Worker（FSDP分布式，可能与Actor共置）                            │
        │  输入: batch包含prompts + responses                                          │
        │  输出: ref_log_probs（参考模型的logprob，用于KL计算）                           │
        │        Shape: (rollout_batch_size * n, response_length)                     │
        └─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  Step 6: 【计算Value】🌐 分卡计算（仅GAE算法）                                  │  # 分micro batch做生成
        │  ─────────────────────────────────────────────────────────────────────────  │
        │  调用: critic_wg.compute_values(batch)                                       │
        │  执行位置: Critic Worker（FSDP分布式）                                          │
        │  输出: values                                                                │
        │        Shape: (rollout_batch_size * n, response_length)                     │
        └─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  Step 7: 【后处理 - 优势计算】🖥️ 统一计算                                      │  ##一次性batch全部生成
        │  ─────────────────────────────────────────────────────────────────────────  │
        │  执行位置: Driver进程（单点计算，无梯度）                                       │
        │  7.1 获取reward_tensor（如果在Step 3是异步执行的）                              │
        │      → data.batch["token_level_scores"]                                     │
        │                                                                             │
        │  7.2 应用KL惩罚（如果use_kl_loss=False且use_reference_policy=True）            │
        │      → apply_kl_penalty()                                                   │
        │      → data.batch["token_level_rewards"]                                    │
        │                                                                             │
        │  7.3 计算优势估计                                                             │
        │      → compute_advantage()                                                  │
        │      根据算法选择:                                                            │
        │        - GRPO: 按uid分组，组内标准化                                           │
        │        - GAE: 用values计算TD-λ优势                                            │
        │        - RLOO: Leave-one-out基线                                              │
        │        - ReMax: 贪婪解码基线                                                  │
        │      → data.batch["advantages"]: (rollout_batch_size * n, response_length)  │
        │      → data.batch["returns"]:   (rollout_batch_size * n, response_length)   │
        └─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  Step 8: 【更新网络】🌐 分卡计算 --- 分布式训练                                    ## 分micro batch计算，因为涉及梯度更新，过大batch容易oom
        │  ─────────────────────────────────────────────────────────────────────────  │
        │  执行位置: 所有Worker（FSDP分布式训练，有梯度回传）                              │
        │  8.1 更新Critic（如果use_critic=True）                                         │
        │      → critic_wg.update_critic(batch)                                        │
        │      目标: 让critic预测的values接近returns                                    │
        │                                                                             │
        │  8.2 更新Actor（策略网络）                                                      │
        │      → actor_rollout_ref_wg.update_actor(batch)                              │
        │      使用PPO裁剪目标函数，根据advantages更新策略                               │
        └─────────────────────────────────────────────────────────────────────────────┘
        
        【为什么有些步骤必须在Driver统一计算？】
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  Step 2 (序列长度平衡): 需要全局视角决定样本分配到哪个rank，Worker无法独立完成      │
        │  Step 7 (优势计算): 涉及跨样本的组内统计（如GRPO的组内均值方差），需要在完整数据上进行 │
        └─────────────────────────────────────────────────────────────────────────────┘
        
        数据Shape总结:
            B  = rollout_batch_size（配置中的batch大小）
            n  = rollout.n（每个问题的生成次数，GRPO通常n>1）
            pl = prompt_length
            rl = response_length
            
            input_ids:          (B, pl)
            responses:          (B*n, rl)
            old_log_probs:      (B*n, rl)
            ref_log_probs:      (B*n, rl)
            token_level_scores: (B*n, rl)
            advantages:         (B*n, rl)
            returns:            (B*n, rl)
            uid:                (B*n,)  - 用于GRPO分组识别
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # 【加载检查点】训练前加载，支持断点续训
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # 【训练前验证】用于观察初始模型性能
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # 【Step 1: 生成数据】
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)  ## 分micro_batch生成，最终合成一个大batch -- 因为显存限制，vllm生成需要kv cache，所以不能一次生成太大batch，生成后再合成一个大batch进行后续计算
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # 【Step 2: 序列长度平衡】
                # NOTE: 这会打乱batch内的数据顺序，GRPO/RLOO依赖uid识别group
                self._balance_batch(batch, metrics=metrics)

                # 计算全局有效token数（用于监控）
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # 【Step 3: 计算奖励】（异步执行）
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)  # 直接一次性计算整个batch的奖励，避免重复调用RPC

                # 【Step 4: 重计算log probs】（用于PPO裁剪）
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)  # 直接计算整个batch的log probs，避免重复调用RPC
                    batch = batch.union(old_log_probs)

                # 【Step 5: 计算参考模型log probs】（用于KL）
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch) # 直接计算整个batch的参考log probs，避免重复调用RPC
                        batch = batch.union(ref_log_probs)

                # 【Step 6: 计算Value】（仅GAE算法需要）
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)  # 直接计算整个batch的values，避免重复调用RPC
                        batch = batch.union(values)

                # 【Step 7: 后处理 - 应用KL惩罚 + 计算优势】
                with timer("adv", timing_raw):
                    # 7.1 获取reward_tensor（如果在Step 3是异步执行的）
                    if "token_level_scores" not in batch.batch:
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}  
                        metrics.update(reward_metrics)

                    # 7.2 应用KL惩罚（如果不用KL loss模式）
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # 7.3 计算优势估计（driver进程上执行）
                    batch = compute_advantage(   # 直接在整个batch上计算优势，避免分多次计算带来的重复开销
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # 【Step 8.1: 更新Critic】（仅GAE算法）
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # 【Step 8.2: 更新Actor】（策略网络）
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):    
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)   ## 分micro_batch更新，内部会进行多轮迭代，最终返回一个大batch的输出metrics

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # 【验证】按配置频率执行
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                # 【保存检查点】按配置频率执行
                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # 【收集Metrics】计算各类指标并记录
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # 【训练结束后验证】
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        # 【保存最终检查点】
        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
