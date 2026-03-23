# Copyright 2022 The HuggingFace Team
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """自适应 KL 控制器
    
    功能: 根据当前 KL 散度动态调整惩罚系数，使 KL 保持在目标值附近
    原理: 当前 KL > 目标值 → 增大惩罚；当前 KL < 目标值 → 减小惩罚
    
    参考: https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """固定 KL 控制器
    
    功能: KL 惩罚系数始终保持不变
    适用: 大多数场景，尤其是不需要精细控制 KL 的情况
    """

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    """优势估计器枚举
    
    支持的算法:
        GAE: PPO 的广义优势估计（需要 Critic 网络）
        GRPO: Group Relative Policy Optimization（最常用，无需 Critic）
        GRPO_PASSK: Pass@k 版本的 GRPO（只奖励最好样本）
        REINFORCE_PLUS_PLUS: REINFORCE++ 算法
        REMAX: 使用贪婪基线的 RL 算法
        RLOO: Leave-One-Out 基线估计
    """

    GAE = "gae"
    GRPO = "grpo"
    GRPO_PASSK = "grpo_passk"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


ADV_ESTIMATOR_MAP: dict[str, Any] = {}


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """获取 KL 控制器
    
    功能: 根据配置创建自适应或固定 KL 控制器
    
    输入:
        algorithm_config: 算法配置（包含 kl_type, kl_coef 等）
    
    输出:
        AdaptiveKLController 或 FixedKLController 实例
    """
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


def register_adv_estimator(name: AdvantageEstimator):
    """优势估计器注册装饰器
    
    功能: 将函数注册到指定的算法名称，用于 @register_adv_estimator(AdvantageEstimator.GRPO) 装饰器
    """

    def decorator(fn):
        wrapped_fn = torch.no_grad()(fn)
        ADV_ESTIMATOR_MAP[getattr(name, "value", name)] = wrapped_fn
        return wrapped_fn

    return decorator


def compute_advantage_return(name: AdvantageEstimator, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """计算优势和返回值
    
    功能: 根据算法名称调用对应的优势估计函数（统一入口）
    
    输入:
        name: 算法名称（如 "grpo", "gae" 等）
        **kwargs: 对应算法的参数
    
    输出:
        advantages: shape: (bs, response_length)
        returns: shape: (bs, response_length)
    """
    return ADV_ESTIMATOR_MAP[getattr(name, "value", name)](**kwargs)


@register_adv_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:   ## ad和return一起返回！！！
    """GAE (广义优势估计)
    
    功能: 使用 Critic 网络的价值估计，通过 TD-λ 方法计算优势
    特点: 需要额外的 Critic 网络，计算更稳定但增加显存开销
    公式: A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
    
    输入:
        token_level_rewards: shape: (bs, response_length)
        values: shape: (bs, response_length)
        response_mask: shape: (bs, response_length)
        gamma: 折扣因子
        lam: GAE lambda 参数
    
    输出:
        advantages: shape: (bs, response_length)
        returns: shape: (bs, response_length)
    
    参考: https://arxiv.org/abs/1506.02438
    """
    nextvalues = 0
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        gaelam = delta + gamma * lam * lastgaelam

        if response_mask[:, t]:  # skip values and TD-error on observation tokens
            nextvalues = values[:, t]
            lastgaelam = gaelam

        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)  
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO 优势估计
    
    功能: 对每个问题的多个回答（group）计算相对优势，无需 Critic 网络
    原理: 同一问题的不同回答互相比较，用组内均值和标准差做标准化
    公式: advantage = (score - group_mean) / (group_std + eps)
    
    优势: 不需要训练 Critic，节省显存，适合大模型 RL
    
    输入:
        token_level_rewards: shape: (bs, response_length)
        response_mask: shape: (bs, response_length)
        index: shape: (bs,) - 每个样本属于哪个 group（问题）
        eps: 防止除零的小数
    
    输出:
        advantages: shape: (bs, response_length)
        returns: shape: (bs, response_length)
    
    注意: rollout.n 必须 > 1（每个问题需要多个样本）
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO-Pass@k 优势估计
    
    功能: 每组只有最高分样本获得非零优势，鼓励产生"明显更好"的回答    有每组最好的那个样本获得优势，其他样本优势为0 -- 赢家通吃
    公式: advantage = (r_max - r_second_max) / group_std  这里是最好的样本的优势公式
    
    适用: 需要高质量采样、过滤低质量回答的场景
    
    输入:
        token_level_rewards: shape: (bs, response_length)
        response_mask: shape: (bs, response_length)
        index: shape: (bs,)
        eps: 防止除零的小数
    
    输出:
        advantages: shape: (bs, response_length)
        returns: shape: (bs, response_length)
    
    参考: https://arxiv.org/abs/2503.19595
    """
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)
    id2score = defaultdict(list)
    id2indices = defaultdict(list)

    bsz = scores.shape[0]  ## 这里做初始化了
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
        id2indices[index[i]].append(i)

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        rewards = torch.tensor(id2score[idx])
        topk, topk_idx = torch.topk(rewards, k=2)
        r_max, r_second_max = topk[0], topk[1]
        i_max = id2indices[idx][topk_idx[0]]
        advantages[i_max] = (r_max - r_second_max) / (torch.std(torch.tensor(id2score[idx])) + eps)

    returns = advantages.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """RLOO (Leave-One-Out) 优势估计
    
    功能: 用组内其他样本的均值作为 baseline，减少方差
    公式: baseline = (group_sum - current_score) / (n - 1)
          advantage = current_score - baseline
    
    原理: 排除当前样本自身的影响，更公平的比较
    
    输入:
        token_level_rewards: shape: (bs, response_length)
        response_mask: shape: (bs, response_length)
        index: shape: (bs,)
    
    输出:
        advantages: shape: (bs, response_length)
        returns: shape: (bs, response_length)
    
    参考: https://arxiv.org/abs/2402.14740
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """REINFORCE++ 优势估计
    
    功能: 从后往前累加折扣奖励，配合 reward mask 处理序列结束
    原理: 蒙特卡洛方法估计回报，从每个位置计算未来折扣奖励总和
    公式: G_t = r_t + γ * G_{t+1}
    
    输入:
        token_level_rewards: shape: (bs, response_length)
        response_mask: shape: (bs, response_length)
        gamma: 折扣因子
    
    输出:
        advantages: shape: (bs, response_length)
        returns: shape: (bs, response_length)
    
    参考: https://arxiv.org/abs/2501.03262
    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """ReMax 优势估计
    
    功能: 用贪婪解码（temperature=0）的输出奖励作为 baseline
    公式: advantage = reward_sample - reward_greedy
    
    优势: 比 GRPO 样本效率更高（每个问题只需一个采样 + 一个贪婪输出）
    
    输入:
        token_level_rewards: shape: (bs, response_length)
        reward_baselines: shape: (bs,) - 贪婪解码的奖励
        response_mask: shape: (bs, response_length)
    
    输出:
        advantages: shape: (bs, response_length)
        returns: shape: (bs, response_length)
    
    参考: https://arxiv.org/abs/2310.10505
    """
    advantages = (token_level_rewards.sum(dim=-1) - reward_baselines) * response_mask
    returns = (token_level_rewards * response_mask).flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    return advantages, returns

## 以上都是计算优势和回报的函数，下面是计算损失和 KL 的函数

def compute_rewards(  ## 最终的奖励 = 预估奖励+kl惩罚
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    """计算最终奖励
    
    功能: 在原始奖励基础上加入 KL 正则化，防止策略偏离参考模型太远
    公式: reward = token_level_scores - kl_ratio * KL(log_probs, ref_log_probs)
    
    输入:
        token_level_scores: shape: (bs, response_length) - 奖励函数给出的原始分数
        log_probs: shape: (bs, response_length) - 当前策略的对数概率
        ref_log_probs: shape: (bs, response_length) - 参考模型的对数概率
        kl_ratio: KL 惩罚系数
    
    输出:
        rewards: shape: (bs, response_length)
    """
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def average_loss(  ## 支持俩种模式!!!
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    """损失平均计算
    
    功能: 支持两种平均模式，影响梯度更新的尺度
    
    模式:
        "token": 整个 batch 所有 token 一起平均（默认）
        "seq": 先对每个序列平均，再对序列平均（更公平处理长短序列）
    
    输入:
        values: shape: (bs, response_length) - 待平均的损失值
        mask: shape: (bs, response_length) - 有效位置掩码
        mode: 平均模式 ("token" 或 "seq")
        eps: 防止除零的小数
    
    输出:
        loss: 标量
    """
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


## policy loss 计算
def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    tau_positive: float,
    tau_negative: float,
    loss_type: Literal["default", "gspo", "gppo_token", "cispo", "sapo"],
    loss_avg_mode: Literal["token", "seq"],
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """策略梯度损失计算
    
    功能: 计算带裁剪的策略梯度损失，支持多种 RL 算法变种
    
    支持的 loss_type:
        "default": 标准 PPO（带 dual clip） grpo也能用
        "gspo"/"gspo_token": GSPO 算法（序列级重要性采样）
        "cispo": CISPO 算法（置信度加权）
        "sapo": SAPO 算法（动态门控裁剪）
    
    核心公式:
        ratio = exp(log_probs - old_log_probs)
        clipped_ratio = clip(ratio, 1-clip_low, 1+clip_high)
        loss = -min(ratio * A, clipped_ratio * A)
    
    输入:
        old_log_probs: shape: (bs, response_length)
        log_probs: shape: (bs, response_length)
        advantages: shape: (bs, response_length)
        response_mask: shape: (bs, response_length)
        clip_ratio_low: 下裁剪范围
        clip_ratio_high: 上裁剪范围
        clip_ratio_dual: dual clip 范围
        tau_positive: SAPO 正样本温度
        tau_negative: SAPO 负样本温度
        loss_type: 损失类型
        loss_avg_mode: 平均模式 ("token" 或 "seq")
    
    输出:
        pg_loss: 标量 - 策略梯度损失
        metrics: dict - 包含 ppo_kl, entropy_loss, pg_clipfrac 等
    
    参考: https://arxiv.org/abs/1707.06347 (PPO)
    """
    negative_approx_kl = log_probs - old_log_probs
    if loss_type in ["gspo", "gspo_token"]:
        # compute sequence-level importance ratio
        negative_approx_kl_in_seq = VF.masked_mean(negative_approx_kl, response_mask, dim=-1)
        # combined ratio at token level
        if loss_type == "gspo_token":
            log_importance_ratio = negative_approx_kl_in_seq.detach().unsqueeze(-1) + log_probs - log_probs.detach()
        else:
            log_importance_ratio = negative_approx_kl_in_seq.unsqueeze(-1) * response_mask
    else:
        log_importance_ratio = negative_approx_kl

    # clamp the ratio before exp to avoid nan grad
    # see: https://github.com/pytorch/pytorch/issues/10729
    ratio = torch.exp(torch.clamp(log_importance_ratio, -20.0, 20.0))
    clipped_ratio = torch.exp(
        torch.clamp(log_importance_ratio, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    # pg metrics
    metrics = {"ppo_kl": -negative_approx_kl}
    # use negative log probs as an estimator of entropy loss
    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)

    if loss_type == "cispo":
        final_pg_loss = -advantages * log_probs * clipped_ratio.detach()
    elif loss_type == "sapo":
        positive_token_mask =  (advantages >= 0).float()
        negative_token_mask =  (advantages < 0).float()
        gate_negative = 4.0 / tau_negative * torch.sigmoid(tau_negative * (ratio - 1.0))
        gate_positive = 4.0 / tau_positive * torch.sigmoid(tau_positive * (ratio - 1.0))
        final_pg_loss = -advantages * (positive_token_mask * gate_positive + negative_token_mask * gate_negative)
    else:
        pg_loss = -advantages * ratio  # -ratio * A
        pg_loss2 = -advantages * clipped_ratio  # -clip(ratio, 1-clip_low, 1+clip_high) * A
        pg_loss3 = -advantages * clip_ratio_dual  # -clip_dual * A

        clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
        metrics["pg_clipfrac_higher"] = (pg_loss < pg_loss2).float()
        clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
        final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
        metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}
    return final_pg_loss, metrics


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> tuple[torch.Tensor, dict[str, float]]:
    """价值网络损失计算
    
    功能: 训练 Critic 网络准确估计状态价值（用于需要 Critic 的算法如 PPO+GAE）
    原理: 最小化预测值与回报之间的差距，使用裁剪防止更新过大
    
    公式:
        v_clipped = clip(vpreds, values-cliprange, values+cliprange)
        loss = 0.5 * max((vpreds - returns)^2, (v_clipped - returns)^2)
    
    输入:
        vpreds: shape: (bs, response_length) - Critic 预测的值
        returns: shape: (bs, response_length) - 真实回报
        values: shape: (bs, response_length) - 旧值（用于裁剪）
        response_mask: shape: (bs, response_length)
        cliprange_value: 值网络裁剪范围
        loss_avg_mode: 平均模式
    
    输出:
        vf_loss: 标量 - 价值函数损失
        metrics: dict - 包含 vf_clipfrac, vpred_mean
    
    注意: GRPO 等算法不需要此损失（无 Critic 网络）
    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)  # clip if vf_loss1 < vf_loss2
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)  ## 这里也支持两种平均模式
    metrics = {
        "vf_clipfrac": VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item(),
        "vpred_mean": VF.masked_mean(vpreds, response_mask).detach().item(),
    }
    return vf_loss, metrics


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
) -> torch.Tensor:
    """KL 散度计算
    
    功能: 计算当前策略与参考模型之间的 KL 散度
    
    支持的计算方式:
        "kl": 简单对数差 log_p - log_p_ref
        "abs": 绝对值 |log_p - log_p_ref|
        "mse": 均方误差 0.5 * (log_diff)^2
        "low_var_kl": 低方差近似（推荐，数值稳定）
        "full": 完整 KL 散度公式
    
    输入:
        log_probs: shape: (bs, response_length) - 当前策略对数概率
        ref_log_probs: shape: (bs, response_length) - 参考模型对数概率
        kl_penalty: 计算方式 ("kl", "abs", "mse", "low_var_kl", "full")
    
    输出:
        kl_div: shape: (bs, response_length)
    
    参考: http://joschu.net/blog/kl-approx.html (low_var_kl)
    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        # For numerical stability
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
