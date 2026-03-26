# Worker (Driver+TP+FSDP) 工作流详解

> 本文档详细解释 Driver → Worker 的完整数据流程，包括推理（生成，使用 TP）和训练（前向/反向传播，使用 FSDP）两个阶段。

---

## 一、场景配置

```
硬件配置：
  - 2 台机器
  - 每台机器 8 张 GPU
  - 共 16 张 GPU (world_size = 16)

训练配置：
  - rollout_batch_size = 512           # 目标问题数
  - mini_rollout_batch_size = 64       # 每次从 DataLoader 取的问题数
  - rollout.n = 5                      # 每个问题生成几个回答
  - tensor_parallel_size = 2           # vLLM TP 大小（生成阶段）
  - global_batch_size_per_device = 32  # 训练时，mini_batch 大小
  - micro_batch_size_per_device_for_update = 4  # 训练时，micro_batch 大小

计算：
  - DP 组数 = world_size / tensor_parallel_size = 16 / 2 = 8 个 DP 组 -- 也叫TP组！！！
  - 每个 DP 组包含 2 个 GPU（TP 组）
```

---

## 二、GPU 分组方式

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           GPU 分组（TP=2, world_size=16）                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

共 16 个 GPU，TP=2，所以有 8 个 TP 组：

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│  TP 组 0:  [GPU 0,  GPU 1]   ←── 2 个 GPU 共同处理一部分数据                         │
│  TP 组 1:  [GPU 2,  GPU 3]                                                         │
│  TP 组 2:  [GPU 4,  GPU 5]                                                         │
│  TP 组 3:  [GPU 6,  GPU 7]                                                         │
│  TP 组 4:  [GPU 8,  GPU 9]                                                         │
│  TP 组 5:  [GPU 10, GPU 11]                                                        │
│  TP 组 6:  [GPU 12, GPU 13]                                                        │
│  TP 组 7:  [GPU 14, GPU 15]                                                        │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

每个 TP 组内：
  - 两个 GPU 存不同的权重分片（各存 1/2）
  - 共同处理相同的数据
  - 通过 NCCL 通信协作计算

不同 TP 组之间：
  - 处理不同的数据（数据并行）
  - 独立计算，最后收集结果
```

---

## 三、Batch 层级关系

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           Batch 层级关系（重要！）                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

【生成阶段】使用 TP：

  rollout_batch_size (512)
       │
       │ 循环累积，每次取 mini_rollout_batch_size
       ▼
  mini_rollout_batch_size (64)  ─→  DataLoader 每次取的数量
       │
       │ Driver 切分: chunk(world_size=16)
       │ 每个 GPU 分到 4 个样本
       ▼
  每个 GPU: 4 个样本
       │
       │ TP All-Gather: TP 组内共享数据
       │ 每个 TP 组有 8 个样本
       ▼
  vLLM 生成（TP 组内协作）
       │
       │ TP Chunk: 每个 GPU 取回自己那部分结果
       ▼
  生成完成


【训练阶段】使用 FSDP：

  总样本数 (512 × 5 = 2560)
       │
       │ Driver 切分: chunk(world_size=16)
       ▼
  每个 GPU: 160 个样本
       │
       │ 切分成 mini_batch (32)
       │ 160 / 32 = 5 个 mini_batch
       ▼
  mini_batch (32 样本)
       │
       │ 切分成 micro_batch (4)
       │ 32 / 4 = 8 个 micro_batch
       ▼
  micro_batch (4 样本)
       │
       │ FSDP All-Gather: 收集完整权重
       │ 前向传播
       │ 反向传播
       │ FSDP Reduce-Scatter: 同步梯度
       ▼
  梯度累积，更新参数
```

---

## 四、完整流程图：生成阶段（使用 TP）

### 5.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           生成阶段完整流程（使用 TP）                                  │
│                           特点：TP 组内协作，没有 micro_batch                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

配置：
  rollout_batch_size = 512          # 目标：512 个问题
  mini_rollout_batch_size = 64      # 每次：64 个问题
  world_size = 16                   # 16 个 GPU
  tensor_parallel_size = 2          # TP 大小
  n = 5                             # 每个问题生成 5 个回答

DP 组数 = 16 / 2 = 8 个
循环次数 = 512 / 64 = 8 轮
最终样本数 = 512 × 5 = 2560 个样本


=======================================================================================
【循环第 1 轮】
=======================================================================================

【位置：Driver 进程】
─────────────────────────────────────────────────────────────────────────────────────

Step 1: DataLoader 取数据
─────────────────────────
    
    batch_dict = next(self.data_iterator)  # 64 个问题

Step 2: 构造 DataProto
─────────────────────────

    new_batch = DataProto.from_single_dict(batch_dict)
    
    # 为每个样本生成唯一 uid（用于 GRPO 分组）
    new_batch.non_tensor_batch["uid"] = [uuid.uuid4() for _ in range(64)]

Step 3: pop 出生成需要的字段
─────────────────────────────

    gen_batch = new_batch.pop(
        batch_keys=["input_ids", "attention_mask", "position_ids"],
        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
    )

Step 4: Driver 切分数据并发送给各 GPU
─────────────────────────────────────────

    gen_batch.chunk(chunks=16)   # world_size = 16  -- 这个是在driver里就分好的 然后通过ray远程调用发数据给其他机器的卡上！！！
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  把 64 个样本切成 16 份，每份 4 个样本:                                        │
    │                                                                              │
    │  chunk[0]:  4 样本  →  GPU 0  (TP 组 0)                                      │
    │  chunk[1]:  4 样本  →  GPU 1  (TP 组 0)                                      │
    │  chunk[2]:  4 样本  →  GPU 2  (TP 组 1)                                      │
    │  chunk[3]:  4 样本  →  GPU 3  (TP 组 1)                                      │
    │  ...                                                                         │
    │  chunk[14]: 4 样本  →  GPU 14 (TP 组 7)                                      │
    │  chunk[15]: 4 样本  →  GPU 15 (TP 组 7)                                      │
    └─────────────────────────────────────────────────────────────────────────────┘
    
                                    │
                                    │ Ray.remote() 并行发送到各 GPU
                                    ▼

【位置：各 GPU Worker】
─────────────────────────────────────────────────────────────────────────────────────

Step 5: Sharding Manager 预处理（TP All-Gather）
─────────────────────────────────────────────────────

    preprocess_data(data):
        all_gather_data_proto(data, size=tp_size, group=tp_group)
    
    作用：让 TP 组内每个 GPU 有相同的数据
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  TP 组 0 (GPU 0, GPU 1):                                                     │
    │                                                                              │
    │  GPU 0 原本有: chunk[0] = 4 samples                                          │
    │  GPU 1 原本有: chunk[1] = 4 samples                                          │
    │                                                                              │
    │  All-Gather 后:                                                              │
    │  GPU 0 有: [chunk[0], chunk[1]] = 8 samples                                  │
    │  GPU 1 有: [chunk[0], chunk[1]] = 8 samples                                  │
    │                                                                              │
    │  ⚠️ TP 组内每个 GPU 都有相同的 8 个样本！                                      │
    │                                                                              │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  TP 组 1 (GPU 2, GPU 3):                                                     │
    │  GPU 2, GPU 3 都有: [chunk[2], chunk[3]] = 8 samples                         │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    ... 其他 TP 组同理 ...


Step 6: vLLM 生成（TP 组内协作）
─────────────────────────────────

    每个 TP 组内的 GPU 共同处理数据：
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  TP 组 0 (GPU 0, GPU 1) 处理 8 个样本:                                        │
    │                                                                              │
    │  ┌───────────────────────────────────────────────────────────────────────┐  │
    │  │  权重存储（TP=2）:                                                      │  │
    │  │                                                                         │  │
    │  │    7B 模型权重总量 ≈ 14GB                                               │  │
    │  │                                                                         │  │
    │  │    GPU 0: 存 1/2 权重 ≈ 7GB (W_0 部分)                                  │  │
    │  │    GPU 1: 存 1/2 权重 ≈ 7GB (W_1 部分)                                  │  │
    │  │                                                                         │  │
    │  │  ⚠️ 每个 GPU 只存一半权重，节省显存！                                    │  │
    │  └───────────────────────────────────────────────────────────────────────┘  │
    │                                                                              │
    │  ┌───────────────────────────────────────────────────────────────────────┐  │
    │  │  生成过程（TP 协作）:                                                    │  │
    │  │                                                                         │  │
    │  │    输入: 8 个样本                                                       │  │
    │  │                                                                         │  │
    │  │    for each layer:                                                      │  │
    │  │        # 每个 GPU 用自己的权重分片计算                                   │  │
    │  │        GPU 0: hidden_0 = input × W_0                                   │  │
    │  │        GPU 1: hidden_1 = input × W_1                                   │  │
    │  │                                                                         │  │
    │  │        # All-Reduce 同步结果                                            │  │
    │  │        All-Reduce(hidden_0, hidden_1) → hidden                          │  │
    │  │                                                                         │  │
    │  │    输出: 8 × 5 = 40 个回答                                               │  │
    │  │                                                                         │  │
    │  │  ⚠️ GPU 0 和 GPU 1 都有相同的 40 个回答                                  │  │
    │  └───────────────────────────────────────────────────────────────────────┘  │
    │                                                                              │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    同理，其他 7 个 TP 组也各自生成 40 个回答


Step 7: Sharding Manager 后处理（TP Chunk）
─────────────────────────────────────────────

    postprocess_data(data):
        data.chunk(chunks=tp_size)[tp_rank]
    
    作用：每个 GPU 只取回自己那部分结果
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  TP 组 0 (GPU 0, GPU 1):                                                     │
    │                                                                              │
    │  GPU 0 和 GPU 1 都有: 40 个回答                                               │
    │                                                                              │
    │  Chunk 后:                                                                   │
    │    GPU 0 取: chunk[0] = 20 个回答                                            │
    │    GPU 1 取: chunk[1] = 20 个回答                                            │
    │                                                                              │
    │  每个 GPU 返回自己那 20 个回答给 Driver                                        │
    └─────────────────────────────────────────────────────────────────────────────┘


【位置：Driver 进程】
─────────────────────────────────────────────────────────────────────────────────────

Step 8: 收集并拼接结果
─────────────────────────

    DataProto.concat([output_0, output_1, ..., output_15])
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  GPU 0:  20 个回答                                                           │
    │  GPU 1:  20 个回答                                                           │
    │  GPU 2:  20 个回答                                                           │
    │  ...                                                                         │
    │  GPU 15: 20 个回答                                                           │
    │                                                                              │
    │  总计: 16 × 20 = 320 个回答 (64 问题 × 5 回答)                                │
    └─────────────────────────────────────────────────────────────────────────────┘

Step 9: 对齐原始数据和生成结果
─────────────────────────────

    new_batch = new_batch.repeat(repeat_times=5, interleave=True)
    new_batch = new_batch.union(gen_batch_output)

Step 10: 累积 batch
─────────────────────────

    batch = DataProto.concat([batch, new_batch])
    
    if current_batch_size < rollout_batch_size (512):
        继续循环 Step 1-10
    else:
        返回 batch


=======================================================================================
【最终结果】
=======================================================================================

    512 个问题 × 5 个回答 = 2560 个样本
```

### 5.2 TP 关键代码

```python
# verl/workers/sharding_manager/fsdp_vllm.py:44-62
class FSDPVLLMShardingManager:
    def __init__(self, ...):
        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()  # TP 大小
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()        # TP rank
        self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group

# verl/workers/sharding_manager/fsdp_vllm.py:217-226
def preprocess_data(self, data: DataProto) -> DataProto:
    """All gather across tp group to make each rank has identical input."""
    all_gather_data_proto(data, size=self.tp_size, group=self.tp_group)
    return data

def postprocess_data(self, data: DataProto) -> DataProto:
    """Get chunk data of this tp rank since we do all gather in preprocess."""
    if self.tp_size > 1:
        data = data.chunk(chunks=self.tp_size)[self.tp_rank]
    return data
```

---

## 五、完整流程图：训练阶段（使用 FSDP）

### 6.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           训练阶段完整流程（使用 FSDP）                                │
│                           特点：有 mini_batch 和 micro_batch                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

输入：
  - 总样本数: 2560 (512 问题 × 5 回答)
  - 已计算好: rewards, advantages, old_log_probs, ref_log_probs

配置：
  world_size = 16
  global_batch_size_per_device = 32    # mini_batch 大小
  micro_batch_size_per_device_for_update = 4  # micro_batch 大小


=======================================================================================
【Step 1: Driver 切分数据并发送给 Worker】
=======================================================================================

    batch (2560 samples)
         │
         │ batch.chunk(chunks=16)
         │
         ▼
    每个 GPU 分到: 2560 / 16 = 160 个样本


=======================================================================================
【Step 2: Worker 内部切分 mini_batch 和 micro_batch】
=======================================================================================

    收到: 160 samples
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Step 2.1: 切分 mini_batch                                                  │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │      mini_batches = data.split(global_batch_size_per_device=32)             │
    │      160 / 32 = 5 个 mini_batch                                             │
    │                                                                              │
    │  Step 2.2: 每个 mini_batch 切分 micro_batch                                 │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │      micro_batches = mini_batch.split(micro_batch_size=4)                   │
    │      32 / 4 = 8 个 micro_batch                                              │
    └─────────────────────────────────────────────────────────────────────────────┘


=======================================================================================
【Step 3: 训练循环（前向 + 反向传播）】
=======================================================================================

    for mini_batch in mini_batches:               # 5 个 mini_batch
        │
        ▼
    for micro_batch in micro_batches:             # 8 个 micro_batch
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                              │
    │  Step 3.1: FSDP All-Gather 收集完整权重                                      │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │                                                                              │
    │  ┌───────────────────────────────────────────────────────────────────────┐  │
    │  │  权重存储（FSDP）:                                                      │  │
    │  │                                                                         │  │
    │  │    7B 模型权重总量 ≈ 14GB                                               │  │
    │  │                                                                         │  │
    │  │    GPU 0:  存 1/16 权重 ≈ 0.875GB (shard_0)                            │  │
    │  │    GPU 1:  存 1/16 权重 ≈ 0.875GB (shard_1)                            │  │
    │  │    ...                                                                  │  │
    │  │    GPU 15: 存 1/16 权重 ≈ 0.875GB (shard_15)                           │  │
    │  │                                                                         │  │
    │  │  ⚠️ 每个 GPU 只存 1/16 权重，节省显存！                                  │  │
    │  └───────────────────────────────────────────────────────────────────────┘  │
    │                                                                              │
    │  ┌───────────────────────────────────────────────────────────────────────┐  │
    │  │  All-Gather 过程:                                                       │  │
    │  │                                                                         │  │
    │  │    GPU 0: [shard_0]                                                     │  │
    │  │         ↓ 发送 shard_0，接收 shard_1~15                                 │  │
    │  │    GPU 0: [shard_0, shard_1, ..., shard_15] = 完整权重 (14GB)          │  │
    │  │                                                                         │  │
    │  │  ⚠️ 此时所有 GPU 都有完整权重，显存占用最大！                            │  │
    │  └───────────────────────────────────────────────────────────────────────┘  │
    │                                                                              │
    │  Step 3.2: 前向传播                                                         │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │      log_probs = forward(完整权重, micro_batch)  # 4 样本                   │
    │                                                                              │
    │  Step 3.3: 计算 PPO Loss                                                    │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │      loss = compute_policy_loss(log_probs, old_log_probs, advantages, ...)  │
    │                                                                              │
    │  Step 3.4: 反向传播                                                         │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │      loss.backward()                                                        │
    │                                                                              │
    │  Step 3.5: FSDP Reduce-Scatter 同步梯度                                     │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │                                                                              │
    │  ┌───────────────────────────────────────────────────────────────────────┐  │
    │  │  Reduce-Scatter 过程:                                                   │  │
    │  │                                                                         │  │
    │  │    GPU 0: [完整梯度 14GB]                                                │  │
    │  │         ↓ 只保留 shard_0 对应的梯度                                      │  │
    │  │    GPU 0: [grad_shard_0] ≈ 0.875GB                                      │  │
    │  │                                                                         │  │
    │  │  ⚠️ 每个 GPU 只保留自己那部分权重的梯度                                   │  │
    │  └───────────────────────────────────────────────────────────────────────┘  │
    │                                                                              │
    │  Step 3.6: 累积梯度（不更新参数）                                            │
    │                                                                              │
    └─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
    所有 8 个 micro_batch 处理完毕，梯度已累积
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Step 4: 梯度裁剪 + 优化器更新                                                │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │      clip_grad_norm_(parameters, max_norm=1.0)                              │
    │      optimizer.step()                                                       │
    │      optimizer.zero_grad()                                                  │
    └─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 FSDP 关键代码

```python
# verl/workers/actor/dp_actor.py:219-297
def update_policy(self, data: DataProto) -> dict[str, Any]:
    
    # Step 1: 切分 mini_batch
    mini_batches = data.select(...).split(self.config.global_batch_size_per_device)
    
    for _ in range(self.config.ppo_epochs):
        for mini_batch in mini_batches:
            
            # Step 2: 切分 micro_batch
            micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
            
            for micro_batch in micro_batches:
                # Step 3: 前向传播（FSDP 自动 All-Gather）
                log_probs = self._forward_micro_batch(model_inputs, temperature)
                
                # Step 4: 计算 Loss
                loss = compute_policy_loss(...)
                
                # Step 5: 反向传播（FSDP 自动 Reduce-Scatter）
                loss.backward()
            
            # Step 6: 梯度裁剪 + 优化器更新
            grad_norm = self._optimizer_step()
    
    return metrics
```

---

## 六、总结：生成 vs 训练

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           生成阶段 vs 训练阶段对比                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

                          │  生成阶段（TP）            │  训练阶段（FSDP）
────────────────────────┼───────────────────────────┼───────────────────────────────
并行方式                 │  Tensor Parallel          │  Data Parallel
────────────────────────┼───────────────────────────┼───────────────────────────────
权重存储                 │  每个 GPU 存 1/TP_size    │  每个 GPU 存 1/world_size
────────────────────────┼───────────────────────────┼───────────────────────────────
数据分配                 │  TP 组内相同数据          │  每个 GPU 不同数据
────────────────────────┼───────────────────────────┼───────────────────────────────
通信方式                 │  All-Reduce（每层）       │  All-Gather + Reduce-Scatter
────────────────────────┼───────────────────────────┼───────────────────────────────
是否有 mini_batch        │  无                       │  有（global_batch_size）
────────────────────────┼───────────────────────────┼───────────────────────────────
是否有 micro_batch       │  无                       │  有（micro_batch_size）
────────────────────────┼───────────────────────────┼───────────────────────────────
是否需要梯度             │  否                       │  是
────────────────────────┼───────────────────────────┼───────────────────────────────
引擎                     │  vLLM                     │  PyTorch FSDP


为什么训练阶段需要 micro_batch？
─────────────────────────────────────────────────────────────────────────────────────
  - FSDP All-Gather 后，每个 GPU 都有完整权重（14GB）
  - 激活值 + 梯度 也需要显存
  - micro_batch 小一些，可以减少激活值显存
  - 多个 micro_batch 梯度累积后，一次更新参数
```

---

## 七、关键代码位置汇总

| 功能 | 文件 | 说明 |
|------|------|------|
| DataLoader 创建 | `verl/trainer/data_loader.py:58-66` | 创建 train_dataloader |
| 数据切分 | `verl/protocol.py:546-572` | `DataProto.chunk()` |
| 数据拼接 | `verl/protocol.py:589-606` | `DataProto.concat()` |
| Ray 远程调用 | `verl/single_controller/ray/base.py:372-389` | `execute_all_async()` |
| 生成入口 | `verl/trainer/ray_trainer.py:641-787` | `_make_batch_data()` |
| TP 预处理 | `verl/workers/sharding_manager/fsdp_vllm.py:217-220` | `preprocess_data()` |
| TP 后处理 | `verl/workers/sharding_manager/fsdp_vllm.py:222-226` | `postprocess_data()` |
| vLLM 生成 | `verl/workers/rollout/vllm_rollout_spmd.py:181-277` | `generate_sequences()` |
| 训练入口 | `verl/workers/actor/dp_actor.py:219-297` | `update_policy()` |
| mini/micro batch 切分 | `verl/workers/actor/dp_actor.py:229,245` | `data.split()` |