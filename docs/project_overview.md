# EasyR1 项目概述

**EasyR1** 是一个高效、可扩展的**多模态强化学习（RL）训练框架**，基于字节跳动 [veRL](https://github.com/volcengine/verl) 开发，专注于视觉语言模型（VLM）的 RL 训练。

---

## 一句话理解

> 用 **GRPO** 等 RL 算法，让多模态模型（如 Qwen-VL）通过"试错+奖励"自我提升推理能力，**不需要标准答案**，只需要能判断对错的奖励函数。

---

## 能做什么？

| 场景 | 输入 | 输出 | 示例 |
|-----|-----|-----|-----|
| **视觉数学** | 几何图形 + 问题 | 推理过程 + 答案 | 看图计算角度 |
| **GUI 自动化** | 手机截图 + 指令 | 点击/滑动操作 | 自动完成 App 任务 |
| **多模态问答** | 多图 + 问题 | 综合回答 | 旅游攻略规划 |
| **纯文本推理** | 数学题文字 | 分步解答 | 代数方程求解 |

---

## 技术能力

### 支持的模型

| 类型 | 模型系列 | 说明 |
|-----|---------|-----|
| 语言模型 | Llama3、Qwen2/2.5/3 | 纯文本 RL 训练 |
| 视觉语言模型 | Qwen2-VL、Qwen2.5-VL、Qwen3-VL | **主打**，支持图文输入 |
| 蒸馏模型 | DeepSeek-R1 distill | 已有推理能力的模型 |

### 支持的 RL 算法

| 算法 | 特点 | 推荐度 |
|-----|-----|-------|
| **GRPO** | 基线算法，最稳定 | ⭐⭐⭐ 入门首选 |
| **DAPO** | 动态采样，效果提升 | ⭐⭐⭐ 进阶 |
| **SAPO** | 目前 Geometry3k 最佳效果 | ⭐⭐⭐ 追求效果 |
| GSPO | 无 KL 惩罚 | ⭐⭐ 特定场景 |
| CISPO | 置信度加权 | ⭐⭐ 特定场景 |

---

## 核心概念：多模态 vs 纯文本

**Q: Qwen2.5-VL（多模态模型）能做纯文本任务吗？**

**A: 能。** 多模态模型兼容纯文本，反之不行。

| 能力 | Qwen2.5-7B（纯文本） | Qwen2.5-VL-7B（多模态） |
|-----|---------------------|------------------------|
| 数学题（文字） | ✅ | ✅ |
| 几何题（看图） | ❌ | ✅ |
| **建议** | - | **直接用这个，兼顾两者** |

---

## 官方 Baseline 数据集

EasyR1 在这些数据集上验证了效果，你可以直接复现：

### 核心数据集（推荐学习）

| 数据集 | 类型 | 任务描述 | 典型效果 |
|-------|-----|---------|---------|
| **Math12k** | 纯文本 | 数学推理（代数、应用题） | 75% → **77%** |
| **Geometry3k** | 多模态 | 看几何图形计算角度/边长 | 37% → **48%** |

### 其他数据集

| 数据集 | 类型 | 任务描述 | 来源 |
|-------|-----|---------|-----|
| CLEVR-70k | 多模态 | 合成图像物体计数 | R1-V 复现 |
| GeoQA-8k | 多模态 | 地理空间推理 | R1-V 复现 |
| DAPO-17k | 纯文本 | 高难度数学题 | DAPO 论文 |
| Android GUI | 多模态 | 手机界面自动化 | GUI Agent |

---

## 学习路径：重点脚本与文件

如果你是初学者，建议按以下顺序阅读和运行：

### 第一阶段：跑通 Baseline（必做）

这两步帮你建立对 RL 训练的直观认知：

| 顺序 | 脚本 | 数据集 | 目的 |
|-----|-----|-------|-----|
| 1 | `examples/qwen2_5_7b_math_grpo.sh` | Math12k | **理解纯文本 RL 流程**，最简单 |
| 2 | `examples/qwen2_5_vl_7b_geo3k_grpo.sh` | Geometry3k | **理解多模态输入**，官方主打 benchmark |

**运行方式**：
```bash
# 1. 纯文本入门
bash examples/qwen2_5_7b_math_grpo.sh

# 2. 多模态进阶
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

---

### 第二阶段：深入理解核心机制（必读源码）

跑通 baseline 后，重点看这些文件理解原理：

| 文件路径 | 核心内容 | 阅读重点 |
|---------|---------|---------|
| `examples/config.yaml` | **所有配置参数** | 了解每个参数的作用和默认值 |
| `examples/reward_function/math.py` | **奖励函数实现** | 如何给模型回答打分（格式 + 准确率）|
| `examples/format_prompt/math.jinja` | **提示词模板** | 模型输入的格式定义 |
| `verl/trainer/ray_trainer.py` | **主训练循环** | RL 训练的完整流程 |
| `verl/trainer/core_algos.py` | **GRPO/DAPO 算法** | 算法核心实现 |

**为什么奖励函数最重要？**

```python
# examples/reward_function/math.py 的核心逻辑
def compute_score(reward_inputs):
    # 1. 格式奖励：是否按 <think>...</think>\boxed{答案} 格式输出
    format_score = format_reward(response)
    
    # 2. 准确率奖励：答案是否正确
    accuracy_score = accuracy_reward(response, ground_truth)
    
    # 3. 加权组合
    return 0.9 * accuracy_score + 0.1 * format_score
```

**奖励函数决定了：**
- 模型输出什么格式
- 如何判断对错
- RL 训练的优化目标

---

### 第三阶段：对比不同算法（进阶）

对比这些脚本，理解不同 RL 算法的配置差异：

| 脚本 | 算法 | 对比重点 |
|-----|-----|---------|
| `examples/qwen2_5_vl_7b_geo3k_grpo.sh` | GRPO | 基线配置 |
| `examples/qwen2_5_vl_7b_geo3k_dapo.sh` | DAPO | 看 `algorithm` 部分的差异 |
| `examples/qwen2_5_vl_7b_geo3k_sapo.sh` | SAPO | 目前效果最好的算法 |

**对比方法**：
```bash
# 用 diff 看配置差异
diff examples/qwen2_5_vl_7b_geo3k_grpo.sh examples/qwen2_5_vl_7b_geo3k_dapo.sh
```

主要差异在 `algorithm` 部分的参数，如 `adv_estimator`、`online_filtering` 等。

---

### 第四阶段：资源优化（显存不足时）

| 脚本 | 技术 | 适用场景 |
|-----|-----|---------|
| `examples/qwen3_vl_4b_geo3k_grpo_lora.sh` | LoRA | 显存不足（如 24GB 跑 7B 模型）|

LoRA 可以将显存需求降低 30%~50%，适合资源有限的场景。

---

### 第五阶段：自定义任务（实战）

想训练自己的任务？参考这些：

| 文件 | 参考用途 |
|-----|---------|
| `examples/reward_function/r1v.py` | 视觉计数任务的奖励设计 |
| `examples/reward_function/android_gui.py` | GUI 操作任务的奖励设计 |

**自定义任务三步走：**

1. **准备数据**：按 `hiyouga/geometry3k` 格式组织 JSONL
2. **写奖励函数**：参考 `math.py` 实现 `compute_score` 函数
3. **改配置**：在 `config.yaml` 基础上修改路径和参数

---

## 学习建议：我该跑哪些？

**最少必要学习路径：**

```
Step 1: 跑 qwen2_5_7b_math_grpo.sh（Math12k）
        ↓ 理解 RL 训练流程
Step 2: 跑 qwen2_5_vl_7b_geo3k_grpo.sh（Geometry3k）
        ↓ 理解多模态输入
Step 3: 读 examples/reward_function/math.py
        ↓ 理解奖励函数设计
Step 4: 对比 grpo.sh vs dapo.sh
        ↓ 理解算法差异
Step 5: 尝试自定义数据集
```

**时间有限？只跑这两个：**

| 优先级 | 脚本 | 原因 |
|-------|-----|-----|
| ⭐⭐⭐ | `qwen2_5_vl_7b_geo3k_grpo.sh` | 最核心，涵盖多模态 + RL |
| ⭐⭐⭐ | `examples/reward_function/math.py` | 理解如何定义任务 |

---

## 硬件需求参考

基于 Geometry3k 任务的估算：

| 方法 | 精度 | 1.5B | 7B | 32B | 72B |
|-----|-----|-----|-----|-----|-----|
| GRPO Full | AMP | 2×24GB | 8×40GB | 16×80GB | 32×80GB |
| GRPO Full | BF16 | 1×24GB | 4×40GB | 8×80GB | 16×80GB |
| **GRPO LoRA** | AMP | **1×12GB** | **2×32GB** | **2×80GB** | **4×80GB** |

**建议**：显存不够直接上 LoRA，效果损失不大 ！！！ 只要试跑整个流程就行！！！

---

## 常见问题速答

**Q: 我只有 24GB 显存，能跑什么？**
> 7B 模型用 LoRA，或跑 3B 模型。

**Q: 自定义数据集怎么准备？**
> 参考 `hiyouga/geometry3k` 的格式：JSONL，每行包含 `problem`、`answer`、`images` 字段。

**Q: 奖励函数怎么写？**
> 看 `examples/reward_function/math.py`，核心是定义 `compute_score` 函数，返回 0-1 的分数。

**Q: 训练和 SFT 有什么区别？**
> SFT 是模仿标准答案，RL 是通过奖励信号试错学习。RL 不需要标准答案，只需要能判断对错的规则。

---

## 相关项目

| 项目 | 关系 | 用途 |
|-----|-----|-----|
| [veRL](https://github.com/volcengine/verl) | 上游框架 | EasyR1 基于它开发 |
| [LlamaFactory](https://github.com/hiyouga/LlamaFactory) | 同作者 | SFT 和推理，配合 EasyR1 使用 |
| [R1-V](https://github.com/deep-agent/R1-V) | 先驱项目 | EasyR1 复现了它的 CLEVR/GeoQA baseline |

---

## 总结

EasyR1 是**目前最完善的多模态 RL 训练框架**之一，如果你要：

- **研究 VLM 推理能力提升** → 直接用 Geometry3k 做基准
- **开发 GUI Agent** → 参考 Android GUI 示例
- **快速上手 RL** → 先跑 `qwen2_5_vl_7b_geo3k_grpo.sh`，再读 `math.py`

核心优势：**工程完善、算法齐全、社区活跃**。
