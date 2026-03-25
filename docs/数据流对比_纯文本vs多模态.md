# 数据流对比：纯文本 vs 多模态

> 本文档详细对比 EasyR1 中纯文本数据（如 Math12k）和多模态数据（如 Geometry3k）的处理流程差异，帮助理解 VLM 训练的核心区别。

---

## 一、原始数据格式对比

### 1.1 纯文本数据 (Math12k)

```json
{
  "problem": "What is 2+2?",
  "answer": "4"
}
```

### 1.2 多模态数据 (Geometry3k)

```json
{
  "problem": "What shape is shown in <image>?",
  "answer": "triangle",
  "images": ["/path/to/triangle.jpg"]
}
```

**关键差异**：
- 多模态数据包含 `images` 或 `videos` 字段
- 问题文本中包含 `<image>` 或 `<video>` 占位符

---

## 二、完整数据流对比

### 2.1 纯文本数据流

```

提示词模板让模型学会"先思考再回答"的推理模式，而不是直接给答案！！！ 总结，让模型学会思考！！！
这样奖励函数就能：
1. 检查有没有 ... 标签 → 格式奖励
2. 检查 \boxed{} 里的答案 → 准确性奖励


┌─────────────────────────────────────────────────────────────────────────────────┐
│                          纯文本数据流 (Math12k 为例)                               │
└─────────────────────────────────────────────────────────────────────────────────┘

Step 1: 数据加载 (RLHFDataset)
─────────────────────────────────────────────────────────────────────────────────
    原始数据:
    {"problem": "What is 2+2?", "answer": "4"}
    
    ↓ load_dataset()
    
    Dataset: 12000 条数学题

Step 2: 格式化提示词 (Jinja2 Template)
─────────────────────────────────────────────────────────────────────────────────
    Template (math.jinja):
    "{{ content }} You FIRST think about the reasoning process..."
    
    ↓ Template.render(content=problem)
    
    格式化后:
    "What is 2+2? You FIRST think about the reasoning process as an internal 
     monologue and then provide the final answer. The reasoning process MUST 
     BE enclosed within <think> </think> tags. The final answer MUST BE put 
     in \boxed{}."

Step 3: 构建 Chat 消息
─────────────────────────────────────────────────────────────────────────────────
    messages = [{"role": "user", "content": "What is 2+2? ..."}]
    
    ↓ tokenizer.apply_chat_template(messages, add_generation_prompt=True)

Step 4: Tokenization
─────────────────────────────────────────────────────────────────────────────────
    tokenizer([prompt], return_tensors="pt")
    
    输出:
    ┌─────────────────────────────────────────────────────────────────┐
    │ input_ids:      [151644, 8940, 332, 1688, 332, 267, ...]        │
    │                 shape: (seq_len,)  例如 (128,)                  │
    │                                                                 │
    │ attention_mask: [1, 1, 1, 1, 1, 1, ...]                         │
    │                 shape: (seq_len,)                               │
    │                                                                 │
    │ position_ids:   [0, 1, 2, 3, 4, 5, ...]                         │
    │                 shape: (seq_len,)  ← 一维位置编码               │
    └─────────────────────────────────────────────────────────────────┘

Step 5: Left Padding & 截断
─────────────────────────────────────────────────────────────────────────────────
    postprocess_data(input_ids, attention_mask, position_ids, 
                     max_length=2048, left_pad=True)
    
    输出 (左填充后):
    ┌─────────────────────────────────────────────────────────────────┐
    │ input_ids:      [PAD, PAD, ..., 151644, 8940, 332, ...]         │
    │                 shape: (2048,)                                  │
    │                                                                 │
    │ attention_mask: [0, 0, ..., 1, 1, 1, ...]                       │
    │                 shape: (2048,)                                  │
    │                                                                 │
    │ position_ids:   [0, 0, ..., 0, 1, 2, ...]                       │
    │                 shape: (2048,)                                  │
    └─────────────────────────────────────────────────────────────────┘

Step 6: 生成阶段 (vLLM)
─────────────────────────────────────────────────────────────────────────────────
    vllm_inputs = [{"prompt_token_ids": [151644, 8940, ...]}]
    
    ↓ llm.generate(vllm_inputs, sampling_params)
    
    输出:
    ┌─────────────────────────────────────────────────────────────────┐
    │ response_ids:   [789, 456, 123, ...]                            │
    │                 shape: (response_len,)  例如 (512,)             │
    │                                                                 │
    │ old_log_probs:  [-2.3, -1.5, -0.8, ...]                         │
    │                 shape: (response_len,)                          │
    └─────────────────────────────────────────────────────────────────┘

Step 7: 前向传播 (Actor)
─────────────────────────────────────────────────────────────────────────────────
    model(input_ids, attention_mask, position_ids)
    
    注意: 纯文本只需要 3 个参数！
    
    输出:
    ┌─────────────────────────────────────────────────────────────────┐
    │ logits:         shape: (seq_len, vocab_size)                    │
    │                 例如: (2048, 152000)                            │
    └─────────────────────────────────────────────────────────────────┘

Step 8: 计算 log_probs
─────────────────────────────────────────────────────────────────────────────────
    log_probs = log_probs_from_logits(logits, labels)
    
    输出:
    old_log_probs: shape: (response_len,)

Step 9: PPO 更新
─────────────────────────────────────────────────────────────────────────────────
    loss = compute_policy_loss(old_log_probs, log_probs, advantages)
    loss.backward()
    optimizer.step()
```

---

### 2.2 多模态数据流 (Qwen2-VL)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      多模态数据流 (Geometry3k 为例)                                │
└─────────────────────────────────────────────────────────────────────────────────┘

Step 1: 数据加载 (RLHFDataset)
─────────────────────────────────────────────────────────────────────────────────
    原始数据:
    {
        "problem": "What shape is shown in <image>?",
        "answer": "triangle",
        "images": ["/path/to/triangle.jpg"]
    }
    
    ↓ load_dataset()
    
    Dataset: 3000 条几何题 + 图像

Step 2: 格式化提示词 (Jinja2 Template)
─────────────────────────────────────────────────────────────────────────────────
    同纯文本，但保留 <image> 占位符
    
    格式化后:
    "What shape is shown in <image>? You FIRST think about..."

Step 3: 构建 Chat 消息 (多模态格式)
─────────────────────────────────────────────────────────────────────────────────
    根据 <image> 占位符拆分，构建多模态消息:
    
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "What shape is shown in "},
                {"type": "image"},                                    ← 图像标记
                {"type": "text", "text": "? You FIRST think..."}
            ]
        }
    ]
    
    ↓ processor.apply_chat_template(messages, add_generation_prompt=True)

Step 4: 图像预处理 ⭐ 关键差异
─────────────────────────────────────────────────────────────────────────────────
    process_image(image, min_pixels=262144, max_pixels=4194304)  ## 这里主要是把原图调整到合适的像素/分辨率大小，较少显存压力，加速计算
    
    处理流程:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 加载图像: PIL.Image.open(image_path)                         │
    │                                                                 │
    │ 2. 分辨率调整:                                                   │
    │    if width * height > max_pixels:                              │
    │        resize_factor = sqrt(max_pixels / (width * height))      │
    │        image = image.resize((new_width, new_height))            │
    │                                                                 │
    │    if width * height < min_pixels:                              │
    │        resize_factor = sqrt(min_pixels / (width * height))      │
    │        image = image.resize((new_width, new_height))            │
    │                                                                 │
    │ 3. 格式转换: image.convert("RGB")                               │
    └─────────────────────────────────────────────────────────────────┘
    
    输出:
    PIL.Image: (H, W, 3)  例如 (512, 512, 3)

Step 5: Processor 处理 ⭐ 关键差异
─────────────────────────────────────────────────────────────────────────────────
    processor(images=[PIL.Image], text=[prompt], return_tensors="pt")
    
    输出:
    ┌─────────────────────────────────────────────────────────────────┐
    │ input_ids:      [151644, 8940, <|image_pad|>, ..., <|image_pad|>, ...]   │
    │                 shape: (seq_len,)                                    │
    │                 包含 N 个 <|image_pad|> token 作为图像占位符           │
    │                                                                 │
    │ attention_mask: [1, 1, 1, ..., 1, ...]                          │
    │                 shape: (seq_len,)                               │
    │                                                                 │
    │ pixel_values:   tensor(shape=(num_patches, embed_dim))          │
    │                 例如: (256, 1280)                               │
    │                 图像的 patch embeddings                         │
    │                                                                 │
    │ image_grid_thw: tensor(shape=(1, 3))                            │
    │                 例如: [[1, 28, 28]]                             │
    │                 表示图像的 temporal, height, width 网格         │
    └─────────────────────────────────────────────────────────────────┘

Step 6: MRoPE 位置编码 ⭐ 关键差异 (Qwen2-VL 特有)
─────────────────────────────────────────────────────────────────────────────────
    get_rope_index(processor, input_ids, image_grid_thw)
    
    原理: 多模态需要 3D 位置编码 (时间, 高度, 宽度)
    
    输出:
    ┌─────────────────────────────────────────────────────────────────┐
    │ vision_position_ids: shape: (3, seq_len)                        │
    │                      [[t0, t1, t2, ...],                        │
    │                       [h0, h1, h2, ...],                        │
    │                       [w0, w1, w2, ...]]                        │
    │                                                                 │
    │ text_position_ids:   shape: (1, seq_len)                        │
    │                      [[0, 1, 2, ...]]                           │
    │                                                                 │
    │ position_ids:        shape: (4, seq_len)  ← 四维位置编码！       │
    │                      concat(text_position_ids, vision_position_ids) │
    └─────────────────────────────────────────────────────────────────┘

Step 7: Left Padding & 截断
─────────────────────────────────────────────────────────────────────────────────
    同纯文本，但 position_ids 是 2D tensor
    
    输出:
    ┌─────────────────────────────────────────────────────────────────┐
    │ input_ids:      shape: (2048,)                                  │
    │ attention_mask: shape: (2048,)                                  │
    │ position_ids:   shape: (4, 2048)  ← 注意是 2D                   │
    │ pixel_values:   shape: (num_patches, embed_dim)                 │
    │ image_grid_thw: shape: (1, 3)                                   │
    └─────────────────────────────────────────────────────────────────┘

Step 8: 生成阶段 (vLLM) ⭐ 关键差异
─────────────────────────────────────────────────────────────────────────────────
    vllm_inputs = [{
        "prompt_token_ids": [151644, 8940, ...],
        "multi_modal_data": {"image": [PIL.Image]}  ← 传入原始图像
    }]
    
    ↓ llm.generate(vllm_inputs, sampling_params)
    
    注意: vLLM 内部会再次处理图像，生成 pixel_values
    
    输出:
    ┌─────────────────────────────────────────────────────────────────┐
    │ response_ids:   shape: (response_len,)                          │
    │ old_log_probs:  shape: (response_len,)                          │
    └─────────────────────────────────────────────────────────────────┘

Step 9: 多模态输入预处理 (Worker 内部) ⭐ 关键差异
─────────────────────────────────────────────────────────────────────────────────
    _process_multi_modal_inputs(data)
    
    在 Worker 上处理图像 (避免重复处理):
    
    processor.image_processor(images=[PIL.Image], return_tensors="pt")
    
    输出:
    ┌─────────────────────────────────────────────────────────────────┐
    │ multi_modal_inputs = {                                          │
    │     "pixel_values": tensor(shape=(num_patches, embed_dim)),     │
    │     "image_grid_thw": tensor(shape=(1, 3))                      │
    │ }                                                               │
    └─────────────────────────────────────────────────────────────────┘

Step 10: 前向传播 (Actor) ⭐ 关键差异
─────────────────────────────────────────────────────────────────────────────────
    model(
        input_ids,           # (batch, seq_len)
        attention_mask,      # (batch, seq_len)
        position_ids,        # (4, batch, seq_len)  ← 注意 shape！
        pixel_values,        # (num_patches, embed_dim)
        image_grid_thw       # (num_images, 3)
    )
    
    注意: 多模态需要 5 个参数！position_ids 是 3D tensor
    
    内部流程:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 获取文本 embeddings:                                          │
    │    inputs_embeds = embedding(input_ids)                         │
    │                                                                 │
    │ 2. 获取图像 embeddings:                                          │
    │    image_embeds = visual_encoder(pixel_values, grid_thw)        │
    │                                                                 │
    │ 3. 替换图像占位符:                                               │
    │    inputs_embeds[image_token_positions] = image_embeds          │
    │                                                                 │
    │ 4. Transformer forward:                                         │
    │    hidden_states = transformer(inputs_embeds, position_ids)     │
    │                                                                 │
    │ 5. 计算 logits:                                                 │
    │    logits = lm_head(hidden_states)                              │
    └─────────────────────────────────────────────────────────────────┘
    
    输出:
    logits: shape: (batch, seq_len, vocab_size)

Step 11: 计算 log_probs
─────────────────────────────────────────────────────────────────────────────────
    同纯文本
    
    log_probs = log_probs_from_logits(logits, labels)

Step 12: PPO 更新
─────────────────────────────────────────────────────────────────────────────────
    同纯文本
    
    loss = compute_policy_loss(old_log_probs, log_probs, advantages)
    loss.backward()
    optimizer.step()
```

---

## 三、关键差异总结

| 步骤 | 纯文本 | 多模态 (Qwen2-VL) |
|------|--------|-------------------|
| **数据格式** | `{"problem", "answer"}` | `{"problem", "answer", "images"}` |
| **消息构建** | `[{"content": "text"}]` | `[{"content": [{"type": "text"}, {"type": "image"}]}]` |
| **图像处理** | 无 | `process_image()` → PIL.Image |
| **Processor** | `tokenizer()` | `processor(images, text)` |
| **额外输出** | 无 | `pixel_values`, `image_grid_thw` |
| **位置编码** | 1D: `(seq_len,)` | 4D: `(4, seq_len)` (MRoPE) |
| **vLLM 输入** | `prompt_token_ids` | `prompt_token_ids` + `multi_modal_data` |
| **前向传播参数** | 3 个 | 5 个 |
| **position_ids shape** | `(batch, seq_len)` | `(4, batch, seq_len)` |

---

## 四、核心代码位置

| 功能 | 文件路径 | 关键函数 |
|------|---------|---------|
| 数据加载 | `verl/utils/dataset.py` | `RLHFDataset.__getitem__()` |
| 图像处理 | `verl/utils/dataset.py` | `process_image()` |
| 消息构建 | `verl/utils/dataset.py` | `_build_messages()` |
| MRoPE 编码 | `verl/models/transformers/qwen2_vl.py` | `get_rope_index()` |
| 多模态前向 | `verl/models/transformers/qwen2_vl.py` | `qwen2_vl_base_forward()` |
| Worker 处理 | `verl/workers/fsdp_workers.py` | `_process_multi_modal_inputs()` |
| vLLM 生成 | `verl/workers/rollout/vllm_rollout_spmd.py` | `generate_sequences()` |

---

## 五、常见问题

### Q1: 为什么多模态的 position_ids 是 4D？

**A**: Qwen2-VL 使用 MRoPE (Multimodal Rotary Position Embedding)，需要为图像的每个 patch 编码 3D 位置 (时间 t, 高度 h, 宽度 w)，加上文本的 1D 位置，共 4 维。

```
position_ids: (4, batch, seq_len)
              [0]: 文本位置 (0, 1, 2, ...)
              [1]: 时间维度 t
              [2]: 高度维度 h
              [3]: 宽度维度 w
```

### Q2: pixel_values 是什么？

**A**: 是图像经过 Vision Encoder 处理后的 patch embeddings。

```
原始图像 (H, W, 3)
    ↓ patch 切分
patches (num_patches, patch_size, patch_size, 3)
    ↓ flatten + projection
pixel_values (num_patches, embed_dim)
```

### Q3: image_grid_thw 是什么？

**A**: 记录图像在 LLM 序列中的网格形状，用于位置编码。

```
image_grid_thw: [[t, h, w]]
- t: 时间维度 (图像为 1，视频为帧数)
- h: 高度方向的 patch 数
- w: 宽度方向的 patch 数

例如: [[1, 28, 28]] 表示 1×28×28 = 784 个 patch
```

### Q4: 为什么 vLLM 生成时传入原始图像？

**A**: vLLM 内部有自己的图像处理流程，传入原始 PIL.Image 可以让 vLLM 优化 KV Cache 的分配。

---

## 六、调试技巧

### 6.1 检查数据格式

```python
# 在 RLHFDataset.__getitem__ 中添加
print(f"input_ids shape: {input_ids.shape}")
print(f"position_ids shape: {position_ids.shape}")
if "pixel_values" in model_inputs:
    print(f"pixel_values shape: {model_inputs['pixel_values'].shape}")
    print(f"image_grid_thw: {model_inputs['image_grid_thw']}")
```

### 6.2 检查 MRoPE

```python
# 检查 position_ids 是否正确
print(f"position_ids[0] (text): {position_ids[0, :20]}")  # 文本位置
print(f"position_ids[1] (t): {position_ids[1, :20]}")     # 时间
print(f"position_ids[2] (h): {position_ids[2, :20]}")     # 高度
print(f"position_ids[3] (w): {position_ids[3, :20]}")     # 宽度
```

### 6.3 常见错误

```
ValueError: Image features and image tokens do not match
```

**原因**: `input_ids` 中的 `<|image_pad|>` token 数量与 `pixel_values` 的 patch 数不匹配。

**解决**: 检查 `max_pixels` 和 `min_pixels` 配置，或增加 `max_prompt_length`。