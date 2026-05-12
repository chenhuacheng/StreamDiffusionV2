# StreamDiffusionV2 Pipeline 并行架构分析

> 基于 4 卡（GPU 4/5/6/7）Pipeline Parallel 方案的深度分析
> 生成时间: 2026-05-12

---

## 目录

1. [数据流概述](#1-数据流概述)
2. [hidden_states 传递机制](#2-hidden_states-传递机制)
3. [环回依赖分析](#3-环回依赖分析)
4. [流水线并行的实际效果](#4-流水线并行的实际效果)
5. [NCCL 通信精确耗时](#5-nccl-通信精确耗时)
6. [瓶颈分析：为什么加卡不提升吞吐](#6-瓶颈分析为什么加卡不提升吞吐)
7. [调优实验记录](#7-调优实验记录)

---

## 1. 数据流概述

### VAE encoder 与 hidden_states 完全无关

```
VAE encoder 的输入输出：
  输入: images (原始像素图片)  [B, C, T, H, W]
  输出: latents (潜空间表示)   [B, C, T, h, w]

hidden_states 的用户：
  输入到: DiT (transformer blocks)
  用途: 作为 DiT 推理时的"上一帧记忆"
```

### Rank 0 的两阶段处理

| 阶段 | 操作 | 需要 hidden_states？ | 需要等 Rank 3 回传？ |
|------|------|---------------------|---------------------|
| **阶段1: VAE 编码** (`prepare_demo_input_batch`) | images → vae.stream_encode → latents → noisy_latents | ❌ 不需要 | ❌ 不需要 |
| **阶段2: DiT 推理** (`run_demo_input_step`) | 先更新 hidden_states，再做 DiT 推理 | ✅ 需要 | ✅ 必须等 |

---

## 2. hidden_states 传递机制

### Rank 0（`block_mode='input'`）

```python
if block_mode == 'input':
    self.hidden_states[1:] = self.hidden_states[:-1].clone()  # 历史往后推
    self.hidden_states[0] = noise[0]                           # 新帧插入队首
```

Rank 0 的 `hidden_states` 是**自己维护的滑动窗口**，必须从 Rank 3 环回拿到最终态来更新。

### Rank 1/2/3（`block_mode='middle'/'output'`）

```python
else:
    self.block_x.copy_(block_x)              # 接收上游的中间计算结果
    self.hidden_states.copy_(noise)           # 直接用上游传来的 hidden_states
    self.kv_cache_starts.copy_(current_start)
    self.kv_cache_ends.copy_(current_end)
```

### 各 Rank 对比

| | Rank 0 | Rank 1/2/3 |
|---|---|---|
| `hidden_states` 来源 | 自己的滑动窗口 + **Rank 3 环回更新** | **上游 rank 发来的**（`latent_data.original_latents`） |
| 是否需要环回 | ✅ 必须等 Rank 3 传回最终态 | ❌ 不需要，从 NCCL receive 直接拿 |
| 数据流方向 | Rank 3 → Rank 0 | Rank 0→1, Rank 1→2, Rank 2→3 |

### hidden_states 透传链

```
Rank 0:
  hidden_states = [环回的最终态]  ← 从 Rank 3 拿的
  DiT → denoised_pred
  send: { latents=denoised_pred, original_latents=hidden_states }

Rank 1 收到:
  self.hidden_states.copy_(noise)  → noise = latent_data.original_latents = Rank 0 的 hidden_states
  self.block_x.copy_(block_x)     → block_x = latent_data.latents = Rank 0 的 denoised_pred
  DiT → denoised_pred
  send: { latents=denoised_pred, original_latents=latent_data.original_latents (透传!) }

Rank 2 收到:
  self.hidden_states.copy_(noise)  → 还是同一份 original_latents（从 Rank 0 一路透传下来的）
  ...
```

**结论：环回机制只是为了解决 Rank 0 这个"环形起点"的问题**——其他 rank 的 hidden_states 随着正向流水线自然传递下来。

---

## 3. 环回依赖分析

### DiT 是串行的吗？

**单个 chunk 看确实是串行的**——chunk 必须经过 Rank 0 → 1 → 2 → 3 全部处理完才算结束。

但流水线并行的收益来自**时间维度的重叠**：

```
时间→  T1    T2    T3    T4    T5    T6    T7
Rank0: [C1]  [C2]  [C3]  [C4]  [C5]  [C6]  [C7]
Rank1:       [C1]  [C2]  [C3]  [C4]  [C5]  [C6]
Rank2:             [C1]  [C2]  [C3]  [C4]  [C5]
Rank3:                   [C1]  [C2]  [C3]  [C4]
```

### 环回不是"等上一帧"

关键发现：**Rank 0 用的是"延迟了 world_size 轮"的 hidden_states**！

```python
if pipeline_manager.processed >= pipeline_manager.world_size:
    latent_data = pipeline_manager.data_transfer.receive_latent_data_async(num_steps)
```

- Rank 0 处理 chunk 5 时，收到的环回是 **chunk 1** 的结果（延迟 4 轮 = world_size）
- 稳态下，Rank 3 每个时间槽都会有结果环回给 Rank 0
- 流水线确实是满的！4 个 rank 各自在处理不同的 chunk

### 类比：工厂流水线

| | 单工人做全部 | 4 人流水线 |
|---|---|---|
| 做一件产品的延时 | 10 分钟 | 12 分钟（多了传递时间）更慢 |
| 每分钟产出 | 1/10 件 | **1/3 件**（快 3 倍多）|
| 代价 | - | 单件延时增加 |

---

## 4. 流水线并行的实际效果

### 多卡没有空闲

实测观察：GPU 5/6/7 利用率均为 96-97%，说明**流水线确实是满的**，多个 chunk 确实在同时被处理。

### 吞吐公式

```
稳态吞吐 = 1 / max(各 rank 的单 chunk 处理时间)
```

瓶颈 rank 决定整体吞吐。

---

## 5. NCCL 通信精确耗时

### 4 卡实测数据（schedule_block 开启时，block 分配 [0,1],[1,14],[14,27],[27,30]）

| 操作 | 耗时 | 说明 |
|------|------|------|
| **Rank 0 send** | **0.3 ms** | 异步发出，几乎瞬间 |
| **Rank 1 recv** | **~6 ms** | 从 Rank 0 接收，很快 |
| **Rank 1 send** | **~7 ms** | 发给 Rank 2 |
| **Rank 0 recv（环回）** | **350-540 ms** | ⚠️ 不是通信慢，而是在等 Rank 1/2/3 处理完 |

### 关键发现

**NCCL 本身的数据传输只要 ~6ms。** Rank 0 的 recv 等了 350-540ms 本质上是在等下游处理完毕。

---

## 6. 瓶颈分析：为什么加卡不提升吞吐

### 瓶颈 1: schedule_block 把 blocks 分配搞坏了

开启 `SCHEDULE_BLOCK=True` 后的分配：

```
Rank 0: blocks [0, 1]   → 1 block  + VAE encode
Rank 1: blocks [1, 14]  → 13 blocks  ← 瓶颈！
Rank 2: blocks [14, 27] → 13 blocks  ← 瓶颈！
Rank 3: blocks [27, 30] → 3 blocks  + VAE decode
```

### 瓶颈 2: VAE encode 是不可并行的串行瓶颈

```
s08_vae_encode = 394ms ！！！
```

各 stage 时间（schedule_block 开启时）：

| Stage | 耗时 | 说明 |
|-------|------|------|
| s07 read_queue | ~45 ms | 读取输入队列 |
| s08 vae_encode | **394 ms** | ⚠️ VAE 编码，只在 Rank 0 |
| s09 dit_rank0 | ~93 ms | DiT Rank 0（只有 1 block） |
| s10 nccl_send | ~0.3 ms | NCCL 异步发送 |
| s11 dit_last_rank | ~10 ms | DiT 最后一个 rank |
| s12 vae_decode | ~34 ms | VAE 解码 |

### 根本原因

```
Rank 0 每个 chunk 时间 = VAE encode(394ms) + DiT(93ms) = 487ms/chunk
```

**无论怎么分 DiT blocks，Rank 0 每个 chunk 都要花 394ms 做 VAE 编码。这个时间只在 Rank 0 上，不可并行，不可拆分。**

理论最大吞吐 = 1/394ms ≈ **2.5 chunks/s ≈ 10 fps**（VAE 编码上限）

### 均匀分配也无法解决

关闭 schedule_block 后均匀分配 [0,8],[8,15],[15,23],[23,30]：

```
Rank 0: VAE(394ms) + DiT 8 blocks(~310ms) ≈ 700ms/chunk ← 新瓶颈（更差！）
Rank 1: DiT 7 blocks ≈ 280ms/chunk
Rank 2: DiT 8 blocks ≈ 310ms/chunk
Rank 3: DiT 7 blocks(280ms) + VAE decode(34ms) ≈ 314ms/chunk
```

### GPU 0 (Rank 0) 利用率低

实测 GPU 4（Rank 0）利用率只有 73%，而 GPU 5/6/7 都是 96-97%。这是因为 Rank 0 大量时间花在 VAE 编码上，而 VAE 编码相比 DiT 推理的 GPU 利用率更低（VAE 计算密度不如 transformer attention/FFN）。

---

## 7. 调优实验记录

### 实验：max_outstanding=4, 关闭 schedule_block, 去掉输入帧率限制

**修改内容：**

| 参数 | 之前 | 之后 |
|------|------|------|
| max_outstanding | 2 | **4** |
| schedule_block | True（[0,1],[1,14],[14,27],[27,30]） | **False**（均匀 [0,8],[8,15],[15,23],[23,30]） |
| 输入帧率限制 | 10 FPS（WebSocket 接收端 throttle） | **无限制** |

**当前实测结果（待补充）：**

- s08 vae_encode: **null**（尚未获取到有效值）
- s09 dit_rank0: **112.7 ms**（8 blocks，符合预期）
- s10 nccl_send: **0.4 ms**
- s11 dit_last_rank: **20.4 ms**（7 blocks）
- s12 vae_decode: **34.0 ms**
- GPU 4 利用率: **73%**（仍然偏低）
- GPU 5/6/7 利用率: **96-97%**

### 已知 GPU 0 利用率低的原因

Rank 0 承担了 **VAE encode + DiT 8 blocks** 的双重负载。VAE encode 本身不是纯 transformer 计算，GPU 利用率天然低于 DiT 推理。当 VAE encode 时间占 Rank 0 总时间的 ~55% 时，整体 GPU 利用率自然被拉低。

---

## 潜在优化方向

| 方案 | 效果 | 可行性 |
|------|------|--------|
| **Tensor Parallel (TP)** | 每个 block 内部拆到多卡，单 chunk 延时真正降低 | 需改模型代码 |
| **VAE encode 异步化** | VAE 编码与 NCCL receive 并行 | 工程改动小 |
| **VAE encode 分离到独立 rank** | Rank 0 专做 VAE，其他 rank 做 DiT | 需要 5 卡 |
| **去掉/延迟环回** | 牺牲质量，但流水线真正流起来 | 质量风险 |
| **减少环回频率** | 每 N 帧才环回一次 | 质量可控 |
| **单卡（显存允许时）** | 零通信开销 | 需要大显存卡 |
| **Batch Parallel** | 多个独立视频同时生成 | 适合多用户场景 |

---

*本文档基于 StreamDiffusionV2 repo_up 分支 4 卡 Pipeline Parallel 方案的代码阅读和实测数据整理。*
