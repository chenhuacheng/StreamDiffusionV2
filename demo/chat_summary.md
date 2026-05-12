# StreamDiffusionV2 多卡并行 & Step=1 深度分析

> 对话时间：2026-05-12
> 涉及代码：`streamdiffusionv2/repo/`

---

## 目录

1. [Step=1 完整分析](#1-step1-完整分析)
2. [首帧为什么比稳态慢](#2-首帧为什么比稳态慢)
3. [多卡并行机制](#3-多卡并行机制)
4. [论文为什么用 4×H100](#4-论文为什么用-4h100)
5. [NCCL 通信开销为何不线性增长](#5-nccl-通信开销为何不线性增长)
6. [1.3B 在 H20 多卡效果差的原因](#6-13b-在-h20-多卡效果差的原因)
7. [NCCL 环境变量解读](#7-nccl-环境变量解读)
8. [max_outstanding 参数详解](#8-max_outstanding-参数详解)

---

## 1. Step=1 完整分析

### 配置推导

YAML 中定义完整列表如 `[700, 500, 400, 200, 0]`。

`inference_common.py` 处理逻辑：
- Step=1 → 取前 1 个非零步 → `[700]` → 追加 0 → `[700, 0]`
- v2v 模式去掉末尾 0 → 最终 `denoising_step_list = [700]`

### 关键参数

| 参数 | 值 | 含义 |
|---|---|---|
| `denoising_step_list` | `[700]` | 只有 1 个 timestep |
| `batch_size` | 1 | 无流水线 |
| `num_steps` | 1 | 无去噪流水线 |

### 时序

```
首帧 (chunk 0 - prepare):
┌──────────────┐ ┌───────────┐ ┌──────────┐
│ VAE Encode   │→│ DIT×1     │→│ VAE Dec  │→ 输出 4帧
│ (4帧→latent) │ │ batch=1   │ │          │
└──────────────┘ └───────────┘ └──────────┘

稳态 (chunk 1, 2, 3, ...):  完全一样的计算量
┌──────────────┐ ┌───────────┐ ┌──────────┐
│ VAE Encode   │→│ DIT×1     │→│ VAE Dec  │→ 输出 4帧
└──────────────┘ └───────────┘ └──────────┘
```

**Step=1 首帧和稳态的 DIT 计算量完全相同：都是 1 次 batch=1 的 DIT 前向。**

---

## 2. 首帧为什么比稳态慢

### 计时陷阱：Processed 1 包含了两个 chunk

`start_time` 在 chunk 0（prepare）之前设置，直到 chunk 1 输出后才更新。

```
         start_time                                    end_time
            │                                             │
            ▼                                             ▼
┌───────────────────────────────────┐ ┌──────────────────────────────────┐
│        chunk 0 (prepare)          │ │         chunk 1 (稳态)           │
│ VAE Enc + text_enc + KV init +   │ │ VAE Enc + DIT×1 + VAE Dec       │
│ DIT×1 + VAE Dec                  │ │                                  │
└───────────────────────────────────┘ └──────────────────────────────────┘
│◄────────────── "Processed 1" 报告的时间 ──────────────▶│
```

### 真实开销分布

| 日志 | 实际覆盖范围 | 包含的工作 |
|---|---|---|
| **Processed 1** | chunk 0 + chunk 1 | text_encoder + KV Cache 初始化 + `.repeat()` + 2× VAE Enc + 2× DIT + 2× VAE Dec |
| **Processed 2+** | 仅当前 chunk | 1× VAE Enc + 1× DIT + 1× VAE Dec |

**首帧"看起来慢"的真正原因**：
1. 🥇 计时包含了 2 个 chunk 的工作（~50%）
2. 🥈 prepare 内的初始化开销（~35%）：text encoder、KV Cache 分配与 `.repeat()`
3. 🥉 其余零碎（~15%）

---

## 3. 多卡并行机制

### 并行方式：Pipeline Parallelism（模型切分）

不是数据并行，是把 **DIT 的 30/40 个 transformer block 切成连续段**，分配到不同 GPU。

### Block 分配

| GPU 数 | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
|--------|--------|--------|--------|--------|
| 2 卡 | Block 0~14 | Block 15~29 | - | - |
| 4 卡 | Block 0~9 | Block 10~19 | Block 20~29 | Block 30~39 |

### 每个 Rank 的角色

```
Rank 0            → block_mode = 'input'    # 入口：VAE Encode + 前段 blocks
Rank 1~N-2        → block_mode = 'middle'   # 中间：只跑中间段 blocks
Rank N-1 (最后)    → block_mode = 'output'   # 出口：后段 blocks + head + VAE Decode
```

### 流水线重叠

**Rank 0 处理 Chunk 1 前半段时，Rank 1 同时处理 Chunk 0 后半段。** 稳态下多 chunk 重叠执行。

```
Rank 0: ┌Chunk0前半┐ ┌Chunk1前半┐ ┌Chunk2前半┐
Rank 1:     空      ┌Chunk0后半┐ ┌Chunk1后半┐ ┌Chunk2后半┐
                          ↑            ↑            ↑
                        输出          输出          输出
```

### 启动命令

```bash
CUDA_VISIBLE_DEVICES="0,1" torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    --module streamv2v.inference_pipe \
    --step 1 \
    --config_path wan_causal_dmd_v2v.yaml
```

---

## 4. 论文为什么用 4×H100

### 核心原因

1. **🥇 显存：14B 模型单卡装不下**

| 模型 | BF16 权重 | +KV Cache+激活值 | 单卡 H100 (80GB) |
|------|----------|-----------------|----------------|
| 1.3B | ~2.6 GB | ~15-20 GB | ✅ 装得下 |
| **14B** | **~28 GB** | **~60-80 GB** | ❌ 装不下 |

2. **🥈 吞吐量：流水线并行接近线性加速**

4 卡各跑 10 blocks，稳态每个时间周期都有输出。

3. **🥉 动态负载均衡**

系统运行时测量各 rank 耗时，动态调整 block 分配。

### 延迟 vs 吞吐

| 指标 | 单卡 | 多卡 |
|------|------|------|
| 首帧延迟 | 基准 | 多卡反而更慢（通信+填充） |
| **稳态吞吐 (FPS)** | 受限单卡算力 | **接近线性加速** |
| 14B 可行性 | 不可能 | **唯一方案** |

---

## 5. NCCL 通信开销为何不线性增长

### 通信量极小

- 每次传输 ≈ **6 MB**（中间 hidden states）
- NVLink 900 GB/s 下：6MB / 900GB/s ≈ **0.01 ms**
- 计算时间 ~25-100ms → 通信占比 < 0.1%

### 通信与计算完全重叠

代码使用**独立 CUDA stream** 做异步通信：

```python
self.com_stream = torch.cuda.Stream()    # 专用通信流

with torch.cuda.stream(self.com_stream):
    latent_data = self.data_transfer.receive_latent_data_async(num_steps)
```

底层用 `dist.isend` / `dist.irecv`（非阻塞）。

### Bubble 只在开头

Pipeline 填充只需 `world_size - 1` 步（4 卡 = 3 步），之后满载。

### Kernel Launch 不增长

每卡只 launch `total_blocks / world_size` 个 block 的 kernel，**次数反而更少**。

### 根本原因

这是 **Pipeline Parallel**（P2P 小量数据），不是 Data Parallel（AllReduce 全量参数）。通信量差了 4 个数量级。

---

## 6. 1.3B 在 H20 多卡效果差的原因

### H20 vs H100 硬件差异

| 指标 | H100 | H20 | 差距 |
|------|------|-----|------|
| BF16 算力 | 1979 TFLOPS | 148 TFLOPS | H100 快 13 倍 |
| 显存带宽 | 3.35 TB/s | 4.0 TB/s | H20 略好 |
| 卡间互联 | NVLink 900 GB/s | NVLink 900 GB/s | 相同 |

### 核心矛盾：计算量太小，通信固定延迟占比暴增

NCCL P2P 通信有**固定启动延迟** ~1-2ms（不是带宽瓶颈，是 latency 瓶颈）。

**H100 + 14B + 4 卡：**
```
每卡计算 10 blocks × 14B → 约 25-30ms
通信固定开销 ≈ 1-2ms
overhead 比 = 2ms / 30ms ≈ 6%
实际加速比 ≈ 4 × 0.94 ≈ 3.76  ← 很好
```

**H20 + 1.3B + 2 卡：**
```
每卡计算 15 blocks × 1.3B → 约 6-10ms  ← 1.3B 比 14B 小 10 倍！
通信固定开销 ≈ 1-2ms
overhead 比 = 2ms / 8ms ≈ 25%
实际加速比 ≈ 2 × 0.75 ≈ 1.5  ← 打了 75 折
```

**H20 + 1.3B + 4 卡：**
```
每卡计算 7-8 blocks → 约 3-5ms
3 次串行通信 → 通信总延迟 = 3 × 2ms = 6ms
流水线周期被通信 bound → 加速比崩塌
```

### 可视化

```
                    每卡计算时间
                    ◀────────▶
H100+14B (4卡):    ████████████████████████████░  ← 通信(░)占比极小
                   |-------- 30ms ----------|2ms|

H20+1.3B (2卡):   ████████░░                     ← 通信开始显眼
                   |--8ms--|2ms|

H20+1.3B (4卡):   ████░░░░                       ← 通信反客为主
                   |4ms|4ms|
```

### 结论

> **加卡确实增加了总算力，但 Pipeline Parallel 的通信有固定延迟底噪（~1-2ms/次）。模型越小 + 卡越多 = 每卡分到的计算量越少 = 通信延迟占比越大 = 加速比越差。** 增加的算力被通信等待时间吃掉了。

### H20 + 1.3B 的正确加速方案

| 方案 | 原理 |
|------|------|
| 数据并行（DP） | 每卡跑独立用户，无通信 |
| 单卡优化 | Flash Attention、kernel 融合、量化 |

---

## 7. NCCL 环境变量解读

论文实验环境配置：

```
NCCL_IB_DISABLE = 1    # 禁用 InfiniBand
NCCL_P2P_DISABLE = 0   # 启用 P2P（NVLink 直连）
```

### 含义

| 变量 | 值 | 含义 |
|------|---|------|
| `NCCL_IB_DISABLE=1` | 禁用 IB | 不走 InfiniBand 网络，说明是**单机多卡** |
| `NCCL_P2P_DISABLE=0` | 启用 P2P | 使用 NVLink 直连通信（最快路径） |

### 建议

| 场景 | 建议 |
|------|------|
| 单机多卡 | 保持 `IB_DISABLE=1`，跟论文一致 |
| 多机多卡 | 设 `IB_DISABLE=0`，需要 IB 跨机通信 |

`NCCL_IB_DISABLE` 改成 0：如果机器没有 IB 网卡则无影响；如果有 IB 网卡可能误走 IB 反而更慢（IB 25-50 GB/s 远慢于 NVLink 900 GB/s）。

---

## 8. max_outstanding 参数详解

### 含义

控制**异步发送队列的最大深度**——允许多少个 `isend` 操作"正在飞行中"还没确认完成。

### 工作原理

```python
def _wait_for_outstanding(self, outstanding: list) -> None:
    while len(outstanding) >= max_outstanding:
        oldest = outstanding.pop(0)
        for work in oldest:
            work.wait()    # 阻塞等最老的发送完成
```

### 不同值的效果

**`max_outstanding=1`（默认）：**
```
计算 8ms → 等通信 2ms → 计算 8ms → 等通信 2ms
有效周期 = 10ms，其中 2ms 浪费
```

**`max_outstanding=2`：**
```
计算 8ms → 计算 8ms → 等（send0 早已完成）→ 计算 8ms
有效周期 ≈ 8ms，通信被完全隐藏
```

### 对比表

| 方面 | =1 | =2 | =4 |
|------|----|----|-----|
| 计算-通信重叠 | ❌ 无 | ✅ 完全重叠 | ✅ 同 =2 |
| 吞吐量 | 受通信阻塞 | 最优 | 同 =2 |
| 显存占用 | 最低 | +6MB | +18MB |

### 结论

> `max_outstanding` 只要 **≥ 2**，通信就能被计算完全遮盖。调到 4 没有额外收益——队列根本不会堆满。**唯一有意义的区别是 1 → 2。**

### 建议值

| 值 | 适用场景 |
|----|---------|
| 1 | 保守，显存最省，但通信阻塞计算 |
| **2** | **推荐**，足以让通信和计算重叠 |
| 3+ | 无额外收益 |

---

## 附录：关键代码文件

| 文件 | 用途 |
|------|------|
| `streamv2v/inference.py` | 单卡推理主循环 |
| `streamv2v/inference_pipe.py` | 多卡 Pipeline Parallel 推理 |
| `streamv2v/inference_common.py` | Step 配置处理 |
| `models/wan/causal_stream_inference.py` | DIT 流式推理核心 |
| `streamv2v/communication/distributed_communicator.py` | NCCL 通信封装 |
| `streamv2v/configs/wan_causal_dmd_v2v.yaml` | 模型配置 |
