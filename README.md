# RapidGEMM

An Efficient Tile-agnostic GEMM for Mixture-of-Experts Training

## Kernels

The two B200 kernels are in [`src/gemm.py`](src/gemm.py).

- `Mgemm(x, w, cnt, max_M_per_E, transpose_B)` computes the grouped forward
  or input-gradient GEMM. `x` has shape `[M, K]`, `w` has shape `[E, N, K]`,
  and `cnt` is the cumulative row-offset tensor with a leading zero.
- `Kgemm(grad, X, cnt, descriptor)` computes the grouped FP32 weight gradient.
  `grad` and `X` contain expert-grouped rows, `cnt` stores cumulative expert
  row ends, and `descriptor` is `gw13` or `gw2`.

## E2E Performance

### Model layout

The experiment uses the 30B-A3B MoE layout. The full model contains 20 Transformer layers; because the experiment runs on four GPUs, we execute the same model geometry with one and two layers and use their per-token time difference to estimate the incremental cost of model depth.

| Parameter | Value |
|---|---:|
| Model width (`d_model`) | 3,072 |
| Dense FFN width (`d_ffn`) | 9,216 |
| Attention heads / KV heads | 24 / 4 |
| Head dimension | 128 |
| MoE experts | 128 |
| Experts selected per token (`top_k`) | 8 |
| Expert FFN width (`moe_ffn_dim`) | 1,280 |
| Shared-expert width | 1,280 |
| Full-model depth | 20 layers |
| Measured depths | 1 and 2 layers |

### Settings

The runs use four NVIDIA B200 GPUs with the following model partition and training settings:

| Setting | Value |
|---|---|
| Expert parallelism | EP = 4; 128 experts are split evenly across four GPUs (32 per GPU) |
| Tensor parallelism | TP = 1; attention projections are not tensor-parallelized |
| Data parallelism | DP = 4 with ZeRO-1 for replicated attention and dense parameters over the same four ranks; expert-DP = 1 |
| Sequence length | 4,096 |
| Micro-batch size / update frequency | 1 / 4 |
| Effective steady-state tokens per update | 65,279 on average |
| Training steps | 100 |
| Learning rate | constant `2e-4`, no warmup |
| Trainer precision | `--precision fp32`; MoE grouped GEMMs use BF16 (`quant_mode=bfloat16`) |
| Backends | RapidGEMM and DeepGEMM |

Each backend is run at one and two layers for 100 updates. We discard the first 20 warmup updates and summarize the steady-state near-peak envelope as the mean of the fastest eight updates (the top decile) from steps 21–100. All four runs have identical per-step token counts.

### Results

RapidGEMM is faster at both measured depths under the steady-state top-decile envelope, and its advantage increases with depth:

| Backend | 1-layer throughput | 2-layer throughput |
|---|---:|---:|
| **RapidGEMM** | **449.17k tokens/s** | **345.69k tokens/s** |
| DeepGEMM | 442.69k tokens/s | 336.71k tokens/s |
| **RapidGEMM improvement** | **+1.46%** | **+2.67%** |

To isolate the cost that scales with model depth, we convert throughput to per-token time before taking the layer difference:

$$
t_L = \frac{1}{T_L}, \qquad
\Delta t_{\mathrm{layer}} = t_2 - t_1, \qquad
R_{\mathrm{model}} = \frac{1}{\Delta t_{\mathrm{layer}}}.
$$

Here, $T_L$ is the measured throughput at depth $L$, and $\Delta t_{\mathrm{layer}}$ is the incremental per-token layer cost. The **model-size rate** measures how quickly the system processes the work added by one more model layer. It is reported in token-layers/s, and higher is better: 1M token-layers/s means processing one million tokens through one additional layer per second.

| Backend | Incremental layer cost | Model-size rate | Estimated 20-layer throughput |
|---|---:|---:|---:|
| **RapidGEMM** | **0.666 µs/token** | **1.500M token-layers/s** | **67.2k tokens/s** |
| DeepGEMM | 0.711 µs/token | 1.406M token-layers/s | 63.4k tokens/s |
| **RapidGEMM improvement** | **6.27% lower** | **+6.68%** | **+5.90%** |

The 20-layer estimate is

$$
t_{20} = t_1 + 19\left(t_2 - t_1\right), \qquad
T_{20} = \frac{1}{t_{20}}.
$$

Across top-fraction choices from 5% through 25%, RapidGEMM's incremental model-size rate remains 4.6%–7.8% higher than DeepGEMM.
