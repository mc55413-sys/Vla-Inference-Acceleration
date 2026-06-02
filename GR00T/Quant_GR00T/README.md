# QuantVLA：GR00T N1.5 量化推理加速

对 GR00T N1.5 模型运行 Baseline BF16、DuQuant W4A8、QuantVLA Full、以及 Selective FP8 四种变体的量化评估。仓库提供脚本化管理入口 — 支持 Docker 与原生 Conda 两种运行方式。

容器不依赖本地 conda 安装即可运行。原始的 `groot_test` 与 `libero_test` 依赖已合并至单个镜像，便于跨机器复现。

## 图像选择

| 设备 | Dockerfile | 用途 |
|---|---|---|
| x86_64 NVIDIA GPU, e.g. RTX 5090/H100/B200 | `Dockerfile` | 推理服务、LIBERO 真机评测、本地延迟/显存/FLOPs 基准测试 |
| Jetson Orin / JetPack 6 | `orin.Dockerfile` | 边缘侧推理服务与本地基准测试；不推荐在 Orin 上运行完整 LIBERO 仿真 |

## 前置条件

宿主机需要：

```bash
docker --version
nvidia-smi
```

并确认 Docker 可以访问 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

首次运行会从 Hugging Face 下载模型，建议挂载宿主机的 Hugging Face 缓存目录以避免重复下载。

## 构建 x86 GPU 镜像

在项目根目录运行：

```bash
cd /path/to/Quant_GR00T
docker build -t quantvla:cuda13 .
```

如果需要 Hugging Face token：

```bash
export HF_TOKEN=your_token
```

## 构建 Jetson Orin 镜像

在 Jetson Orin 设备上：

```bash
cd /path/to/Quant_GR00T
docker build -t quantvla:orin -f orin.Dockerfile .
```

运行时使用 NVIDIA runtime：

```bash
docker run --rm -it --runtime nvidia --network host --ipc host \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:orin bash
```

## 任务与运行模式

支持的任务：

```bash
libero_spatial
libero_goal
libero_object
libero_90
libero_10
```

支持的运行模式：

```bash
baseline   # BF16 基线（已默认启用 torch.compile + SDPA）
duquant    # DuQuant W4A8 权重量化（LLM 线性层，4-bit packed 存储）
full       # QuantVLA Full: DuQuant W4A8 + ATM（自适应时序调制）+ OHB（输出头平衡）
fp8        # Selective FP8：仅对大型 LLM matmul（≥30M ops）使用 FP8 tensor-core 加速
```

## 环境变量控制

Docker 和原生 Conda 运行均使用以下环境变量进行变体切换。可手动 `export` 或由启动脚本自动设置。

### 核心模式变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GR00T_DUQUANT_WBITS_DEFAULT` | 空（不启用） | 设为 `4` 启用 DuQuant W4A8 |
| `GR00T_DUQUANT_INCLUDE` | — | DuQuant 层名正则 include |
| `GR00T_DUQUANT_EXCLUDE` | — | DuQuant 层名正则 exclude |
| `GR00T_DUQUANT_BLOCK` | `64` | DuQuant block 大小 |
| `GR00T_DUQUANT_STORAGE` | `packed` | DuQuant 存储格式：`packed` / `fake` / `unpacked` |
| `GR00T_DUQUANT_ACT_MODE` | `off` | 激活量化模式：`off` / `dynamic` |
| `GR00T_DUQUANT_PACKDIR` | — | DuQuant packed cache 目录 |
| `GR00T_DUQUANT_CALIB_STEPS` | `32` | 标定步数 |
| `GR00T_FP8_MODE` | `0` | 设为 `1` 启用选择性 FP8 Linear（覆盖 DuQuant/ATM） |
| `GR00T_ATM_ENABLE` | `0` | 设为 `1` 启用 ATM（须配合 alpha json 使用） |
| `GR00T_ATM_ALPHA_PATH` | — | ATM alpha/beta 缩放因子 JSON 路径 |
| `GR00T_ATM_SCOPE` | `dit` | ATM 作用域 |
| `GR00T_OHB_ENABLE` | `0` | 设为 `1` 启用 OHB |
| `GR00T_OHB_SCOPE` | `dit` | OHB 作用域 |
| `GR00T_OHB_FALLBACK` | `1.0` | OHB fallback 值 |

### torch.compile 控制

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GR00T_TORCH_COMPILE` | `1` | 设为 `1` 启用 torch.compile 加速 |
| `GR00T_TORCH_COMPILE_MODE` | `default` | compile 模式：`default` / `reduce-overhead` / `max-autotune` |
| `GR00T_DUQUANT_COMPILE_BACKBONE` | `1` | 设为 `0` 跳过 DuQuant 模式下的 backbone compile |

torch.compile 策略（`GR00T_TORCH_COMPILE=1` 时启用）：
- **DiT (action_head)**：使用 `max-autotune` + CUDA graphs（固定输入形状可获 ~1.45x 加速）
- **LLM (backbone)**：使用 `default` 模式。DuQuant 自定义层可能导致 graph breaks，但周围的 Eagle blocks 在 Blackwell/Ada GPU 上仍可受益。通过 `GR00T_DUQUANT_COMPILE_BACKBONE=0` 可跳过 DuQuant 模式下的 backbone compile。

### 运行时变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `QVLA_DENOISING_STEPS` | `8` | 去噪步数 |
| `QVLA_HOST` | `localhost` | 评测客户端连接推理服务的主机地址 |
| `QVLA_PORT` | `5556` | 推理服务端口 |
| `QVLA_WARMUP` | `5` | 本地基准测试预热迭代 |
| `QVLA_ITERS` | `20` | 本地基准测试计时迭代 |
| `QVLA_ACTION_SAMPLES` | `32` | 本地基准测试 action dump 数量 |
| `QVLA_PRIMARY_LATENCY_KEY` | `dual_stage_sum_ms` | 汇总用的主要延迟键 |
| `HF_TOKEN` | 空 | Hugging Face token（用于私有或限速模型访问） |

## 运行 Baseline BF16

需要两个终端。第一个启动推理服务，第二个运行 LIBERO 评测。

终端 1：

```bash
cd /path/to/Quant_GR00T
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_inference_server.sh baseline libero_10
```

终端 2：

```bash
cd /path/to/Quant_GR00T
docker run --rm -it --gpus all --network host --ipc host \
  -e QVLA_HOST=localhost \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_real_libero_benchmark.sh baseline libero_10
```

## 运行 Selective FP8

Selective FP8 仅将超大 LLM matmul（≥30M ops，如 gate/up/down proj）替换为 FP8 tensor-core 加速。小 matmul保持在 BF16，避免了 FP8 转换开销在小型操作上的性能退化。

终端 1：

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_inference_server.sh fp8 libero_10
```

终端 2：

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e QVLA_HOST=localhost \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_real_libero_benchmark.sh fp8 libero_10
```

> **关于 FP8 阈值：** 默认阈值为 30M ops（`out_features × in_features`），在 GR00T N1.5 + Qwen3 上 Gate/Up/Down proj（~51M ops）被转换，而较小的 FFN（~14.7M ops）和 DiT FFN（~16.8M ops）保持在 BF16。实测在 RTX 5090 上，51M ops 的 matmul 在 FP8 下达到 52μs vs BF16 70μs（1.35x），而 14.7M ops 的 matmul 在 FP8 下 42μs vs BF16 30μs（反而更慢）。阈值可在 `gr00t/quantization/fp8_linear.py` 中的 `_MIN_FP8_OPS` 常量调整。

## 运行 QuantVLA Full 量化

终端 1：

```bash
cd /path/to/Quant_GR00T
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_inference_server.sh full libero_10
```

终端 2：

```bash
cd /path/to/Quant_GR00T
docker run --rm -it --gpus all --network host --ipc host \
  -e QVLA_HOST=localhost \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_real_libero_benchmark.sh full libero_10
```

## 运行 DuQuant Only

终端 1：

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_inference_server.sh duquant libero_10
```

终端 2：

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e QVLA_HOST=localhost \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_real_libero_benchmark.sh duquant libero_10
```

## 实时指标输出

`tools/run_quantvla_real_libero_benchmark.sh` 自动启用：

```bash
--profile-server
--print_step_latency
```

运行时你会看到类似输出：

```text
[latency] task=0 episode=0 step=10 Data=... | Preprocess=... | System-2 Vision=... | System-2 Reasoning=... | Bridge=... | System-1 Vision=... | System-1 Action=... | End to End Latency=... avg=... | Model Latency=... avg=...
[tflops] task=0 episode=0 step=10 dense_equiv=...TFLOPs/call
```

### 服务端时序分解

启用 `--print_step_latency` 时，服务端会在每个 action 请求中返回详细的时序分解（可在运行时日志中查看）：

**预处理阶段：**
- `server_preprocess_video_ms`：视频变换
- `server_preprocess_state_action_ms`：状态/动作变换
- `server_preprocess_concat_ms`：拼接操作
- `server_preprocess_gr00t_vlm_ms`：GR00T VLM 预处理
- `server_preprocess_other_ms`：其他预处理操作

**模型推理阶段：**
- `server_system2_vision_model_ms`：System-2 视觉 backbone
- `server_system2_vision_projector_ms`：System-2 视觉投影器
- `server_system2_reasoning_ms`：System-2 LLM 推理
- `server_system2_to_system1_bridge_ms`：System-2 → System-1 桥接
- `server_system1_vision_norm_ms`：System-1 视觉归一化
- `server_system1_vision_attention_ms`：System-1 视觉注意力
- `server_system1_action_head_ms`：System-1 动作头总耗时
- `server_model_total_ms`：模型总耗时（准备+backbone+action+验证）

**后处理阶段：**
- `server_postprocess_untransform_ms`：逆变换

**汇总维度：**
- `server_system2_vision_ms` = vision_model + vision_projector
- `server_system2_other_ms` = backbone - (vision + reasoning + bridge)
- `server_system1_vision_ms` = vision_norm + vision_attention
- `server_system1_action_ms` = action_head - system1_vision
- `server_policy_total_ms`：策略总耗时（输入打包 → 逆变换完成）

### 服务端内存快照

每次请求还会返回以下组件级内存指标：

```text
[model component memory] llm=...GiB | dit=...GiB | llm+dit=...GiB | vision=...GiB
| total=...GiB | duquant_layers=... | packed_layers=... | atm_env=1 | ohb_env=1
```

字段说明：
- `server_model_component_llm_bytes`：LLM（语言模型）组件显存
- `server_model_component_dit_bytes`：DiT（动作头）组件显存
- `server_model_component_llm_plus_dit_bytes`：LLM + DiT 合计
- `server_model_component_vision_bytes`：视觉 backbone 显存
- `server_model_component_backbone_other_bytes`：backbone 其他部分
- `server_model_component_action_head_other_bytes`：动作头其他部分
- `server_duquant_linear_modules`：总 DuQuant 层数
- `server_duquant_packed_modules`：packed 存储的 DuQuant 层数
- `server_atm_env_enabled`：ATM 是否启用
- `server_ohb_env_enabled`：OHB 是否启用

## 本地基准测试（无需 LIBERO 仿真）

此模式仅加载模型并使用合成 LIBERO 观测值测量延迟、显存、FLOPs 以及 action dump — 适用于快速检查设备是否能运行流水线。

Baseline：

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_local_benchmarks.sh baseline libero_10
```

Selective FP8：

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_local_benchmarks.sh fp8 libero_10
```

QuantVLA Full：

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_local_benchmarks.sh full libero_10
```

可通过环境变量控制本地基准测试的迭代次数：

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e QVLA_WARMUP=3 \
  -e QVLA_ITERS=10 \
  -e QVLA_ACTION_SAMPLES=16 \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_local_benchmarks.sh full libero_10
```

本地基准测试会运行以下工具：
- `benchmark_gr00t_latency.py`：延迟测量
- `benchmark_gr00t_memory.py`：显存测量
- `benchmark_gr00t_flops.py`：FLOPs 估算
- `collect_gr00t_actions.py`：Action dump（用于精度对比）

## 输出文件

LIBERO 真机评测输出：

```bash
results/benchmarks/<variant>/<task>/flops.json
results/benchmarks/<variant>/<task>/memory.json
results/benchmarks/<variant>/<task>/success.json
results/benchmarks/<variant>/<task>/real_libero_metrics.json
results/benchmarks/summary.csv
results/benchmarks/summary.md
```

运行时日志：

```bash
/tmp/logs/libero_eval_<variant>_<task>.log
```

日志中每行 `[latency]` 前缀记录了一个 step 的详细耗时；每行 `[tflops]` 记录了该 step 的等效密集 FLOPs。

由于 `/tmp/logs` 在容器内部，挂载外部目录以持久化：

```bash
-v "$PWD/logs:/tmp/logs"
```

## torch.compile 调优

### 启用/禁用

默认所有变体（baseline / duquant / full）都启用 `torch.compile`（`GR00T_TORCH_COMPILE=1`）。如需禁用：

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e GR00T_TORCH_COMPILE=0 \
  ... \
  quantvla:cuda13 \
  tools/run_quantvla_inference_server.sh baseline libero_10
```

### compile 模式

通过 `GR00T_TORCH_COMPILE_MODE` 控制：

- `default`（默认）：平衡编译时间与性能
- `reduce-overhead`：使用 CUDA graphs 减少 kernel launch 开销
- `max-autotune`：DiT action_head 默认使用此模式（固定形状可获最佳加速）

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e GR00T_TORCH_COMPILE=1 \
  -e GR00T_TORCH_COMPILE_MODE=reduce-overhead \
  ... \
  quantvla:cuda13 \
  tools/run_quantvla_inference_server.sh baseline libero_10
```

### DuQuant 模式下的 compile

DuQuant 自定义层在 traced graph 中可能产生 graph breaks，但周围的 Eagle blocks 仍可从编译中受益。如遇不稳定，可单独关闭 backbone compile：

```bash
-e GR00T_DUQUANT_COMPILE_BACKBONE=0
```

## 原生 Conda 运行

推荐使用 Docker。如果已有本地 conda 环境，也可直接运行：

```bash
conda activate groot_test
tools/run_quantvla_inference_server.sh baseline libero_10
```

另一个终端：

```bash
tools/run_quantvla_real_libero_benchmark.sh baseline libero_10
```

脚本会自动检测 conda；在 Docker 内部则直接使用容器 Python。

## 故障排查

如果容器无法看到 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

如果评测客户端无法连接到服务器，确认两个容器都包含：

```bash
--network host
```

如果首次运行非常缓慢，很可能是在下载 Hugging Face 模型或生成 DuQuant packed cache。始终挂载：

```bash
-v "$HOME/.cache/huggingface:/root/.cache/huggingface"
-v "$PWD/results:/workspace/QuantVLA/results"
```

如果 FP8 模式下性能无明显提升，检查是否在非 Blackwell GPU 上运行：`torch._scaled_mm` 在 Ada/Hopper 上的 FP8 加速有限。阈值 `_MIN_FP8_OPS` 可按需在 `gr00t/quantization/fp8_linear.py` 中调整。
