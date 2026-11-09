# TinyInfiniTrain

TinyInfiniTrain 是一个以 C++20 编写的教学型深度学习训练框架。仓库包含 Tensor/autograd、CPU/CUDA kernel dispatcher、Tiny GPT-2 训练，以及基于 NCCL 的张量并行、流水线并行和 ZeRO。

## 已实现能力

- Tensor 与 autograd：FP32、FP16、BF16 和 E4M3 FP8 存储，shape/view/transpose/slice；显式拓扑排序反向传播、分支梯度累积、`retain_graph`、`create_graph` 和 activation checkpoint。
- CPU/CUDA dispatcher：通过 `REGISTER_KERNEL` 按 device 路由；cuBLAS FP32/FP16/BF16 GEMM 与 strided-batched GEMM、FP32 master-weight Adam、RMSNorm、RoPE 和 SwiGLU。FP32 SwiGLU 在满足对齐条件时使用 `float4` 路径。
- 端到端 Tiny GPT-2：二进制 Tiny Shakespeare datapipe/tokenizer、FP16 forward/backward/Adam、启用 block activation recomputation 的 3 epoch 训练回归、BF16 + FP32 master-weight 训练，以及 PyTorch 单步更新前后的 logits parity fixture。
- 分布式训练：Megatron 风格 Column/Row Parallel Linear 已接入完整 GPT-2 attention/MLP；GPipe 和非交错 1F1B；PipeDream 每 micro-batch 更新与 weight stashing；ZeRO-1/2/3 参数、梯度和 optimizer-state 分片。
- 混合精度与显存：BF16/FP16 FP32 master weights；E4M3 动态缩放 fake quant、STE backward、FP32 accumulation 与多步优化回归；真实 CUDA event/allocator high-watermark 流水线 benchmark。

FP8 路径是软件 E4M3 fake quant（用于可移植的混合精度实验），不是 Hopper 硬件 FP8 GEMM。当前不包含 sequence parallel 和 interleaved virtual pipeline stages。

## CPU 构建与验证

```bash
cmake -S . -B build/Release \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TEST=ON
cmake --build build/Release -j
ctest --test-dir build/Release --output-on-failure
```

仓库包含 `Data/tiny_gpt2_fp16_reference.bin`。如需重新生成确定性 fixture，所用 Python 环境需已安装 PyTorch：

```bash
python3 tools/generate_gpt2_reference.py \
  --output Data/tiny_gpt2_fp16_reference.bin
./build/Release/test_gpt2_parity
```

## CUDA/NCCL 构建与多卡验证

需要 CUDA Toolkit、cuBLAS、NCCL，以及至少两张可见 NVIDIA GPU；BF16 测试需要 Ampere（`sm_80`）或更新架构。
默认编译 `sm_75/sm_80/sm_86`；Ada/Hopper 可在配置时显式传入例如 `-DCMAKE_CUDA_ARCHITECTURES=89` 或 `90`。

```bash
cmake -S . -B build/cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TEST=ON \
  -DUSE_CUDA=ON \
  -DUSE_NCCL=ON
cmake --build build/cuda -j

# 单进程 CUDA kernel、Transformer primitive、mixed-precision 与 Adam
ctest --test-dir build/cuda --output-on-failure

# 多进程 TP/PP/PipeDream/ZeRO（test_distributed_nccl 不作为普通 CTest 运行）
python3 tools/launch_distributed_tests.py \
  --binary build/cuda/test_distributed_nccl \
  --gpus 0,1
```

分布式 launcher 会依次运行 primitive TP、完整 GPT-2 TP、ZeRO-1/2/3、BF16 ZeRO-2，以及 GPipe/1F1B/PipeDream 集成测试。每个模式使用独立的文件 rendezvous 和 run id，避免旧 NCCL unique-id 文件污染新进程。

## 实测流水线 bubble/显存

`pipeline_benchmark` 执行真实 Tiny GPT-2 forward/backward，通过 CUDA events 统计 compute/elapsed time，并读取默认 CUDA memory pool 的当前值和 high watermark：

```bash
python3 tools/benchmark_pipeline.py \
  --binary build/cuda/pipeline_benchmark \
  --pipeline-depths 2 4 \
  --microbatches 1 2 4 8 16 \
  --schedules gpipe 1f1b \
  --activation-recomputation off on \
  --micro-batch-size 2 \
  --sequence-length 128 \
  --layers-per-stage 2 \
  --n-head 8 \
  --n-embd 256 \
  --vocab-size 512 \
  --gpus 0,1,2,3 \
  --output pipeline_benchmark.csv
```

输出包含 `measured_bubble`、`activation_delta_bytes`、`peak_inflight` 和 `activation_recomputation`，可直接比较 pipeline depth、micro-batch 数量以及 block 重计算带来的 compute-memory 取舍。`tools/parallel_analysis.cc` 仍保留为解析公式模型，不作为实测结果。

无多卡 GPU 时可以先运行 `./build/Release/parallel_analysis` 检查解析公式表；该输出明确是容量规划估算，不能替代上述 CUDA/NCCL 实测 CSV。

## 代码入口

- Tensor/autograd：`infini_train/include/tensor.h`、`infini_train/src/autograd/function.cc`
- Dispatcher 与 kernels：`infini_train/include/dispatcher.h`、`infini_train/src/kernels/`
- GPT-2：`example/gpt2/net.cc`
- TP/PP/NCCL：`infini_train/src/autograd/linear_parallel.cc`、`infini_train/src/distributed/`
- ZeRO/Adam：`infini_train/src/optimizer.cc`
- 测试：`test/`

其他背景文档见：[环境部署](docs/项目部署.md)、[训练营作业介绍](docs/训练营作业介绍.md)、[作业报告模版](docs/TinyInfiniTrain%20作业报告.md)。
