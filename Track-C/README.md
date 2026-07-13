# 赛道 C：编译器 & Runtime & 算子调度

> 编译 AEC ISA 机器码、驱动虚拟 GPGPU 设备、部署深度学习模型。
> 三个子题 C1/C2/C3 各为独立评分，归一化后等权平均到 100 分。

## 赛题列表

| 编号 | 赛题 | 赛题说明 | 评分细则 | 公开测试集 |
|------|------|----------|----------|-----------|
| C1 | AEC IR 编译器 | [spec.md](C1-compiler/spec.md) | [scoring.md](C1-compiler/scoring.md) | [testcases/](C1-compiler/testcases/) |
| C2 | 主机侧驱动与 Runtime | [spec.md](C2-runtime/spec.md) | [scoring.md](C2-runtime/scoring.md) | [starter-kit/](C2-runtime/starter-kit/) |
| C3 | 算子调度与模型部署 | [spec.md](C3-scheduler/spec.md) | [scoring.md](C3-scheduler/scoring.md) | [testcases/](C3-scheduler/testcases/) |

## 性能参考目标参数

Track-C 可参考统一的性能目标参数进行优化，详见：[hint.md](hint.md)

## 赛道总览

赛道 C 聚焦于 **AEC GPGPU 的软件栈**，从编译器到底层驱动再到高层算子调度：

```text
C1 编译器        -> PTX 风格 IR -> AEC ISA 机器码
C2 Runtime       -> libaec.so + 虚拟设备驱动
C3 算子调度      -> ONNX 模型 -> AEC GPGPU 推理
```

## 评分总则

**赛道 C 分为 C1、C2、C3 三道赛题，每道原始满分均为 100 分。最终赛道总分为三道题归一化后等权平均，满分为 100 分。**

```text
赛道总分 = (C1归一化分 + C2归一化分 + C3归一化分) / 3
C1归一化分 = (C1原始得分 / 100) x 100
C2归一化分 = (C2原始得分 / 100) x 100
C3归一化分 = (C3原始得分 / 100) x 100
```

> **示例**：C1 得 75 分、C2 得 60 分、C3 得 80 分，则赛道总分 = (75 + 60 + 80) / 3 = 71.67 分。

| 赛题 | 原始满分 | 评分构成 |
|------|----------|----------|
| C1 AEC IR 编译器 | 100 | 正确性 50 + 性能 35 + 鲁棒 5 + Agent 10 |
| C2 主机侧驱动 | 100 | Runtime 30 + 计算库 30 + Driver 20 + Agent 20 |
| C3 算子调度 | 100 | 图解析 10 + 分解 15 + 融合 15 + 内存 10 + 端到端 50 |

---

## C1：AEC IR 编译器

### 任务

将 PTX 风格中间表示编译为 AEC ISA 机器码，支持指令调度、寄存器分配、多精度 GEMM 优化等编译 pass。

### 命令行接口

```bash
aec-cc input.ptx -O2 -o output.aecbin
aec-objdump output.aecbin
```

### 测试类别

| 类别 | 数量 | 主要测试能力 | 性能分 |
|------|------|--------------|--------|
| T1 Basic Lowering | 20 | PTX 解析、基础指令 Lowering | 0 |
| T2 Control & Scalar Opt | 20 | CFG、谓词、DCE、CSE、LICM | 5 |
| T3 Memory Opt | 20 | 内存访问、合并访问、Shared Memory | 9 |
| T4 Register & Scheduling | 20 | 寄存器分配、Spill、DDG、List Scheduling | 10 |
| T5 Tensor / GEMM | 20 | TMUL、Tensor Load/Store、Tiling | 11 |

### 公开测试

5 道代表性 PTX 题，见 [testcases/](C1-compiler/testcases/)：

| 编号 | 文件 | 类别 | 主题 |
|------|------|------|------|
| PTX-01 | `PTX-01_vector_add.ptx` | T1 | Vector Add, FP32 |
| PTX-02 | `PTX-02_invariant_poly.ptx` | T2 | Loop Invariant, CSE, DCE |
| PTX-03 | `PTX-03_repeated_reuse.ptx` | T3 | Load Reuse, Shared Memory |
| PTX-04 | `PTX-04_reg_schedule.ptx` | T4 | Live Interval, DDG, Dual Issue |
| PTX-05 | `PTX-05_gemm_f16.ptx` | T5 | TMUL Lowering, FP16 Tiling |

---

## C2：主机侧驱动与 Runtime

### 任务

实现 AEC 虚拟 GPGPU 的 Host Runtime：`libaec.so`。

冲击 Excellent 时再提交 Agent 代码。

本题使用**确定性虚拟设备**，不使用真实 FPGA、Linux 内核驱动或物理 PCIe 性能。所有性能分均使用虚拟周期。

### Starter Kit

完整 starter kit 位于 [starter-kit/](C2-runtime/starter-kit/)，包含：

- 公共头文件（`include/aec_runtime.h`、`aec_isa.h`、`aec_device_abi.h`）
- 固定 Kernel image（34 个 image）
- 受控虚拟设备（`lib/libaec_device.so`）
- 起始代码（`src/aec_runtime.cpp`）
- 示例程序（`examples/01_device_query.c` ~ `06_registered_copy.c`）
- 公开测试（`cases/test_r101.py` ~ `test_r402.py`）
- 评分脚本（`grader/public_grade.py`）
- 详细文档（`docs/`）

### 快速开始

```bash
cd starter-kit
make -j2
make examples
./bin/01_device_query
python3 grader/public_grade.py --submission . --profile public
```

### GEMM 支持矩阵

| 精度 | API |
|------|-----|
| FP4 E2M1 | `aecMatmulF4` |
| FP8 E4M3/E5M2 | `aecMatmulF8` |
| FP16 | `aecMatmulF16` |
| BF16 | `aecMatmulBF16` |
| FP32 | `aecMatmulF32` |
| FP64 | `aecMatmulF64` |
| INT4 | `aecMatmulI4` |
| INT8 | `aecMatmulI8` |
| INT32 | `aecMatmulI32` |

### 等级门槛

| 等级 | 必需能力 |
|------|----------|
| Basic | 查询/错误、内存、同步复制、Vector Add、FP32/INT32 GEMM |
| Good | Basic + Stream/Event、异步 DMA、注册内存、全部计算、故障恢复 |
| Excellent | Good + 两个合法 Agent，并取得足够的性能分 |

---

## C3：算子调度与模型部署

### 任务

实现算子调度层，将典型深度学习模型的计算图解析、优化并调度到 AEC GPGPU 上执行。

### 子任务分解

| 子任务 | 分值 | 任务 |
|--------|------|------|
| C3.1 | 10 | 计算图解析与表示（ONNX -> DAG JSON） |
| C3.2 | 15 | 算子分解与内核选择（微基准测试） |
| C3.3 | 15 | 算子融合与图优化（微基准测试） |
| C3.4 | 10 | 内存规划与调度（Code Review） |
| C3.5 | 50 | 典型模型部署（端到端测试） |

### 命令行接口

```bash
# C3.1 计算图解析
<选手程序> --onnx <model.onnx> --output <dag.json>

# C3.5 端到端推理
<选手程序> --onnx <model.onnx> --input <input_dir> --output <output_dir> --batch-size 256
```

### 评测模型

| 模型 | 任务 | 输入形状 | 准确率阈值 |
|------|------|----------|-----------|
| MLP | MNIST 手写数字分类 | `[N, 1, 28, 28]` | >= 98% |
| ResNet-18（简化） | CIFAR-10 图像分类 | `[N, 3, 32, 32]` | >= 85% |
| Transformer（decoder-only） | 合成序列任务 | `[N, 18]`（int64） | -- |

### 支持的 ONNX 算子（17 种）

`Add`、`Constant`、`Conv`、`Div`、`Erf`、`Flatten`、`Gather`、`Gemm`、`GlobalAveragePool`、`LayerNormalization`、`MatMul`、`Mul`、`Relu`、`Reshape`、`Softmax`、`Split`、`Transpose`

---

## 提交规范

| 子题 | 必交文件 |
|------|----------|
| C1 | `compiler/aec-cc` + `disassembler/aec-objdump` + `agent/run_agent` + README |
| C2 | `libaec.so`（+ Agent 代码冲击 Excellent） |
| C3 | 推理程序 + 命令模板 + 源码 |

## 环境约束

| 子题 | 环境 |
|------|------|
| C1 | 8 cores / 16 GB / 180s 编译超时，Docker 运行环境 |
| C2 | little-endian Linux，C ABI，C++17 |
| C3 | GPU 可用（NVML 监控），Python 环境 |
