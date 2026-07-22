# U280-GPGPU 研究竞赛赛题说明

## 1. 赛题背景与目标

本赛题要求参赛队伍在单张 Alveo U280 FPGA 加速卡上实现一套可编程 GPGPU 原型。参赛系统应覆盖从上层模型到硬件执行的完整软件栈：

```text
PyTorch 模型/算子
      ↓
图或算子后端、运行时
      ↓
PTX 受限子集 / AEC 汇编 / AEC 机器码
      ↓
编译、调度与 kernel launch
      ↓
XDMA 驱动与主机运行时
      ↓
U280 上的 GPGPU RTL
      ↓
HBM2 / DDR4
```

竞赛重点不是复刻一颗商品 GPU，而是在明确接口和正确性约束下，研究软硬件协同设计。参赛者可以自主决定计算单元数量、SIMT 宽度、寄存器容量、缓存层级、GEMM 阵列规模、数据精度和 HBM 通道分配，并通过端到端工作负载证明设计选择。

本赛题有两个固定接口边界：

1. **AEC-G ISA 是软件栈与硬件之间的固定接口**。RTL 必须执行本文定义的 AEC-G v1.0 指令语义；可以在硬件内部使用微码、专用阵列或私有控制信号，但不得改变 AEC 指令编码、ABI 或可观察行为来适配特定模型。
2. **AEC runtime 是 PyTorch/应用与设备之间的固定 CUDA-like 接口**。参赛的 PyTorch 算子库和端到端调度必须通过本文定义的 runtime API 完成内存管理、拷贝、module load、kernel launch、同步、错误和计数器访问；不得要求评测程序调用队伍私有的模型专用接口。

比赛目标包括：

1. 在 U280 上实现可综合、可布局布线并稳定运行的 GPGPU RTL；
2. 支持基本 VALU、寄存器堆、取指/译码、SIMT 分支发散与收敛、访存及 kernel launch；
3. 至少实现一种面向神经网络的低精度 GEMM 路径，必选 FP8，允许 FP16/BF16/FP32 累加；
4. 使用 U280 HBM2 作为主要设备内存，允许使用 DDR4 保存大容量权重、数据集或溢出数据；
5. 完成基于 XDMA 的驱动、固定 runtime API、PTX-to-AEC 编译工具和 PyTorch 自定义算子接入；
6. 以算子测试、ResNet 类网络和 Transformer/大模型子图或模型进行正确性、性能、能效和可编程性评测。

## 2. 规范等级、边界与基本原则

### 2.1 规范等级

本文使用四类规范词，所有裁决均按这些等级执行：

| 等级 | 含义 |
|---|---|
| **必须** | 资格、正确性、接口或评分所需的强制条款；不满足时按本文件规定扣分、记 0 分或取消成绩 |
| **禁止** | 规则不接受的行为；出现结果伪造、越权计算或绕过接口时取消成绩 |
| **允许** | 参赛者可以采用的实现方式；采用后仍必须满足正确性、计时和可审查要求 |
| **参考** | 赛题给出的可实现配置、工程经验或优化方向；不作为资格上限，也不直接计分 |

未使用上述规范词的说明性文字只用于解释上下文；若说明性文字与规范条款冲突，以规范条款、机器可执行评分脚本和固定版本测试 manifest 为准。

### 2.1.1 术语表

本文区分体系结构可见的逻辑线程组织和硬件内部的物理执行宽度：

| 术语 | 定义 |
|---|---|
| `logical_warp_width` | 每个 warp 的体系结构线程数。AEC-G v1.0 正式评分配置固定为 32。 |
| `physical_simd_lanes` | 硬件每个周期实际执行的 SIMD 线程数。它是微架构参数，不改变 ISA 可观察语义。 |
| `issue_beats_per_warp` | 执行一个逻辑 warp 所需的 issue beat 数，必须满足 `issue_beats_per_warp = logical_warp_width / physical_simd_lanes`，且除法必须整除。 |

除非特别说明，本文中的 lane、thread、`%laneid`、predicate 位、active mask 位和 warp 内交换/归约均指逻辑 warp 内的逻辑线程。物理 SIMD lane 只是在多个 issue beat 中复用的执行资源，不能改变寄存器、predicate、mask、shuffle、reduction 或 MMA fragment 的体系结构含义。

### 2.2 必须完成

- 必须在同一张 U280 的指定 XDMA 动态区内放置所有计分硬件逻辑；
- 必须使用 RTL 或可审查的 RTL 生成流程实现核心处理器；
- 必须实现 AEC-G v1.0 固定 ISA/ABI；允许的扩展只能通过 capability 声明，不得改变基础指令编码、语义或错误行为；
- 必须由 AEC 指令流触发模型计算，包括访存、SIMT 标量指令、MMA、SFU、reduction 和同步；
- 必须实现本文定义的固定 CUDA-like runtime API，提供统一 kernel launch、内存分配、数据传输、同步和错误返回接口；
- 必须自行实现 PTX 到 AEC-G ISA 的映射、汇编/装载和报告生成；CUDA `.cu` 到 PTX 的前端步骤允许调用 `nvcc`；
- 必须自行实现 runtime 以及基于 XDMA 的驱动/控制路径，完成主机到 U280 动态区的命令提交、DMA、同步、错误恢复和计数器读取；
- 必须通过公开测试、随机测试、隐藏测试和赛后生成的未知 kernel 测试；
- 必须提交可复现 RTL 代码以及工程、bitstream/xclbin、软件栈源码、资源/时序报告、评分日志和设计说明。

### 2.3 禁止事项与反取巧

- 禁止根据测试用例名称、模型名称、shape、kernel 名称、`.aecbin` 机器码特征、输入哈希、输入摘要、样本 ID、文件偏移或隐藏数据统计特征返回预计算结果；
- 禁止把 ResNet、Transformer 或固定 shape 实现为绕过 AEC 指令语义的硬连线状态机；专用 MMA/SFU/reduction 单元只有通过 AEC 指令或由 AEC kernel 写入的描述符启动时才计分；
- 禁止在计时区间由 CPU 执行依赖输入、权重、激活或 KV cache 的数值计算，主机侧白名单见 8.2 节；
- 禁止直接实例化完整闭源 DPU/GPU/NPU 核代替参赛设计；
- 禁止修改板卡电气参数、散热保护、平台静态区、驱动计时工具或评分脚本；
- 禁止联网、读取隐藏参考输出、跨轮次复用隐藏输入的中间结果，或只上报成功轮次。

隐藏测试包含不可枚举的私有输入、随机 shape、随机边界值、赛后生成的 AEC/PTX kernel，以及与公开模型同分布但不同样本顺序的数据。出现结果伪造、越权 CPU 计算或硬编码输出时，取消成绩。

### 2.4 允许事项

- 允许调用 AMD/Xilinx 提供的存储、PCIe/XDMA、时钟、FIFO、浮点、AXI、HBM/DDR 控制器等基础 IP；
- 允许 HLS 实现非核心辅助模块；取指、调度、SIMT、寄存器访问、主要执行通路、MMA/SFU 接口和内存一致性逻辑必须可审查；
- 允许按 manifest 声明的 capability 选择不同 AEC kernel、tile shape、数据布局或量化模式；
- 允许使用专用 MMA、SFU、reduction、copy engine、prefetcher 和 DMA engine，但其输入、输出、地址、shape 和同步必须由 AEC 程序或 ABI 描述符显式给出；
- 允许使用 DDR4 保存大容量权重、数据集或溢出数据；计分模型的主执行路径必须报告 HBM 与 DDR 的实际访问量。

### 2.5 非必需范围

- 允许不完整实现 CUDA、PTX 或 IEEE 754 的所有模式；
- 允许不实现图形流水线、纹理单元、光栅化、虚拟内存分页或抢占；
- 允许不在每个计算单元内同时放置完整 FP32、FP16、FP8、整数和 SFU 数据通路；
- 允许不支持训练；基础赛道以推理为主，训练或反向传播只计入开放加分；
- 允许不支持任意 PyTorch 模型无修改运行，但必须覆盖公布的算子集合和模型评测入口。

最终以指定平台上的 routed timing、板上正确性、端到端计时和评分脚本输出为准；理论资源估算和仿真性能不能代替实测。

### 2.6 固定接口与参赛边界

本赛题只固定体系结构接口和 runtime 接口，不固定参赛队伍的微架构实现。

**固定 ISA 接口**：

- AEC-G v1.0 是计分硬件唯一接受的程序接口，隐藏测试可以直接生成 AEC 指令流并在板上执行；
- RTL 可以把 AEC 指令翻译为内部微操作，也可以用专用 MMA/SFU/reduction 单元加速，但提交系统对外必须表现为执行同一套 AEC 指令；
- 参赛队伍不得重定义 opcode、寄存器语义、predicate/active-mask 语义、MMA fragment 布局、地址空间、异常码或 `.aecbin` 格式；
- 可选指令或更宽访存只能作为 capability 声明的扩展，不能成为通过基础测试和正式模型的必要条件。

**固定 runtime 接口**：

- 第 8 节定义的 AEC runtime API 是计分程序调用参赛系统的固定 CUDA-like 接口；
- PyTorch package、算子库、模型调度、autotuning 和 benchmark 脚本都必须经由该 runtime API 进行设备内存管理、H2D/D2H、module load、kernel launch 和同步；
- 允许 runtime 内部使用队伍自定义的命令队列、描述符、缓存策略和 DMA 分块策略，但这些实现细节不得改变 API 语义；
- 禁止要求评测脚本绕过 runtime 直接写 MMIO、调用私有 ioctl、调用模型专用驱动入口，或以非公开接口触发特定网络的硬连线执行。

**提交边界**：

1. PyTorch 侧提交面向参赛硬件设计优化的算子库，以及 ResNet/Transformer 等端到端网络的调度、融合、分块、布局和量化方法；
2. 编译侧允许用 `nvcc` 完成 CUDA `.cu` 到 PTX，PTX 到 AEC-G ISA 的解析、lowering、寄存器分配、指令选择和 `.aecbin` 生成必须由参赛队伍提交源码实现；
3. 系统软件侧必须提交 runtime 实现，以及基于 XDMA 的驱动/控制路径实现；
4. 硬件侧必须提交 RTL、U280 XDMA 平台集成工程和在 U280 FPGA 上通过实现、时序与板上测试的 bitstream/xclbin。

## 3. 官方平台与可用资源

U280 量产卡的公开规格为：1,304K LUT、2,607K 寄存器、9,024 DSP、2,016 个 BRAM、960 个 URAM；板载 8 GB HBM2，标称总带宽 460 GB/s；板载 32 GB DDR4，标称总带宽 38 GB/s；PCIe 支持 Gen3 x16 或 Gen4 x8。来源见 AMD [U280 数据手册 DS963](https://docs.amd.com/r/en-US/ds963-u280/Alveo-Product-Details)。

芯片资源并不等于用户逻辑可用资源。官方 U280 Gen3x16 XDMA base 平台扣除静态区后，动态区按三个 SLR 给出的资源如下（不同平台和工具版本可能略有差异）：

| 资源 | SLR0 | SLR1 | SLR2 | 合计 |
|---|---:|---:|---:|---:|
| CLB LUT | 386K | 364K | 381K | 1,131K |
| CLB Register | 773K | 729K | 763K | 2,265K |
| BRAM36 | 600 | 576 | 600 | 1,776 |
| URAM | 320 | 320 | 320 | 960 |
| DSP48E2 | 2,664 | 2,784 | 2,856 | 8,304 |

上述动态区数字来自 AMD [U280 Gen3x16 XDMA base 平台说明](https://docs.amd.com/r/en-US/ug1120-alveo-platforms/U280-Gen3x16-XDMA-base_1-Platform)。必须使用 3.1 节的固定环境，内容包括卡型、shell、Vivado/Vitis 版本、XRT 版本、时钟约束、散热条件和实际动态区预算；固定版本报告是资源裁决的唯一依据。

### 3.1 固定评测环境

正式评测、复测、资源报告和时序报告必须使用以下固定环境。参赛提交的构建脚本、驱动/runtime 和 bitstream 必须与该环境兼容；使用其他版本得到的综合、实现或性能结果只能作为非计分参考。

| 类别 | 项目 | 固定值 |
|---|---|---|
| 服务器 | Model | NF5468M5 |
| 服务器 | CPU Cores | 64 |
| 服务器 | Memory | 768G |
| 操作系统 | OS Name | Linux |
| 操作系统 | Distribution | CentOS Linux 7.9 (Core) |
| 操作系统 | Release | `3.10.0-1160.108.1.el7.x86_64` |
| 操作系统 | Version | `#1 SMP Thu Jan 25 16:17:31 UTC 2024` |
| 操作系统 | Machine | `x86_64` |
| 操作系统 | GLIBC | `2.17` |
| FPGA 工具 | Vivado | `2022.2` |
| FPGA 工具 | Vitis | `2022.2` |
| XRT | Version | `2.13.479` |
| XRT | Branch | `2022.1` |
| XRT | Hash | `5e92a513c6950e79638b1a879ddb882da34fc683` |
| XRT | Hash Date | `2022-06-25 09:05:04` |
| XRT | XOCL | `2.13.479, 5e92a513c6950e79638b1a879ddb882da34fc683` |
| XRT | XCLMGMT | `2.13.479, 5e92a513c6950e79638b1a879ddb882da34fc683` |
| Shell | Platform | U280 Gen3x16 XDMA base_1 |

在评分日志中记录 `xbutil examine`、shell/platform 名称、shell UUID、BDF、XRT 版本、XOCL/XCLMGMT 版本、Vivado/Vitis 版本和 Linux kernel 版本。若实际 shell UUID 或动态区资源与本节不一致，必须在正式评测前更新固定环境表和 3.2 节资源表；否则该次结果不能作为正式成绩。

### 3.2 参考的保守实现

U280 是三 SLR 器件。GPGPU 的寄存器堆、宽数据通路、HBM AXI 互连和跨 SLR 控制网络具有较高布线压力，因此不能只根据全卡资源利用率判断可实现性。为给 platform、AXI/HBM 互连、clock/reset、跨 SLR pipeline、调试探针和后期 ECO 留出余量，参考 starter design 控制在以下保守范围内：

| 项目 | 参考包络 | 约占 XDMA 动态区 | 说明 |
|---|---:|---:|---|
| LUT | ≤ 565K | ≤ 50% | 参考设计保留约一半 LUT 余量 |
| Register | ≤ 1,133K | ≤ 50% | 包括复制寄存器和跨 SLR pipeline |
| BRAM36 | ≤ 888 | ≤ 50% | 避免 BRAM 列局部拥塞，不能只看全卡合计 |
| URAM | ≤ 480 | ≤ 50% | 每个 SLR 参考不超过 160 个 |
| DSP48E2 | ≤ 4,152 | ≤ 50% | 每个 SLR 参考不超过该 SLR 动态区的 55% |
| 用户时钟 | 基线 180 MHz，参考目标 200–225 MHz | — | 250 MHz 作为高级优化目标 |
| 卡功耗 | 禁止越过平台保护阈值 | — | 统一功耗模式和散热条件实测 |

表中 50% 是参考包络，不是资格上限；超过后仍允许参赛，但必须自行承担布局布线、跨 SLR 和功耗风险。资源利用率本身不计分，性能、正确性和效率才计分。

AMD 官方案例也表明需要保守规划：U280 上一个 DPUCAHX8L 使用约 212,860 LUT、299,342 registers、459 BRAM、312 URAM 和 2,452 DSP，见 [DPUCAHX8L Resource Utilization（PG366）](https://docs.amd.com/r/en-US/pg366-dpucahx8l/Resource-Utilization)；官方 Vitis BLAS 单 CU GEMM 示例使用约 198,418 LUT、66 BRAM、24 URAM 和 1,235 DSP，并在 300 MHz 构建，见 [Vitis BLAS GEMM Profiling](https://docs.amd.com/r/en-US/Vitis_Libraries/blas/user_guide/L2/L2_benchmark_gemm.html_0_1)。这些数字不是本赛题模块的直接估算，但说明一个成熟计算核本身即可消耗显著资源；再叠加 SIMT、cache 和多端口 HBM 后，需要保留充足余量。

### 3.3 资源自由分配规则

除 U280 XDMA 动态区的物理上限、平台保留资源、时序、功耗和温度保护外，组织方不规定各模块的资源配额。参赛者可以自由决定：

- CU 数量、每 CU `physical_simd_lanes`、`logical_warp_width` 和 resident warp 数；
- DSP 在 VALU、FP8 GEMM、SFU 插值和地址计算之间的分配；
- BRAM/URAM 在寄存器堆、L1、共享存储、L2、指令存储、SFU LUT 和 DMA buffer 之间的分配；
- HBM pseudo-channel 在权重、激活、KV cache、指令和中间结果之间的映射；
- 是否使用 DDR4 作为第二级容量存储；
- 通用 SIMT 执行与专用 GEMM/SFU/reduction 单元的面积比例。

不按“使用了多少 CU/DSP/缓存”直接加分，也不设置统一面积惩罚。只要实现后的 bitstream 在指定 shell 上通过时序、功耗、稳定性和正确性要求，资源可以由参赛者自主分配。该规则鼓励参赛者根据端到端瓶颈进行软硬件协同，而不是照抄参考架构。

## 4. 可行的参考 GPGPU 基线

**4 CU** 是正式 starter design 的参考规模；8 CU 仅作为参赛者自行探索的性能配置：

| 模块 | 参考配置 |
|---|---|
| Compute Unit（CU） | 4 个；SLR0/1/2 按 1/2/1 分布，或将控制/HBM 较重逻辑单独放置 |
| SIMT 宽度 | `logical_warp_width=32` threads；参考 `physical_simd_lanes=8` lanes；每个 warp 分 4 个 issue beat 执行 |
| Warp slots | 每 CU 4 个 resident warps；验证稳定后扩展到 8 个 |
| VALU | 每 CU 8×INT32 基本 lane；FP32 add/mul 可按 2–4 lane 共享或分时复用 |
| 寄存器堆 | 每 CU 64 KiB，32-bit word，banked；4 CU 共 256 KiB |
| Predicate | 每个逻辑线程至少 8 个 predicate bit；每个逻辑 warp 1 个 32-bit active mask |
| GEMM | 全卡 2–4 个 16×16 FP8 逻辑 tile，按 CU 共享或一一对应；允许多周期/分块实现 |
| L1D / Shared Memory | 每 CU 16 KiB shared + 8–16 KiB L1D，或统一 24–32 KiB |
| L1I | 每 CU 4–8 KiB，参考做法是两个 CU 共享一个 instruction cache |
| L2 | 全卡 1–2 MiB，4–8 bank；starter design 可先做 1 MiB 或显式 scratchpad |
| SFU | 每 2–4 个 CU 共享一个近似单元，最低支持 reciprocal、exp2 或其软件/查表实现 |
| HBM | 起步使用 4–8 个 AXI master 端口；验证后再扩展，数据按 bank 交错 |
| DDR | 可选，用于大容量权重或冷数据；基础正确性禁止依赖主机反复介入 |

### 4.1 粗略资源预算

以下为赛题规划量级，不是综合承诺。最终使用量强烈依赖位宽、复制端口、FP8 格式、累加精度、缓存 associativity、IP 选项和频率。

| 模块 | 参考规模 | LUT | BRAM36 | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| 4×SIMT 前端、调度、VALU、LSU | 4 CU × 8 lanes | 80K–140K | 64–128 | 16–48 | 120–300 |
| 4×寄存器堆及 scoreboard | 256 KiB 总容量 | 20K–45K | 0–64 | 24–48 | 0 |
| FP8 GEMM 数据通路 | 2–4×16×16 逻辑 tile | 50K–110K | 32–80 | 0–24 | 512–1,024 |
| L1/共享存储 | 约 96–128 KiB | 20K–50K | 32–96 | 0–16 | 0 |
| 分 bank L2 | 1–2 MiB | 30K–75K | 32–128 | 32–80 | 0 |
| HBM/DDR/AXI、DMA、命令处理 | 4–8 HBM ports | 60K–120K | 64–160 | 0–16 | 0–32 |
| SFU、性能计数器、调试 | 1–2 个共享 SFU | 15K–40K | 8–32 | 0–8 | 16–80 |
| **合计估计** |  | **275K–580K** | **232–688** | **72–240** | **648–1,436** |

starter design 必须以约 **350K–450K LUT、350–550 BRAM36、120–200 URAM、800–1,200 DSP 和 180–200 MHz** 为可实现目标区间。只有 starter design 完成 routed timing、板上长稳和端到端闭环后，参赛者才允许逐步提高到 225–250 MHz、增加 GEMM tile 或扩展 CU。上述区间上沿已经接近参考 LUT 包络，官方参考设计禁止同时把所有模块都按上限配置。

参考扩展策略为：首先实现 2 CU/8-lane + 2 个 GEMM tile；随后扩展到 4 CU；最后根据实现报告只增加真正受限的资源。8 CU、16-lane、2–4 MiB L2 或 8 个 16×16 tile 均属于高风险性能方案。

### 4.2 理论吞吐量的统一口径

理论峰值仅用于报告，不直接替代实测。若设计有 `Nmac` 个每周期可接受一组输入的 FP8 MAC，时钟为 `f`，并按乘和加各算一次操作：

```text
Peak_OPS = 2 × Nmac × f
```

例如 4 个 16×16 tile、200 MHz、每个逻辑 MAC 每周期接收一组操作时，理论峰值为：

```text
2 × 4 × 256 × 200 MHz = 0.4096 TOPS
```

若一个 DSP 通过 packing 每周期完成多个 FP8 操作，必须给出位级正确性证明、实际启动间隔和实现报告。吞吐量按实际 initiation interval 修正。

## 5. 硬件架构要求

### 5.1 命令处理与 kernel launch

设备至少支持以下命令：

- 分配/登记设备 buffer；
- H2D、D2H 和可选 D2D copy；
- 加载 `.aecbin`；
- 写入参数区；
- 设置 `gridDim`、`blockDim` 和动态共享存储大小；
- launch、等待、超时、中止或复位；
- 读取状态、错误码、周期数、访存字节数和 cache miss 等计数器。

命令队列、doorbell、completion queue 和寄存器映射由平台 ABI 中固定。禁止以主机逐条驱动指令执行。

### 5.2 SIMT 与控制流

每个 CU 必须维护 warp PC、active mask 和 predicate 状态，并正确处理：

- 多 warp 驻留与切换；
- 逻辑线程级 predication；
- 条件分支发散；
- 分支收敛；
- 逻辑线程的 `HALT`；
- block 完成与 kernel 完成通知。

active mask 的位数必须等于 `logical_warp_width`。AEC-G v1.0 中 active mask 为 32 bit，每一位对应同一逻辑 warp 内的一个逻辑线程。若硬件用多个 issue beat 执行一个 warp，每个 beat 只读取与本 beat 逻辑线程编号对应的一段 mask，但完整 mask 始终属于同一个逻辑 warp，并由同一套分支收敛语义维护。

参考方案可采用 reconvergence stack、IPDOM、编译器插入的 `SSY/SYNC`，或显式 active-mask 指令。具体方案可自由选择，但同一 kernel 的可观察结果必须符合定义。

### 5.3 寄存器堆与 scoreboard

- 每个逻辑线程至少支持 32-bit GPR；64-bit 值使用相邻寄存器对；
- 每个逻辑线程至少支持 8 个 predicate bit；
- 允许寄存器 bank conflict 导致停顿，但禁止破坏结果；
- 必须处理 RAW/WAW 等相关；
- 允许编译器静态调度或硬件 scoreboard，须在报告中说明；
- 若支持 spill，必须明确 local memory ABI；基础赛允许限制无 spill kernel。

### 5.4 存储系统

必选地址空间：

| 空间 | 用途 | 最低要求 |
|---|---|---|
| `.gmem` | HBM/DDR 全局内存 | byte address，至少 32-bit 可用地址；参考内部使用 64-bit |
| `.pmem` | kernel 参数 | 只读或 launch 前写入 |
| `.smem` | block 内共享存储 | 同一 block 可见，支持 barrier 后通信 |
| `.lmem` | spill/线程私有 | 可选 |
| `.cmem` | 常量 | 可选 |

必须定义未对齐访问、越界访问、缓存一致性和 DMA 同步规则。最低实现可以只保证自然对齐访问；不支持的访问必须返回错误或在编译期拒绝，禁止静默产生错误结果。

L1/L2 的具体结构开放。至少需要一种可工作的缓存或显式 scratchpad 路径。若 DMA 与设备 cache 不保持硬件一致，运行时必须在 launch 边界执行 flush/invalidate 或使用不可缓存窗口。

### 5.5 HBM 使用要求

U280 的 8 GB HBM 由多个 pseudo-channel 构成；Vitis 文档给出的 U280 单个 pseudo-channel 容量为 256 MB，详见 AMD [HBM Configuration and Use](https://docs.amd.com/r/2024.1-English/ug1393-vitis-application-acceleration/HBM-Configuration-and-Use)。参赛设计必须：

- 至少使用 4 个彼此独立的 HBM pseudo-channel；
- 参考性能配置使用 8–16 个端口，并在报告中给出地址到 bank 的映射；
- 避免所有 CU 经单一 AXI master 串行访问 HBM；
- 报告连续、随机、读写混合三类带宽及 bank conflict；
- 区分 HBM 标称 460 GB/s 与设计可达有效带宽。

### 5.6 FP8、MMA 与转换数值规范

必须支持 FP8 `E4M3FN` 输入格式，允许扩展 `E5M2`。禁止只实现名为 FP8、实为整数且无公开 scale 语义的路径。允许 block-wise、tensor-wise 或 channel-wise scaling；scale 必须作为 ABI 的显式输入，并在 golden model 中按同一顺序复现。

#### 5.6.1 FP8 编码

`E4M3FN` 为基础必选格式：

```text
bit[7]      sign
bit[6:3]    exponent
bit[2:0]    fraction
bias        7
```

其数值定义为：

| exponent | fraction | 数值 |
|---:|---:|---|
| 0 | 0 | signed zero |
| 0 | 1..7 | `(-1)^sign × 2^-6 × (fraction / 8)` |
| 1..14 | 0..7 | `(-1)^sign × 2^(exponent-7) × (1 + fraction/8)` |
| 15 | 0..6 | `(-1)^sign × 2^8 × (1 + fraction/8)` |
| 15 | 7 | canonical NaN；sign bit ignored |

`E4M3FN` 没有 `Inf`。有限最大绝对值为 448。转换到 `E4M3FN` 时，必须使用 round-to-nearest-even；上溢必须饱和为带符号最大有限值；NaN 必须映射为 canonical NaN；下溢按最近可表示 subnormal 或 signed zero 舍入。

`E5M2` 是可选格式：

```text
bit[7]      sign
bit[6:2]    exponent
bit[1:0]    fraction
bias        15
```

`E5M2` 遵循 IEEE-like 编码：`exponent=0` 表示 zero/subnormal，`1..30` 表示 normal，`31,fraction=0` 表示 `Inf`，`31,fraction!=0` 表示 NaN。启用 `E5M2` 的实现必须在 capability 中声明，并通过同一转换测试。

#### 5.6.2 舍入、异常值和转换

- 必须将 `.f32` 标量算术定义为 IEEE-754 binary32、round-to-nearest-even；本赛题不暴露 IEEE exception flags；
- 必须将 `.f16` 定义为 IEEE-754 binary16，将 `.bf16` 定义为 bfloat16，二者与 `.f32` 互转均使用 round-to-nearest-even；
- 必须在 `aec_g_isa_v1.json` 中列出所有合法 `CVT/PACK/UNPACK` pair；未列出的 pair 必须在编译期拒绝或在板上触发 `ILLEGAL_INSTRUCTION`；
- 必须对 NaN 使用 canonical NaN 输出，除非指令语义明确要求保留 payload；
- 必须在 manifest 中声明是否对 subnormal 使用 full subnormal、输入 flush-to-zero 或输出 flush-to-zero；正式评分按 manifest 和 golden simulator 的 feature bit 选择对应参考；
- 禁止在同一提交中对公开测试和隐藏测试使用不同舍入、饱和或 flush 规则。

#### 5.6.3 MMA 语义和专用单元边界

基础必选矩阵指令为：

```text
MMA.m16n16k16.e4m3.f32 D, A, B, C
```

必须满足：

- `A` 和 `B` 片段为 `E4M3FN`，`C` 和 `D` 片段为 FP32；
- 语义为 `D = A × B + C`，`m/n/k` 固定为 `16/16/16`；
- 每个元素的参考值按 `k=0..15` 的固定顺序，用 FP32 FMA 语义累加；
- scale 的应用顺序固定为 `real_A = fp8_to_f32(A) × scale_A`、`real_B = fp8_to_f32(B) × scale_B`，再执行乘加；
- `scale_A/scale_B` 可以是 tensor-wise、block-wise 或 channel-wise；scale 索引函数必须写入 manifest，并由 golden simulator 复现；
- 部分逻辑线程 predication 对 MMA 非法；MMA 必须由完整 32-thread 逻辑 warp 一致执行，任一逻辑线程因 active mask 或 predicate 未执行该 MMA 时必须触发 `ILLEGAL_INSTRUCTION` 或由编译器拒绝；
- 专用 MMA 单元允许使用 DSP packing、多周期 tile、脉动阵列或共享执行单元；启动、操作数描述符、scale、shape、同步和写回必须由 AEC 指令或 AEC kernel 写入的描述符决定；
- 禁止 MMA 单元根据模型名称、kernel 名称、shape 或输入 fingerprint 选择预计算输出。

#### 5.6.4 数值误差公式

默认公式为：

```text
abs_err_i = abs(y_i - y_ref_i)
rel_err_i = abs(y_i - y_ref_i) / max(abs(y_ref_i), epsilon)
max_abs_err = max_i(abs_err_i)
max_rel_err = max_i(rel_err_i)
NRMSE = sqrt(mean_i((y_i - y_ref_i)^2)) / max(stddev_i(y_ref_i), epsilon)
```

分类模型必须输出 logits，并由评分脚本统一计算 Top-1/Top-5。LLM 的 perplexity、token match 和 EOS 处理见 10.3.2。缺失输出、NaN/Inf 分类错误、shape 不匹配或返回码未成功的测试项记 0 分。

### 5.7 SFU

SFU 允许不覆盖商品 GPU 的全部超越函数。基础赛必须提供可被 AEC 指令调用的 SFU 路径，并至少支持 `RCP` 和 `EXP2`；`RSQRT`、`TANH`、`SIGMOID`、`LOG2`、`SQRT`、`SIN/COS` 为可选扩展。模型中的 GELU、SiLU、Softmax、LayerNorm/RMSNorm 可以由 VALU、SFU 和 reduction 指令组合实现。

#### 5.7.1 允许的实现方式

允许采用查找表实现 SFU。参考实现形式包括：

1. 单级 LUT；
2. 粗查表加线性插值；
3. 分段 LUT 加低阶多项式；
4. LUT 提供初值，再进行一次 Newton-Raphson 迭代；
5. 可编程微码或上述方法的组合。

查找表可使用 BRAM、URAM、distributed ROM 或组合逻辑实现。表内容可以在综合时固化，也可以在 bitstream 加载或设备初始化阶段写入；运行过程中若允许更新，必须限制为特权 runtime 操作，并保证正式评测时内容固定。允许多个 CU 共享 SFU，也允许每个 CU 独立配置，但共享结构必须提供仲裁、反压和结果 tag；禁止因不同 warp 请求交错而写回错误 lane。

#### 5.7.2 最低数据格式与执行语义

- SFU 输入和输出必须支持 FP32；允许内部使用 FP16、定点或自定义尾数格式；
- SFU 指令必须服从逻辑线程的 active mask 和 predicate；inactive 逻辑线程禁止改变目标寄存器；
- 必须正确处理流水线 back-pressure、warp 切换和多条在途请求；
- 允许可变延迟，但硬件必须通过 scoreboard 或 ready/valid 机制阻止过早读取结果；
- 编译器和模拟器必须使用相同的特殊值、舍入、饱和及 flush-to-zero 规则；
- NaN、正负 Inf、正负零、负数非法定义域和 subnormal 的行为必须在 ISA 附录中列出。

基础语义固定为：

| 指令 | 正常输入域 | 期望结果 | 特殊情况 |
|---|---|---|---|
| `RCP.f32` | 有限非零 `x` | `1/x` | `±0 → ±Inf`，`±Inf → ±0` |
| `EXP2.f32` | 有限 `x` | `2^x` | 上溢为 `+Inf`，下溢可按规定 flush 为 `+0` |
| `RSQRT.f32` | `x > 0` | `1/sqrt(x)` | `+0 → +Inf`，负数 → canonical NaN |

#### 5.7.3 LUT 区间归约要求

参赛者必须在设计说明中给出区间归约和重构方法。例如：

```text
EXP2: x = n + f, n = floor(x), f ∈ [0, 1)
      2^x = 2^n × LUT_or_poly(f)

RCP:  x = sign × 2^e × m, m ∈ [1, 2)
      1/x = sign × 2^(-e) × LUT_or_refine(m)

RSQRT: x = 2^e × m
       1/sqrt(x) = 2^(-floor(e/2)) × parity_scale × LUT_or_refine(m)
```

禁止只对模型中少数固定输入建立离散表。隐藏测试将在完整正常输入域内随机采样，并在分段边界、零附近、极大/极小值和特殊值处进行定向测试。

#### 5.7.4 精度要求

基础门槛固定如下；若正式评分镜像调整门槛，必须同步更新版本号、测试 manifest 和 golden simulator：

| 函数 | 基础精度门槛 | 性能组参考目标 |
|---|---:|---:|
| `RCP.f32` | 最大相对误差 ≤ `2^-10` | ≤ `2^-14` |
| `EXP2.f32` | 正常结果范围内最大相对误差 ≤ `2^-9` | ≤ `2^-13` |
| `RSQRT.f32`（若实现） | 最大相对误差 ≤ `2^-9` | ≤ `2^-13` |

相对误差定义为：

```text
relative_error = abs(y_dut - y_ref) / max(abs(y_ref), epsilon)
```

对接近零、上溢、下溢和特殊值的输入使用绝对误差或分类一致性单独判断。对由 SFU 构成的 Softmax、GELU、SiLU 和归一化算子将设置端到端误差门槛，防止单函数满足误差但组合后误差不可接受。

#### 5.7.5 吞吐量与资源报告

SFU 不要求每个逻辑线程每周期一份结果。参赛者可在吞吐量和资源之间折中，但必须报告：

- 每种函数的 pipeline latency 和 initiation interval；
- 每 CU 或全卡的每周期最大请求数；
- LUT 深度、表项位宽、插值或迭代次数；
- SFU 使用的 LUT、BRAM、URAM、DSP 和寄存器数量；
- 单 warp、多个 warp 和多个 CU 竞争共享 SFU 时的实测吞吐量；
- 不同输入分布下的最大误差、P99 误差和均方误差。

 starter design 可采用“每 2 个 CU 共享一个 SFU、BRAM ROM + 线性插值、每周期接收 1 个逻辑线程请求”的低成本实现。参赛者可通过多 bank 查表、复制 ROM、向量化查询、提高流水深度或编译器调度提升吞吐量。

## 6. AEC-G 固定 ISA 与可声明扩展

AEC-G v1.0 是本赛题固定的硬件接口。编译器、assembler、loader、runtime、公开测试和隐藏测试均以本章定义的编码、ABI、错误码和可观察语义为准。参赛硬件可以在内部采用任意微架构、微码或专用执行单元，但对外必须完整执行同一套 AEC-G v1.0 指令语义。

### 6.1 固定标量兼容集

参赛处理器必须兼容本文定义的 AEC 128-bit 定长格式：

```text
bits [127:112]  Opcode      16 bits
bits [111:96]   Pred/Ctrl   16 bits
bits [95:80]    Dest        16 bits
bits [79:64]    Src1        16 bits
bits [63:32]    Src2/Imm32  32 bits
bits [31:0]     ImmExt      32 bits
```

必选 opcode：`ADD/SUB/MUL/MAD/FMA`、`AND/OR/XOR/SHL/SHR`、`CMPP`、`LD/ST`、`BR/BRX/HALT`、`CPY/LOADI/LOADI64`。必选类型至少为 `.b32/.b64/.u32/.s32/.f32`；必选空间至少为 `.gmem/.pmem/.smem`。

原参考文本采用“全局地址高 32 位恒为 0”的抽象地址规则。硬件基础测试允许保留该规则，但运行时 API 必须使用 64-bit device address，并通过 buffer table、base register 或地址转换窗口映射到 AEC 地址，防止 8 GB HBM 和 32 GB DDR 在系统层面无法完整寻址。

### 6.2 正式固定扩展集

完整模型执行必须支持下列由赛题固定的扩展。编码由 AEC-G v1.0、机器可读 ISA 描述和 golden simulator 固定，禁止由参赛队伍各自解释：

| 类别 | 指令 | 语义 |
|---|---|---|
| 同步 | `BAR.SYNC id, count` | block 内 barrier |
| SIMT | `SSY pc`、`SYNC` 或等价 mask 指令 | 发散收敛 |
| 访存 | `LD/ST` 的 8/16/32/64-bit 类型 | 标量和成组搬运；128-bit 向量访存为可选扩展 |
| 原子 | `ATOM.ADD.u32` | 最低原子能力；可选 FP32 |
| 矩阵 | `MMA.FP8 dst, a, b, acc` | tile 级 FP8 乘加 |
| 转换 | `CVT`、`PACK`、`UNPACK` | FP8/FP16/FP32/整数转换 |
| reduction | `SHFL` 或 `REDUCE` | warp 内求和/最大值 |
| SFU | `RCP`、`EXP2`、可选 `RSQRT` | 允许 LUT/插值/迭代近似，语义和误差按 5.7 节 |
| 缓存 | `FENCE`、可选 prefetch | DMA/设备可见性 |

`MMA.FP8` 的 tile shape、寄存器布局、scale、累加精度和异常值规则必须由机器可读 ISA 描述和 golden model 同时定义。

AEC-G v1.0 基础 profile 不定义 `.b128` 访存类型。需要 128-bit 搬运的 kernel 必须由编译器 lowering 为多条 32/64-bit `LD/ST`，或使用设备在 capability 中声明的可选向量访存扩展；正式基础测试不得要求未声明扩展的实现支持 128-bit 单指令访存。

### 6.3 二进制、ABI 与 manifest

继续采用原参考格式：`.aecbin` 为无 header 的 128-bit 指令流，每条指令按四个 little-endian 32-bit word 写入。提交必须包含 manifest：

```json
{
  "kernel": "gemm_fp8",
  "gridDim": [128, 128, 1],
  "blockDim": [256, 1, 1],
  "dynamic_smem_bytes": 32768,
  "registers_per_thread": 48,
  "required_capability": {
    "isa_major": 1,
    "isa_minor": 0,
    "logical_warp_width": 32
  },
  "params": [],
  "buffers": {},
  "numeric_mode": {
    "input": "e4m3fn",
    "accumulate": "fp32",
    "output": "fp16"
  }
}
```

### 6.4 AEC-G ISA 正式编码

本节定义赛题使用的规范性指令集，命名为 **AEC-G v1.0**。若本节与前述概述或参考附件存在歧义，以本节为准。正式评分镜像发布后不再改变编码或语义。

#### 6.4.1 指令字与存储顺序

所有指令固定为 128 bit：

```text
127                    112 111                     96
+-------------------------+--------------------------+
| opcode[15:0]            | pred_ctrl[15:0]          |
+-------------------------+--------------------------+
95                      80 79                      64
+-------------------------+--------------------------+
| dst[15:0]               | src1[15:0]               |
+-------------------------+--------------------------+
63                                              32
+--------------------------------------------------+
| src2_or_imm32[31:0]                            |
+--------------------------------------------------+
31                                               0
+--------------------------------------------------+
| src3_or_immext[31:0]                           |
+--------------------------------------------------+
```

`.aecbin` 不包含文件头。每条指令按以下 32-bit little-endian word 顺序写入：

```text
w0 = instruction[31:0]
w1 = instruction[63:32]
w2 = instruction[95:64]
w3 = instruction[127:96]
```

PC 的单位是指令而不是 byte，顺序执行时 `PC_next = PC + 1`。

#### 6.4.2 通用寄存器编码

- `R0..R255`：256 个 32-bit GPR 编号，合法编码为 `0x0000..0x00ff`；
- `P0..P7`：每个逻辑线程独立拥有的 8 个 1-bit predicate，由 `pred_ctrl.pred` 或 `dst[2:0]` 编码；
- 64-bit 标量使用偶数对齐寄存器对 `{Rk+1, Rk}`，`Rk` 保存低 32 bit；
- 普通标量指令中寄存器字段的 `[15:8]` 必须为 0，否则为非法编码；
- `R0` 是普通可写寄存器，不是硬连线零；
- 每个逻辑线程拥有独立 GPR 和 predicate 状态；每个逻辑 warp 拥有独立 PC、active mask 和收敛状态。物理 SIMD lane 仅在不同 issue beat 中复用执行资源，不拥有体系结构可见状态。

Special register selector 使用 `src1` 字段：

| 编码 | 名称 | 编码 | 名称 |
|---:|---|---:|---|
| `0x0100` | `%tid.x` | `0x0110` | `%tid.y` |
| `0x0101` | `%ntid.x` | `0x0111` | `%ntid.y` |
| `0x0102` | `%ctaid.x` | `0x0112` | `%ctaid.y` |
| `0x0103` | `%nctaid.x` | `0x0113` | `%nctaid.y` |
| `0x0104` | `%laneid` | `0x0120` | `%tid.z` |
| `0x0105` | `%warpid` | `0x0121` | `%ntid.z` |
| `0x0106` | `%smid` | `0x0122` | `%ctaid.z` |
| `0x0107` | `%clock_lo` | `0x0123` | `%nctaid.z` |

`%laneid` 必须返回当前逻辑线程在完整逻辑 warp 内的编号，范围为 `0..logical_warp_width-1`。AEC-G v1.0 中该范围固定为 `0..31`。多 issue beat 执行时，`%laneid` 不得返回当前物理 beat 内的 `0..physical_simd_lanes-1`。

#### 6.4.3 `pred_ctrl` 字段

```text
bit  [2:0]   pred       predicate register P0..P7
bit  [6:3]   type       element/scalar type
bit  [7]     imm_en     src2 使用 imm32，而不是寄存器
bit  [10:8]  subop      compare、convert、atomic 或 SFU 子操作
bit  [13:11] space      memory address space
bit  [14]    pred_neg   predicate 取反
bit  [15]    pred_en    启用指令级 predicate
```

每个逻辑线程的执行条件为：

```text
execute_lane = active_mask[lane] &&
               (!pred_en || (P[pred][lane] XOR pred_neg))
```

这里的 `lane` 是逻辑 warp 内的线程编号。active mask 位数必须等于 `logical_warp_width`；AEC-G v1.0 固定为 32 bit。多 issue beat 执行时，每个 beat 只读取与本 beat 逻辑线程编号对应的 mask 切片，但完整 active mask 始终属于同一个逻辑 warp，并由控制流指令按本节 SIMT 语义更新。

未执行的逻辑线程禁止修改寄存器、predicate、内存或异常状态。控制流指令对 warp 的完整 active mask 执行本节定义的 SIMT 语义。

#### 6.4.4 类型编码

| type | 名称 | 位宽 | 说明 |
|---:|---|---:|---|
| `0x0` | `.b32` | 32 | 无类型 bit pattern |
| `0x1` | `.b64` | 64 | 寄存器对 |
| `0x2` | `.u32` | 32 | 无符号整数 |
| `0x3` | `.s32` | 32 | 有符号整数 |
| `0x4` | `.u8` | 8 | load/store/convert 使用 |
| `0x5` | `.s8` | 8 | load/store/convert 使用 |
| `0x6` | `.u16` | 16 | load/store/convert 使用 |
| `0x7` | `.s16` | 16 | load/store/convert 使用 |
| `0x8` | `.f32` | 32 | IEEE-754 binary32 |
| `0x9` | `.f16` | 16 | IEEE-754 binary16，寄存器低 16 bit |
| `0xa` | `.bf16` | 16 | bfloat16，寄存器低 16 bit |
| `0xb` | `.e4m3` | 8 | FP8 E4M3FN，寄存器低 8 bit |
| `0xc` | `.e5m2` | 8 | FP8 E5M2，寄存器低 8 bit，可选 |
| `0xd` | `.v4e4m3` | 32 | 4 个 packed FP8 E4M3FN |
| `0xe` | reserved | — | 必须拒绝 |
| `0xf` | `.none` | — | 无数据类型的控制指令 |

整数算术按位宽取模。`.f32` 的基础 `ADD/SUB/MUL/FMA` 使用 round-to-nearest-even；其他舍入和特殊值规则由 5.6 节、5.7 节和 `aec_g_isa_v1.json` 固定。

#### 6.4.5 地址空间编码

| space | 名称 | 含义 |
|---:|---|---|
| `0` | `.gmem` | HBM/DDR 全局设备内存 |
| `1` | `.smem` | 当前 thread block 共享存储 |
| `2` | `.cmem` | 只读常量存储，可选 |
| `3` | `.lmem` | thread 私有 local/spill 存储，可选 |
| `4` | `.pmem` | kernel 参数存储 |
| `5` | `.mmio` | 设备控制空间，仅特权 runtime 使用 |
| `6..7` | reserved | 必须拒绝 |

所有地址均为 byte address。基础 ISA 的 `LD/ST` 使用 `src1` 指定的 32-bit 地址寄存器。运行时通过 buffer base/window 将 64-bit API 地址映射到可见地址；可选的 64-bit 地址模式使用偶数寄存器对，并通过 ISA feature bit 声明。

#### 6.4.6 Opcode 总表

| Opcode | Mnemonic | 类别 | 基础/可选 |
|---:|---|---|---|
| `0x0001` | `ADD` | 算术 | 基础 |
| `0x0002` | `SUB` | 算术 | 基础 |
| `0x0003` | `MUL` | 算术 | 基础 |
| `0x0004` | `MAD` | 三源算术 | 基础 |
| `0x0005` | `FMA` | 融合乘加 | 基础 |
| `0x0010` | `AND` | 位运算 | 基础 |
| `0x0011` | `OR` | 位运算 | 基础 |
| `0x0012` | `XOR` | 位运算 | 基础 |
| `0x0013` | `NOT` | 位运算 | 基础 |
| `0x0014` | `SHL` | 移位 | 基础 |
| `0x0015` | `SHR` | 逻辑右移 | 基础 |
| `0x0016` | `SAR` | 算术右移 | 基础 |
| `0x0020` | `SETP` | predicate 赋值 | 基础 |
| `0x0021` | `CMPP` | 比较写 predicate | 基础 |
| `0x0022` | `SEL` | predicate 选择 | 基础 |
| `0x0030` | `LD` | load | 基础 |
| `0x0031` | `ST` | store | 基础 |
| `0x0032` | `ATOM` | atomic RMW | `ADD.u32` 基础，其余可选 |
| `0x0033` | `PREFETCH` | cache hint | 可选 |
| `0x0034` | `FENCE` | memory ordering | 基础 |
| `0x0040` | `BR` | 无条件分支 | 基础 |
| `0x0041` | `BRX` | predicate 分支 | 基础 |
| `0x0042` | `SSY` | 设置收敛点 | 基础 |
| `0x0043` | `SYNC` | 分支收敛 | 基础 |
| `0x0044` | `BAR` | block barrier | 基础 |
| `0x0045` | `HALT` | 逻辑线程结束 | 基础 |
| `0x0054` | `CPY` | move/special register | 基础 |
| `0x0055` | `LOADI` | 32-bit 立即数 | 基础 |
| `0x0056` | `LOADI64` | 64-bit 立即数 | 基础 |
| `0x0057` | `CVT` | 类型转换 | 基础 |
| `0x0058` | `PACK` | 打包 | FP8 路径基础 |
| `0x0059` | `UNPACK` | 解包 | FP8 路径基础 |
| `0x0060` | `SHFL` | warp lane 交换 | 基础 |
| `0x0061` | `REDUCE` | warp reduction | 基础 |
| `0x0070` | `MMA` | tile 矩阵乘加 | FP8 路径基础 |
| `0x0080` | `SFU` | 特殊函数 | 基础 |
| `0x00f0` | `NOP` | 空操作 | 基础 |

未列出的 opcode 为 reserved。硬件遇到 reserved opcode、reserved type/space 或不支持的 feature，必须停止当前 kernel 并设置 `ILLEGAL_INSTRUCTION`，禁止当作 `NOP`。

#### 6.4.7 通用操作数编码

除本文件单独规定外，指令使用下列格式：

| 格式 | 字段含义 |
|---|---|
| R 型二源 | `dst=Rd, src1=Rs1, src2_or_imm32[15:0]=Rs2` |
| I 型二源 | `dst=Rd, src1=Rs1, imm_en=1, src2_or_imm32=imm32` |
| R 型三源 | `dst=Rd, src1=Rs1, src2_or_imm32[15:0]=Rs2, src3_or_immext[15:0]=Rs3` |
| 分支 | `src3_or_immext=target_pc` |
| load | `dst=Rd, src1=Raddr, src2_or_imm32=signed_byte_offset` |
| store | `dst[15:0]=Rvalue, src1=Raddr, src2_or_imm32=signed_byte_offset` |

寄存器格式中未使用的高位必须写 0。基础实现中，除明确列出的 I 型算术、load/store offset 和 `LOADI` 外，`imm_en` 必须为 0。

#### 6.4.8 标量算术、逻辑与选择

```text
ADD.type Rd, Rs1, Rs2/imm   Rd = Rs1 + operand2
SUB.type Rd, Rs1, Rs2/imm   Rd = Rs1 - operand2
MUL.type Rd, Rs1, Rs2/imm   Rd = Rs1 × operand2
MAD.type Rd, Rs1, Rs2, Rs3  Rd = round(round(Rs1 × Rs2) + Rs3)
FMA.f32 Rd, Rs1, Rs2, Rs3   Rd = fused(Rs1 × Rs2 + Rs3)
AND/OR/XOR.b32              对应 bitwise 运算
NOT.b32 Rd, Rs1             Rd = ~Rs1
SHL.b32                     Rd = Rs1 << (operand2 & 31)
SHR.u32                     Rd = unsigned(Rs1) >> (operand2 & 31)
SAR.s32                     Rd = signed(Rs1) >> (operand2 & 31)
```

`MAD.f32` 必须执行非融合的两次舍入，`FMA.f32` 只执行一次最终舍入。基础 `MUL.u32` 返回乘积低 32 bit；`mul.wide.u32` 仍由编译器 lowering 为低位乘法和高位处理。

`SETP` 的 `dst[2:0]` 指定目标 predicate，`src1` 指定源寄存器，`src2_or_imm32` 和 `src3_or_immext` 必须为 0：

```text
SETP Pd, Rs
    if execute_lane:
        Pd[lane] = (Rs[lane] != 0)
    else:
        Pd[lane] 保持原值
```

对整数和 bit 类型，`Rs != 0` 按 32-bit bit pattern 判断；对 `.f32`，`+0.0` 和 `-0.0` 为 false，非零有限值、Inf 和 NaN 为 true。`SETP` 服从 `pred_ctrl` 的指令级 predicate，若源 predicate 与目标 predicate 相同，读取旧 predicate 后再写回新 predicate。

`CMPP` 的 `dst[2:0]` 指定目标 predicate，`subop` 为：

| subop | 比较 | subop | 比较 |
|---:|---|---:|---|
| `0` | `eq` | `3` | `le` |
| `1` | `ne` | `4` | `gt` |
| `2` | `lt` | `5` | `ge` |
| `6` | reserved | `7` | reserved |

```text
CMPP.type.op Pd, Rs1, Rs2
    Pd[lane] = compare_op(Rs1[lane], Rs2[lane])

SEL.type Rd, Rs_true, Rs_false, Pn
    Rd[lane] = Pn[lane] ? Rs_true[lane] : Rs_false[lane]
```

`SEL` 的 `pred_ctrl.pred` 指定选择 predicate，但 `pred_en` 必须为 0；`src1` 是 true source，`src2_or_imm32[15:0]` 是 false source。

#### 6.4.9 Move、立即数和转换

```text
CPY.type Rd, Rs              Rd = Rs
CPY.u32 Rd, %special         Rd = 当前逻辑线程/warp/CU 的 special register
LOADI Rd, imm32              Rd = imm32
LOADI64 Rk, imm64            Rk = imm64[31:0]; Rk+1 = imm64[63:32]
```

`LOADI` 的立即数为 `src2_or_imm32`。`LOADI64` 的低 32 bit 位于 `src2_or_imm32`，高 32 bit 位于 `src3_or_immext`，且 `dst` 必须为偶数且不大于 R254。

`CVT` 使用 `src1` 作为源寄存器，`dst` 作为目标寄存器，`type` 表示目标类型，`subop[2:0]` 表示源类型。`imm_en` 必须为 0，`src2_or_imm32` 和 `src3_or_immext` 必须为 0。AEC-G v1.0 的合法转换表固定如下，未列出的 `subop/type` pair 必须在编译期拒绝或在板上触发 `ILLEGAL_INSTRUCTION`：

| subop | 源类型 | 允许目标类型 `type` | 舍入方式 | 溢出、下溢和特殊值规则 |
|---:|---|---|---|---|
| `0` | `.u32` | `.f32` | round-to-nearest-even | 所有 `u32` 输入转换为有限 `.f32`；超过 24 bit 精度时按 RNE 舍入，无上溢。 |
| `1` | `.s32` | `.f32` | round-to-nearest-even | 所有 `s32` 输入转换为有限 `.f32`；超过 24 bit 精度时按 RNE 舍入，无上溢。 |
| `2` | `.f32` | `.u32`、`.s32`、`.f16`、`.bf16`、`.e4m3`、可选 `.e5m2` | 到整数为 round-toward-zero；到浮点/FP8 为 round-to-nearest-even | 到 `.u32/.s32` 时，NaN 输出 0，超出目标范围饱和到目标最小/最大值；到 `.f16/.bf16` 时上溢为带符号 Inf，下溢按目标 subnormal/zero 规则舍入；到 `.e4m3` 时按 5.6.1 饱和到最大有限值并输出 canonical NaN；到 `.e5m2` 时按 IEEE-like E5M2 输出 Inf 或 canonical NaN。 |
| `3` | `.f16` | `.f32`、`.e4m3`、可选 `.e5m2` | 到 `.f32` 精确扩展；到 FP8 为 round-to-nearest-even | 到 `.e4m3` 时按 5.6.1 饱和和 NaN 规则；到 `.e5m2` 时按 IEEE-like E5M2 规则。 |
| `4` | `.bf16` | `.f32` | 精确扩展 | NaN 输出 canonical `.f32` NaN；Inf 和 signed zero 保留。 |
| `5` | `.e4m3` | `.f32`、`.f16` | 精确扩展，若目标不能精确表示则 RNE | `0x7f` 类 canonical NaN 输入输出目标 canonical NaN；E4M3FN 不产生 Inf。 |
| `6` | `.e5m2` | `.f32`、`.f16` | 精确扩展，若目标不能精确表示则 RNE | 仅在 capability 声明 `E5M2` 时合法；Inf 和 NaN 按目标格式保留或 canonicalize。 |
| `7` | reserved | — | — | 必须触发 `ILLEGAL_INSTRUCTION`。 |

`CVT` 不暴露 IEEE exception flags。若 manifest 声明 subnormal flush-to-zero 模式，则 5.6.2 的 feature bit 选择对应 golden simulator；同一提交中不得按测试集改变转换规则。

`PACK.v4e4m3` 将 4 个连续寄存器低 8 bit 打包到一个 GPR；`UNPACK.v4e4m3` 执行逆操作。输入或输出寄存器组越界属于非法指令。

#### 6.4.10 Load、store、atomic 与内存序

```text
LD.space.type Rd, [Raddr + offset32]
ST.space.type [Raddr + offset32], Rs
```

- 有效地址按 32-bit 无符号 byte address 计算；
- offset 为有符号 32-bit two's-complement；
- `.u8/.s8/.u16/.s16` load 分别零扩展或符号扩展到 32 bit；
- FP8/FP16/BF16 load 放入寄存器低位，其余高位清零；
- `.b64` load/store 使用偶数对齐寄存器对 `{Rk+1, Rk}`，`Rk` 保存低 32 bit，`Rk` 必须为偶数且不大于 R254；
- AEC-G v1.0 基础访存宽度仅为 8/16/32/64 bit，不定义 `.b128` 编码；未声明可选向量访存扩展时，单指令 128-bit `LD/ST` 必须触发 `ILLEGAL_INSTRUCTION`；
- 基础测试只生成自然对齐访问；未对齐访问必须正确拆分或触发 `MISALIGNED_ACCESS`；
- `.pmem` 和 `.cmem` store 为非法；
- 对同一逻辑线程，单线程程序顺序必须成立；不同 warp 的可见性由 `FENCE` 和 `BAR` 定义。

`ATOM` 的 `subop` 编码：

| subop | 操作 | 要求 |
|---:|---|---|
| `0` | `ADD.u32` | 基础 |
| `1` | `CAS.u32` | 可选 |
| `2` | `MAX.u32` | 可选 |
| `3` | `MIN.u32` | 可选 |
| `4` | `ADD.f32` | 可选 |
| `5..7` | reserved | — |

`ATOM` 原子读取旧值写入 `dst`，地址来自 `src1`，操作数来自 `src2_or_imm32[15:0]`；CAS 的 compare/new-value 另用 `src3_or_immext[15:0]` 和 `dst`，其精确编码在启用该可选 feature 时由附录定义。

`FENCE` 的 `subop`：`0=CTA`、`1=DEVICE`、`2=SYSTEM`。基础实现必须支持 CTA 和 DEVICE；SYSTEM 可通过 runtime/DMA 同步实现。`PREFETCH` 只是性能提示，禁止改变程序可观察结果。

#### 6.4.11 分支、收敛、barrier 与结束

```text
BR target_pc
BRX Pn, target_pc
SSY reconverge_pc
SYNC
BAR.SYNC barrier_id, expected_warps
HALT
```

- `BR` 和 `BRX` 的绝对 target PC 位于 `src3_or_immext`；
- `BRX` 使用 `pred_ctrl.pred/pred_neg` 选择条件，`pred_en` 必须为 1；
- `SSY` 将 `reconverge_pc` 和当前 active mask 压入当前 warp 的 reconvergence stack；
- 发散 `BRX` 必须保证 taken 和 fall-through 两条路径均执行其对应逻辑线程，执行次序可由实现决定；
- `SYNC` 在收敛点合并保存在栈中的 active mask；栈下溢或溢出触发 `SIMT_STACK_FAULT`；
- `BAR.SYNC` 的 `dst[7:0]` 为 barrier ID，`src2_or_imm32[15:0]` 为参与 warp 数；0 表示当前 block 的全部未结束 warp；
- barrier 只同步同一 thread block。部分 active 逻辑线程到达、而同一 warp 其他未结束逻辑线程永不到达 barrier 的程序行为未定义，编译器必须拒绝明显的 divergent barrier；
- `HALT` 仅清除执行该指令的 active 逻辑线程；warp 所有逻辑线程结束后释放 warp，block 所有 warp 结束后产生 block completion。

硬件可以采用不同的内部收敛机制，但必须与上述可观察行为一致。

#### 6.4.12 Warp exchange 与 reduction

`SHFL` 的 `subop`：

| subop | 操作 |
|---:|---|
| `0` | `IDX`：从指定逻辑 lane 取值 |
| `1` | `XOR`：从 `laneid XOR delta` 取值 |
| `2` | `UP`：从 `laneid - delta` 取值 |
| `3` | `DOWN`：从 `laneid + delta` 取值 |

源值在 `src1`，逻辑 lane index/delta 在 `src2_or_imm32`。所有源逻辑 lane 编号均基于完整逻辑 warp，范围为 `0..logical_warp_width-1`；AEC-G v1.0 固定为 `0..31`。`SHFL` 允许跨 issue beat 读取同一逻辑 warp 内其他逻辑线程的源值，硬件不得把交换范围限制在当前物理 SIMD lane 或当前 issue beat 内。源逻辑 lane 未激活或超出 `logical_warp_width` 时返回调用逻辑线程自身的源值。

`REDUCE` 必须在完整逻辑 warp 的当前 active mask 内执行 reduction，将结果广播到所有 active 逻辑线程。`subop` 为 `0=ADD`、`1=MAX`、`2=MIN`、`3=AND`、`4=OR`、`5=XOR`；支持 `.u32/.s32/.f32` 中与操作匹配的类型。多 issue beat 实现必须跨 beat 收集同一逻辑 warp 的所有 active 逻辑线程，不能只归约当前物理 SIMD lane 范围。

AEC-G v1.0 golden simulator 固定 reduction 树为按逻辑 lane 编号升序的平衡二叉树：步长依次为 `1, 2, 4, ...`，每一步由较小逻辑 lane 编号的部分结果与 `lane + step` 的部分结果合并，未激活的逻辑 lane 被跳过且不引入额外单位元。FP32 `REDUCE.ADD` 的每一次合并均使用 `.f32` round-to-nearest-even，因此浮点运算顺序固定为该跨 beat 树形次序，不保证与任意串行求和 bit-identical。

#### 6.4.13 FP8 MMA 指令

基础矩阵指令为：

```text
MMA.m16n16k16.e4m3.f32  D, A, B, C
```

编码：

- `opcode = 0x0070`；
- `type = 0xb` 表示 E4M3FN 输入和 FP32 累加；
- `subop = 0` 表示 `m16n16k16`；
- `dst`、`src1`、`src2_or_imm32[15:0]`、`src3_or_immext[15:0]` 分别为 D/A/B/C fragment 的起始 GPR 编号，寄存器字段高位必须为 0；A/B fragment 起始寄存器必须为偶数且不大于 R254，C/D fragment 起始寄存器必须按 8 个 GPR 对齐且不大于 R248；
- fragment 描述符按固定的 32-thread 逻辑 warp 分配。令逻辑 lane `l ∈ [0,31]`，`row = l >> 1`，`half = l & 1`，`col_base = 8 × half`，`k_base = 8 × half`，`b_col = l >> 1`。A fragment 的 `R[A+0]` 四个 byte 依次保存 `A[row][col_base+0..3]`，`R[A+1]` 保存 `A[row][col_base+4..7]`；B fragment 的 `R[B+0]` 四个 byte 依次保存 `B[k_base+0..3][b_col]`，`R[B+1]` 保存 `B[k_base+4..7][b_col]`；C fragment 的 `R[C+i]` 保存 `C[row][col_base+i]`，D fragment 的 `R[D+i]` 保存 `D[row][col_base+i]`，`i=0..7`。`aec_g_isa_v1.json` 必须逐 logical lane 列出同一布局，作为 assembler、compiler 和 golden simulator 的机器可读来源；
- 语义为 `D = A × B + C`。每个 A/B FP8 输入先按 5.6.1 转换为 FP32 并应用 scale；对每个输出元素，初值为对应 FP32 `C`，随后严格按 `k=0..15` 顺序执行 FP32 FMA 累加，最终 FP32 结果写入 D；
- `D` 与 `C` 允许相同 fragment 起始寄存器；部分逻辑线程 predication 对 MMA 非法，MMA 必须由完整 32-thread 逻辑 warp 一致执行。若任一逻辑线程因 active mask 或 predicate 未执行该 MMA，必须触发 `ILLEGAL_INSTRUCTION` 或由编译器拒绝；
- K/N/M 尾块由软件 padding、mask 后的 load/store 或标量路径处理。

可选 subop：`1=m16n16k32.e4m3.f32`、`2=m16n16k16.e5m2.f32`、`3=m8n8k16.e4m3.f16`。未声明 feature 的实现必须拒绝这些编码。

#### 6.4.14 SFU 指令编码

SFU 使用统一 opcode `0x0080`：

| subop | 汇编 | 语义 | 要求 |
|---:|---|---|---|
| `0` | `SFU.RCP.f32 Rd, Rs` | `1/x` | 基础 |
| `1` | `SFU.EXP2.f32 Rd, Rs` | `2^x` | 基础 |
| `2` | `SFU.RSQRT.f32 Rd, Rs` | `1/sqrt(x)` | 可选 |
| `3` | `SFU.LOG2.f32 Rd, Rs` | `log2(x)` | 可选 |
| `4` | `SFU.TANH.f32 Rd, Rs` | `tanh(x)` | 可选 |
| `5` | `SFU.SIN.f32 Rd, Rs` | `sin(x)` | 可选 |
| `6` | `SFU.COS.f32 Rd, Rs` | `cos(x)` | 可选 |
| `7` | reserved | — | — |

`dst=Rd`、`src1=Rs`，其余操作数字段必须为 0。允许 LUT、插值、多项式或迭代近似；特殊值、误差、吞吐量和验证要求见 5.7 节。SFU 结果在体系结构上与普通寄存器写回等价，scoreboard 必须覆盖其实际延迟。

#### 6.4.15 `NOP`、错误和 feature discovery

`NOP` 除 predicate/active-mask 的正常取指行为外不产生任何状态变化；其未使用字段必须为 0。

设备只读 capability 寄存器必须至少报告：

```text
ISA major/minor
logical_warp_width
physical_simd_lanes
issue_beats_per_warp
number of CU
GPR count and predicate count
supported type bitmap
supported opcode/feature bitmap
MMA tile bitmap
SFU function bitmap
maximum block threads and resident warps
shared-memory size
address width
```

设备必须保证 `issue_beats_per_warp = logical_warp_width / physical_simd_lanes` 且整除。编译器把所需 feature 和 `required_capability.logical_warp_width` 写入 module manifest。runtime 在 module load 或 launch 前必须比较 module 与 device capability；逻辑 warp 宽度不匹配、feature 不支持或 capability 字段自相矛盾时返回 `AEC_ERROR_UNSUPPORTED_FEATURE`。`physical_simd_lanes` 可以随设备实现不同，但不得改变 AEC-G v1.0 的逻辑 warp 语义。板上非法指令、地址错误、barrier deadlock、SIMT stack fault 和 watchdog timeout 必须保留 fault PC、CU/warp/block ID 及错误码，供主机读取。

### 6.5 汇编语法与示例

汇编器忽略 `#` 后的行内注释，label 以 `:` 结尾，寄存器写作 `R0..R255`，predicate 写作 `P0..P7`。示例：

```asm
# C[i] = A[i] + B[i]
CPY.u32       R0, %tid.x
CPY.u32       R1, %ctaid.x
CPY.u32       R2, %ntid.x
MAD.u32       R3, R1, R2, R0

LOADI         R10, 0
LD.pmem.u32   R11, [R10 + 0]      # A base/window address
LD.pmem.u32   R12, [R10 + 4]      # B base/window address
LD.pmem.u32   R13, [R10 + 8]      # C base/window address
SHL.b32       R4, R3, 2
ADD.u32       R5, R11, R4
ADD.u32       R6, R12, R4
ADD.u32       R7, R13, R4
LD.gmem.f32   R20, [R5 + 0]
LD.gmem.f32   R21, [R6 + 0]
ADD.f32       R22, R20, R21
ST.gmem.f32   [R7 + 0], R22
HALT
```

SFU 示例：

```asm
# y = exp2(x) / sum，R1=x，R2=sum
SFU.EXP2.f32  R3, R1
SFU.RCP.f32   R4, R2
MUL.f32       R5, R3, R4
```

## 7. 编译器要求

### 7.1 输入与输出

基础编译器的固定输入是 NVIDIA PTX 9.3 的受限子集，固定输出是 `.aecbin` 和 JSON 编译报告。参赛队伍允许在构建流程中调用 `nvcc` 将 CUDA `.cu` kernel 编译为 PTX；该前端步骤不要求自研，也不作为编译器核心能力计分。

从 PTX 开始到 AEC-G v1.0 的 ISA 映射必须由参赛队伍自行实现并提交源码，包括 PTX 解析、CFG、lowering、寄存器与 predicate 分配、AEC 指令选择、ABI 映射、`.aecbin` 生成和编译报告。禁止只提交手写 `.aecbin`、只提交公开 kernel 的离线转换结果，或针对 kernel 名称/shape 做表驱动替换。

```bash
nvcc -ptx kernel.cu -o kernel.ptx      # 允许借用
compiler/aec-cc kernel.ptx -O2 -o kernel.aecbin --report compile_report.json
```

可选高级入口包括：

- AEC 汇编；
- MLIR/LLVM IR 后端；
- Triton-like kernel DSL；
- PyTorch FX/Inductor 自定义后端；
- 离线算子库选择和 autotuning。

无论采用哪种高级入口，计分硬件执行的最终程序都必须是 AEC-G v1.0 指令流，并通过第 8 节固定 runtime API launch。高级入口可以绕过 PTX 直接生成 AEC-G，但不能替代必选的 PTX-to-AEC 编译能力，也不能使隐藏 PTX kernel 的可移植性测试失效。

### 7.2 必选功能

- PTX 受限子集解析、类型检查、CFG 和基本块；
- PTX-to-AEC-G v1.0 lowering 和指令选择；
- 物理寄存器与 predicate 分配；
- 分支 target、参数 ABI 和 64-bit pointer pair 处理；
- 基本优化：常量折叠、DCE、简单 CSE、地址计算简化；
- 指令调度或延迟槽/相关停顿的正确处理；
- 编译错误必须可诊断，禁止生成未定义机器码；
- 编译报告列出指令数、寄存器、spill、共享存储、估计 occupancy 和使用的扩展。

性能组参考实现 tiling、vectorization、双缓冲、软件流水、GEMM intrinsic、算子融合、布局变换和静态/动态 shape specialization。

## 8. XDMA 驱动与运行时

组织方固定 U280 XDMA shell/platform 和评测环境，使参赛队伍不需要重新实现 PCIe endpoint 或板卡静态区。参赛队伍必须自行实现计分可见的基于 XDMA 的驱动/控制路径和 AEC runtime，包括 BAR/MMIO 访问、DMA 提交、命令队列、同步、错误恢复和计数器读取。允许使用操作系统和 Xilinx/AMD 平台提供的基础内核接口完成设备枚举、BAR 映射或底层 DMA 能力，但不得把计分所需的 runtime/driver 语义交给闭源模型框架或私有不可审查组件。

### 8.1 固定 CUDA-like Runtime API

以下 C API 是固定接口，不是参考接口。正式评分使用组织方随评测 SDK 发布的 `aec_runtime.h` 编译，接口名称、参数语义、返回码和同步语义固定；参赛队伍可以在内部增加辅助函数，但不得要求评分程序使用替代 API。评测程序、PyTorch package 和 benchmark 脚本必须通过这些接口调用参赛系统。

```c
aecContextCreate(...);
aecMalloc(...);
aecFree(...);
aecMemcpyH2D(...);
aecMemcpyD2H(...);
aecModuleLoad(...);
aecKernelLaunch(...);
aecSynchronize(...);
aecGetLastError(...);
aecReadCounters(...);
```

最小固定语义如下：

| 接口 | 固定语义 |
|---|---|
| `aecContextCreate` | 打开指定 U280 设备，初始化驱动/runtime 状态，读取 capability 和错误状态 |
| `aecMalloc` / `aecFree` | 分配/释放 64-bit device address 可表示的设备 buffer，并记录 HBM/DDR 放置策略 |
| `aecMemcpyH2D` / `aecMemcpyD2H` | 在 host tensor/storage 与 device buffer 之间传输字节，遵守同步和 cache 可见性规则 |
| `aecModuleLoad` | 加载 `.aecbin` 与 manifest，校验 ISA 版本、capability、参数 ABI 和资源需求 |
| `aecKernelLaunch` | 按固定 `gridDim`、`blockDim`、参数区和动态 shared memory 语义提交 kernel |
| `aecSynchronize` | 等待已提交 DMA/kernel 完成并返回首个可见错误 |
| `aecGetLastError` | 返回并可清除当前 context 的最后错误码 |
| `aecReadCounters` | 读取周期数、访存字节数、cache miss、DMA 时间、kernel 时间和 fault 信息 |

运行时必须：

- 提交完整的 API header、实现源码和链接方式，并与固定评测环境兼容；
- 使用异步 DMA 或明确说明同步行为；
- 对齐并分块大传输；
- 正确处理 pin memory、IOMMU 和超时；
- 不信任用户提供的长度、地址和机器码；
- 在 `aecModuleLoad` 或 launch 前校验 `.aecbin`、manifest 和 device capability，禁止按模型名称选择私有执行路径；
- kernel hang 后能在规定时间内返回错误并复位动态区；
- 记录 H2D、kernel、D2H 三段时间，端到端计分禁止隐藏传输开销。

### 8.2 主机 CPU 操作白名单与 fallback

计分运行中，CPU 只允许执行以下控制类操作：

- Python/C++ 调度、参数检查、shape 推导、stride/layout 元数据管理；
- buffer 分配、pin memory、页锁定、IOMMU 映射、DMA 描述符生成和提交；
- `.aecbin` 加载、manifest 校验、capability 检查、kernel launch、同步、超时处理和错误恢复；
- 将已经由官方预处理产生的输入张量或已固定权重按字节拷贝、分块、打包为传输格式；
- 读取设备输出并交给评分脚本计算 accuracy、perplexity、Top-K、token match、NRMSE 和日志统计。

计分运行中，CPU 禁止执行以下输入或权重相关的数值计算：

- convolution、GEMM/linear、attention、MLP、normalization、pooling、softmax、activation、reduction、argmax 或采样；
- 动态量化、反量化、scale 估计、clamp、bias add、residual add、layout transpose 中的数值重排以外的算术；
- KV cache 更新、token logits 计算、CNN logits 计算或任何模型层输出计算；
- 依据输入、权重、kernel 或 shape 选择预存结果、近似表输出或主机端快捷路径。

允许存在 CPU fallback 代码用于非计分调试或组织方明确标记的非计分算子。正式计分时，runtime 必须逐算子输出 `fallback=false/true`、fallback 原因、CPU 时间、输入/输出字节数和调用次数。未披露 fallback 视为越权使用 CPU。对计分模型，主计算算子 fallback 比例必须为 0；非主计算且白名单允许的 fallback 只计入端到端时间，不额外加分。超过 manifest 允许范围的 fallback 使对应模型性能分记 0 分。

## 9. PyTorch 对接

参赛提交须提供面向参赛硬件优化的 PyTorch Python package。该 package 的核心交付不是通用 CUDA 兼容层，而是针对硬件设计的优化算子库，以及针对端到端网络的调度、融合、分块、布局转换、量化和 kernel launch 方法。所有计分路径必须通过第 8 节固定 runtime API 访问设备。

示例：

```python
import torch
import aec_torch

y = aec_torch.ops.gemm_fp8(x, w, x_scale, w_scale)
```

最低要求：

- 通过固定 AEC runtime API 完成 device buffer、module load、kernel launch、同步和错误返回；
- PyTorch Tensor 与设备 buffer 的显式转换；
- CPU fallback 仅限 8.2 节白名单；
- 算子级正确性与异常信息；
- 提供固定 batch/shape 的 ResNet/Transformer 端到端推理脚本和调度说明；
- 提供面向硬件优化的算子库，至少覆盖公开测试和正式模型需要的 P0/P1 主计算路径；
- 运行时统计和 warm-up 控制；
- 禁止在计时区间由 CPU 计算参赛算子的主要结果。

参考支持的算子优先级：

| 级别 | 算子 |
|---|---|
| P0 | copy、elementwise add/mul、ReLU、FP8 GEMM/linear |
| P1 | Conv2d（可 lowering 为 GEMM）、bias、pooling、residual add |
| P2 | LayerNorm/RMSNorm、Softmax、GELU/SiLU、transpose/reshape |
| P3 | batched GEMM、QKV projection、attention、KV cache |

## 10. 测试工作负载

### 10.1 分层测试

1. **ISA 单元测试**：所有 opcode、predicate、branch、边界数值和异常路径；
2. **微架构测试**：发散、barrier、bank conflict、cache miss、并发 warp、长延迟访存、SFU 反压与乱序写回保护；
3. **基础 kernel**：vector add、copy、SAXPY、reduction、transpose、softmax，以及 `RCP/EXP2/RSQRT` 随机与边界输入；
4. **核心算子**：不同 M/N/K、batch、对齐与尾块的 FP8 GEMM；
5. **CNN**：ResNet-18 为基础端到端项，ResNet-50 为主要性能项；
6. **Transformer**：固定约 1B decoder-only 模型或结构等价模型；权重分块和层评测只用于非正式分析，不能替代正式 LLM 任务；
7. **鲁棒性**：随机 shape、随机机器码拒绝、kernel timeout、连续运行和温度稳定性。

### 10.2 端到端口径

正式计时必须使用统一口径：

- 算子性能：输入、权重和输出 buffer 已驻留设备，计时从 `aecKernelLaunch` 可见开始，到该 kernel 的 completion 可见并完成必要 device fence 结束；
- 模型端到端性能：计时从评测程序把已定义格式的 CPU Tensor 或 token batch 交给参赛后端开始，到最终 logits 或 token IDs 已同步到 CPU 可读内存结束；
- 吞吐率：CNN 使用固定 batch 下 `images/s`；LLM 使用固定 batch、prompt length、生成长度下 `tokens/s`；
- latency：报告 P50、P95 和最大值；P95 定义为按单请求 latency 升序排序后的第 `ceil(0.95 × N)` 个样本，`N` 至少为 100 个请求或 30 秒内的全部请求，取较大者；
- queue depth：正式 latency 项固定 `queue_depth=1`、`concurrency=1`；正式 throughput 项固定使用 manifest 给出的 batch tensor，默认 `queue_depth=1`，除非测试 manifest 显式启用异步队列；
- 动态 batching：latency 项禁止动态 batching；throughput 项只允许在评测程序已经给出的 batch 内做静态 batching，禁止等待未来请求拼 batch；
- warm-up：warm-up 次数由 10.3 节固定，warm-up 输入禁止与隐藏计分输入相同；warm-up 结束后必须清空输入相关缓存和中间结果；
- 状态管理：bitstream、权重常驻、已编译 AEC kernel、合法 autotune 结果可以保留；输入相关 activation、输出、KV cache、随机测试结果、哈希表、lookup cache 和隐藏样本中间结果必须在测试阶段之间清空；
- 准确率：与 PyTorch FP32/量化参考比较，具体门槛由测试 manifest 固定；
- bitstream 配置、编译产物、权重版本、tokenizer、量化参数和 autotune cache 必须在正式计时前固定并写入日志。

### 10.3 正式端到端public任务

正式比赛固定三个端到端模型：ResNet-50、约 1B decoder-only Transformer 和 ResNet-18。模型文件、量化校准集、输入集、参考输出、PyTorch 版本、tokenizer、EOS 规则和预处理结果均随评测镜像发布。正式评测使用公开输入和同分布隐藏输入，禁止根据样本 ID 缓存输出。缺失模型、缺失 shape、输出接口不匹配或未达到精度门槛时，对应模型或 shape 的分数按 0 处理，禁止选择性只提交高分 shape。

#### 10.3.0 公开测试例子的 PyTorch 结构

随 public 测试发布两个可直接导入的 PyTorch 网络定义，作为端到端流程、算子 lowering、量化、runtime 和 fallback 日志的公开 sanity case。权威源码路径为 `tests/public_models.py`；评分镜像中的同名文件、权重文件和输入 manifest 必须固定 SHA-256。两个 public 例子不替代 10.3.1、10.3.2 和 ResNet-18 的正式模型分，但必须在公开测试中通过。

Public CNN 例子固定为 `PublicCNNv1`：

| 项目 | 固定值 |
|---|---|
| 输入 | `float32` NCHW Tensor `[batch, 3, 64, 64]` |
| 输出 | logits `[batch, 10]` |
| stem | `Conv2d(3,16,3,padding=1,bias=False)` + `BatchNorm2d(16, eps=1e-5)` + ReLU |
| block1 | 两个 `Conv2d(16,16,3,padding=1,bias=False)` + BN，identity residual add，ReLU |
| block2 | `Conv2d(16,32,3,stride=2,padding=1,bias=False)` + BN + ReLU + `Conv2d(32,32,3,padding=1,bias=False)` + BN；skip 为 `Conv2d(16,32,1,stride=2,bias=False)` + BN；residual add 后 ReLU |
| head | `AdaptiveAvgPool2d((1,1))` + flatten + `Linear(32,10)` |

Public Transformer 例子固定为 `PublicTinyDecoderLMv1`：

| 项目 | 固定值 |
|---|---|
| 输入 | `token_ids` `[batch, seq_len]`，`attention_mask` `[batch, seq_len]`，`seq_len <= 128` |
| 输出 | logits `[batch, seq_len, 4096]` |
| embedding | `Embedding(4096,128)` token embedding + `Embedding(128,128)` learned absolute position embedding |
| blocks | 2 个 decoder block |
| attention | pre-LN causal self-attention，`hidden_size=128`，`num_heads=4`，QKV 为 `Linear(128,384)`，输出投影 `Linear(128,128)`，softmax mask 使用 causal mask 和 `attention_mask` |
| MLP | pre-LN 后 `Linear(128,256)` + exact GELU + `Linear(256,128)` |
| final | `LayerNorm(128, eps=1e-5)` + `Linear(128,4096,bias=False)` |

参考导入方式：

```python
from tests.public_models import PublicCNNv1, PublicTinyDecoderLMv1

cnn = PublicCNNv1().eval()
lm = PublicTinyDecoderLMv1().eval()
```

public 权重、BatchNorm running statistics、tokenizer 或 synthetic token 生成规则必须随 public manifest 发布。所有 dropout 概率固定为 0；forward 中禁止随机采样。参赛实现可对这些网络做合法量化、算子融合和布局转换，但必须保持与上述 PyTorch `eval()` 参考在 manifest 误差门槛内一致。

#### 10.3.1 E2E-CNN：ResNet-50 图像分类

| 项目 | 固定设置 |
|---|---|
| 模型 | torchvision ResNet-50 v1.5，1000 类，inference/eval 模式 |
| 数据 | ImageNet-1K validation；公开 2,000 张，隐藏 5,000 张 |
| 输入 | 已完成 resize/crop/normalize 的 `NCHW FP32` CPU Tensor，`224×224` |
| 精度 | convolution/linear 允许 FP8，累加至少 FP16；其他层允许 FP16/FP32 |
| batch | latency：1；throughput：16 和 64；缺失任一 batch shape 时该 shape 记 0 分 |
| 输出 | 必须返回 `float32` 或 `float16` 的 `[batch,1000]` logits；Top-1/Top-5 由评分脚本计算 |
| 预热 | 每个 batch 10 次，不计分 |
| 正式运行 | 每个 batch 至少 100 个 batch，或持续 30 秒，取时间较长者 |

参考准确率由固定的 FP32 PyTorch 模型测得。参赛结果必须同时满足：

- 隐藏集 Top-1 相对 FP32 参考下降不超过 1.5 个百分点；
- Top-5 相对下降不超过 1.0 个百分点；
- 禁止输入相关 CPU fallback 执行 convolution、linear、pooling 或 activation 主计算。

计时开始于评测程序把一批已经预处理完成的 CPU Tensor 交给参赛后端，结束于该批 logits 已同步回到 CPU 可读内存。计时包含布局转换、量化、H2D、所有 kernel、同步和 D2H；不包含 JPEG 解码和 resize/crop。

吞吐率定义为：

```text
CNN_throughput(batch) = processed_images / measured_wall_time
```

#### 10.3.2 E2E-LLM：约 1B 参数 decoder-only Transformer

给出正式固定一个开放权重、约 1B 参数的 decoder-only 模型。参考候选为 **TinyLlama-1.1B-Chat 的固定 checkpoint**；若发布许可或工具兼容性不满足，则在开赛前以结构等价的模型替换，并同步发布权重 SHA-256。FP8/INT8 后的权重、激活和 KV cache 可以完整放入 8 GB HBM，因此该任务能够在单张 U280 上执行，不要求多卡。

| 项目 | 固定设置 |
|---|---|
| 模型 | 约 1B decoder-only Transformer，固定 checkpoint 和 tokenizer |
| 权重 | FP8、FP16、BF16 或混合量化；禁止改变层数、hidden size、词表或跳过层 |
| 输入 | 已 tokenized 的 `int32 token_id` 和 attention mask CPU Tensor |
| batch | 1 和 4 |
| prompt 长度 | 128、512；隐藏测试包含 64–768 范围内的长度 |
| 生成长度 | 固定生成 128 tokens |
| 解码 | greedy decoding，`temperature=0`；argmax 必须在设备端完成，tie 取最小 token ID |
| KV cache | 必须保存在 U280 HBM/DDR；禁止由 CPU 计算 attention |
| 输出 | 生成接口必须返回 `[batch,128] int32 token_ids`；perplexity 接口必须返回 `[batch,seq,vocab]` logits 或评分 manifest 允许的等价 next-token logprob |
| 预热 | 每个 shape 3 次，不计分 |
| 正式运行 | 每个 shape 至少 20 个 prompt，重复 3 轮取中位数 |

分别报告：

```text
TTFT_ms      = prompt 提交到第一个输出 token 可见的时间
Prefill_tok_s = 所有 prompt token 数 / prefill wall time
Decode_tok_s  = 除首 token 外生成 token 数 / decode wall time
E2E_tok_s     = 所有生成 token 数 / 从输入提交到最终 token 返回的 wall time
```

LLM 精度必须满足以下门槛：

- 在固定语言建模验证集上，perplexity 相对 FP16 参考恶化不超过 5%；
- 固定确定性 prompt 的 greedy token match ratio 不低于 95%；
- 对数值敏感的公开层测试，归一化输出误差 `NRMSE ≤ 3%`；
- 任何为通过精度门槛而在 CPU 上执行主要 Transformer 层的方案，端到端项记 0 分。

perplexity 计算由评分脚本在 CPU 上执行，输入为设备返回的 next-token logits 或 manifest 明确允许的 next-token logprob。默认公式为：

```text
NLL = -sum_t log_softmax(logits_t)[label_t] / valid_token_count
perplexity = exp(NLL)
```

padding、prompt mask 和 label shift 由测试 manifest 固定。生成接口固定执行 `max_new_tokens=128` 个 decode step；某个样本生成 `eos_token_id` 后，该样本之后的输出位置必须填充 `eos_token_id`，token match 只在参考序列的有效区间内计算。正式 throughput 的分母固定为 `batch × 128` 个生成位置，隐藏 prompt 可以选择在 128 步内不产生 EOS，禁止利用早停样本虚增吞吐。

计时开始于 token IDs/attention mask CPU Tensor 交给后端，结束于最后一个生成 token ID 已同步到 CPU。权重允许在计时前一次性加载并常驻设备；权重加载时间须另行报告，但不计入 steady-state inference。KV cache 初始化、prompt H2D、prefill、逐 token launch/调度、decode 和 token D2H 均计时。tokenizer 不计时。

### 10.4 统一评测流程

每支队伍提交一个容器和一个 bitstream。在同一服务器、同一 U280 卡型、相同驱动、相同电源模式和规定散热条件下执行：

```text
1. 冷启动并加载 bitstream；
2. 查询 capability，校验 ISA/模型 feature；
3. 加载权重，记录但不计 steady-state 的加载时间；
4. 执行公开正确性与边界测试；
5. 执行隐藏精度测试；
6. 清空设备和软件中的输入相关缓存、中间结果、KV cache、输出缓存和哈希索引；
7. 达到精度门槛后执行 warm-up；
8. 再次清空输入相关缓存，只保留 10.2 节允许的常驻状态；
9. 按固定 shape 随机顺序执行吞吐率测试；
10. 在不同模型、不同 shape 和公开/隐藏阶段切换时重复执行缓存清理；
11. 执行至少 30 分钟连续稳定性测试；
12. 导出 runtime trace、温度、资源、频率和原始计时日志。
13. 按照上述流程评测隐藏测试集
```

正式吞吐测试期间：

- CPU 频率、线程数和 NUMA 绑定固定；
- 同一时刻只允许被测任务使用 U280；
- 禁止联网和读取隐藏参考输出；
- 所有异步工作必须在停止计时前完成；
- 取 3 轮中位数；若最大值与最小值相差超过 5%，增加到 7 轮并取中位数；
- 发生超时、复位或错误输出的轮次吞吐率记为 0，禁止只挑选成功轮次。

## 11. 评分规则（100 分 + 10 分开放加分）

### 11.1 基础正确性与完整性：25 分

| 项目 | 分值 |
|---|---:|
| AEC 标量 ISA、ABI 和随机指令测试 | 6 |
| SIMT 发散/收敛、同步和多 warp 正确性 | 6 |
| HBM/DDR、DMA、缓存及异常恢复 | 5 |
| FP8 GEMM 与 SFU 数值正确性 | 5 |
| 可复现构建、文档和自动化测试 | 3 |

任一隐藏测试出现结果伪造、越权使用 CPU 或硬编码测试输出，取消成绩。基础正确性低于 15/25、FP8 GEMM 不通过，或不能完成至少一个规定端到端模型的提交，不进入端到端性能排名。

### 11.2 算子性能：15 分

- FP8 GEMM：7 分；
- memory/reduction/normalization/activation/SFU：5 分；
- 多 shape 几何平均与尾块鲁棒性：3 分。

算子性能必须采用有界归一化：

```text
score_i = weight_i × min(1, log(1 + perf_team/perf_base) / log(1 + perf_target/perf_base))
```

`perf_base`、`perf_target` 和各 shape 权重在赛前固定。使用几何平均，避免单一 shape 特化取得不成比例优势。

### 11.3 端到端模型性能和可靠性：40 分

| 项目 | 分值 |
|---|---:|
| ResNet-50 多 batch 端到端吞吐率 | 12 |
| 约 1B Transformer prefill/decode/E2E 吞吐率 | 18 |
| ResNet-18 多 batch 端到端吞吐率 | 4 |
| 包含 H2D/D2H、量化、布局转换和 launch 的全流程性能 | 4 |
| 长时间连续推理的 P95 latency 与稳定性 | 2 |

端到端项目是本赛题的主要排名依据。准确率是取得性能分的硬门槛，不以牺牲必要精度换取吞吐率：分类 Top-1/Top-5、Transformer perplexity、token match 和 NRMSE 任一超过 10.3 节门槛，则对应模型全部性能分记 0 分。通过精度门槛后，端到端吞吐率越高，分数严格越高。缺失 ResNet-50、LLM 或 ResNet-18 时，对应的 12、18 或 4 分记 0；缺失某个 batch、prompt length、生成长度或隐藏 shape 时，该 shape 的吞吐率取 0 并参与几何平均。

每个 shape 的有效吞吐率直接采用 10.3 节的 `images/s` 或 `tokens/s`。为避免某个 shape 的极端特化，模型综合吞吐率使用加权几何平均：

```text
T_model = exp(sum_i(weight_i × ln(max(T_i, epsilon))) / sum_i(weight_i))
```

shape 权重固定为：

```text
ResNet-50: batch1=20%, batch16=40%, batch64=40%
ResNet-18: batch1=20%, batch16=40%, batch64=40%
LLM: prefill=30%, decode=50%, full E2E=20%
LLM 内部各 batch/prompt length 等权几何平均
```

性能分在所有通过精度和稳定性门槛的有效提交中按最佳成绩归一化：

```text
S_CNN = 12 × (T_CNN_team / T_CNN_best)^0.5
S_LLM = 18 × (T_LLM_team / T_LLM_best)^0.5
S_R18 = 4 × (T_R18_team / T_R18_best)^0.5
```

其中 `T_best` 是复测后的该届最佳有效成绩。若需要在比赛开始前即可计算绝对分，可以提前固定参考目标 `T_target`，但禁止封顶：超过目标时仍按公开的延伸曲线增加，最后统一归一化到该项满分。相同精度等级下，吞吐率更高的队伍禁止获得更低分。

P95 latency 单独用于 2 分稳定性/交互项，不与主要吞吐率重复计分。LLM 首 token latency 必须低于固定的最大门槛，否则 LLM 吞吐分乘以 0.8；该门槛用于防止无限增大 batch 或缓冲时间来虚增吞吐。

纯设备 kernel 时间只能用于分析，不能代替端到端计分。端到端计时从接收已定义格式的 PyTorch CPU Tensor 或数据 batch 开始，到生成可被 PyTorch CPU 侧消费的最终输出结束，计入必要的量化、反量化、布局转换、H2D/D2H、kernel launch、同步和设备计算。模型权重的首次离线编译不计入单次推理，但权重上传、缓存预热和常驻条件必须按规定统一处理。

CPU fallback 按 8.2 节统一处理。计分模型的主计算算子 fallback 比例必须为 0；未披露 fallback 或超过白名单范围时，对应模型性能分记 0，情节严重时取消成绩。

### 11.4 软件栈与可编程性：12 分

| 项目 | 分值 | 机器可执行口径 |
|---|---:|---|
| PTX-to-AEC 编译器正确性和优化 | 5 | `5 × weighted_pass_rate(compiler_manifest)` |
| PyTorch 优化算子库、端到端调度与 fallback 日志 | 3 | `3 × weighted_pass_rate(pytorch_manifest)` |
| XDMA 驱动/runtime 的异步、错误处理和可观测性 | 2 | `2 × weighted_pass_rate(runtime_manifest)` |
| 未公开 kernel 的可移植性与文档 | 2 | `2 × weighted_pass_rate(portability_manifest)` |

`weighted_pass_rate(manifest)` 由评分脚本按测试项权重计算：

```text
weighted_pass_rate = sum_i(weight_i × pass_i) / sum_i(weight_i)
pass_i ∈ {0, 1}
```

禁止通过手工修改 RTL、驱动或硬编码 kernel 名称来通过可移植性测试；未知 kernel 必须经提交的编译器、assembler、loader 和 AEC 执行路径完成。

### 11.5 能效与工程质量：8 分

| 项目 | 分值 | 机器可执行口径 |
|---|---:|---|
| 实测能效 | 4 | `S_eff = S_eff_R18 + S_eff_R50 + S_eff_LLM` |
| 时序收敛、长稳测试和热稳定 | 2 | `2 × weighted_pass_rate(stability_manifest)` |
| 设计清晰度、验证覆盖率和资源效率 | 2 | `2 × weighted_pass_rate(quality_manifest)` |

能效按端到端模型分别计算，不再把不同单位的原始能效做统一几何平均：

```text
Eff_R18 = T_R18 / P_avg_R18      # images/J
Eff_R50 = T_R50 / P_avg_R50      # images/J
Eff_LLM = T_LLM / P_avg_LLM      # tokens/J

S_eff_R18 = 1 × min(1, (Eff_R18 / Eff_R18_ref)^0.5)
S_eff_R50 = 1 × min(1, (Eff_R50 / Eff_R50_ref)^0.5)
S_eff_LLM = 2 × min(1, (Eff_LLM / Eff_LLM_ref)^0.5)
```

`T_R18` 和 `T_R50` 使用 11.3 节对应模型的 `images/s`，`T_LLM` 使用对应 LLM 综合 `tokens/s`。`P_avg_m` 是该模型正式计时窗口内板上传感器平均功耗，单位 W，因此 CNN 能效单位为 `images/J`，LLM 能效单位为 `tokens/J`。三项只在各自模型内部与 `Eff_*_ref` 比较；`Eff_*_ref` 由评分 manifest 固定为赛前参考目标，或在复测后替换为该届对应模型最佳有效能效，禁止直接混合 CNN 和 LLM 原始能效数值。

某个模型未完成、精度不通过、吞吐率为 0、运行错误缺失、传感器读数异常，该模型的能效分直接记 0。剩余模型的能效权重不得重新归一化。`stability_manifest` 至少包含 WNS/TNS、30 分钟连续运行、温度上限、错误恢复和 P95 latency；`quality_manifest` 至少包含 lint、仿真覆盖率、资源报告完整性、第三方 IP 清单和复现实验脚本。

### 11.6 开放加分：最多 10 分

开放加分必须由提交物声明，并由 `open_bonus_manifest` 评分。默认上限如下：

| 项目 | 上限 |
|---|---:|
| 稀疏 GEMM、结构化稀疏或压缩访存 | 2 |
| 多 kernel 并发、异步流水或图执行 | 2 |
| Attention/KV cache 的专门优化 | 2 |
| 训练/反向传播 | 1 |
| 形式验证、故障注入或安全隔离 | 1 |
| 编译器 autotuning、自动映射或架构搜索 | 2 |

每项开放加分按以下公式计算：

```text
bonus_i = cap_i × weighted_pass_rate(open_bonus_manifest_i)
S_open = min(10, sum_i(bonus_i))
```

开放加分不能弥补基础正确性门槛。

## 12. 提交物

最终提交是一次完整系统提交，按以下四类核心内容验收，缺失任一类都会影响基础完整性和对应性能项：

1. **PyTorch 优化算子库与端到端调度**：提交针对参赛硬件设计优化的 PyTorch 算子库，以及 ResNet、Transformer 等端到端网络的 kernel 编排、融合、分块、布局转换、量化、warm-up 和 benchmark 脚本。
2. **PTX-to-AEC 编译工具链**：CUDA `.cu` 到 PTX 可以借用 `nvcc`，但 PTX 到 AEC-G v1.0 ISA 的映射必须由参赛队伍自行编写并提交源码，包括 compiler、assembler、loader 所需元数据、反汇编器或可读报告。
3. **固定 runtime 与 XDMA 驱动**：提交第 8 节固定 CUDA-like runtime API 的完整实现，以及基于 XDMA 的驱动/控制路径实现，覆盖内存管理、DMA、module load、kernel launch、同步、错误恢复和计数器。
4. **U280 RTL 实现**：提交可综合 RTL、U280 XDMA 平台集成工程、约束、bitstream/xclbin、资源/时序/功耗报告，并保证在指定 U280 FPGA 上实现通过、计分时钟 WNS ≥ 0 且板上测试通过。

```text
rtl/                 # 可综合 RTL 与 IP 配置
constraints/         # 时钟、管脚、pblock/SLR 约束
platform/            # U280 XDMA 集成脚本
driver/              # 基于 XDMA 的驱动/用户态驱动控制路径
runtime/              # 固定 CUDA-like runtime API 与实现
compiler/             # PTX-to-AEC 编译器、汇编器、反汇编器
pytorch/              # 硬件优化算子库和端到端模型调度脚本
tests/                # 单元、随机、板上测试
bitstream/            # 指定平台的 xclbin/bitstream
reports/              # utilization、timing、power、benchmark
docs/                 # 架构、ISA 扩展、ABI、复现说明
```

同时必须包含：

- RTL 设计和 U280 XDMA 集成说明；
- 一键或分阶段构建脚本；
- 软件版本、license、第三方 IP 清单；
- 最终 Vivado utilization、timing summary，且所有计分时钟 WNS ≥ 0；
- 每项成绩对应的原始日志和可复现实验命令；
- `design.json`，记录 CU、`logical_warp_width`、`physical_simd_lanes`、`issue_beats_per_warp`、cache、GEMM、频率、数值模式、runtime capability 和驱动版本。

### 时间表
- 7月20日，赛题发布
- 7月22日，组队完毕
- 7月23日，竞赛开始
- 8月6日上午10点，最终提交

## 13. 赛题调整
赛题可能存在瑕疵或错误，可能会做小的调整，但最终评分以相同硬件评分为准以保证公平性。
