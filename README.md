# Agentic4Systems暑期学校GPGPU智能体加速设计竞赛 2026

> 面向本科生与研究生的综合性芯片设计竞赛
> 基于自定义 AEC (Array Execution Core) ISA，完成 GPGPU RTL 设计、EDA 工具链与软件栈开发
>
> 2026 年度 · 三赛道独立评分

---

## 竞赛简介

本竞赛围绕一个完整的 **GPGPU 芯片全栈设计**，设置三个平行赛道，参赛队伍选择其一完成。三个赛道覆盖：

- **赛道 A**：自研 EDA 工具（仿真、验证、综合）
- **赛道 B**：AEC GPGPU RTL 设计（基于自研 128-bit 定长 ISA）
- **赛道 C**：编译器、Runtime 与算子调度

每个赛道独立评分、独立评审。

```text
+--------------------------------------------------------------+
|                 AEC GPGPU 竞赛全栈结构                         |
+--------------------------------------------------------------+
|                                                               |
|  +------------------------------------------------------+   |
|  |  赛道 A：EDA 软件                                     |   |
|  |                                                       |   |
|  |  A1 轻量 RTL 仿真器              --+                  |   |
|  |  A2 验证环境自动生成              --+ 为赛道 B        |   |
|  |  A3 轻量 RTL 逻辑综合工具        --+ 提供工具链       |   |
|  +------------------------------------------------------+   |
|                          |                                    |
|                          v                                    |
|  +------------------------------------------------------+   |
|  |  赛道 B：GPGPU RTL 设计                               |   |
|  |                                                       |   |
|  |  基于 AEC 128-bit 定长 ISA                            |   |
|  |  RTL + CModel + 构建脚本 + 报告                        |   |
|  |                                                       |   |
|  |  为赛道 C 提供：精确 C 模型 (Golden Model)             |   |
|  +------------------------------------------------------+   |
|                          |                                    |
|                          v                                    |
|  +------------------------------------------------------+   |
|  |  赛道 C：编译器 & Runtime & 算子调度                   |   |
|  |                                                       |   |
|  |  C1 AEC IR 编译器 (PTX 风格 -> AEC ISA)              |   |
|  |  C2 主机侧驱动 (libaec.so + 虚拟设备)                 |   |
|  |  C3 算子调度 (ONNX -> AEC GPGPU 推理)                |   |
|  +------------------------------------------------------+   |
|                                                               |
+--------------------------------------------------------------+
```

---

## 竞赛规则与学术诚信

### 原创性要求

本竞赛要求所有参赛作品为**队伍原创**。参赛队伍必须遵守以下规则：

1. **禁止抄袭**：不得抄袭其他队伍的代码、设计方案或文档。引用公开学术资料或开源项目必须如实披露。
2. **禁止硬编码**：不得根据测试用例 ID、文件名、输入 hash 或固定数据直接生成答案。所有输出必须由通用算法逻辑产生。
3. **禁止绕过评测**：不得绕过 result script 或评测基础设施。不得修改官方提供的评测组件。
4. **禁止预计算**：不得提交预计算的网表、预编译的二进制或其他预生成的结果文件。所有产物必须由提交的源码在评测环境中实时生成。
5. **禁止针对性优化**：不得读取隐藏测试用例的标识进行专门优化。工具必须对所有合法输入具备通用处理能力。
6. **禁止未披露的第三方依赖**：使用任何第三方代码、工具库、AI 辅助生成代码均必须在提交文档中如实声明。隐瞒来源可能导致取消资格。

### 允许的辅助开发方式

- **开源工具**：允许使用 Yosys、Mockturtle、Berkeley ABC、Z3、PyVerilog 等开源项目，但必须声明版本、许可证和调用边界
- **LLM 辅助开发**：允许使用大语言模型辅助编码，但必须在原创性声明中披露，且参赛者必须能够解释和维护所有生成代码
- **学术参考**：允许参考公开学术论文和教科书中的算法，必须在文档中注明出处

### 违规处理

违反以上规则可能导致：

- 相关子题得 0 分
- 取消参赛资格

### 评测期间限制

- 评测在指定 Docker 容器或服务器中进行
- **评测期间禁止网络访问**
- 所有第三方依赖必须包含在提交包中
- 使用固定的硬件规格（详见各赛道说明）

---

## 快速导航

| 赛道 | 主题 | 子题 | 总分 |
|------|------|------|------|
| [Track-A](Track-A/) | EDA 软件 | A1 + A2 + A3 | 100 分（归一化） |
| [Track-B](Track-B/) | GPGPU RTL 设计 | 单一赛题 | 100 分 |
| [Track-C](Track-C/) | 编译器 & Runtime | C1 + C2 + C3 | 100 分（归一化） |

---

## 赛道 A：EDA 软件

> 参赛队伍选择本赛道后完成 **A1 + A2 + A3 全部题目**。

| 编号 | 赛题 | 核心目标 | 原始满分 |
|------|------|----------|----------|
| A1 | [轻量 RTL 仿真器](Track-A/A1-simulator/) | 事件驱动 Verilog RTL 仿真器 | 100 |
| A2 | [验证环境自动生成](Track-A/A2-verification/) | 自动生成 testbench + 约束随机测试 + 覆盖率 | 100 |
| A3 | [RTL 逻辑综合工具](Track-A/A3-synthesis/) | RTL -> Nangate45 门级网表 | 100 |

### A1：轻量 RTL 仿真器

设计并实现轻量级事件驱动 Verilog RTL 仿真器，支持 Verilog-2001 核心语法子集、增量编译和并行仿真。

- **评分**：语言解析 (F1) + 仿真正确性 (F2) + 编译性能 (P1) + 仿真性能 (P2) + 多核加速比 (P3)
- **公开测试**：12 个 case（basic01-05、alu、priority_encoder、i2c、ip、axis_fifo、sha256、GEMM）
- **接口**：统一 Makefile（`build` / `compile_sim` / `run` / `parallel_run`）

详见 [Track-A/A1-simulator/](Track-A/A1-simulator/)。

### A2：验证环境自动生成

读取给定 RTL 设计，自动生成可运行的验证环境，支持约束随机测试和覆盖率反馈。

- **评分**：验证骨架（3 分/电路，门禁项）+ 综合覆盖率（7 分/电路）
- **评测对象**：10 个隐藏 RTL benchmark，每题 10 分
- **覆盖率公式**：`C = 0.4 x 行覆盖 + 0.3 x 分支覆盖 + 0.3 x 功能覆盖`

详见 [Track-A/A2-verification/](Track-A/A2-verification/)。

### A3：轻量 RTL 逻辑综合工具

构建可离线运行的 RTL 综合工具，读取 Verilog + SDC + Nangate45 Liberty，生成合法门级网表。

- **评分**：PPA Hypervolume (90) + Runtime (5) + 原创性 (5)
- **评测对象**：20 个电路（10 公开 + 10 隐藏），每题最多 7 个 point
- **公开题库**：LSV01-LSV10，见 [testcases/](Track-A/A3-synthesis/testcases/)

详见 [Track-A/A3-synthesis/](Track-A/A3-synthesis/)。

---

## 赛道 B：GPGPU RTL 设计

> 参赛队伍选择本赛道后完成**单一赛题**：AEC GPGPU RTL 设计。

基于 AEC 128-bit 定长指令集，设计并实现 GPGPU 系统。提交完整、可综合的 RTL 和与其架构行为一致的 CModel。

### AEC ISA 特性

| 特性 | 规格 |
|------|------|
| 指令宽度 | 128-bit 定长 |
| 寄存器文件 | 256 寄存器 x 32-bit / thread |
| 谓词寄存器 | 8 个独立谓词 P0-P7 |
| 执行模型 | Warp = 32 lanes |
| CTA 规模 | 最多 256 threads / CTA (8 warps) |
| 内存空间 | `.gmem`、`.smem`、`.cmem`、`.lmem`、`.pmem` |

### 评分细则

- **正确性 50 分**：RTL 指令 (40) + CModel (5) + RTL/CModel alignment (5)
- **PPA 40 分**：Performance benchmark (15) + 频率 (10) + Perf/Watt (8) + Perf/Area (7)
- **报告 10 分**：设计说明 (4) + 验证说明 (3) + PPA/限制/合规 (3)

### 公开测试

36 个公开测试用例，覆盖 ABI、算术、逻辑、内存、控制流、SFU 和完整 kernel（vadd、gemm_naive、histogram）。

详见 [Track-B/](Track-B/) 和 [testcases/](Track-B/testcases/)。

---

## 赛道 C：编译器 & Runtime & 算子调度

> 参赛队伍选择本赛道后完成 **C1 + C2 + C3 全部题目**。

| 编号 | 赛题 | 核心目标 | 原始满分 |
|------|------|----------|----------|
| C1 | [AEC IR 编译器](Track-C/C1-compiler/) | PTX 风格 IR -> AEC ISA 机器码 | 100 |
| C2 | [主机侧驱动](Track-C/C2-runtime/) | `libaec.so` + 虚拟设备驱动 | 100 |
| C3 | [算子调度](Track-C/C3-scheduler/) | ONNX 模型 -> AEC GPGPU 推理 | 100 |

### C1：AEC IR 编译器

将 PTX 风格中间表示编译为 AEC ISA 机器码，支持指令调度、寄存器分配和多精度 GEMM 优化。

- **评分**：正确性 (50) + 性能 (35) + 鲁棒性 (5) + Agent 优化 (10)
- **公开测试**：5 道 PTX 题（PTX-01 ~ PTX-05），覆盖 T1-T5 类别
- **接口**：`aec-cc input.ptx -O2 -o output.aecbin`

详见 [Track-C/C1-compiler/](Track-C/C1-compiler/)。

### C2：主机侧驱动与 Runtime

实现 AEC 虚拟 GPGPU 的 Host Runtime：内存管理、kernel 启动、stream/event、计算库（10 种 GEMM dtype）和虚拟驱动。

- **评分**：Runtime (30) + 计算库 (30) + 驱动 (20) + Agent (20)
- **Starter Kit**：完整开发包，见 [starter-kit/](Track-C/C2-runtime/starter-kit/)
- **快速开始**：
  ```bash
  cd starter-kit && make -j2 && make examples
  ./bin/01_device_query
  python3 grader/public_grade.py --submission . --profile public
  ```

详见 [Track-C/C2-runtime/](Track-C/C2-runtime/)。

### C3：算子调度与模型部署

实现算子调度层，将深度学习模型（MLP、ResNet-18、Transformer）部署到 AEC GPGPU 上推理。

- **评分**：图解析 (10) + 分解 (15) + 融合 (15) + 内存规划 (10) + 端到端 (50)
- **评测模型**：MLP（MNIST，>=98%）、ResNet-18（CIFAR-10，>=85%）、Transformer（decoder-only）
- **支持算子**：17 种 ONNX 算子

详见 [Track-C/C3-scheduler/](Track-C/C3-scheduler/)。

---

## 跨赛道关联

```text
赛道 A: A1 仿真器        --+
        A2 验证生成      --+ 为赛道 B 提供 RTL 仿真/验证/PPA 工具链
        A3 综合工具      --+

赛道 B: CModel           --- 为赛道 C 的编译器、Runtime、算子调度提供精确参考
```

**统一 ISA 规范**：[Track-B/spec.md](Track-B/spec.md)

> **注**：由于设计难度原因，跨赛道 ISA 一致性不做强制要求。赛道 C 中使用的部分指令（如 TMUL）可能不在赛道 B 的必须实现范围内。

---

## 评分归一化

**每个赛道总分为 100 分。**

- **赛道 A** 和 **赛道 C** 各分为 3 道赛题，每道赛题的原始满分均为 100 分。最终赛道总分为三道题归一化后等权平均，满分为 100 分。
- **赛道 B** 为单一赛题，满分 100 分。

### 归一化公式

赛道 A 和 C 的赛道总分按各子题归一化后等权平均：

```text
赛道总分 = (子题1归一化分 + 子题2归一化分 + 子题3归一化分) / 3
子题归一化分 = (子题原始得分 / 100) x 100
```

> 即三道题各 100 分，取平均后赛道总分也是 100 分。

### 赛道 A：100 分

| 赛题 | 原始满分 | 评分构成 |
|------|----------|----------|
| A1 轻量 RTL 仿真器 | 100 | F1 + F2 + P1 + P2 + P3（按 case 权重累加） |
| A2 验证环境自动生成 | 100 | 10 电路 x 10 分 |
| A3 逻辑综合工具 | 100 | PPA 90 + Runtime 5 + 原创性 5 |

**示例**：A1 得 70 分、A2 得 80 分、A3 得 60 分，则赛道总分 = (70 + 80 + 60) / 3 = 70 分。

### 赛道 B：100 分

| 赛题 | 总分 | 评分构成 |
|------|------|----------|
| AEC GPGPU RTL 设计 | 100 | 正确性 50 + PPA 40 + 报告 10 |

### 赛道 C：100 分

| 赛题 | 原始满分 | 评分构成 |
|------|----------|----------|
| C1 编译器 | 100 | 正确性 50 + 性能 35 + 鲁棒 5 + Agent 10 |
| C2 Runtime | 100 | Runtime 30 + 计算库 30 + Driver 20 + Agent 20 |
| C3 算子调度 | 100 | 图解析 10 + 分解 15 + 融合 15 + 内存 10 + 端到端 50 |

**示例**：C1 得 75 分、C2 得 60 分、C3 得 80 分，则赛道总分 = (75 + 60 + 80) / 3 = 71.67 分。

---

## 决赛题目：U280-GPGPU 集成设计

> 本赛道面向完成初赛并进入决赛的队伍，要求在单张 Alveo U280 FPGA 上实现完整的可编程 GPGPU 原型系统。

决赛题目要求在 U280 FPGA 上完成 GPGPU 的全栈集成设计：基于 AEC 128-bit 定长 ISA，实现可综合的 RTL 设计（SIMT 执行、FP8 GEMM、寄存器堆、存储层级、SFU），配合 PTX→AEC 编译器、XDMA 驱动/runtime 和 PyTorch 后端，在 ResNet-50/Transformer 等端到端模型上完成板上验证与性能评测。详见 [决赛题目](Final/contest.md)。

---

## 开发与评测环境

| 工具 | 版本 | 适用赛道 |
|------|------|----------|
| Verilator | 5.049 devel | B（RTL 仿真） |
| Yosys | 0.64+308 | B（综合） |
| OpenSTA | v2.2.0-2121 | B（时序分析） |
| GCC/G++ | 13.3.0 | B、C |
| Python | 3.10+ | A、C |
| ASAP7 PDK | 1.7, 7.5-track v28 | B |
| Z3 SMT Solver | 4.12+ | A2 |
| Nangate45 | typical Liberty | A3 |

---

## 提交规范

每个赛道有独立的提交目录结构要求，详见各赛道 README。

### 通用要求

- 提交必须包含完整可复现源码
- 评测在指定镜像、无网络环境中进行
- 第三方依赖必须如实披露
- 大模型辅助开发允许，但必须能够解释和维护生成代码
- 禁止硬编码、预计算、针对性优化和绕过评测

---

## 仓库结构

```text
.
|-- README.md                              # 本文件（竞赛总览）
|-- LICENSE
|-- Track-A/                                # 赛道 A：EDA 软件
|   |-- README.md                          # A 赛道总览
|   |-- A1-simulator/                      # A1：轻量 RTL 仿真器
|   |   |-- spec.md                        # 赛题说明
|   |   |-- scoring.md                     # 评分细则
|   |   +-- testcases/                     # 公开测试
|   |-- A2-verification/                   # A2：验证环境自动生成
|   |   |-- spec.md
|   |   |-- scoring.md
|   |   +-- testcases/
|   +-- A3-synthesis/                      # A3：逻辑综合工具
|       |-- spec.md
|       |-- scoring.md
|       +-- testcases/                     # 公开电路 LSV01-LSV10
|-- Track-B/                                # 赛道 B：GPGPU RTL 设计
|   |-- README.md                          # B 赛道总览
|   |-- spec.md                            # AEC ISA 完整规范
|   |-- scoring.md                         # 评分细则
|   |-- sram/                              # ASAP7 SRAM wrapper
|   +-- testcases/                         # 公开参考测试
+-- Track-C/                                # 赛道 C：编译器 & Runtime
    |-- README.md                          # C 赛道总览
    |-- C1-compiler/                       # C1：AEC IR 编译器
    |   |-- spec.md
    |   |-- scoring.md
    |   +-- testcases/                     # PTX-01 ~ PTX-05
    |-- C2-runtime/                        # C2：主机侧驱动
    |   |-- spec.md
    |   |-- scoring.md
    |   +-- starter-kit/                   # 完整开发包
    +-- C3-scheduler/                      # C3：算子调度
        |-- spec.md
        |-- scoring.md
        +-- testcases/
```

---

## 联系方式

如有任何关于赛题的疑问，请通过官方渠道联系组委会。

许可证：MIT（见 [LICENSE](LICENSE)）
