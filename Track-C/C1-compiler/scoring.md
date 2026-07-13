# C1：PTX 到 AEC 标量机器码编译器 - 评分细则

## 总分：100 分

| 类别 | 分值 |
|------|-----:|
| A. 编译与执行正确性 | 50 |
| B. 生成代码效率 | 40 |
| C. 泛化与鲁棒性 | 10 |

---

## A. 编译与执行正确性（50 分）

每个测试用例的评测流程如下：

1. 参赛者编译器从 PTX 输入生成 `.aecbin`；
2. 评测系统加载 `.aecbin`；
3. 根据测试用例的 manifest 初始化 kernel 参数、grid/block 配置和输入输出 buffer；
4. 执行生成的 AEC 指令流；
5. 将输出结果与参考结果进行比对；
6. 仅正确结果纳入性能评分。

正确性在 100 个隐藏测试用例上评测：

| 类别 | 数量 | 重点 |
|------|------:|------|
| T1 基础指令 Lowering | 20 | PTX 解析、参数加载、special register、基础算术、load/store、branch、ret lowering |
| T2 标量优化 | 20 | 常量传播、死代码删除、公共子表达式消除、循环不变量外提、基本块合并 |
| T3 内存访问优化 | 20 | global memory 访问、重复 load、load hoisting、简单内存复用、地址计算优化 |
| T4 寄存器分配与指令调度 | 20 | GPR/predicate 分配、live range 管理、寄存器压力、load/compute interleaving、依赖调度 |
| T5 FP32 Scalar GEMM | 20 | FP32 标量 GEMM、二维索引、K 维循环、地址计算、标量 multiply-add 调度 |

每个测试用例均由 PTX 文件和对应 manifest 文件共同描述。manifest 用于指定 kernel 名称、grid/block 维度、kernel 参数、输入输出 buffer 和正确性检查规则。

正确性得分按通过测试用例数量计算：

```text
correctness_score = 50 * passed_cases / total_cases
```

若某测试用例编译失败、执行失败、输出错误或超时，则该测试用例不得分，并且不参与该测试用例的性能评分。

---

## B. 生成代码效率（40 分）

性能评分衡量参赛编译器生成 AEC 指令流的执行效率。性能只在正确性通过的测试用例上计算。

| 类别 | 性能分 |
|------|-------:|
| T1 基础指令 Lowering | 0 |
| T2 标量优化 | 8 |
| T3 内存访问优化 | 10 |
| T4 寄存器分配与指令调度 | 10 |
| T5 FP32 Scalar GEMM | 12 |

T1 主要用于检查基础 lowering 和执行正确性，不单独计入性能分。T2 到 T5 用于衡量优化能力。

### B.1 性能度量

对每个性能测试用例，评测系统记录生成代码的执行开销。具体度量可以采用评测平台提供的执行时间、周期数或等价性能指标。评分时以官方基线编译器结果作为参照。

对测试用例 `i`，定义加速比：

```text
r_i = baseline_i / participant_i
```

其中：

```text
baseline_i     = 官方基线编译器在该测试上的性能指标
participant_i  = 参赛编译器在该测试上的性能指标
```

指标越小表示性能越好。因此当 `r_i > 1` 时，表示参赛结果优于官方基线。

### B.2 性能评分原则

性能分按各类别性能测试的几何平均加速比计算。仅正确性通过的测试用例参与性能统计。

对于每个性能测试类别，先计算该类别内部的几何平均加速比：

```text
R_category = geometric_mean(r_i)
```

然后按照类别权重映射为该类别分数：

| 几何平均加速比 | 类别性能得分 |
|---:|---:|
| `< 1.00` | 0 |
| `1.00 - 1.10` | 线性映射到该类别分数的 40% |
| `1.10 - 1.25` | 线性映射到该类别分数的 80% |
| `>= 1.25` | 该类别满分 |

具体映射可由评测脚本固定实现。若某类别中没有正确通过的测试用例，则该类别性能分为 0。

### B.3 诊断指标

以下指标可作为评测报告中的诊断信息，但不直接计分：

```text
instruction_count
register_count
predicate_count
spill_count
branch_count
load_count
store_count
memory_instruction_ratio
estimated_dependency_depth
```

---

## C. 泛化与鲁棒性（10 分）

泛化与鲁棒性在自动生成的变体测试上评测。变体测试仍然遵循 C1 输入语言和 AEC opcode 范围。

变体包括：

```text
参数规模变化；
grid/block 维度变化；
寄存器重命名；
基本块顺序变化；
循环次数变化；
死代码插入；
无关计算插入；
寄存器压力增加；
地址计算形式变化；
内存访问模式变化；
标量 GEMM 矩阵大小变化。
```

泛化与鲁棒性测试共 50 个变体测试用例：

| 类别 | 数量 | 重点 |
|------|------:|------|
| T1 变体 | 10 | 基础 lowering 稳定性 |
| T2 变体 | 10 | 标量优化对语法和结构变化的鲁棒性 |
| T3 变体 | 10 | 内存访问模式变化 |
| T4 变体 | 10 | 寄存器压力和调度变化 |
| T5 变体 | 10 | FP32 Scalar GEMM 尺寸和边界变化 |

泛化与鲁棒性得分按通过变体数量计算：

```text
robustness_score = 10 * passed_variant_cases / total_variant_cases
```

参赛编译器不得假定公开测试用例的变量名、寄存器名、基本块顺序、循环结构或矩阵大小固定不变。

---

## 评分汇总

最终得分为：

```text
total_score = correctness_score + performance_score + robustness_score
```

其中：

```text
correctness_score <= 50
performance_score <= 40
robustness_score  <= 10
total_score       <= 100
```

---

## 测试题类别说明

### T1：基础指令 Lowering

T1 主要评测从 PTX 到 AEC 的基础 lowering 能力，包括：

```text
PTX 文件解析；
kernel 参数加载；
special register 读取；
整数和 FP32 基础运算；
global memory load/store；
predicate 比较；
条件分支；
kernel ret 到 HALT 的 lowering。
```

典型测试包括：

```text
vector_add
copy
saxpy
```

### T2：标量优化

T2 主要评测常规标量编译优化，包括：

```text
常量传播；
死代码删除；
公共子表达式消除；
循环不变量外提；
基本块合并。
```

典型测试包括：

```text
repeated expression
loop-invariant polynomial
dead computation
```

### T3：内存访问优化

T3 主要评测 global memory 访问相关优化，包括：

```text
重复 global load；
load hoisting；
简单内存复用；
地址计算优化；
memory instruction reduction。
```

典型测试包括：

```text
repeated global memory reuse
stencil-like scalar load pattern
```

### T4：寄存器分配与指令调度

T4 主要评测后端代码生成能力，包括：

```text
虚拟寄存器到 AEC GPR 的分配；
predicate 分配；
live range 管理；
寄存器压力处理；
load/compute interleaving；
基本依赖调度。
```

典型测试包括：

```text
long arithmetic dependency chain
mixed load and compute sequence
moderate register pressure kernel
```

### T5：FP32 Scalar GEMM

T5 评测 FP32 标量 GEMM 的代码生成与优化能力。测试形式为：

```text
C = A x B
A, B, C 均为 FP32
每个 thread 计算一个 C[i, j]
使用 scalar for-loop over K
使用 global memory load/store
使用 FP32 scalar multiply-add
```

T5 主要考察：

```text
二维索引计算；
K 维循环 lowering；
global memory 地址计算；
FP32 multiply-add 调度；
loop-level scalar optimization；
寄存器分配与 live range 管理。
```

T5 不使用 Tensor 指令，不要求 Tensor tile、Tensor register 或低精度矩阵乘支持。

---

## 提交与评测要求

参赛队伍提交：

```text
1. C1 编译器源码；
2. 编译器可执行入口 compiler/aec-cc；

评测系统对每个测试用例执行：

```bash
compiler/aec-cc kernel.ptx -O2 -o output.aecbin --report compile_report.json
```

每个测试用例的 manifest 由测试集提供。评测系统根据 manifest 运行生成的 `.aecbin` 并检查输出结果。
