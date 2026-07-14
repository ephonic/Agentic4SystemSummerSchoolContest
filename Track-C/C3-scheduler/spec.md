# C3：算子调度与模型部署 - 赛题说明

## 赛题概述

本文档面向参赛选手，规定 C3 赛道各子任务的提交要求、命令行接口、数据格式与自动评测规则。请在开发前完整阅读。

评测采用固定的命令行模板调用选手程序。选手在报名时须提交对应的命令模板，其中以占位符表示评测机在运行时填入的路径。

## 子任务与分值

| 子任务 | 分值 | 评测方式 |
|--------|-----:|----------|
| C3.1 计算图解析与表示 | 10 | 自动检查 |
| C3.2 算子分解与内核选择 | 15 | 微基准测试 |
| C3.3 算子融合与图优化 | 15 | 微基准测试 |
| C3.4 内存规划与调度 | 10 | Code Review |
| C3.5 典型模型部署 | 50 | 端到端测试 |
| **合计** | **100** | |

---

## C3.1 计算图解析与表示（10 分）

### 任务描述

实现一个命令行程序，读取指定的 ONNX 模型文件，解析其计算图，并将计算图以有向无环图（DAG）的形式导出为一个 JSON 文件。

本任务测试计算图的解析与表示能力，完成模型加载（4 分）、正确的计算图解析（6 分）。

### 命令行接口

```bash
<选手程序> --onnx <model.onnx> --output <dag.json>
```

- `--onnx`：输入 ONNX 模型文件路径
- `--output`：输出 DAG JSON 文件路径，程序须将结果写入该路径
- 程序须以退出码 0 结束。非零退出码视为该模型处理失败
- 标准输出（stdout）的内容不参与评测，评测仅读取 `--output` 指定的文件

报名时须提交命令模板，使用 `{onnx}` 与 `{output}` 作为占位符，例如：

```bash
python export_dag.py --onnx {onnx} --output {output}
```

### 输出格式

输出文件须为一个合法的 JSON 文件。建议采用下述结构表示计算图，字段命名建议直接沿用 ONNX 图中的原始名称（节点名与张量名）。

```json
{
  "format_version": "1.0",
  "graph_inputs":  [ { "name": "input",  "dtype": "FLOAT", "shape": ["batch", 1, 28, 28] } ],
  "graph_outputs": [ { "name": "logits", "dtype": "FLOAT", "shape": ["batch", 10] } ],
  "nodes": [
    {
      "name": "/fc1/Gemm",
      "op_type": "Gemm",
      "inputs": ["/flatten/Flatten_output_0", "fc1.weight", "fc1.bias"],
      "outputs": ["/fc1/Gemm_output_0"]
    }
  ],
  "edges": [
    { "src_node": "/flatten/Flatten", "dst_node": "/fc1/Gemm", "tensor": "/flatten/Flatten_output_0" }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `graph_inputs` | 模型的输入张量列表（不含权重等 initializer） |
| `graph_outputs` | 模型的输出张量列表 |
| `nodes` | 计算图节点列表，每个节点包含节点名、算子类型、输入张量名列表、输出张量名列表 |
| `edges` | 数据依赖边列表，表示张量在节点间的流动 |

---

## C3.2 算子分解与内核选择（15 分）

### 任务描述

子任务目标为将高层算子分解为 GPGPU 内核执行序列，同时支持多精度的自动选择。

> 评测脚本：`benchmarks/c32_c33/bench_c32_c33.py`
> 评测模型：MNIST MLP、CIFAR-10 简化 ResNet-18

### 评测输入

评测脚本**只**通过以下公共 API 抓信号，避免依赖私有实现：

| 信号 | 抓取方式 |
|------|----------|
| 原始算子 DAG | `import_onnx_graph(model.onnx)` |
| 算子精度决策 | `strategy.select_precision(node, graph) → PrecisionProfile`，结果与 `hardware.supported_precisions()` 取交集 |
| 算子分解结果 | `strategy.decompose(node, graph, precision) → List[KernelSpecRef]` |
| 内核启动参数 | `strategy.tune_kernel(ref, precision, problem_size) → KernelTuningParams`，**必须填全 `block_x` / `grid_x` / `smem_bytes`** |
| 中间张量 | 通过 `KernelSpecRef.outputs` 与 `node.outputs` 差集识别，名称形如 `__c3_inter_N__` |

### 评分维度

| 维度 | 分值 |
|------|-----:|
| D1. 多精度路由正确性 / 覆盖度 | 3 |
| D2. 内核序列完整性 | 3 |
| D3. 中间张量跟踪 | 3 |
| D4. 内核调优参数有效性 | 3 |
| D5. 硬件能力覆盖度 | 3 |
| **合计** | **15** |

#### D1. 多精度路由正确性 / 覆盖度（3 分）

| 子项 | 满分 | 检查内容 |
|------|-----:|----------|
| 敏感算子强制 FP32 | 1.5 | Softmax / LayerNorm / BatchNorm / ReduceMax / ReduceSum / ReduceMean 的 `precision == "fp32"` 占比 × 1.5 |
| 精度多样度 | 1.0 | 出现 fp32 / fp16 / fp8 / fp4 中的 N 种，得分 = `N / 4`（**目标 ≥ 4 种**） |
| 非敏感算子走可用精度 | 0.5 | MatMul / Linear / Conv2d 的精度 ∈ `hardware.supported_precisions()` 占比 × 0.5 |

> **硬指标**：在 `FULL_FP32` 模式下与 PyTorch 参考做 `max_abs_diff ≤ 1e-3`、`top1_match ≥ 0.99`。强行对敏感算子开 FP8/FP4 导致超阈，直接扣光 D1 的 3 分。

#### D2. 内核序列完整性（3 分）

| 算子 | 关键 kernel（前缀） | 命中权重 |
|------|---------------------|---------:|
| MatMul / Linear | `matmul_*` | 0.5 |
| Softmax | `reduce_max` + `exp` + `reduce_sum` + `div` 全套 | 0.5 |
| LayerNorm | `reduce_mean` + `sub` + `mul` + `sqrt` 全套 | 0.5 |
| Conv2d（3×3） | `winograd_forward_*` 或 `im2col_*` | 0.5 |
| 全部节点 | `len(kernel_sequence) > 0`（非空覆盖率） | 1.0 |

公式：`score = seq_coverage × 1.0 + key_seq_score × 2.0`，最高 3.0。

#### D3. 中间张量跟踪（3 分）

公式：`score = key_intermediate_ratio × 2.0 + total_intermediate_ratio × 1.0`，最高 3.0。
识别规则：`len(KernelSpecRef.outputs \ node.outputs) > 0`，关键算子 = Softmax / LayerNorm / Conv2d。

#### D4. 内核调优参数有效性（3 分）

| 子项 | 满分 | 检查内容 |
|------|-----:|----------|
| `tuning_coverage` | 1.5 | 产出非空 `tuning_params` 的算子占比 = `nodes_with_tuning / total_tunable`（**目标 ≥ 90%**） |
| `tuning_validity` | 1.5 | 每个有 tuning 的算子跑 3 条断言：① `0 < block_x ≤ max_threads_per_block`，② `grid_x > 0`，③ `smem_bytes ≤ hardware.smem_bytes`（`-1` 视为合规）；得分 = 通过断言数 ÷ (3 × `nodes_with_tuning`) |

> **常见扣分点**：漏写 `grid_x`（或设成 0）、`block_x` 大于 `max_threads_per_block`、smem 超过硬件预算且没标 `-1`。

#### D5. 硬件能力覆盖度（3 分）

| 子项 | 满分 | 检查内容 |
|------|-----:|----------|
| 精度种类 | 1.0 | ≥ 2 种使用得 0.5，3–4 种满分 |
| GEMM kernel 多样度 | 1.0 | 至少出现 `matmul_f32` 和 `matmul_f16`，再出现 `matmul_f8`/`matmul_f4` 各 +0.25 |
| Conv2d 策略选择 | 1.0 | im2col 与 Winograd 都被选过（按硬件能力切换） |

---

## C3.3 算子融合与图优化（15 分）

### 任务描述

子任务的目标是实现算子融合及计算图优化。

### 评测输入

`GraphPassPipeline(enable_fusion=True, …)` 在原始图上运行一遍，从 `pass_results['Fusion']['stats']['fusion_log']` 拿到每条匹配记录。

### 评分维度

| 维度 | 分值 |
|------|-----:|
| F1. 融合 pattern 覆盖 | 5 |
| F2. Kernel launch 数减少 | 3 |
| F3. 中间 buffer 数减少 | 3 |
| F4. 融合正确性 | 4 |
| **合计** | **15** |

#### F1. 融合 pattern 覆盖（5 分）

5 个目标 pattern，命中 1 个 +1 分；得分 = 命中数，最高 5 分。

| Pattern | 触发条件 |
|---------|----------|
| `FusedMatMulBias` | MatMul → AddBias |
| `FusedConv2dBatchNorm` | Conv2d → BatchNorm |
| `FusedEWChain` | 2–5 个相邻 elementwise（Add → Mul → ReLU 等） |
| `FusedSoftmaxDropout` | Softmax → Dropout |
| `FusedResidualNorm` | skip-Add → LayerNorm |

> **实现备注**：当前 ResNet-18 训练时把 BN 折进了 conv 权重，导出的 ONNX 里**没有 BN 节点**——`FusedConv2dBatchNorm` 自然不会命中。要拿这 1 分，需要在 `scheduler/graph_passes/fusion.py` 里加一个**预融合** pass（从 BN 参数 + conv 权重反向算回 merged conv）；或者可以要求做code review。

#### F2. Kernel launch 数减少（3 分）

公式：`score = min((raw_launches − opt_launches) / raw_launches × 5.0, 3.0)`

锚点：`reduction ≥ 60%` 即满分。

#### F3. 中间 buffer 数减少（3 分）

公式：`score = min((raw_buffers − opt_buffers) / raw_buffers × 5.0, 3.0)`

锚点：`reduction ≥ 60%` 即满分。

#### F4. 融合正确性（4 分）

| 检查项 | 分值 |
|--------|-----:|
| `graph.outputs` 保留可解析 | 1 |
| `graph.inputs` 保留 | 1 |
| `graph.validate()` 通过（无环 / 张量引用一致） | 1 |
| 优化图节点数 ≤ 原始图节点数 | 1 |

外加一次**数值对齐检查**：用 `MockRuntime` 跑原始图 + 优化图，与 FP32 参考做 `max_abs_diff` 比较；任一条路径 `> 1e-3` 则 F4 全扣。

---

## C3.4 内存规划与调度（10 分）

**评分方式：Code Review**
**评分原则：检测功能是否实现；性能提升与实测效果统一在 C3.5 中体现**

| 子项 | 题目要求 | 分值 |
|------|----------|-----:|
| A | 设备内存池 / 权重预加载路径 | 2 |
| B | 中间张量 lifetime 内存复用 | 2 |
| C | 内存碎片整理（池化回收与再分配） | 2 |
| D | 计算与传输重叠：权重预取 | 2 |
| E | 计算与传输重叠：流级并行 | 2 |

> - **实现即得分**：代码中存在清晰、可定位的实现路径，且与调度/执行计划打通，即给对应分
> - **空壳不得分**：仅有注释、接口声明、未接线的 stub，或仅打印日志而无真实逻辑 → 0 分
> - **等价实现可接受**：命名可与参考仓库不同，但须能对应到功能检查点
> - **五项等权**：A / B / C / D / E 各满分 2 分，互不替代

### A. 设备内存池与权重预加载路径（2 分）

| 分 | 审查要点 |
|----|----------|
| 0 | 无设备侧分配封装；权重仍只在 host 侧临时数组中，无 H2D/device buffer 路径 |
| 1 | 有 `malloc/free`（或等价）封装，或有 H2D 上载步骤，但二者未形成完整链路 |
| 2 | 同时具备：① 设备内存分配/释放接口；② 模型权重/常量经计划或初始化步骤上传到 device buffer，并被后续计算步骤引用 |

### B. 中间张量 Lifetime 内存复用（2 分）

| 分 | 审查要点 |
|----|----------|
| 0 | 无 lifetime 分析；每个中间 tensor 独立分配且无复用逻辑 |
| 1 | 有 lifetime 分析（first/last use 或等价），但**未**改写执行计划/分配策略 |
| 2 | 将生命周期不重叠的中间张量映射到同一逻辑 slot / 同一物理缓冲，且该逻辑接入执行计划生成路径 |

### C. 内存池碎片整理 / 池内复用（2 分）

| 分 | 审查要点 |
|----|----------|
| 0 | `free` 后立即归还后端且无 free-list / 缓存块；无池语义 |
| 1 | `free` 后块进入可复用结构（free list / cache），后续 `malloc` 可命中已释放块 |
| 2 | 在 1 的基础上，有明确的块管理策略之一：best-fit / 按 size class / coalesce（合并相邻空闲块）/ 分段整理 |

### D. 权重预取（2 分）

| 分 | 审查要点 |
|----|----------|
| 0 | 全部权重仍在首个 kernel **之前** bulk 上传；无预取 / 异步传输语义 |
| 1 | 有 `async` H2D 标注，但权重仍全在计算前上载，计划顺序未体现「边算边传」 |
| 2 | 至少将部分层权重的 `alloc/h2d` 相对消费 kernel **前移到前序计算附近**（「当前层算、下一层传」的计划语义） |

### E. 流级并行（2 分）

| 分 | 审查要点 |
|----|----------|
| 0 | 无 stream / 多队列概念；所有计算步骤隐含单流顺序执行 |
| 1 | 有 stream 字段、copy/compute 双队列或 stream API 封装，但无依赖分析 |
| 2 | 无数据依赖的算子/层被分配到**不同** compute stream（或等价并发队列），计划中可见多 compute stream |

---

## C3.5 典型模型部署（50 分）

### 任务描述

实现一个推理 worker（常驻进程），读取指定的 ONNX 模型与一批输入张量，在 GPU 上完成模型推理，并将推理结果写出。评测从精度、准确率、峰值显存与运行时间四个方面进行考核。

满分 50 分：精度测试占 15 分（通过门槛），运行时间占 25 分，峰值显存占 10 分。

### 运行方式：持久化 Worker

C3.5 的推理程序以**常驻进程（worker）**方式运行：评测机启动一次 worker，通过标准输入/输出多次下发任务，计时只覆盖"加载模型 + 推理"，不包含进程启动与框架初始化。完整协议见 `C35_WORKER_PROTOCOL.md`，此处摘录要点：

- 评测机以选手报名时提交的**启动命令**（不带任务参数，如 `python infer_worker.py`）启动 worker.
- worker 完成初始化后，向 stdout 输出一行 `READY`。
- 评测机经 stdin 逐行下发任务 JSON：`{"onnx": "...", "input": "...", "output": "...", "batch_size": 256}`；worker 完成后经 stdout 回一行 `{"status": "ok", "samples": N}`。
- 评测机发送 `{"cmd": "exit"}`，worker 干净退出。
- **stdout 仅用于协议信号**（`READY` 与结果行）；所有日志走 stderr。

选手可用资料包中的 `selfcheck_worker.py` 自测自己的 worker 协议实现。

### 模型规格

评测使用**四类模型**，每类各一个公开版本（供调试）与一个隐藏版本（供评分）。两个版本结构相同、权重不同。

| 模型 | 任务 | 输入张量 | 输入形状 | 输出张量 | 输出形状 |
|------|------|----------|----------|----------|----------|
| MLP | MNIST 手写数字分类 | `input` (float32) | `[N, 1, 28, 28]` | `logits` (float32) | `[N, 10]` |
| ResNet-18（简化） | CIFAR-10 图像分类 | `input` (float32) | `[N, 3, 32, 32]` | `logits` (float32) | `[N, 10]` |
| Transformer（decoder-only） | 合成序列任务 | `input_ids` (int64) | `[N, 18]` | `logits` (float32) | `[N, 18, 14]` |
| BigFormer（大尺寸双向 Transformer） | 显存卸载压力测试 | `input_ids` (int64) | `[N, 32]` | `logits` (float32) | `[N, 32, 14]` |

说明：
- 四个模型的批量维 `N` 均为动态维，支持任意批量大小
- 输入数据已完成预处理：MNIST 与 CIFAR-10 图像已按标准均值与标准差归一化；Transformer 与 BigFormer 的 `input_ids` 为取值 0 至 13 的 token id
- 选手程序无须再做任何预处理，直接输入模型即可

#### BigFormer 与显存卸载

BigFormer 是一个大尺寸、单次前向（非自回归）的双向 Transformer，其 **fp32 权重约 18 GB，超过评测 GPU 的 16 GB 显存**。因此：

- 权重以 **ONNX 外部数据（external data）**形式存储：模型为 `bigformer_*.onnx`（图结构）+ 同名 `bigformer_*.onnx.data`（权重）两个文件，二者须放在同一目录，用 `onnx.load` 会自动关联。
- 一次性把全部权重上载到 GPU 会 **OOM**。

### 输入格式

`<input_dir>/manifest.json` 描述输入目录中的各张量：

```json
{
  "tensors": [
    { "name": "input", "file": "input.npy", "dtype": "float32", "shape": [10000, 1, 28, 28] }
  ]
}
```

- 每个张量对应一个 `.npy` 文件，其第 0 维为样本数 `N`
- `name` 为模型的输入张量名（MLP 与 ResNet 为 `input`，Transformer 为 `input_ids`）

### 输出格式

选手程序须在 `<output_dir>/` 下写入：

```text
manifest.json
<output_name>.npy
```

- `manifest.json` 采用与输入相同的结构：`{"tensors": [{"name", "file", "dtype", "shape"}, ...]}`
- 输出张量的 `name` 须使用模型的输出张量名（三个模型均为 `logits`）
- 输出须覆盖全部 `N` 个样本，且第 0 维的顺序与输入一致
- 输出 dtype 为 `float32`

### 评分规则

各模型参与的测试项如下：**四个模型都参与精度校准**；**MLP 与 ResNet 参与准确率门槛**；**性能测试（运行时间 + 峰值显存）只针对 ResNet 与 BigFormer**。MLP 与 Transformer 推理耗时极短，不计入性能分。

| 模型 | 精度校准 | 准确率门槛 | 性能测试（时间 + 显存） |
|------|:---:|:---:|:---:|
| MLP | ✅ | ≥ 98% | ✗ |
| Transformer | ✅ | — | ✗ |
| ResNet-18 | ✅ | ≥ 85% | ✅ |
| BigFormer | ✅ | — | ✅ |

#### （1）精度测试（通过门槛，全部模型）

将选手输出张量与标准答案逐元素比较。参考标准为 PyTorch 在 fp32 精度下计算的参考输出。

通过条件：

```text
对所有元素：|out - golden| <= atol + rtol * |golden|
（等价于 numpy.allclose(out, golden, rtol, atol)）
```

阈值统一为 `rtol = atol = 1e-3`。

> **注意**：标准答案以 fp32 精度计算。若在计算中使用 TF32、FP16、BF16 等低精度加速，ResNet 等较深网络的输出容易超出 1e-3 的阈值。如需以低精度换取性能，须自行确认精度仍在阈值范围内。

#### （2）准确率测试（分类模型的通过门槛）

MLP 与 ResNet：对输出 `logits` 取 argmax，与 `labels.npy` 中的真值标签比较，计算 top-1 准确率。

| 模型 | 准确率阈值 |
|------|-----------|
| MLP（MNIST） | ≥ 98% |
| ResNet-18（CIFAR-10） | ≥ 85% |

#### （3）运行时间（仅 ResNet 与 BigFormer）

评测机按持久化 worker 协议对每个模型下发 `2 次 warmup + 5 次计时` 任务，计时窗口为每个任务的"加载模型 + 推理 + 写输出"（不含进程启动与初始化），取 5 次计时的**中位数**作为该模型的运行时间。

#### （4）峰值显存（仅 ResNet 与 BigFormer）

评测机在计时期间通过 NVML 采样 GPU 已用显存，采样间隔 20 ms，取**进程绝对峰值**（5 次计时的最大值）。

#### 通过判定

精度测试通过，且准确率测试通过（若该模型有准确率门槛），即判定该模型的 C3.5 通过。运行时间与峰值显存不设硬性门槛，用于 ResNet 与 BigFormer 的评分排序。

### 支持的算子清单（17 种）

| 模型 | 使用的算子 |
|------|------------|
| MLP | `Flatten`、`Gemm`、`Relu` |
| ResNet-18（简化） | `Conv`、`Relu`、`Add`、`GlobalAveragePool`、`Flatten`、`Gemm` |
| Transformer | `Gather`、`Add`、`LayerNormalization`、`MatMul`、`Constant`、`Split`、`Reshape`、`Transpose`、`Div`、`Softmax`、`Erf`、`Mul` |
| BigFormer | `Gather`、`Add`、`LayerNormalization`、`MatMul`、`Constant`、`Split`、`Reshape`、`Transpose`、`Div`、`Softmax`、`Erf`、`Mul` |

全部 17 种算子的并集：`Add`、`Constant`、`Conv`、`Div`、`Erf`、`Flatten`、`Gather`、`Gemm`、`GlobalAveragePool`、`LayerNormalization`、`MatMul`、`Mul`、`Relu`、`Reshape`、`Softmax`、`Split`、`Transpose`

补充说明：
- Transformer 与 BigFormer 的 GELU 被分解为 `Div`、`Erf`、`Add`、`Mul` 的组合，图中无单独的 Gelu 节点
- `Gather` 仅用于词嵌入查表（按 `input_ids` 索引词向量）
- `Constant` 为图内嵌常量（如注意力缩放因子）
- BigFormer 与 Transformer 算子集相同，但 BigFormer 为双向注意力（无因果掩码），且权重以外部数据文件存储

### 调试数据

每个公开模型均提供一份调试数据包：

```text
input/
  manifest.json      # 描述各输入张量（名称、文件、dtype、形状）
  <name>.npy         # 输入张量，第 0 维为样本数 N
golden/
  manifest.json
  <output>.npy       # 标准答案（PyTorch fp32 参考输出）
labels.npy           # 真值标签（仅分类模型：MLP 与 ResNet）
thresholds.json      # 该模型的精度与准确率阈值
```

选手可使用 `golden/` 自测精度、使用 `labels.npy` 自测准确率。评分所用的隐藏模型采用相同的目录结构。BigFormer 无 `labels.npy`（无准确率门槛），其模型文件为 `bigformer_v1.onnx` + `bigformer_v1.onnx.data` 两部分。

### 提交清单

提交材料须包含：

1. 程序源码及构建与运行说明
2. 命令行 / 启动模板：
   - C3.1：`... --onnx {onnx} --output {output}`
   - C3.5：worker 启动命令（不带任务参数），如 `python infer_worker.py`

建议在提交前完成自测：
- 用 `selfcheck_worker.py` 校验 worker 协议实现（握手、多轮任务、退出）是否正确。
- 用公开模型将输出与 `golden/` 做 `numpy.allclose(rtol=1e-3, atol=1e-3)` 比较精度，并用 `labels.npy` 核对准确率。

## 评测工作流（C3.2 和 C3.3）

### 运行命令

```bash
python3 benchmarks/c32_c33/bench_c32_c33.py \
    --models mnist_mlp cifar_resnet18 \
    --output-dir benchmarks/c32_c33/results
```

输出：

```text
benchmarks/c32_c33/results/
├── bench_mnist_mlp.json         # 该模型每个算子的分解 + tuning + 中间张量
├── bench_cifar_resnet18.json    # 同上
├── scores.json                  # 两模型合并后的最终分
└── BENCHMARK_REPORT.md          # 评审可读的总览
```

### 评分门槛

门槛仅供**自评参考**，不作为排序唯一依据。最终评级以评审结论为准。

| 总分区间（C3.2 + C3.3） | 评语 |
|--------------------------|------|
| ≥ 25 | S 级 — 多精度齐全、Winograd/im2col 切换 + 至少 3 个融合 pattern |
| 20 – 24 | A 级 — 主要维度齐全，仅个别 pattern 缺失 |
| 14 – 19 | B 级 — 基本分解 + 1–2 个融合 |
| 8 – 13 | C 级 — 仅完成单一算子的分解或单一融合 pattern |
| < 8 | 未达标 |

## 环境

- GPU 可用（NVML 监控）
- Python 环境
- 评测期间无网络访问