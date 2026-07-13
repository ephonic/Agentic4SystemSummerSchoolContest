# Track-C 性能参考目标平台参数

> 本文件给出 Track-C 各赛道（C1、C2、C3 等）共用的性能参考目标参数。所有参赛队伍可依据这些参数对生成代码或调度策略进行优化。

## 参考目标平台参数

### 存储层次

| 参数 | 平台 A | 平台 B |
| :--- | ---: | ---: |
| 寄存器文件 | 256 KB / SM | 256 KB / SM |
| 统一 L1 / Shared Memory 池 | 192 KB / SM | 256 KB / SM |
| 最大 Shared Memory | 164 KB / SM | 228 KB / SM |
| 每线程块最大 Shared Memory | 163 KB | 227 KB |
| Shared Memory Bank 组织 | 32 banks，4 B 宽 | 32 banks，4 B 宽 |
| L2 Cache | 40 MB | 50 MB |
| 设备显存 | 80 GB HBM2e | 80 GB HBM3 |
| 峰值 HBM 带宽 | 2,039 GB/s | 3.35 TB/s |
| 主机互联 | PCIe Gen4，64 GB/s | PCIe Gen5，128 GB/s |
| GPU 互联 | 600 GB/s | 900 GB/s |

### 各级访问延迟参考

| 存储层级 | 参考延迟 |
| :--- | ---: |
| 寄存器 | 1 个指令周期附近 |
| Shared Memory | 约 20 cycles |
| L1 Cache | 约 40 cycles |
| L2 Cache | 约 200 cycles |
| HBM | 约 600 cycles |
| 主机内存（PCIe） | 约 5 µs |

### 两代平台典型提升

- HBM 峰值带宽：3.35 TB/s vs 2,039 GB/s（提升约 64%）
- L2 容量：50 MB vs 40 MB（提升约 25%）
- 统一 L1/Shared 容量：256 KB/SM vs 192 KB/SM（提升约 33%）
- 最大 Shared Memory：228 KB/SM vs 164 KB/SM（提升约 39%）
- GPU 互联带宽：900 GB/s vs 600 GB/s（提升约 50%）
- 主机互联带宽：PCIe Gen5 vs PCIe Gen4（翻倍）

## 性能建模参考

### Roofline 带宽上界

- 平台 A：约 2,039 GB/s
- 平台 B：约 3.35 TB/s

实际可达带宽受访问合并度、请求大小、Cache 命中率、内存分区利用率、指令混合比例及并发度影响。

### 常见优化方向

- 提升全局内存访问合并度，减少非合并事务。
- 利用 Shared Memory 缓存可复用数据，避免 Bank 冲突。
- 提高线程占用率，用独立任务或流水线隐藏 HBM 延迟。
- 控制寄存器压力，减少 Spill，降低 Live Interval。
- 合理选择 GEMM Tile 大小与精度格式。

## PTX 到真实硬件的映射

参赛者可自行将 PTX 映射到真实 GPGPU 上做辅助性能评估，例如使用 `nvcc` 编译 PTX，并通过 `ncu`/`nsys` 等工具观察 `memory_transactions`、`stall_cycles`、`sm__throughput` 等指标，将瓶颈分析结果反馈到 AEC ISA 的优化决策中。

> 该映射仅作为可选调试/验证手段，最终评分以 AEC Cycle Model 为准。
