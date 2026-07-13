# Track-C 性能参考目标平台参数

> 本文件给出 Track-C 的性能参考目标参数。所有参赛队伍可依据这些参数对生成代码或调度策略进行优化。

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

## PTX 到真实硬件的映射

参赛者可自行将 PTX 映射到真实 GPGPU 上做辅助性能评估，但应考虑我们给出的性能参数，可使用 `nvcc` 编译 PTX，并通过 `ncu`/`nsys` 等工具观察 `memory_transactions`、`stall_cycles`、`sm__throughput` 等指标，将瓶颈分析结果反馈到优化决策中。

## PTX 版本

PTX 采用 9.3 版本，具体可以参考：https://docs.nvidia.com/cuda/pdf/ptx_isa_9.3.pdf
