# C1 Testcases

本目录给出 C1 赛题的公开测试题示例。每个测试题由两个文件组成：

```text
kernel.ptx      # 输入 PTX 程序
manifest.json   # 测试运行配置
```

Manifest 用于描述 kernel 名称、grid/block 维度、参数、输入输出 buffer 和正确性检查规则。PTX 文件本身只包含 kernel 代码，不包含这些运行信息。

目录结构：

```text
T1_basic_lowering/
T2_scalar_optimization/
T3_memory_reuse/
T4_register_scheduling/
T5_scalar_gemm/
```
