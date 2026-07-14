# AEC ASAP7 综合、时序和面积流程

本目录提供 Yosys 综合、OpenSTA frequency 搜索、library 锁定校验和面积统计
脚本。Yosys 对 judged RTL 只综合和映射一次；frequency 搜索复用同一份
mapped netlist，仅改变 OpenSTA 时钟周期。

## 发布内容

- `run_ppa.sh`：公开 shell 入口；
- `run_ppa.py`：综合、STA 搜索、面积统计和结果生成主程序；
- `config/default.env`：可搬运的公共默认配置；
- `synth.tcl`：Yosys 综合与 ASAP7 technology mapping；
- `constraints.sdc.tcl`：OpenSTA 约束；
- `sta.tcl`：OpenSTA 检查和时序报告；
- `lib/asap7sc7p5t_28/`：5 个锁定的 RVT/TT NLDM Liberty 文件；
- `lib/asap7_sram_0p0/`：4 个 SRAM macro 的 12 个锁定 view；
- `run_smoke.sh`、`run_sram_smoke.sh` 和 `smoke/`：结构冒烟输入。

## 工具和路径

公共配置默认从 PATH 查找 `yosys` 和 `sta`，库路径均相对于发布包自动解析：

```bash
source release/ppa-flow/config/default.env
```

如需使用其他安装位置，可在 source 前覆盖工具变量：

```bash
export YOSYS=/path/to/yosys
export OPENSTA=/path/to/sta
source release/ppa-flow/config/default.env
```

脚本要求工具报告以下 release commit：

- Yosys：`78e05dfb0`；
- OpenSTA：`66c29303`。

5 个 standard-cell Liberty 必须匹配 `ASAP7_STDCELL_LOCK.sha256`，
`lib/asap7_sram_0p0/` 中的 12 个 SRAM view 必须匹配
`release/sram/ASAP7_SRAM_LOCK.sha256`。缺少文件、版本不匹配或 SHA256
不匹配时，流程在综合前失败。

## RTL 输入

提交仓库提供 `rtl/files.f`，每行一个可综合 SystemVerilog 源文件，或一个
`+incdir+<path>`。相对路径以 filelist 所在目录为基准解析。filelist 不得加入
testbench 或 behavioral SRAM model。默认顶层为 `aec_eval_top`。

`aec_eval_top` 的有效接口由原始规范和 `release/ERRATA.md` 共同定义，其中
E-001 后加了 `mem_req_space` 输出。PPA flow 会检查顶层端口名称、方向和位宽，
缺少该端口、保留旧接口或增加额外顶层端口都会失败。

原生 SRAM macro 可以直接实例化，也可以使用 `release/sram/` 中的 wrapper。
Yosys 以 library/blackbox 方式读取随包提供的 behavioral Verilog；OpenSTA 读取
对应 Liberty；面积按 mapped JSON 中的实例数和锁定 LEF 尺寸累加。

## 固定参数

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `PPA_ABC_DELAY_PS` | 1000 ps | ABC mapping target |
| `PPA_UNCERTAINTY_RATIO` | 0.05 | setup 时钟不确定度占搜索周期比例 |
| `PPA_IO_DELAY_RATIO` | 0.10 | input/output max delay 占搜索周期比例 |
| `PPA_INPUT_DELAY_MIN_NS` | 0.050 ns | hold 检查的外部最早输入到达 |
| `PPA_INPUT_TRANSITION_NS` | 0.010 ns | 输入 transition |
| `PPA_OUTPUT_LOAD_FF` | 1.0 fF | 输出负载 |
| `PPA_MIN_PERIOD_NS` | 0.050 ns | 搜索下界 |
| `PPA_MAX_PERIOD_NS` | 1000.000 ns | 搜索上界 |
| `PPA_PERIOD_STEP_NS` | 0.001 ns | 搜索精度 |
| `PPA_RESET_STYLE` | `async` | active-low `rst_n` |

正式评测使用上述默认值。覆盖参数仅用于受控实验，`result.json` 会将
`official_defaults` 标记为 `false`。

异步 reset 的外部 delay 为 0，只对映射后异步 `RESETN` 引脚设置 false path；
若 `rst_n` 被当作普通数据使用，该路径仍参与计时。recovery/removal 需要实现
阶段的 reset-tree 约束，本综合级流程不报告该项。

## 命令

只检查工具、版本和锁定库：

```bash
source release/ppa-flow/config/default.env
python3 release/ppa-flow/run_ppa.py \
  --output build/ppa-check \
  --check-only
```

运行发布包内的结构冒烟：

```bash
bash release/ppa-flow/run_smoke.sh
bash release/ppa-flow/run_sram_smoke.sh
```

评测提交：

```bash
release/ppa-flow/run_ppa.sh --output build/ppa
```

仓库根部的 `scripts/run_ppa.sh` 是兼容入口，会转发到同一脚本。主入口要求显式
提供 `--output`，避免覆盖已有结果。

## 输出和失败条件

输出目录包含解析后的 filelist、生成的 ABC mapping Liberty、唯一 mapped
netlist、Yosys/OpenSTA 日志、综合统计和 `result.json`。结果 JSON 记录周期、
fmax、setup/hold WNS、standard-cell/SRAM/总面积、运行时工具版本、实际配置和
已使用库文件的哈希。power 字段当前为 `not_implemented`。

以下任一情况会令流程失败：

- 工具、库或 RTL 文件缺失，或工具版本、库哈希不匹配；
- hierarchy、综合、technology mapping 或 OpenSTA 执行失败；
- mapped netlist 残留未映射内部 cell 或未知面积 cell；
- `check_setup` 报告约束缺失、组合环、多时钟等问题；
- 存在 unconstrained endpoint；
- 最终 setup WNS 或 hold WNS 小于 0；
- 搜索上界仍无法通过 setup。

## 范围限制

本脚本使用综合后理想时钟 netlist，不包含 placement/routing 后的时钟树、寄生
参数和 recovery/removal 分析。workload activity、switching、internal、leakage
和 SRAM internal power 由后续组织者 power flow 处理，不在本脚本中估算。
