# C1 赛题说明：PTX 到 AEC 标量机器码编译器

## 1. 赛题目标

C1 要求参赛队伍实现一个面向 AEC 指令集的编译器。编译器接收 NVIDIA PTX ISA 9.3 的受限标量子集作为输入，生成 AEC 128-bit 定长指令流机器码。

整体流程如下：

```text
PTX 输入程序
    ↓
C1 编译器
    ↓
AEC 128-bit 指令流 .aecbin
    ↓
评测系统加载 .aecbin
    ↓
正确性与性能评测
```

参赛队伍需要完成的主要工作包括：

```text
1. 解析受限 PTX 9.3 标量子集；
2. 构建内部 IR、基本块和控制流图；
3. 将 PTX 指令 lowering 到 AEC 标量指令；
4. 执行标量编译优化；
5. 完成寄存器分配和指令调度；
6. 生成 AEC 128-bit 定长机器码；
7. 按规定格式输出 .aecbin 文件；
8. 输出编译报告。
```

---

## 2. 输入语言

C1 输入采用 NVIDIA PTX ISA 9.3 的受限标量子集。评测用例使用本文档列出的 PTX directive、类型、寄存器声明和指令形式。

### 2.1 PTX 文件结构

输入文件采用如下结构：

```ptx
.version 9.3
.target sm_90
.address_size 64

.visible .entry kernel_name(
    .param .u64 param_a,
    .param .u32 param_n
)
{
    .reg .pred %p<4>;
    .reg .u32  %r<16>;
    .reg .u64  %rd<16>;
    .reg .f32  %f<16>;

LABEL:
    ...
    ret;
}
```

评测用例采用单个 `.visible .entry` kernel。kernel 名称、参数列表、launch 配置和输入输出 buffer 由测试用例 manifest 给出。

### 2.2 Directive

参赛编译器需要处理以下 PTX directive：

| Directive | 说明 |
|---|---|
| `.version 9.3` | PTX 版本声明。 |
| `.target sm_90` | 目标架构声明。 |
| `.address_size 64` | PTX 层面的指针和地址使用 64-bit 形式。 |
| `.visible .entry` | kernel 入口声明。 |
| `.param` | kernel 参数声明。 |
| `.reg` | 虚拟寄存器声明。 |
| label | 基本块标签和分支目标。 |

### 2.3 数据类型

评测用例使用以下 PTX 数据类型：

| PTX 类型 | 用途 |
|---|---|
| `.pred` | predicate register。 |
| `.b32` | 32-bit bit pattern。 |
| `.b64` | 64-bit bit pattern 或指针值。 |
| `.u32` | 32-bit unsigned integer。 |
| `.s32` | 32-bit signed integer。 |
| `.u64` | 64-bit pointer/address。 |
| `.f32` | FP32 标量运算。 |

### 2.4 寄存器声明

评测用例使用如下寄存器声明形式：

```ptx
.reg .pred %p<N>;
.reg .u32  %r<N>;
.reg .s32  %s<N>;
.reg .u64  %rd<N>;
.reg .b32  %b<N>;
.reg .b64  %bd<N>;
.reg .f32  %f<N>;
```

---

## 3. PTX 指令子集

### 3.1 参数加载

```ptx
ld.param.u32 dst, [param_name];
ld.param.u64 dst, [param_name];
ld.param.b32 dst, [param_name];
ld.param.b64 dst, [param_name];
```

### 3.2 Special Register 读取

```ptx
mov.u32 dst, %tid.x;
mov.u32 dst, %tid.y;
mov.u32 dst, %tid.z;

mov.u32 dst, %ntid.x;
mov.u32 dst, %ntid.y;
mov.u32 dst, %ntid.z;

mov.u32 dst, %ctaid.x;
mov.u32 dst, %ctaid.y;
mov.u32 dst, %ctaid.z;

mov.u32 dst, %nctaid.x;
mov.u32 dst, %nctaid.y;
mov.u32 dst, %nctaid.z;

mov.u32 dst, %laneid;
```

### 3.3 Move

```ptx
mov.u32 dst, src;
mov.u32 dst, imm;
mov.u64 dst, imm;
mov.b32 dst, src;
mov.b64 dst, src;
```

### 3.4 32-bit 整数算术

```ptx
add.u32 dst, a, b;
sub.u32 dst, a, b;
mul.lo.u32 dst, a, b;
mad.lo.u32 dst, a, b, c;
```

整数运算采用 32-bit modulo arithmetic。`mul.lo.u32` 保留乘法结果低 32 bits；`mad.lo.u32` 计算 `low32(a * b + c)`。

### 3.5 地址计算相关指令

```ptx
mul.wide.u32 dst64, a32, b32;
add.u64 dst64, a64, b64;
```

这两类指令用于 PTX 中常见的地址计算，lowering 规则见第 8 节。

### 3.6 位运算与移位

```ptx
and.b32 dst, a, b;
or.b32  dst, a, b;
xor.b32 dst, a, b;
shl.b32 dst, a, b;
shr.u32 dst, a, b;
```

### 3.7 FP32 标量运算

```ptx
add.f32 dst, a, b;
add.rn.f32 dst, a, b;

sub.f32 dst, a, b;
sub.rn.f32 dst, a, b;

mul.f32 dst, a, b;
mul.rn.f32 dst, a, b;

mad.f32 dst, a, b, c;
mad.rn.f32 dst, a, b, c;

fma.rn.f32 dst, a, b, c;
```

### 3.8 比较与分支

```ptx
setp.eq.u32 p, a, b;
setp.ne.u32 p, a, b;
setp.lt.u32 p, a, b;
setp.le.u32 p, a, b;
setp.gt.u32 p, a, b;
setp.ge.u32 p, a, b;

bra LABEL;
@%p bra LABEL;
@!%p bra LABEL;
```

### 3.9 Global Memory Load/Store

```ptx
ld.global.f32 dst, [addr];
st.global.f32 [addr], src;

ld.global.u32 dst, [addr];
st.global.u32 [addr], src;

ld.global.b32 dst, [addr];
st.global.b32 [addr], src;
```

### 3.10 Kernel Return

```ptx
ret;
```

kernel 顶层 `ret` lowering 为 AEC `HALT`。

---

## 4. 目标 AEC Opcode 集合

C1 输出机器码使用以下 AEC opcode：

| Opcode | Mnemonic | 说明 |
|---:|---|---|
| `0x0001` | `ADD` | 加法。 |
| `0x0002` | `SUB` | 减法。 |
| `0x0003` | `MUL` | 乘法。 |
| `0x0004` | `MAD` | 乘加。 |
| `0x0005` | `FMA` | 融合乘加。 |
| `0x0010` | `AND` | 按位与。 |
| `0x0011` | `OR` | 按位或。 |
| `0x0012` | `XOR` | 按位异或。 |
| `0x0014` | `SHL` | 左移。 |
| `0x0015` | `SHR` | 逻辑右移。 |
| `0x0021` | `CMPP` | 比较并写入 predicate。 |
| `0x0030` | `LD` | load。 |
| `0x0031` | `ST` | store。 |
| `0x0040` | `BR` | 无条件分支。 |
| `0x0041` | `BRX` | predicate 分支。 |
| `0x0045` | `HALT` | 终止当前 thread/lane。 |
| `0x0054` | `CPY` | 寄存器拷贝或 special register 读取。 |
| `0x0055` | `LOADI` | 32-bit 立即数加载。 |
| `0x0056` | `LOADI64` | 64-bit 立即数加载。 |

---

## 5. AEC 指令格式

### 5.1 128-bit 指令字段

每条 AEC 指令为 128-bit，字段布局如下：

```text
bits [127:112]  Opcode      16 bits
bits [111:96]   Pred/Ctrl   16 bits
bits [95:80]    Dest        16 bits
bits [79:64]    Src1        16 bits
bits [63:32]    Src2/Imm32  32 bits
bits [31:0]     ImmExt      32 bits
```

### 5.2 Pred/Ctrl 字段

| Bits | 名称 | 说明 |
|---:|---|---|
| `[2:0]` | `pred` | predicate register index，`P0..P7` |
| `[6:3]` | `type` | type code |
| `[10:8]` | `subop` | compare subop |
| `[13:11]` | `space` | memory space |
| `[14]` | `pred_neg` | predicate negate |
| `[15]` | `pred_en` | predicate enable |

谓词执行规则：

```text
execute_lane = active_lane && (!pred_en || (P[pred] XOR pred_neg))
```

### 5.3 Type Code

| Code | Type |
|---:|---|
| `0x0` | `.b32` |
| `0x1` | `.b64` |
| `0x2` | `.u32` |
| `0x3` | `.s32` |
| `0x8` | `.f32` |
| `0xf` | `.none` |

### 5.4 Memory Space

| Code | Space |
|---:|---|
| `0` | `.gmem` |
| `1` | `.smem` |
| `2` | `.cmem` |
| `3` | `.lmem` |
| `4` | `.pmem` |
| `5..7` | reserved |

C1 评测主要使用 `.gmem` 和 `.pmem`。

### 5.5 Special Register Selector

| Selector | Register |
|---:|---|
| `0x0100` | `%tid.x` |
| `0x0101` | `%ntid.x` |
| `0x0102` | `%ctaid.x` |
| `0x0103` | `%nctaid.x` |
| `0x0104` | `%laneid` |
| `0x0110` | `%tid.y` |
| `0x0111` | `%ntid.y` |
| `0x0112` | `%ctaid.y` |
| `0x0113` | `%nctaid.y` |
| `0x0120` | `%tid.z` |
| `0x0121` | `%ntid.z` |
| `0x0122` | `%ctaid.z` |
| `0x0123` | `%nctaid.z` |

---

## 6. AEC 指令语义

### 6.1 ADD / SUB

```asm
ADD.type Rd, Rs1, Rs2
SUB.type Rd, Rs1, Rs2
```

语义：

```text
ADD.u32: Rd = low32(Rs1 + Rs2)
SUB.u32: Rd = low32(Rs1 - Rs2)
ADD.f32: FP32 add
SUB.f32: FP32 sub
```

### 6.2 MUL / MAD / FMA

```asm
MUL.type Rd, Rs1, Rs2
MAD.type Rd, Rs1, Rs2, Rs3
FMA.f32 Rd, Rs1, Rs2, Rs3
```

语义：

```text
MUL.u32: Rd = low32(Rs1 * Rs2)
MAD.u32: Rd = low32(Rs1 * Rs2 + Rs3)
MUL.f32: FP32 multiply
MAD.f32: non-fused FP32 multiply then FP32 add
FMA.f32: fused multiply-add, single rounding
```

### 6.3 AND / OR / XOR / SHL / SHR

```text
AND.b32: bitwise and
OR.b32 : bitwise or
XOR.b32: bitwise xor
SHL.b32: Rs1 << (Rs2 & 31)
SHR.u32: logical right shift Rs1 >> (Rs2 & 31)
```

### 6.4 CMPP

```asm
CMPP.u32.op Pd, Rs1, Rs2
```

支持：

```text
eq, ne, lt, le, gt, ge
```

结果写入 predicate `Pd`。

Compare subop 编码如下：

| Code | Op |
|---:|---|
| `0` | `.eq` |
| `1` | `.ne` |
| `2` | `.lt` |
| `3` | `.le` |
| `4` | `.gt` |
| `5` | `.ge` |

### 6.5 LD / ST

```asm
LD.space.type Rd, [Raddr]
ST.space.type [Raddr], Rs
```

语义：

```text
LD: Rd = memory[space][Raddr]
ST: memory[space][Raddr] = Rs
```

访问字节数由 type 决定。`.gmem` 是 global memory，`.pmem` 是 kernel 参数内存。

### 6.6 BR / BRX / HALT

```text
BR target:
  PC = target

BRX Pn, target:
  if Pn: PC = target
  else : PC = PC + 1

HALT:
  terminate current thread/lane
```

PC 单位是一条 AEC 128-bit 指令，不是 byte。

### 6.7 CPY / LOADI / LOADI64

```text
CPY Rd, Rs:
  Rd = Rs

CPY Rd, %special:
  Rd = special register value

LOADI Rd, imm32:
  Rd = imm32

LOADI64 Rd, imm64:
  Rd   = imm64[31:0]
  Rd+1 = imm64[63:32]
```

---

## 7. 参数 ABI

### 7.1 参数内存

PTX `.param` 映射到 AEC `.pmem`。`.pmem` 是 kernel 参数内存，按 byte offset 访问。

### 7.2 参数布局规则

参数布局采用如下规则：

```text
1. 参数按照 .entry 声明顺序放入 .pmem；
2. 每个参数按照自然对齐；
3. 必要时插入 padding；
4. pointer 参数统一表示为 .u64；
5. 参数块总大小向 8 bytes 对齐；
6. .param 名称在编译期解析为 .pmem byte offset。
```

类型大小与对齐：

| Type | Size | Align |
|---|---:|---:|
| `.u32` / `.s32` / `.b32` / `.f32` | 4 | 4 |
| `.u64` / `.b64` / pointer | 8 | 8 |

### 7.3 参数布局示例

```ptx
.visible .entry vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
```

布局：

| 参数 | Offset | Size |
|---|---:|---:|
| `param_a` | 0 | 8 |
| `param_b` | 8 | 8 |
| `param_c` | 16 | 8 |
| `param_n` | 24 | 4 |
| padding | 28 | 4 |

总大小：

```text
32 bytes
```

### 7.4 `ld.param` Lowering

```ptx
ld.param.u32 %r1, [param_n];
```

lower 为：

```asm
LOADI Rtmp, 24
LD.pmem.u32 R1, [Rtmp]
```

```ptx
ld.param.u64 %rd1, [param_a];
```

lower 为：

```asm
LOADI Rtmp, 0
LD.pmem.u32 Rrd1_low, [Rtmp]
LOADI Rtmp, 4
LD.pmem.u32 Rrd1_high, [Rtmp]
```

---

## 8. 地址 ABI

### 8.1 地址单位

所有地址均为 byte address：

```text
.gmem: byte address
.pmem: byte offset
```

### 8.2 32-bit Abstract Address 规则

C1 使用如下地址规则：

```text
1. PTX 层面使用 .address_size 64 和 .u64 pointer；
2. 评测用例保证所有 global memory address 的高 32 bits 为 0；
3. AEC LD.gmem / ST.gmem 使用地址低 32 bits；
4. 高 32 bits 不参与基础评测语义。
```

### 8.3 `.u64` Register Pair

PTX `.u64` / `.b64` 虚拟寄存器映射为两个 AEC GPR：

```text
low 32 bits  -> Rk
high 32 bits -> Rk+1
```

约束：

```text
Rk 建议偶数对齐。
R255 不能作为 64-bit pair low register。
```

### 8.4 `mul.wide.u32` Lowering

```ptx
mul.wide.u32 %rd4, %r5, 4;
```

lower 为：

```asm
MUL.u32 Rrd4_low, R5, 4
LOADI Rrd4_high, 0
```

### 8.5 `add.u64` Lowering

```ptx
add.u64 %rd5, %rd1, %rd4;
```

lower 为：

```asm
ADD.u32 Rrd5_low, Rrd1_low, Rrd4_low
LOADI Rrd5_high, 0
```

### 8.6 Global Memory 访问

```ptx
ld.global.f32 %f1, [%rd5];
```

如果 `%rd5 = {R10 low, R11 high}`，则 lower 为：

```asm
LD.gmem.f32 Rf1, [R10]
```

---

## 9. PTX-to-AEC Lowering 表

| PTX | AEC |
|---|---|
| `mov.u32 dst, %tid.x` | `CPY.u32 dst, %tid.x` |
| `mov.u32 dst, %ntid.x` | `CPY.u32 dst, %ntid.x` |
| `mov.u32 dst, %ctaid.x` | `CPY.u32 dst, %ctaid.x` |
| `mov.u32 dst, %nctaid.x` | `CPY.u32 dst, %nctaid.x` |
| `mov.u32 dst, %laneid` | `CPY.u32 dst, %laneid` |
| `mov.u32 dst, imm` | `LOADI dst, imm32` |
| `mov.u64 dst, imm` | `LOADI64 dst, imm64` 或两条 `LOADI` |
| `mov.b32/u32 dst, src` | `CPY.b32/u32 dst, src` |
| `add.u32` | `ADD.u32` |
| `sub.u32` | `SUB.u32` |
| `mul.lo.u32` | `MUL.u32` |
| `mad.lo.u32` | `MAD.u32` |
| `mul.wide.u32` | `MUL.u32 low + LOADI high=0` |
| `add.u64` | `ADD.u32 low + LOADI high=0` |
| `and.b32` | `AND.b32` |
| `or.b32` | `OR.b32` |
| `xor.b32` | `XOR.b32` |
| `shl.b32` | `SHL.b32` |
| `shr.u32` | `SHR.u32` |
| `add.f32/add.rn.f32` | `ADD.f32` |
| `sub.f32/sub.rn.f32` | `SUB.f32` |
| `mul.f32/mul.rn.f32` | `MUL.f32` |
| `mad.f32/mad.rn.f32` | `MAD.f32` |
| `fma.rn.f32` | `FMA.f32` |
| `setp.eq/ne/lt/le/gt/ge.u32` | `CMPP.u32.eq/ne/lt/le/gt/ge` |
| `bra label` | `BR label` |
| `@%p bra label` | `BRX P, label` |
| `@!%p bra label` | `BRX !P, label` |
| `ld.param.u32/b32` | `LOADI offset + LD.pmem.u32/b32` |
| `ld.param.u64/b64` | two `LD.pmem.u32` |
| `ld.global.f32/u32/b32` | `LD.gmem.f32/u32/b32` |
| `st.global.f32/u32/b32` | `ST.gmem.f32/u32/b32` |
| `ret` | `HALT` |

---

## 10. .aecbin 格式

### 10.1 格式定义

`.aecbin` 定义为：

```text
raw AEC 128-bit instruction stream
```

即：

```text
无 header
无 data section
无 relocation section
无 symbol table
entry_pc 默认为 0
所有 label 在编译阶段解析完成
```

### 10.2 指令存储顺序

每条 AEC 指令为 128-bit，存储为 4 个 little-endian 32-bit word：

```text
w0 = bits [31:0]
w1 = bits [63:32]
w2 = bits [95:64]
w3 = bits [127:96]
```

文件写入顺序：

```text
w0, w1, w2, w3
```

文本 hex dump 使用 MSB-first：

```text
w3_w2_w1_w0
```

### 10.3 合法性要求

`.aecbin` 需满足：

```text
1. 文件大小是 16 bytes 的倍数；
2. 至少包含一条指令；
3. entry_pc 默认为 0；
4. 所有 opcode 属于 C1 允许集合；
5. 所有 type code、memory space、register index、predicate index 合法；
6. 所有 branch target 在指令范围内；
7. LOADI64 不写入越界 register pair。
```

---

## 11. Test Manifest

每个测试用例由 PTX 文件和 manifest 文件共同描述。

manifest 示例：

```json
{
  "kernel": "vector_add",
  "gridDim": [4096, 1, 1],
  "blockDim": [256, 1, 1],
  "dynamic_smem_bytes": 0,
  "params": [
    {"name": "param_a", "type": "u64", "kind": "gmem_ptr", "buffer": "a"},
    {"name": "param_b", "type": "u64", "kind": "gmem_ptr", "buffer": "b"},
    {"name": "param_c", "type": "u64", "kind": "gmem_ptr", "buffer": "c"},
    {"name": "param_n", "type": "u32", "kind": "value", "value": 1048576}
  ],
  "buffers": {
    "a": {"dtype": "f32", "numel": 1048576, "init": "rand_uniform"},
    "b": {"dtype": "f32", "numel": 1048576, "init": "rand_uniform"},
    "c": {"dtype": "f32", "numel": 1048576, "init": "zero", "output": true}
  },
  "check": {
    "type": "vector_add",
    "output": "c",
    "atol": 1e-6,
    "rtol": 1e-6
  }
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `kernel` | kernel 名称。 |
| `gridDim` | grid 维度。 |
| `blockDim` | block 维度。 |
| `dynamic_smem_bytes` | dynamic shared memory 字节数。 |
| `params` | kernel 参数。 |
| `buffers` | global memory buffer 初始化。 |
| `check` | 输出检查规则。 |

---

## 12. 编译报告

编译命令：

```bash
compiler/aec-cc input.ptx -O2 -o output.aecbin --report compile_report.json
```

编译报告示例：

```json
{
  "status": "ok",
  "input": "input.ptx",
  "output": "output.aecbin",
  "opt_level": "O2",
  "num_ptx_instructions": 42,
  "num_aec_instructions": 58,
  "num_basic_blocks": 5,
  "num_virtual_registers": 24,
  "num_physical_registers": 18,
  "num_predicates": 2,
  "spills": {
    "loads": 0,
    "stores": 0
  },
  "passes": {
    "dce": true,
    "cse": true,
    "licm": true,
    "scheduler": "list"
  },
  "warnings": []
}
```

---

## 13. 测试题设计

测试题作为外部文件夹提供。每个测试题包含：

```text
kernel.ptx
manifest.json
```

### T1：基础指令 Lowering

测试目标：

```text
PTX directive 和 kernel 解析；
参数加载；
special register 读取；
global load/store；
整数和 FP32 基础运算；
比较和条件分支；
ret 到 HALT 的 lowering。
```

典型题目：

```text
vector_add
copy
saxpy
```

### T2：标量优化

测试目标：

```text
常量传播；
死代码删除；
公共子表达式消除；
循环不变量外提；
基本块合并。
```

典型题目：

```text
loop-invariant polynomial
repeated expression
dead computation
```

### T3：内存访问优化

测试目标：

```text
重复 global load；
load hoisting；
简单内存复用；
地址计算优化。
```

典型题目：

```text
repeated global memory reuse
stencil-like scalar load pattern
```

### T4：寄存器分配与指令调度

测试目标：

```text
虚拟寄存器到 AEC GPR 的分配；
predicate 分配；
live range 管理；
load/compute interleaving；
基本依赖调度。
```

典型题目：

```text
long arithmetic dependency chain
mixed load and compute sequence
moderate register pressure kernel
```

### T5：FP32 Scalar GEMM

测试目标：

```text
二维索引计算；
FP32 global load；
FP32 multiply-add；
K 维循环；
地址计算优化；
标量循环调度。
```

题目形式：

```text
每个 thread 计算一个 C[i, j]；
A、B、C 均为 FP32；
使用 scalar for-loop over K；
输出 C = A x B。
```



## 14. 提交内容

参赛队伍提交内容包括：

```text
1. C1 编译器源码；
2. 编译器可执行入口 compiler/aec-cc；
```
