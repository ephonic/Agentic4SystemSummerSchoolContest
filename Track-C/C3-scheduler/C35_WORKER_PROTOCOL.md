# C3.5 持久化 Worker 评测协议

C3.5 的推理程序以**常驻进程（worker）**方式运行。评测机启动一次 worker，通过标准输入/输出（stdin/stdout）多次下发任务并回收结果。计时只覆盖"加载模型 + 推理"，不包含进程启动与框架初始化。

## 通信通道

- **stdin**：评测机 → worker，逐行下发 JSON 任务。
- **stdout**：worker → 评测机，**仅**输出协议信号：一行 `READY` 与每个任务的一行结果 JSON。
- **stderr**：worker 的所有日志、告警、调试输出，必须走这里，不得写入 stdout。

所有消息以换行符 `\n` 分隔，每条消息占一行。

## 启动

评测机以选手报名时提交的启动命令启动 worker，**不带任务参数**：

```bash
<选手 worker 启动命令>
```

例如 `python infer_worker.py`。

## 时序

```
评测机                                   worker
  │  启动进程 ────────────────────────────▶│
  │                                        │ 导入框架、创建 CUDA context
  │  ◀──────────────────────────  READY    │  （不计时）
  │                                        │
  │  任务 JSON ────────────────────────────▶│ ┐ 计时开始、显存采样开始
  │                                        │ │ 加载 onnx、推理、写输出
  │  ◀──────────────────────  结果 JSON     │ ┘ 计时结束、采样结束
  │            …（可重复多次）…             │
  │  {"cmd":"exit"} ──────────────────────▶│ 退出（退出码 0）
```

## 消息定义

### READY（worker → 评测机）

worker 完成一次性初始化后，向 stdout 输出恰好一行：

```
READY
```


### 任务（评测机 → worker，经 stdin）

一行 JSON：

```json
{"onnx": "<模型路径>", "input": "<输入目录>", "output": "<输出目录>", "batch_size": 256}
```

| 字段 | 说明 |
|------|------|
| `onnx` | 本次任务的 ONNX 模型文件路径 |
| `input` | 输入目录，含 `manifest.json` 与 `.npy` |
| `output` | 输出目录，worker 须将结果写入此处 |
| `batch_size` | 批量大小 |

### 结果（worker → 评测机，经 stdout）

worker 在**输出文件全部落盘后**，向 stdout 输出恰好一行 JSON：

成功：

```json
{"status": "ok", "samples": 10000}
```

失败：

```json
{"status": "error", "error": "<原因描述>"}
```

评测机以读到结果行作为计时终点，因此结果行必须晚于输出写盘。

### exit（评测机 → worker，经 stdin）

```json
{"cmd": "exit"}
```

worker 收到后须干净退出（退出码 0）。

## 输入 / 输出格式

与目录约定一致：

- 输入目录含 `manifest.json`（`{"tensors":[{"name","file","dtype","shape"},...]}`）与对应 `.npy`，第 0 维为样本数 N。
- 输出目录须写入相同结构的 `manifest.json` 与 `<output_name>.npy`；输出张量名用模型输出名（本赛道均为 `logits`），dtype 为 `float32`，第 0 维顺序与输入一致。

## 计时与显存

- **计时窗口**：评测机写出任务行后开始，读到结果行后结束。窗口 = 加载 onnx + 推理 + 写输出。
- **重复**：每个模型连续下发 `2 次 warmup + 5 次计时` 任务；丢弃 warmup，时间取 5 次中位数，显存取 5 次最大值。
- **显存口径**：NVML 采样，取进程绝对峰值，采样间隔 20 ms。
- **worker 复用**：每个模型使用一个全新 worker，跑完即发 `exit` 并重启进程再跑下一个模型。

## 超时与失败

- worker 启动后 60 s 内未输出 `READY`：判失败。
- 单个任务超过其超时上限（普通模型 120 s，大模型 1800 s）未返回结果行：评测机终止 worker，该模型判失败。
- worker 中途崩溃或 stdout 关闭：当前模型判失败。

## 协议自测

资料包提供 `selfcheck_worker.py`：模拟评测机跑完整握手与多轮任务流程，校验 worker 的协议实现与输出正确性（不测显存）。用法：

```bash
python selfcheck_worker.py --worker "<你的 worker 启动命令>" --models mlp_v1 resnet_v1
```
