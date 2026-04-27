# TensorFlow MUSA Extension 调试指南

本文档介绍 TensorFlow MUSA Extension 的调试手段：遥测（Telemetry）、环境变量、图与日志诊断等。插件使用常规 Release 构建（`./build.sh` / `pip` 安装的 wheel）即可；仓库不再提供基于内核埋点的 Kernel 计时宏。

---

## 1. 遥测系统（Telemetry）

遥测系统提供全链路追踪能力，用于诊断脏数据（NaN）、内存问题和同步异常。

### 1.1 启用遥测

通过环境变量启用和配置遥测系统：

| 变量名 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `MUSA_TELEMETRY_ENABLED` | 启用遥测（`1` 或 `true`） | `false` | `export MUSA_TELEMETRY_ENABLED=1` |
| `MUSA_TELEMETRY_LOG_PATH` | 日志输出文件路径 | `stderr` | `export MUSA_TELEMETRY_LOG_PATH=/tmp/telemetry.json` |
| `MUSA_TELEMETRY_BUFFER_SIZE` | 事件缓冲区大小 | `10000` | `export MUSA_TELEMETRY_BUFFER_SIZE=50000` |
| `MUSA_TELEMETRY_FLUSH_MS` | 日志刷新间隔（毫秒） | `100` | `export MUSA_TELEMETRY_FLUSH_MS=50` |
| `MUSA_TELEMETRY_STACK_TRACE` | 包含堆栈追踪（`1` 或 `true`） | `false` | `export MUSA_TELEMETRY_STACK_TRACE=1` |

### 1.2 遥测事件类型

遥测系统自动记录以下事件：

| 事件类型 | 说明 | 记录内容 |
|----------|------|----------|
| `tensor_allocate` | Tensor 分配 | 地址、大小、设备 ID、Stream ID、Tensor ID |
| `tensor_free` | Tensor 释放 | 地址、大小、设备 ID、Tensor ID |
| `kernel_launch` | Kernel 启动 | 算子名、输入 Tensor ID、输出 Tensor ID、Stream ID |
| `memcpy_h2d` | Host→Device 拷贝 | 源地址、目标地址、大小、Stream ID |
| `memcpy_d2h` | Device→Host 拷贝 | 源地址、目标地址、大小、Stream ID |
| `memcpy_d2d` | Device→Device 拷贝 | 源地址、目标地址、大小、Stream ID |
| `event_record` | Event 记录 | Event 句柄、Stream ID |
| `event_wait` | Event 等待 | Event 句柄、等待 Stream ID、源 Stream ID |
| `dirty_data_detected` | 脏数据检测 | 地址、大小、描述信息 |

### 1.3 遥测日志格式

遥测日志采用 JSON Lines 格式（每行一个 JSON 对象）：

```json
{"timestamp_ns":1234567890123456,"event_type":"tensor_allocate","correlation_id":42,"device_id":0,"stream_id":123456,"thread_id":789,"memory_addr":"0x7f1234000000","memory_size":1048576,"tensor_id":100,"op_name":"Allocate"}
{"timestamp_ns":1234567890123500,"event_type":"kernel_launch","correlation_id":43,"device_id":0,"stream_id":123456,"thread_id":789,"op_name":"MatMul","input_tensor_ids":[100,101],"output_tensor_ids":[102]}
{"timestamp_ns":1234567890123600,"event_type":"memcpy_d2h","correlation_id":44,"device_id":0,"stream_id":234567,"thread_id":789,"memory_addr":"0x7f1234001000","memory_size":512,"metadata":{"src_addr":"0x7f1234000000"}}
```

### 1.4 脏数据反向追溯

遥测系统提供三种反向追溯 API：

**按地址追溯**：查询指定内存地址最近的 N 次操作

```cpp
// 查询地址 addr 最近 10 次操作
auto records = MusaTelemetry::Instance().BacktraceByAddress(addr, 10);
for (const auto& r : records) {
  LOG(INFO) << "Op: " << r.op_name
            << " Time: " << r.timestamp_ns
            << " Stream: " << r.stream_id;
}
```

**按时间范围追溯**：查询指定时间窗口内的所有操作

```cpp
// 查询时间范围内的操作
auto records = MusaTelemetry::Instance().BacktraceByTime(start_ns, end_ns);
```

**按 Tensor ID 追溯**：查询指定 Tensor 的操作历史

```cpp
// 查询 Tensor ID 100 最近 20 次操作
auto records = MusaTelemetry::Instance().BacktraceByTensorId(100, 20);
```

### 1.5 遥测使用示例

**完整遥测诊断流程**：

```bash
# 1. 启用遥测
export MUSA_TELEMETRY_ENABLED=1
export MUSA_TELEMETRY_LOG_PATH=/tmp/musa_telemetry.json
export MUSA_TELEMETRY_BUFFER_SIZE=50000

# 2. 运行测试
cd test
python test_runner.py 2>&1 | tee /tmp/test_output.log

# 3. 分析遥测日志（检测脏数据）
grep "dirty_data_detected" /tmp/musa_telemetry.json

# 4. 分析特定地址的操作历史
python -c "
import json
addr = '0x7f1234000000'  # 问题地址
with open('/tmp/musa_telemetry.json') as f:
    for line in f:
        event = json.loads(line)
        if event.get('memory_addr') == addr:
            print(json.dumps(event, indent=2))
"
```

### 1.6 遥测预期输出示例

启用遥测后，stderr 或日志文件中会输出 JSON Lines 格式的遥测事件：

```
[MUSA_TELEMETRY] {"timestamp_ns":279876543210987,"event_type":"tensor_allocate","correlation_id":0,"device_id":0,"stream_id":140728345678912,"thread_id":12345678,"memory_addr":"0x7f1234560000","memory_size":2048,"tensor_id":1,"op_name":"Allocate"}
[MUSA_TELEMETRY] {"timestamp_ns":279876543211000,"event_type":"kernel_launch","correlation_id":1,"device_id":0,"stream_id":140728345678912,"thread_id":12345678,"op_name":"MatMul","input_tensor_ids":[1,2],"output_tensor_ids":[3]}
[MUSA_TELEMETRY] {"timestamp_ns":279876543211100,"event_type":"event_record","correlation_id":2,"device_id":0,"stream_id":140728345678912,"thread_id":12345678,"event_handle":"0x7f1234001000","op_name":"EventRecord"}
[MUSA_TELEMETRY] {"timestamp_ns":279876543211200,"event_type":"event_wait","correlation_id":3,"device_id":0,"stream_id":140728345679000,"thread_id":12345678,"event_handle":"0x7f1234001000","source_stream_id":140728345678912,"op_name":"EventWait"}
[MUSA_TELEMETRY] {"timestamp_ns":279876543211300,"event_type":"memcpy_d2h","correlation_id":4,"device_id":0,"stream_id":140728345679000,"thread_id":12345678,"memory_addr":"0x7f6000000000","memory_size":512,"metadata":{"src_addr":"0x7f1234560000"},"op_name":"MemcpyD2H"}
```

**脏数据检测输出**（检测到 NaN 时）：

```
[MUSA_TELEMETRY] {"timestamp_ns":279876543212000,"event_type":"dirty_data_detected","correlation_id":100,"device_id":0,"stream_id":0,"thread_id":12345678,"memory_addr":"0x7f1234560000","memory_size":2048,"op_name":"DirtyDataDetected","metadata":{"description":"NaN detected in MatMul output tensor"}}
[MUSA Telemetry] DIRTY DATA DETECTED at address 0x7f1234560000, size=2048, device_id=0, description=NaN detected in MatMul output tensor
```

**进程退出统计**：

```
[MUSA Telemetry] Shutdown. Events logged: 15234, Events dropped: 0
```

---

## 2. 环境变量汇总

### 2.1 功能控制

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `MUSA_ENABLE_TF32` | 启用 TF32 加速 MatMul/Conv | `export MUSA_ENABLE_TF32=1` |
| `MUSA_DUMP_GRAPHDEF` | 启用图优化调试，dump GraphDef | `export MUSA_DUMP_GRAPHDEF=1` |
| `MUSA_DUMP_GRAPHDEF_DIR` | 指定 GraphDef dump 目录 | `export MUSA_DUMP_GRAPHDEF_DIR=/tmp/graphs` |
| `MUSA_AUTO_MIXED_PRECISION` | 启用自动混合精度（AMP） | `export MUSA_AUTO_MIXED_PRECISION=1` |
| `MUSA_AMP_MODE` | AMP 精度模式（`FP16` 或 `BF16`） | `export MUSA_AMP_MODE=FP16` |
| `MUSA_DISABLE_GRAPPLER` | 禁用 Grappler 图优化 | `export MUSA_DISABLE_GRAPPLER=1` |

### 2.2 日志与调试

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `TF_CPP_MIN_LOG_LEVEL` | 全局日志级别（0=INFO, 1=WARNING, 2=ERROR） | `export TF_CPP_MIN_LOG_LEVEL=1` |
| `TF_CPP_VMODULE` | 精确控制特定文件的 VLOG 级别 | `export TF_CPP_VMODULE="musa_graph_optimizer=1"` |

### 2.3 遥测系统

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `MUSA_TELEMETRY_ENABLED` | 启用遥测系统 | `export MUSA_TELEMETRY_ENABLED=1` |
| `MUSA_TELEMETRY_LOG_PATH` | 遥测日志输出路径 | `export MUSA_TELEMETRY_LOG_PATH=/tmp/telemetry.json` |
| `MUSA_TELEMETRY_BUFFER_SIZE` | 事件缓冲区大小 | `export MUSA_TELEMETRY_BUFFER_SIZE=50000` |
| `MUSA_TELEMETRY_FLUSH_MS` | 日志刷新间隔（毫秒） | `export MUSA_TELEMETRY_FLUSH_MS=50` |

---

## 3. 常用调试组合

### 3.1 性能与热点（TensorFlow 侧）

内核级计时宏已移除。可结合 TensorFlow 自带能力做性能分析，例如：

- 使用 TensorFlow Profiler（`tf.profiler` / TensorBoard）采集算子与时间线
- 使用 `TF_CPP_VMODULE` 打开图优化或融合相关 VLOG，确认优化是否命中
- 通过 `MUSA_ENABLE_TF32`、`MUSA_AUTO_MIXED_PRECISION` 等开关对比数值与耗时

```bash
cd test
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_VMODULE="musa_graph_optimizer=1"
python test_runner.py --single ops/matmul_op_test.py
```

### 3.2 图优化调试

```bash
# 查看图优化器的详细日志
export TF_CPP_VMODULE="musa_graph_optimizer=1,fusion_pattern_manager=1"
python -m fusion.layernorm_gelu_fusion_test

# 查看算子融合详细过程
export TF_CPP_VMODULE="layernorm_fusion=2,gelu_fusion=1"
python -m fusion.layernorm_gelu_fusion_test

# Dump GraphDef
export MUSA_DUMP_GRAPHDEF=1
export MUSA_DUMP_GRAPHDEF_DIR=/tmp/graphs
python test_runner.py
```

### 3.3 脏数据诊断

```bash
# 启用遥测进行脏数据追溯
export MUSA_TELEMETRY_ENABLED=1
export MUSA_TELEMETRY_LOG_PATH=/tmp/telemetry.json
export MUSA_TELEMETRY_BUFFER_SIZE=50000

# 运行你的模型或测试脚本（示例：测试套件）
cd test && python test_runner.py

# 分析遥测日志
grep "dirty_data_detected" /tmp/telemetry.json
```

### 3.4 静音模式（仅显示错误）

```bash
export TF_CPP_MIN_LOG_LEVEL=2
python test_runner.py
```

### 3.5 恢复默认配置

```bash
unset MUSA_TELEMETRY_ENABLED MUSA_TELEMETRY_LOG_PATH
unset TF_CPP_MIN_LOG_LEVEL TF_CPP_VMODULE
```

---

## 4. 内存诊断（Memory Coloring）

内存染色用于检测 Use-After-Free 和内存越界等问题。请先使用常规方式构建插件（`./build.sh` 或安装 wheel），再按下列流程配合遥测使用。

### 4.1 内存染色原理

- **分配时填充**：`0xABABABAB` 模式（标识未初始化内存）
- **释放时填充**：`0xCDCDCDCD` 模式（标识已释放内存）
- **检测机制**：Kernel 执行前验证内存模式，若发现 `0xCDCDCDCD` 则报告 Use-After-Free

### 4.2 内存诊断流程

```bash
# 1. 构建或安装当前版本插件（Release）
#    在仓库根目录: ./build.sh

# 2. 启用遥测（配合内存诊断）
export MUSA_TELEMETRY_ENABLED=1

# 3. 运行测试
cd test
python test_runner.py

# 4. 检查日志中的内存问题报告
grep "Use-After-Free" /tmp/test_output.log
grep "memory_corruption" /tmp/telemetry.json
```

---

## 5. 流同步诊断

遥测系统可追踪 Event 和 Stream 的同步关系，用于诊断跨流竞争条件。

### 5.1 同步事件追踪

遥测日志中的同步事件示例：

```json
{"event_type":"event_record","event_handle":"0x7f1000","stream_id":100,"op_name":"EventRecord"}
{"event_type":"event_wait","event_handle":"0x7f1000","stream_id":200,"source_stream_id":100,"op_name":"EventWait"}
```

通过分析 `event_record` 和 `event_wait` 的时序关系，可以验证：

- H2D Stream → Compute Stream 同步是否正确
- Compute Stream → D2H Stream 同步是否正确
- Event 是否在等待 Stream 完成后才被销毁

### 5.2 同步问题诊断流程

```bash
# 1. 启用遥测
export MUSA_TELEMETRY_ENABLED=1
export MUSA_TELEMETRY_LOG_PATH=/tmp/sync_trace.json

# 2. 运行测试（或你的同步压力用例）
cd test && python test_runner.py

# 3. 分析 Event 同步链
python -c "
import json
with open('/tmp/sync_trace.json') as f:
    events = [json.loads(line) for line in f]
# 找出所有 event_record 和 event_wait
records = [e for e in events if e['event_type'] == 'event_record']
waits = [e for e in events if e['event_type'] == 'event_wait']
print(f'Event records: {len(records)}, Event waits: {len(waits)}')
"
```

---

## 6. 故障排查清单

### 6.1 脏数据（NaN）问题

1. 启用遥测系统记录完整操作链
2. 检测到 NaN 后，使用 `BacktraceByAddress` 追溯
3. 检查最近 10 次内存操作（分配、拷贝、Kernel）
4. 验证 Event 同步链是否完整

### 6.2 OOM 问题

1. 使用 `musaMemGetInfo` 监控显存使用趋势
2. 检查遥测日志中的 `tensor_allocate` 和 `tensor_free` 计数
3. 验证是否存在内存泄漏（分配 > 释放）
4. 检查碎片率（ fragmentation > 40% 为异常）

### 6.3 性能问题

1. 使用 TensorFlow Profiler 或 timeline 分析算子级耗时
2. 使用 `TF_CPP_VMODULE` 确认 Grappler / 融合是否按预期执行
3. 验证是否启用了 TF32/AMP 等加速开关
4. 对比 `MUSA_DISABLE_GRAPPLER=1` 以隔离图优化影响

---

**文档版本**: 2026-04-24
**适用版本**: TensorFlow MUSA Extension v1.0+
