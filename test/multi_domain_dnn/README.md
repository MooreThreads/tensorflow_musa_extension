# Multi Domain DNN TF2.15 Graph Tests

该目录提供两套运行方式：

- 原生 TensorFlow MUSA：`runtime_mode=native`
- OpenXLA-MUSA PJRT：`runtime_mode=pjrt`

默认模型是 JD multi-domain DNN frozen GraphDef：

```text
model/JD/real_models/platform/dnn__multi-domain-task-simplifyv3__2026022622335783__tf_models__frozen_grpah.pb
```

正式输出为 `WeightedSum/add:0`。脚本默认使用确定性 mock input（`seed=20260421`）；真实业务验证应通过 `--input_npz` 提供输入。

## PJRT 安装与加载要求

PJRT 不是只把一个 `.so` 文件复制到 TensorFlow 的 Python 目录就能保证可用。插件必须与以下组件来自同一套构建/运行环境：

- TensorFlow 2.15 的 Python wheel 和 `libtensorflow_framework.so.2`；
- MUSA SDK/runtime 及其 `libmusa.so.1`、muDNN 等依赖；
- OpenXLA-MUSA PJRT 插件 `libmusa_pjrt_plugin_zy.so`；
- TF2.15 registry bridge `libmusa_tf215_registry_bridge.so`（如果该构建提供）。

普通用户路径是先安装 `tensorflow_musa` 与 `openxla_musa_pjrt` 两个 wheel，再只设置：

```bash
export TENSORFLOW_MUSA_ENABLE_PJRT=1
```

此时 `tensorflow_musa` 负责调用 provider，provider 从自身 wheel 的 manifest 自动发现
plugin、bridge 和工具链；不需要手工复制或传入 `.so` 路径。

本目录的 standalone 脚本还保留了开发期的路径覆盖参数，用于调试未安装的构建产物。如果使用
自定义开发位置，才显式指定：

```bash
export TF215_SITE_PACKAGES=/path/to/tf2.15/lib/python3.10/site-packages
export MUSA_PJRT_PLUGIN_PATH=/path/to/libmusa_pjrt_plugin_zy.so
export MUSA_TF215_BRIDGE_PATH=/path/to/libmusa_tf215_registry_bridge.so
export LD_LIBRARY_PATH=/usr/local/musa/lib:$TF215_SITE_PACKAGES/tensorflow:$(dirname "$MUSA_PJRT_PLUGIN_PATH"):$(dirname "$MUSA_TF215_BRIDGE_PATH"):$LD_LIBRARY_PATH
```

先用短命令验证加载和单次编译，再运行长时间性能测试：

```bash
python3 test/multi_domain_dnn/tf215_graph_accuracy.py \
  --device musa --runtime_mode pjrt --device_index 3 \
  --warmup_runs_per_case 0 --runs_per_case 1
```

`accuracy` 使用 host-feed 做 CPU golden 对比；`perf` 的 `input_mode=device`、`--no-fetch_output` 是性能路径，不能替代 accuracy。

已验证的普通 wheel 路径是：两个 wheel 安装到同一 Python 环境后，只设置
`TENSORFLOW_MUSA_ENABLE_PJRT=1`，`tensorflow_musa` 会得到 `phase=ready`，并自动加载
provider wheel 内的 plugin/bridge。下面的 standalone 模型命令用于仓库级功能/性能回归；命令中
不需要用户手工填写 `.so` 路径。环境覆盖变量只在调试未安装的构建产物时使用。

## 目录内运行

建议直接在 [`tensorflow_musa_extension`](../../README.md:1) 仓库根目录运行：

```bash
cd /data/eureka/tensorflow_musa_extension
```

下面示例均使用相对路径 `test/multi_domain_dnn/...`。

## 功能验证

[`tf215_graph_accuracy.py`](tf215_graph_accuracy.py:1) 使用相同输入分别执行 CPU no-XLA 和目标 backend，检查 shape、dtype、finite、重复执行稳定性、数值误差和 strict placement。
功能测试保持 host-feed 语义，适合做 CPU golden 与目标 backend 的端到端正确性验证；device-resident input 只用于性能测试，不用于替代 CPU golden。

### 原生 TF-MUSA

```bash
python3 test/multi_domain_dnn/tf215_graph_accuracy.py \
  --device musa \
  --runtime_mode native \
  --device_index 3 \
  --graph_path model/JD/real_models/platform/dnn__multi-domain-task-simplifyv3__2026022622335783__tf_models__frozen_grpah.pb \
  --output_tensor WeightedSum/add:0 \
  --batches 1
```

```bash
python3 test/multi_domain_dnn/tf215_graph_accuracy.py \
  --device musa \
  --runtime_mode native \
  --device_index 3 \
  --graph_path model/JD/real_models/platform/sim_graph_2_frozen_graph.pb \
  --output_tensor Reshape_2:0 \
  --batches 1 \
  --int_upper_bound 50
```

```bash
python3 test/multi_domain_dnn/tf215_graph_accuracy.py \
  --device musa \
  --runtime_mode native \
  --device_index 3 \
  --graph_path model/JD/real_models/platform/twin_graph_2_frozen_graph.pb \
  --output_tensor WeightedSum/add:0 \
  --batches 1
```

### OpenXLA-MUSA PJRT

```bash
python3 test/multi_domain_dnn/tf215_graph_accuracy.py \
  --device musa \
  --runtime_mode pjrt \
  --device_index 3 \
  --graph_path model/JD/real_models/platform/dnn__multi-domain-task-simplifyv3__2026022622335783__tf_models__frozen_grpah.pb \
  --output_tensor WeightedSum/add:0 \
  --batches 1
```

```bash
python3 test/multi_domain_dnn/tf215_graph_accuracy.py \
  --device musa \
  --runtime_mode pjrt \
  --device_index 3 \
  --graph_path model/JD/real_models/platform/sim_graph_2_frozen_graph.pb \
  --output_tensor Reshape_2:0 \
  --batches 1
```

```bash
python3 test/multi_domain_dnn/tf215_graph_accuracy.py \
  --device musa \
  --runtime_mode pjrt \
  --device_index 3 \
  --graph_path model/JD/real_models/platform/twin_graph_2_frozen_graph.pb \
  --output_tensor WeightedSum/add:0 \
  --batches 1
```

默认测试 batch `1,2,8,32,100`，默认容差为 `atol=1e-4, rtol=1e-3`。mock input 下的跨 backend 对齐不能替代真实业务 input/output golden。

当前 native 算子覆盖限制：SIM 图中的 `UnsortedSegmentMax` 和 Twin 图中的 `OneHot` 尚无
MUSA kernel，因此 strict placement 下 native SIM/Twin accuracy/perf 会在建图或初始化阶段失败。
这不影响 PJRT 路径；三个模型的 PJRT accuracy/perf 使用 OpenXLA-MUSA 后端验证。

## 性能测试

性能入口统一为 [`tf215_graph_perf.py`](tf215_graph_perf.py:1)。

### Device-resident 纯推理

MUSA/CUDA 默认使用 `input_mode=device`、不取回输出；MUSA 关闭逐 cluster host wait。每轮末尾分别调用一次 `musaDeviceSynchronize()`/`cudaDeviceSynchronize()`。初始化 H2D 不在正式计时范围内，计时边界为 device input ready 到最后一个模型 kernel 完成。

### 原生 TF-MUSA

```bash
taskset -c 0-31,64-95 \
python3 test/multi_domain_dnn/tf215_graph_perf.py \
  --device musa \
  --runtime_mode native \
  --device_index 3 \
  --input_mode device \
  --graph_path model/JD/real_models/platform/dnn__multi-domain-task-simplifyv3__2026022622335783__tf_models__frozen_grpah.pb \
  --output_tensor WeightedSum/add:0 \
  --batchsize 100 \
  --warmup_runs 20 \
  --num_runs 200 \
  --no-fetch_output \
  --no-wait_execute_done \
  --json_output tmp/e2e_perf_standalone_native/dnn_perf_report.json
```

```bash
taskset -c 0-31,64-95 \
python3 test/multi_domain_dnn/tf215_graph_perf.py \
  --device musa \
  --runtime_mode native \
  --device_index 0 \
  --input_mode device \
  --graph_path model/JD/real_models/platform/sim_graph_2_frozen_graph.pb \
  --output_tensor Reshape_2:0 \
  --batchsize 100 \
  --warmup_runs 20 \
  --num_runs 200 \
  --no-fetch_output \
  --no-wait_execute_done \
  --json_output tmp/e2e_perf_standalone_native/sim_perf_report.json
```

```bash
taskset -c 0-31,64-95 \
python3 test/multi_domain_dnn/tf215_graph_perf.py \
  --device musa \
  --runtime_mode native \
  --device_index 3 \
  --input_mode device \
  --graph_path model/JD/real_models/platform/twin_graph_2_frozen_graph.pb \
  --output_tensor WeightedSum/add:0 \
  --batchsize 100 \
  --warmup_runs 20 \
  --num_runs 200 \
  --no-fetch_output \
  --no-wait_execute_done \
  --json_output tmp/e2e_perf_standalone_native/twin_perf_report.json
```

### OpenXLA-MUSA PJRT

```bash
taskset -c 0-31,64-95 \
python3 test/multi_domain_dnn/tf215_graph_perf.py \
  --device musa \
  --runtime_mode pjrt \
  --device_index 3 \
  --xla \
  --input_mode device \
  --graph_path model/JD/real_models/platform/dnn__multi-domain-task-simplifyv3__2026022622335783__tf_models__frozen_grpah.pb \
  --output_tensor WeightedSum/add:0 \
  --batchsize 100 \
  --warmup_runs 20 \
  --num_runs 200 \
  --no-fetch_output \
  --no-wait_execute_done \
  --json_output tmp/e2e_perf_standalone_pjrt/dnn_perf_report.json
```

```bash
taskset -c 0-31,64-95 \
python3 test/multi_domain_dnn/tf215_graph_perf.py \
  --device musa \
  --runtime_mode pjrt \
  --device_index 3 \
  --xla \
  --input_mode device \
  --graph_path model/JD/real_models/platform/sim_graph_2_frozen_graph.pb \
  --output_tensor Reshape_2:0 \
  --batchsize 100 \
  --warmup_runs 20 \
  --num_runs 200 \
  --no-fetch_output \
  --no-wait_execute_done \
  --json_output tmp/e2e_perf_standalone_pjrt/sim_perf_report.json
```

```bash
taskset -c 0-31,64-95 \
python3 test/multi_domain_dnn/tf215_graph_perf.py \
  --device musa \
  --runtime_mode pjrt \
  --device_index 3 \
  --xla \
  --input_mode device \
  --graph_path model/JD/real_models/platform/twin_graph_2_frozen_graph.pb \
  --output_tensor WeightedSum/add:0 \
  --batchsize 100 \
  --warmup_runs 20 \
  --num_runs 200 \
  --no-fetch_output \
  --no-wait_execute_done \
  --json_output tmp/e2e_perf_standalone_pjrt/twin_perf_report.json
```

device input 通过一次初始化的 `tf.Variable(use_resource=True)` 驻留在 device 上，再用 `read_value()` 替换 GraphDef Placeholder；初始化 H2D 位于计时外，正式 runs 使用空 `feed_dict`，不再逐轮解析 session handle。每轮仍会读取 resource variables，但输入连接关系和 device buffer 在建图阶段已经固定；该方式不会把输入变成编译期 `Const`。

该 device-input 路径用于固定输入的纯推理性能测试。输入初始化不计入正式计时，正式计时从 `Session.run()` 开始，并在最后一次 device sync 完成后结束。

### Host-feed E2E

`input_mode=host` 每轮从 NumPy feed 输入；配合 `--fetch_output` 时，计时包含 host feed、H2D、模型执行、最终输出 D2H 和 TensorFlow/Python 调度。

```bash
python3 test/multi_domain_dnn/tf215_graph_perf.py \
  --device musa \
  --runtime_mode pjrt \
  --device_index 3 \
  --input_mode host \
  --batchsize 100 \
  --warmup_runs 20 \
  --num_runs 200 \
  --fetch_output \
  --no-wait_execute_done
```

CPU 默认使用 host input。A100 与 S5000 做纯推理对比时，CUDA 使用与 MUSA 相同的 device input、no-fetch 和最终 device sync 口径：

```bash
# CPU no-XLA
python3 test/multi_domain_dnn/tf215_graph_perf.py \
  --device cpu --input_mode host --batchsize 100 \
  --warmup_runs 20 --num_runs 200 --fetch_output

# A100 CUDA XLA device-resident；默认保留 TF32
python3 test/multi_domain_dnn/tf215_graph_perf.py \
  --device cuda --device_index 0 --xla --input_mode device \
  --graph_path /path/to/frozen_grpah.pb --batchsize 100 \
  --warmup_runs 20 --num_runs 200 --no-fetch_output
```

### Profiler Range

`--profiler_range` 只用 profiler API 包围 warmup 后的正式 runs：

```bash
msys profile \
  --capture-range=musaProfilerApi \
  --capture-range-end=stop \
  -o /path/to/report \
  python3 test/multi_domain_dnn/tf215_graph_perf.py \
    --device musa --device_index 3 --runtime_mode pjrt --xla --input_mode device \
    --warmup_runs 20 --num_runs 10 --no-fetch_output \
    --no-wait_execute_done --profiler_range
```

### Multi-stream QPS 对齐

目标 ONNX 是固定 `batch=1`。ONNX Runtime 的 `--streams 4 --querys 1000 --warmup 20` 表示 4 个 host worker 共享一个 query queue，每个 worker 预建一次 IOBinding，输入不在每轮重新构造；MUSA EP 另外通过 ORT 的 stream-aware 接口管理执行 stream。当前 `bind_output(name)` 路径会产生很小的输出 D2H，MSYS 中约为每次推理一个 4-byte payload。QPS 是 `querys / wall_time`。

TF 的 `--streams` 同样表示 host worker 数，不等价于显式创建 GPU stream；PJRT 内部的 stream/event 调度需要通过 MSYS/NSYS 单独确认。TF device-input/no-fetch 路径不取回模型输出，只在正式区间结束后做一次 device sync，因此 TF 与 ORT 的输入驻留和并发模型相近，但输出同步边界并不完全相同。TF 对应命令为：

```bash
python3 test/multi_domain_dnn/tf215_graph_perf.py \
  --device musa --device_index 1 --runtime_mode pjrt --xla \
  --input_mode device --batchsize 1 \
  --streams 4 --querys 1000 --warmup_runs 20 \
  --no-fetch_output --no-wait_execute_done
```

`--streams/--querys` 只支持 `input_mode=device`，并在所有 worker 完成后做一次最终 device sync。MUSA 多流默认将 `MUSA_PJRT_MAX_INFLIGHT_EXECUTES` 设为 `streams`，显式参数可以覆盖。

控制台中的 `qps`/`sample_qps` 表示 `querys / wall_time`，与 ONNX Runtime 保持一致；`run_qps` 表示 `Session.run/s`。只有 batch1 时两者数值相同；batch 大于 1 时，样本吞吐通常约为 `run_qps * batchsize`。

### Host Calibration

跨机器比较 A100/S5000 E2E 前，使用 `--host_calibration` 测量同一批 feed 的 CPU host 调度基线。它不执行模型算子，不代表模型性能。

```bash
python3 test/multi_domain_dnn/tf215_graph_perf.py \
  --device musa --device_index 3 --host_calibration \
  --batchsize 100 --warmup_runs 50 --num_runs 500
```

只有两端 calibration P50 差异不超过约 3%，且 CPU governor/frequency、affinity、NUMA、TF inter/intra-op threads、host load 和 GPU 状态均已记录时，才比较完整 E2E。

## 关键参数

- `--runtime_mode native|pjrt`：MUSA 执行模式；native 使用 `tensorflow_musa` 原生 kernel，pjrt 使用 OpenXLA-MUSA。
- `--input_mode host|device`：MUSA/CUDA 默认 `device`，CPU 默认 `host`。
- `--fetch_output`：取回完整输出；device 纯推理默认关闭。
- `--wait_execute_done`：控制每个 PJRT Execute 是否 host wait；device 纯推理默认关闭。
- `--profiler_range`：只抓 warmup 后正式 runs。
- `--cuda_disable_tf32`：CUDA 禁用 TF32；默认保留 TensorFlow 设置。
- `--input_npz`：加载真实多输入数据。
- `--save_output_path/--compare_output_path`：保存或对比完整输出，要求开启 `--fetch_output`。
