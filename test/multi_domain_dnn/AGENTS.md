# Multi Domain DNN 测试规则

- 详细用法先读本目录 `README.md`。
- 默认模型为 JD multi-domain DNN frozen GraphDef，正式输出为
  `WeightedSum/add:0`。
- 功能验证使用 `tf215_graph_accuracy.py`；性能测试统一使用
  `tf215_graph_perf.py`，不要再创建平行性能入口。
- 功能验证保持 host-feed、CPU no-XLA golden、strict placement 和 backend
  trace；不要为了对齐性能口径改成 device-resident input。
- MUSA/CUDA 纯推理使用 `--input_mode device --no-fetch_output`；MUSA 额外使用
  `--no-wait_execute_done`。该模式排除初始化 H2D，每轮末尾分别执行一次
  `musaDeviceSynchronize()`/`cudaDeviceSynchronize()`，计时到最后一个模型
  kernel 完成。
- device input 固定使用一次初始化的 device resource variables，正式 runs 使用空
  `feed_dict`，每轮只读取已有 resource variables，不再 feed session handles。
- 对齐 ONNX Runtime 四流 QPS 时使用 batch1、`--streams 4 --querys 1000`；
  MUSA 默认令 `MUSA_PJRT_MAX_INFLIGHT_EXECUTES=streams`。读取 JSON 的
  与 ORT 同口径的 `qps`、辅助分析用的 `run_qps`/`sample_qps`，以及
  `wall_time_ms`；不要使用单次 `p50_ms` 直接换算多流 QPS。
- `streams` 在 TF/ORT Python 脚本中表示 host worker 数，不直接等价于 GPU
  stream 数；ORT MUSA EP 是 stream-aware，TF 的 PJRT stream/event 数量必须用
  MSYS/NSYS 确认。
- 包含传输的 E2E 使用 `--input_mode host --fetch_output
  --no-wait_execute_done`。
- 不要用固定 device `Const` 替代 Placeholder 做正式性能测试；XLA 会发生
  constant folding，改变 kernel 图。
- MSYS/NSYS 必须配合 `--profiler_range`，只采集 warmup 后的正式 runs。
- mock input 只用于确定性回归，不能视为真实业务 golden。
- 跨机器 E2E 比较前先运行 `--host_calibration`，P50 差异应不超过约 3%。
- 性能结果必须记录 batch、warmup、runs、input/fetch/wait 模式、CPU
  governor/frequency、affinity、NUMA、TF 线程数、host load 和 GPU 状态。
- 临时日志不纳入该测试目录或提交内容；稳定结论写入根目录 `WORKLOG.md`。
