# MUSA Kernel Timing 总文档（Master）

> 说明：这是本轮 timing 宏开发与调试的汇总文档。  
> 用于开发联调、结果核对、问题上报。PR 前可按需删除。

---

## 1. 功能目标与当前状态

- 已实现宏开关与运行时控制。
- 输出分隔符已改为 `,`。
- 总时间拆分为：
  - `host_total_ms`（CPU 墙钟）
  - `device_total_ms`（MUSA event 设备时间）
- 已支持 `MUSA_KERNEL_TRACE_START()` / `MUSA_KERNEL_TRACE_END()` 分段计时。
- 已支持自定义阶段名（例如 `State1`、`State2`）。
- 已支持未配对 START/END 告警输出。

---

## 2. 代码位置（关键文件）

- 计时宏与实现：
  - `musa_ext/utils/logging.h`
- 当前已埋点算子示例：
  - `musa_ext/kernels/math/musa_matmul_op.cc`
  - `musa_ext/kernels/math/musa_addn_op.cc`
  - `musa_ext/kernels/math/musa_conv2d_op.cc`

---

## 3. 编译模式与开关

### 3.1 Debug（开启 timing 宏）

```bash
cd /workspace/tensorflow_musa_extension
./build.sh debug
```

- 等效核心开关：
  - `CMAKE_BUILD_TYPE=Debug`
  - `MUSA_KERNEL_DEBUG=ON`

### 3.2 Release（关闭 timing 宏）

```bash
cd /workspace/tensorflow_musa_extension
./build.sh
# 或 ./build.sh release
```

- 等效核心开关：
  - `CMAKE_BUILD_TYPE=Release`
  - `MUSA_KERNEL_DEBUG=OFF`

---

## 4. 运行时环境变量

### 4.1 模式控制

```bash
export MUSA_TIMING_KERNEL_LEVEL=1   # 只打总时间
# export MUSA_TIMING_KERNEL_LEVEL=2 # 打总时间 + 各分段
```

### 4.2 名称过滤

```bash
export MUSA_TIMING_KERNEL_NAME=ALL
# 例如只看 MatMul
# export MUSA_TIMING_KERNEL_NAME=MatMul
```

### 4.3 Summary 开关

```bash
export MUSA_TIMING_KERNEL_STATS=1
# 0 表示关闭 summary
```

---

## 5. 宏使用方式

### 5.1 基础 guard

```cpp
MUSA_KERNEL_TIMING_GUARD(ctx);
```

### 5.2 分段埋点

```cpp
MUSA_KERNEL_TRACE_START("Mem Alloc");
// ... code block ...
MUSA_KERNEL_TRACE_END("Mem Alloc");

MUSA_KERNEL_TRACE_START("Kernel");
// ... kernel launch ...
MUSA_KERNEL_TRACE_END("Kernel");
```

### 5.3 自定义阶段命名（调试自由命名）

```cpp
MUSA_KERNEL_TRACE_START("State1");
// ... allocate / pre-process ...
MUSA_KERNEL_TRACE_END("State1");

MUSA_KERNEL_TRACE_START("State2");
// ... main kernel ...
MUSA_KERNEL_TRACE_END("State2");
```

---

## 6. 输出格式（当前）

### 6.1 LEVEL=2 样式

```text
[MUSA_KERNEL_TIMING] MatMul [[10,20],[20,15]], host_total_ms=16.419, device_total_ms=16.004, Kernel=15.832, Other=0.171
```

### 6.2 告警样式（START/END 不匹配）

```text
[MUSA_KERNEL_TIMING_WARNING] END without matching START. kernel=MatMul, stage=Mem Alloc
[MUSA_KERNEL_TIMING_WARNING] Unmatched START without END. kernel=MatMul, stage=Mem
```

### 6.3 设备头信息

```text
[MUSA_KERNEL_TIMING_DEVICE] device_id=0, device_count=8, device_name=MTT S5000
```

### 6.4 Summary 样式

```text
=================================================================================
MUSA Kernel Debug Statistics
=================================================================================
Kernel Name      Input Shape          Count      Total(ms)    Avg(ms)      Min(ms)      Max(ms)
...
=================================================================================
```

---

## 7. 常用测试命令

### 7.1 单算子（MatMul）验证

```bash
cd /workspace/tensorflow_musa_extension
./build.sh debug

export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1

mkdir -p /tmp/musa_timing_logs
python test/test_runner.py --single matmul_op_test.py 2>&1 | tee /tmp/musa_timing_logs/matmul_l2.log

grep "MUSA_KERNEL_TIMING_DEVICE\|MUSA_KERNEL_TIMING_WARNING" /tmp/musa_timing_logs/matmul_l2.log | head -n 20
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_l2.log | head -n 80
grep -n -A30 "MUSA Kernel Debug Statistics" /tmp/musa_timing_logs/matmul_l2.log
```

### 7.2 全量测试

```bash
cd /workspace/tensorflow_musa_extension
./build.sh debug

export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1

mkdir -p /tmp/musa_timing_logs
python test/test_runner.py --quiet 2>&1 | tee /tmp/musa_timing_logs/all_ops_l2.log
```

---

## 8. 已验证结果（本轮）

- 正常输出已验证：
  - `,` 分隔符
  - `host_total_ms` + `device_total_ms`
  - 分段名可自定义（`State1`/`State2`）
  - 缺失配对会打印 warning

---

## 9. 当前已知问题（阻塞项）

### 9.1 Debug 下稳定崩溃（重点）

- 崩溃测试：
  - `apply_resource_op_test.py`
  - `applyadam_op_test.py`（同路径风险）
- 典型报错：

```text
tensorflow/core/platform/refcount.h:90
Check failed: ref_.load() == 0 (1 vs. 0)
Aborted (core dumped)
```

### 9.2 结论

- 该问题在 `MUSA_TIMING_KERNEL_LEVEL=0` 时仍复现。
- 当前证据指向 `ResourceApplyAdam` 算子实现路径问题（资源/生命周期/异步时序），而非 `logging.h` 打印逻辑。

### 9.3 补充说明

- Debug 编译出现的 Eigen `#pragma pop_macro("EIGEN_DEVICE_FUNC")` 警告通常不是此崩溃根因。

---

## 10. 维护建议

- 若仅关注 timing 宏功能验证，优先在稳定算子（MatMul/AddN/Conv2D）回归。
- 对 `ResourceApplyAdam` 建议单独开 issue，并补齐 C++ 调用栈定位（容器内安装 gdb 后抓栈）。
- PR 前若不需保留该文档，可直接删除：

```bash
cd /workspace/tensorflow_musa_extension
rm -f TIMING_KERNEL_MASTER_GUIDE.md
```

