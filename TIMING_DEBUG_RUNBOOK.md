# MUSA Kernel Timing Debug Runbook (Remote + Docker)

> 说明：这是临时调试文档，便于在 remote 端复现计时日志与统计输出。  
> 你合 PR 前可直接删除本文件。

## 0. 你当前环境

你现在是：

- 宿主机：`zhaoye@worker33016:~/workspace/tensorflow_musa_extension`
- 容器：`root@dev:/workspace#`（通过 `docker exec -it zhaoye /bin/bash` 进入）

下面默认在容器里执行（`root@dev:/workspace#`）。

---

## 1. 进入工程目录

```bash
cd /workspace/tensorflow_musa_extension
pwd
```

预期输出：

```text
/workspace/tensorflow_musa_extension
```

---

## 2. 编译 debug 版本（开启宏）

```bash
./build.sh debug
```

关键点：

- `debug` 会把 CMake 选项 `MUSA_KERNEL_DEBUG=ON` 打开。
- 产物在：`/workspace/tensorflow_musa_extension/build/libmusa_plugin.so`

可验证：

```bash
ls -lh build/libmusa_plugin.so
```

---

## 3. 运行前设置环境变量

### 3.1 只看 compute 总耗时（LEVEL=1）

```bash
export MUSA_TIMING_KERNEL_LEVEL=1
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=0
```

### 3.2 看分段耗时（LEVEL=2）

```bash
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=MatMul
export MUSA_TIMING_KERNEL_STATS=0
```

### 3.3 打开汇总统计（进程退出打印）

```bash
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1
```

> 名称过滤是大小写不敏感子串匹配：  
> `MatMul` 可匹配 `MatMul / BatchMatMul / MusaMatMul`。

---

## 4. 运行单测并保存日志

推荐把 stderr+stdout 一起保存，方便 grep：

```bash
mkdir -p /tmp/musa_timing_logs

python test/test_runner.py --single matmul_op_test.py 2>&1 | tee /tmp/musa_timing_logs/matmul_level2.log
```

或：

```bash
python test/test_runner.py --single conv2d_op_test.py 2>&1 | tee /tmp/musa_timing_logs/conv2d_level2.log
python test/test_runner.py --single addn_op_test.py 2>&1 | tee /tmp/musa_timing_logs/addn_level2.log
```

---

## 5. 在哪里看结果

### 5.1 实时终端中看

运行测试时终端会直接打印：

- 单次计时行（`[MUSA_KERNEL_TIMING] ...`）
- 若 `MUSA_TIMING_KERNEL_STATS=1`，结束时会打印汇总表（`MUSA Kernel Debug Statistics`）

### 5.2 日志文件中看

```bash
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_level2.log
```

看 summary：

```bash
grep -n "MUSA Kernel Debug Statistics" /tmp/musa_timing_logs/matmul_level2.log
grep -n -A30 "MUSA Kernel Debug Statistics" /tmp/musa_timing_logs/matmul_level2.log
```

---

## 6. 期望输出样式

### LEVEL=1

```text
[MUSA_KERNEL_TIMING] MatMul [<input-shape>] | Total(ms) 0.123 |
```

### LEVEL=2

```text
[MUSA_KERNEL_TIMING] MatMul [<input-shape>] | Total(ms) 0.234 | Mem Alloc 0.010 | Mem Cpy 0.002 | Kernel 0.220 | Sync 0.000 |
```

### STATS=1

进程退出时会出现：

```text
=================================================================================
MUSA Kernel Debug Statistics
...
=================================================================================
```

---

## 7. 验证 release 下无日志（宏关闭）

```bash
./build.sh release
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1

python test/test_runner.py --single matmul_op_test.py 2>&1 | tee /tmp/musa_timing_logs/matmul_release.log
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_release.log
```

预期：`grep` 无输出（宏在 release 下为空实现）。

---

## 8. 常见问题

1. 看不到计时日志
- 先确认是 `./build.sh debug` 构建。
- 确认环境变量在**同一个 shell**里 export 后再运行 python。
- 确认 `MUSA_TIMING_KERNEL_NAME` 没过滤掉目标 op（先用 `ALL`）。

2. 日志只有测试 runner，没有 timing 行
- 检查是否加载的是刚编译的 `build/libmusa_plugin.so`。
- 重新 `./build.sh debug` 后再跑。

3. 我想看全部测试的 timing

```bash
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1

python test/test_runner.py --quiet 2>&1 | tee /tmp/musa_timing_logs/all_tests.log
```

