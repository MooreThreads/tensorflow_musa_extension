# MUSA Kernel Timing Debug Runbook (Remote + Docker)

> 说明：这是临时调试文档，便于在 remote 端复现计时日志与统计输出。  
> 你合 PR 前可直接删除本文件。

## 0. 当前环境约定

你通常是：

- 宿主机：`zhaoye@worker33016:~/workspace/tensorflow_musa_extension`
- 容器：`root@dev:/workspace#`（通过 `docker exec -it zhaoye /bin/bash` 进入）

下面默认在容器里执行。

---

## 1. 进入项目目录并编译 debug

```bash
cd /workspace/tensorflow_musa_extension
./build.sh debug
ls -lh build/libmusa_plugin.so
```

说明：

- `./build.sh debug` 会开启 `MUSA_KERNEL_DEBUG=ON`。
- 输出库在 `build/libmusa_plugin.so`。

---

## 2. 运行前环境变量模板

### 2.1 LEVEL=1（只看总耗时）

```bash
export MUSA_TIMING_KERNEL_LEVEL=1
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=0
```

### 2.2 LEVEL=2（看分段耗时）

```bash
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=0
```

### 2.3 LEVEL=2 + summary（进程退出打印统计）

```bash
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1
```

### 2.4 只看某类 kernel（示例 MatMul）

```bash
export MUSA_TIMING_KERNEL_NAME=MatMul
```

> 名称过滤是大小写不敏感子串匹配：`MatMul` 可匹配 `MatMul / BatchMatMul / MusaMatMul`。

---

## 3. 运行单测并落盘日志

先创建日志目录：

```bash
mkdir -p /tmp/musa_timing_logs
```

### 3.1 MatMul 示例（多行）

```bash
python test/test_runner.py --single matmul_op_test.py 2>&1 \
  | tee /tmp/musa_timing_logs/matmul_level2.log
```

### 3.2 MatMul 示例（你对话里的一行命令）

```bash
python test/test_runner.py --single matmul_op_test.py 2>&1 | tee /tmp/musa_timing_logs/matmul_level2.log
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_level2.log | head -n 60
```

### 3.3 其他算子示例

```bash
python test/test_runner.py --single conv2d_op_test.py 2>&1 | tee /tmp/musa_timing_logs/conv2d_level2.log
python test/test_runner.py --single addn_op_test.py 2>&1 | tee /tmp/musa_timing_logs/addn_level2.log
```

### 3.4 全量测试示例

```bash
python test/test_runner.py --quiet 2>&1 | tee /tmp/musa_timing_logs/all_tests_level2.log
```

---

## 4. 分析日志（在哪里看、看什么）

### 4.1 看 timing 行

```bash
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_level2.log
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_level2.log | head -n 60
```

### 4.2 看 summary 汇总

```bash
grep -n "MUSA Kernel Debug Statistics" /tmp/musa_timing_logs/matmul_level2.log
grep -n -A30 "MUSA Kernel Debug Statistics" /tmp/musa_timing_logs/matmul_level2.log
```

### 4.3 快速统计记录条数

```bash
grep -c "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_level2.log
```

### 4.4 从全量日志筛指定 kernel

```bash
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/all_tests_level2.log | grep -i matmul
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/all_tests_level2.log | grep -i addn
```

---

## 5. 期望输出样式

### 5.1 LEVEL=1

```text
[MUSA_KERNEL_TIMING] MatMul [<input-shape>], host_total_ms=0.140, device_total_ms=0.123
```

### 5.2 LEVEL=2

```text
[MUSA_KERNEL_TIMING] MatMul [<input-shape>], host_total_ms=0.260, device_total_ms=0.234, Mem Alloc=0.010, Kernel=0.220, Other=0.004
```

### 5.3 STATS=1

进程退出时会出现：

```text
=================================================================================
MUSA Kernel Debug Statistics
...
=================================================================================
```

### 5.4 设备信息 + 告警顺序

每次进程第一次打印 timing 时，会先打印设备信息；若有告警，会紧跟在设备信息后：

```text
[MUSA_KERNEL_TIMING_DEVICE] device_id=0, device_count=1, device_name=MTT S4000, MUSA_VISIBLE_DEVICES=0
[MUSA_KERNEL_TIMING_WARNING] END without matching START. kernel=MatMul, stage=Kernel
```

---

## 6. Release 对照验证（宏关闭）

```bash
cd /workspace/tensorflow_musa_extension
./build.sh release

export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1

python test/test_runner.py --single matmul_op_test.py 2>&1 | tee /tmp/musa_timing_logs/matmul_release.log
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_release.log
```

预期：`grep` 无输出。

---

## 7. 常见问题

1. 看不到 timing 日志
- 确认使用 `./build.sh debug` 构建。
- 确认环境变量在同一个 shell 里 `export` 后再运行测试。
- 先把 `MUSA_TIMING_KERNEL_NAME=ALL`，避免过滤掉目标算子。

2. 只有测试进度，没有 timing 行
- 检查加载的是最新 `build/libmusa_plugin.so`。
- 重新 `./build.sh debug` 后重跑。

3. 路径写错导致 grep 查不到
- 例如你使用 `tee /tmp/musa_timing_logs/matmul_level2.log`，grep 也必须指向同一路径：
  `grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_level2.log`

---

## 8. 提交前可选步骤（宿主机）

```bash
cd ~/workspace/tensorflow_musa_extension
git status -sb
./install-hooks.sh
```

---

## 9. 临时文档清理（可选）

```bash
cd ~/workspace/tensorflow_musa_extension
rm TIMING_DEBUG_SUMMARY.md TIMING_DEBUG_RUNBOOK.md
```
