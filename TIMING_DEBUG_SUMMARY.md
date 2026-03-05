# Timing Debug Summary (Remote + Docker)

> 目的：快速复现 MUSA kernel 计时日志与统计输出。  
> 场景：你在宿主机 SSH 后，通过 `docker exec` 进入开发容器。

---

## 1. 从宿主机进入容器

```bash
cd ~/workspace/tensorflow_musa_extension
docker exec -it zhaoye /bin/bash
```

---

## 2. 在容器里编译 Debug 版本

```bash
cd /workspace/tensorflow_musa_extension
./build.sh debug
ls -lh build/libmusa_plugin.so
```

- `debug` 会开启 `MUSA_KERNEL_DEBUG=ON`。

---

## 3. 设置计时环境变量（按场景）

### 3.1 分段计时 + 全部 kernel + 打印 summary（最常用）

```bash
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1
```

### 3.2 分段计时 + 只看 MatMul 类

```bash
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=MatMul
export MUSA_TIMING_KERNEL_STATS=0
```

### 3.3 只看 compute 总时间

```bash
export MUSA_TIMING_KERNEL_LEVEL=1
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=0
```

---

## 4. 运行并落盘日志

先创建日志目录：

```bash
mkdir -p /tmp/musa_timing_logs
```

### 4.1 推荐写法（多行）

```bash
python test/test_runner.py --single matmul_op_test.py 2>&1 \
  | tee /tmp/musa_timing_logs/matmul_level2.log
```

### 4.2 你这次对话里的一行示例（已写入）

```bash
python test/test_runner.py --single matmul_op_test.py 2>&1 | tee /tmp/musa_timing_logs/matmul_level2.log
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_level2.log | head -n 60
```

### 4.3 其他算子示例

```bash
python test/test_runner.py --single conv2d_op_test.py 2>&1 | tee /tmp/musa_timing_logs/conv2d_level2.log
python test/test_runner.py --single addn_op_test.py 2>&1 | tee /tmp/musa_timing_logs/addn_level2.log
```

---

## 5. 查看与分析结果

```bash
# 1) 查看 timing 行
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_level2.log

# 2) 只看前 60 行（便于快速扫）
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_level2.log | head -n 60

# 3) 看 summary 汇总表（如果 MUSA_TIMING_KERNEL_STATS=1）
grep -n -A30 "MUSA Kernel Debug Statistics" /tmp/musa_timing_logs/matmul_level2.log

# 4) 看一共产生了多少条 timing 记录
grep -c "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_level2.log

# 5) 从全量日志里筛 MatMul
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/all_tests_level2.log | grep -i "matmul"
```

---

## 6. 常见运行组合（可直接复制）

### 6.1 只看总时间（LEVEL=1）

```bash
export MUSA_TIMING_KERNEL_LEVEL=1
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=0

python test/test_runner.py --single matmul_op_test.py 2>&1 | tee /tmp/musa_timing_logs/matmul_level1.log
```

### 6.2 看分段 + 只看 MatMul

```bash
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=MatMul
export MUSA_TIMING_KERNEL_STATS=0

python test/test_runner.py --single matmul_op_test.py 2>&1 | tee /tmp/musa_timing_logs/matmul_only.log
```

### 6.3 跑全量测试并收集 summary

```bash
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1

python test/test_runner.py --quiet 2>&1 | tee /tmp/musa_timing_logs/all_tests_level2.log
```

---

## 7. 验证 Release 下无计时输出

```bash
cd /workspace/tensorflow_musa_extension
./build.sh release

export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1

python test/test_runner.py --single matmul_op_test.py 2>&1 | tee /tmp/musa_timing_logs/matmul_release.log
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul_release.log
```

预期：最后一条 `grep` 无输出。

---

## 8. 提交代码与发 PR（宿主机执行）

```bash
cd ~/workspace/tensorflow_musa_extension

# 1) 确认分支和改动
git status -sb

# 2) （可选）同步 upstream main
git fetch upstream
git rebase upstream/main

# 3) 暂存改动（按需增删）
git add CMakeLists.txt build.sh
git add musa_ext/utils/logging.h
git add musa_ext/kernels/math/musa_matmul_op.cc
git add musa_ext/kernels/math/musa_conv2d_op.cc
git add musa_ext/kernels/math/musa_addn_op.cc
git add README.md README.en.md

# 4) 若需要把临时文档一起提交
# git add TIMING_DEBUG_SUMMARY.md TIMING_DEBUG_RUNBOOK.md

# 5) 提交
git commit -m "feat: add MUSA kernel timing macros and debug controls"

# 6) 推送
git push origin <your_branch>
```

PR 页面（fork -> upstream）：

- `https://github.com/MooreThreads/tensorflow_musa_extension/compare/main...XFDG:<your_branch>`

---

## 9. 常用 `.sh` 脚本

### 9.1 `install-hooks.sh`（宿主机执行一次）

```bash
cd ~/workspace/tensorflow_musa_extension
chmod +x install-hooks.sh
./install-hooks.sh
```

作用：安装 `pre-commit` 和 `commit-msg` hook。

### 9.2 `test/run_all_tests.sh`（容器里跑全量）

```bash
cd /workspace/tensorflow_musa_extension
chmod +x test/run_all_tests.sh
./test/run_all_tests.sh
```

---

## 10. PR 前清理临时文档（如果不想带进 PR）

```bash
cd ~/workspace/tensorflow_musa_extension
rm TIMING_DEBUG_SUMMARY.md TIMING_DEBUG_RUNBOOK.md
```

---

## 11. 备注

- 本文件是精简版。  
- 详细版见：`TIMING_DEBUG_RUNBOOK.md`
