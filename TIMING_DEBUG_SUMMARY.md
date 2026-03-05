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

## 3. 设置计时环境变量

### 3.1 分段计时（推荐）

```bash
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1
```

### 3.2 只看 MatMul 类

```bash
export MUSA_TIMING_KERNEL_NAME=MatMul
```

### 3.3 只看总时间

```bash
export MUSA_TIMING_KERNEL_LEVEL=1
export MUSA_TIMING_KERNEL_STATS=0
```

---

## 4. 运行并落盘日志

```bash
mkdir -p /tmp/musa_timing_logs
python test/test_runner.py --single matmul_op_test.py 2>&1 | tee /tmp/musa_timing_logs/matmul.log
```

你也可以换成：

```bash
python test/test_runner.py --single conv2d_op_test.py 2>&1 | tee /tmp/musa_timing_logs/conv2d.log
python test/test_runner.py --single addn_op_test.py 2>&1 | tee /tmp/musa_timing_logs/addn.log
```

---

## 5. 查看分析结果

```bash
# 逐条 timing 行
grep "MUSA_KERNEL_TIMING" /tmp/musa_timing_logs/matmul.log

# summary 汇总表（如果 MUSA_TIMING_KERNEL_STATS=1）
grep -n "MUSA Kernel Debug Statistics" -A30 /tmp/musa_timing_logs/matmul.log
```

---

## 6. 验证 Release 下无计时输出

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

## 7. 提交代码与发 PR（你当前环境）

你现在在宿主机分支：`feat/timing_macro`，远端有 `origin(XFDG)` 和 `upstream(MooreThreads)`。  
建议在**宿主机 SSH 终端**执行下面命令（不是容器里）。

```bash
cd ~/workspace/tensorflow_musa_extension

# 1) 确认分支和改动
git checkout feat/timing_macro
git status -sb

# 2) （可选）先同步上游 main，再把分支 rebase 上去
git fetch upstream
git rebase upstream/main

# 3) 暂存改动（按需增删文件）
git add CMakeLists.txt build.sh
git add musa_ext/utils/logging.h
git add musa_ext/kernels/math/musa_matmul_op.cc
git add musa_ext/kernels/math/musa_conv2d_op.cc
git add musa_ext/kernels/math/musa_addn_op.cc
git add README.md README.en.md

# 4) 如果你要把临时文档一起提交，再执行：
# git add TIMING_DEBUG_SUMMARY.md TIMING_DEBUG_RUNBOOK.md

# 5) 提交
git commit -m "feat: add MUSA kernel timing macros and debug controls"

# 6) 推送到你的 fork 分支
git push origin feat/timing_macro
```

推送后创建 PR（fork -> upstream）：

- 浏览器打开：  
  `https://github.com/MooreThreads/tensorflow_musa_extension/compare/main...XFDG:feat/timing_macro`
- Base 选 `MooreThreads/tensorflow_musa_extension:main`
- Head 选 `XFDG/tensorflow_musa_extension:feat/timing_macro`

---

## 8. 常用 `.sh` 脚本怎么用

你提到的“另一个 `.sh`”，通常是 `install-hooks.sh`（用于安装 Git hooks）。

### 8.1 `install-hooks.sh`（推荐在宿主机执行一次）

```bash
cd ~/workspace/tensorflow_musa_extension
chmod +x install-hooks.sh
./install-hooks.sh
```

作用：

- 安装 `pre-commit` 和 `commit-msg` hook
- 提交时自动做格式/质量检查
- 校验 commit message 是否符合 conventional commits

### 8.2 `test/run_all_tests.sh`（跑全量测试）

```bash
cd /workspace/tensorflow_musa_extension
chmod +x test/run_all_tests.sh
./test/run_all_tests.sh
```

作用：

- 若 `build/libmusa_plugin.so` 不存在，会先触发 `./build.sh`
- 再执行 `python3 test/test_runner.py --quiet`

---

## 9. PR 前清理临时文档（如果不想带进 PR）

```bash
cd ~/workspace/tensorflow_musa_extension
rm TIMING_DEBUG_SUMMARY.md TIMING_DEBUG_RUNBOOK.md
```

---

## 10. 备注

- 本文件是精简版。  
- 详细版见：`TIMING_DEBUG_RUNBOOK.md`
