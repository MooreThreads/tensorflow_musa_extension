# TensorFlow MUSA Extension

TensorFlow MUSA Extension 是一个高性能的 TensorFlow 插件，专为摩尔线程（Moore Threads）MUSA GPU 架构设计。该扩展通过原生 MUSA 内核实现，为 TensorFlow 提供完整的 GPU 加速支持，充分发挥摩尔线程全功能 GPU 的计算性能。

## 特性

- **完整的算子支持**：涵盖深度学习训练和推理所需的核心算子
- **高性能优化**：针对 MUSA 架构进行深度优化，包括内存访问模式和计算效率
- **自动图优化**：支持 Layout 自动转换、算子融合和自动混合精度（AMP）
- **无缝集成**：与 TensorFlow 生态系统完全兼容，无需修改现有代码
- **设备管理**：完整的 MUSA 设备注册、内存管理和流式处理支持
- **Kernel 调试支持**：内置 Kernel 执行时间统计功能，便于性能分析

## 快速开始

### 目录结构

```
tensorflow_musa_extension/
├── CMakeLists.txt          # CMake 构建配置
├── build.sh                # 构建脚本
├── .clang-format           # 代码格式化配置
├── .pre-commit-config.yaml # pre-commit 钩子配置
├── .gitlab-ci.yml          # CI/CD 配置
├── musa_ext/               # 核心源码目录
│   ├── kernels/            # MUSA 内核实现
│   ├── mu/                 # MUSA 设备和优化器实现
│   └── utils/              # 工具函数
└── test/                   # 测试用例
    ├── musa_test_utils.py  # 测试工具基类
    ├── test_runner.py      # 测试运行器
    ├── ops/                # 算子测试
    └── fusion/             # 融合测试（e2e）
```

### 环境要求

- **构建工具**:
  - CMake (版本 >= 3.10)
  - Make
- **MUSA SDK**:
  - MUSA Runtime (>= 1.0)
  - muBLAS 库
  - muDNN 库
  - 默认安装路径: `/usr/local/musa`
- **Python 依赖**
  - Python: >= 3.7
  - TensorFlow: == 2.6.1
  - protobuf: == 3.20.3
  - NumPy: >= 1.19.0
  - pettytable: >= 3.0.0
- **开发工具**:
  - pre-commit >= 3.0.0
  - pytest >= 6.0.0

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd tensorflow_musa_extension

# 构建插件
./build.sh

# 在 Python 中加载插件
import tensorflow as tf
tf.load_library("./build/libmusa_plugin.so")
```

## 构建指南

### 1. 编译模式

支持 Release 与 Debug 两种模式：

| 模式 | 命令 | 说明 |
|------|------|------|
| **Release** | `./build.sh` 或 `./build.sh release` | 优化性能，无调试开销 |
| **Debug** | `./build.sh debug` | 开启 `MUSA_KERNEL_DEBUG`，启用 kernel timing 宏 |

### 2. 编译流程

执行自动化构建脚本：

```bash
# Release（默认）
./build.sh

# Release（显式）
./build.sh release

# Debug（计时调试）
./build.sh debug
```

构建脚本将自动完成以下步骤：
- 配置 CMake 项目
- 编译 MUSA 内核和主机代码
- 生成动态链接库 `libmusa_plugin.so`

### 3. Kernel 计时（Debug 模式）

仅在 `./build.sh debug` 构建下生效（`MUSA_KERNEL_DEBUG=ON`）：

运行时环境变量请见下方 [环境变量](#环境变量) 章节中的“日志调试”表格。

#### 3.1 宏使用方式

```cpp
// 基础 guard
MUSA_KERNEL_TIMING_GUARD(ctx);

// 分段埋点
MUSA_KERNEL_TRACE_START("Mem Alloc");
// ... code block ...
MUSA_KERNEL_TRACE_END("Mem Alloc");

MUSA_KERNEL_TRACE_START("Kernel");
// ... kernel launch ...
MUSA_KERNEL_TRACE_END("Kernel");

// 自定义阶段名
MUSA_KERNEL_TRACE_START("State1");
// ... allocate / pre-process ...
MUSA_KERNEL_TRACE_END("State1");

MUSA_KERNEL_TRACE_START("State2");
// ... main kernel ...
MUSA_KERNEL_TRACE_END("State2");
```

### 4. 常用验证命令（MatMul）

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

## 环境变量

### 功能控制

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `MUSA_ENABLE_TF32` | 启用 TF32 加速 MatMul/Conv | `export MUSA_ENABLE_TF32=1` |
| `MUSA_DUMP_GRAPHDEF` | 启用图优化调试，dump GraphDef | `export MUSA_DUMP_GRAPHDEF=1` |
| `MUSA_DUMP_GRAPHDEF_DIR` | 指定 GraphDef dump 目录 | `export MUSA_DUMP_GRAPHDEF_DIR=/tmp/graphs` |

### 日志调试

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `MUSA_TIMING_KERNEL_LEVEL` | Timing 模式控制（`1`=仅总耗时，`2`=总耗时+分段耗时） | `export MUSA_TIMING_KERNEL_LEVEL=2` |
| `MUSA_TIMING_KERNEL_NAME` | 仅打印指定 Kernel（大小写不敏感子串匹配，`ALL` 为全部） | `export MUSA_TIMING_KERNEL_NAME=MatMul` |
| `MUSA_TIMING_KERNEL_STATS` | 进程退出时打印 timing 汇总（`1`=开启，`0`=关闭） | `export MUSA_TIMING_KERNEL_STATS=1` |
| `TF_CPP_MIN_LOG_LEVEL` | 全局日志级别（0=INFO, 1=WARNING, 2=ERROR） | `export TF_CPP_MIN_LOG_LEVEL=1` |
| `TF_CPP_VMODULE` | 精确控制特定文件的 VLOG 级别 | `export TF_CPP_VMODULE="musa_graph_optimizer=1,layernorm_fusion=2"` |

**常用调试组合：**

```bash
# 1. 查看图优化器的详细日志
export TF_CPP_VMODULE="musa_graph_optimizer=1,fusion_pattern_manager=1"
python -m fusion.layernorm_gelu_fusion_test

# 2. 查看算子融合详细过程
export TF_CPP_VMODULE="layernorm_fusion=2,gelu_fusion=1"
python -m ops.layernorm_op_test

# 3. 完全静音（只显示错误）
export TF_CPP_MIN_LOG_LEVEL=2
python test_runner.py

# 4. 恢复默认日志
unset TF_CPP_MIN_LOG_LEVEL TF_CPP_VMODULE
```

## 测试

构建完成后，运行测试套件验证功能正确性。测试分为**算子测试**（`test/ops/`）和**融合测试**（`test/fusion/`）两类。

### 运行单个测试

```bash
cd test

# 运行特定算子测试
python -m ops.add_op_test
python -m ops.matmul_op_test

# 运行融合测试
python -m fusion.layernorm_gelu_fusion_test
```

### 使用测试运行器

```bash
cd test

# 运行所有算子测试（默认）
python test_runner.py

# 运行所有融合测试
python test_runner.py --fusion

# 运行单个测试文件
python test_runner.py --single ops/matmul_op_test.py
python test_runner.py --single fusion/layernorm_gelu_fusion_test.py

# 详细模式（显示每个测试的详细输出）
python test_runner.py --detail

# 安静模式（只显示进度条和摘要）
python test_runner.py --quiet
```

### 测试文件命名规范

**算子测试**（`test/ops/`）：
- 使用 `op_name_op_test.py` 格式
- 继承自 `MUSATestCase`（封装了插件加载）
- 测试方法以 `test_` 开头

**融合测试**（`test/fusion/`）：
- 使用 `*_fusion_test.py` 格式
- 继承自 `MUSATestCase`
- 测试端到端的图优化和算子融合

## 支持的算子

当前版本支持以下核心算子：
- **基础运算**：Add, Sub, Multiply, RealDiv, Maximum, Minimum
- **激活函数**：Relu, Sigmoid, Softmax, Erf
- **矩阵运算**：MatMul, FusedMatMul, Transpose
- **数据操作**：Reshape, Concat, Gather, StridedSlice, ExpandDims
- **归一化**：LayerNorm, FusedBatchNorm
- **特殊算子**：TensorInteraction, BiasAdd, Assign

## 贡献指南

欢迎贡献新的算子实现或优化！贡献流程：

1. Fork 仓库并创建特性分支
2. 实现算子或优化功能
3. 添加相应的测试用例
4. 更新文档（如需要）
5. 提交 Pull Request

## 许可证

本项目遵循 Apache 2.0 开源协议。

## 技术支持

如遇问题，请提交 Issue 或联系项目维护者。
