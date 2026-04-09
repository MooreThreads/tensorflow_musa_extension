# TensorFlow MUSA Extension

TensorFlow MUSA Extension 是一个高性能的 TensorFlow 插件，专为摩尔线程（Moore Threads）MUSA GPU 架构设计。该扩展通过原生 MUSA 内核实现，为 TensorFlow 提供完整的 GPU 加速支持，充分发挥摩尔线程全功能 GPU 的计算性能。

## 特性

- **完整的算子支持**：涵盖深度学习训练和推理所需的核心算子 
- **高性能优化**：针对 MUSA 架构进行深度优化，包括内存访问模式和计算效率
- **自动图优化**：支持 Layout 自动转换、算子融合和自动混合精度（AMP）
- **无缝集成**：与 TensorFlow 生态系统完全兼容，无需修改现有代码
- **设备管理**：完整的 MUSA 设备注册、内存管理和流式处理支持
- **Kernel 调试支持**：Debug 模式下输出算子类型、输入类型和输入 shape，并支持终端颜色高亮

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
| **Debug** | `./build.sh debug` | 开启 `MUSA_KERNEL_DEBUG`，输出 Kernel 调试日志 |

### 2. 编译流程

执行自动化构建脚本：

```bash
# Release（默认）
./build.sh

# Release（显式）
./build.sh release

# Debug（Kernel 调试日志）
./build.sh debug
```

构建脚本将自动完成以下步骤：
- 配置 CMake 项目
- 编译 MUSA 内核和主机代码
- 生成动态链接库 `libmusa_plugin.so`

### 3. 调试与诊断

详细的调试指南请参阅 [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md)。当前 README 中与 Kernel 调试相关的说明已经更新为新的 debug 日志方案；旧的 timing 宏路径已移除。

- **Kernel 调试日志**：通过 `MUSA_DEBUG_LOG_KERNEL(ctx)` 输出 `op_type`、`input_types`、`input_shapes`
- **遥测系统（Telemetry）**：全链路追踪和脏数据诊断
- **内存诊断**：Use-After-Free 检测和内存染色
- **环境变量**：完整的环境变量配置表

快速启用遥测系统进行诊断：

```bash
export MUSA_TELEMETRY_ENABLED=1
export MUSA_TELEMETRY_LOG_PATH=/tmp/telemetry.json
python test_runner.py
```

### 4. 本次 Kernel Debug 改动说明

本次调试链路做了如下调整：

- 新增统一调试宏 `MUSA_DEBUG_LOG_KERNEL(ctx)`，在算子 `Compute()` 开始时打印详细调试信息，并在 `Compute()` 退出时自动打印极简结束提示
- 公共格式化和输出逻辑集中在 `musa_ext/kernels/utils_op.h` 与 `musa_ext/kernels/utils_op.cc`
- 当前 `musa_ext/kernels` 下所有已接入 `MUSA_DEBUG_LOG_KERNEL(ctx)` 的算子 `Compute()` 入口都会输出这套日志
- 旧的 timing 宏已经删除，不再使用 `MUSA_KERNEL_TIMING_GUARD`、`MUSA_KERNEL_TRACE_START`、`MUSA_KERNEL_TRACE_END`、`MUSA_KERNEL_TRACE`、`MUSA_PROFILE_OP`

新的日志格式如下：

```txt
[MUSA_KERNEL_DEBUG] op_type=AddV2 input_types=[float, float] input_shapes=[[1024,1024], [1024,1024]]
[MUSA_KERNEL_DEBUG] END AddV2
```

其中：

- 开始日志保留 `op_type`、`input_types`、`input_shapes`
- 结束日志只保留极简提示，成功时打印 `END <OpName>`，失败时打印 `FAIL <OpName>`
- `input_types` 默认使用青色高亮
- `input_shapes` 默认使用黄色高亮
- `END` 日志默认使用亮绿色高亮，`FAIL` 日志默认使用亮红色高亮
- 当输出被重定向到文件时，默认保持纯文本，避免 ANSI 颜色码污染日志
- 如需在 `tee` 或重定向场景中强制保留颜色，可设置 `MUSA_KERNEL_DEBUG_COLOR=1`
- 如需显式关闭颜色，可设置 `NO_COLOR=1`

快速启用新的 Kernel 调试日志：

方式一：在仓库根目录运行。

```bash
./build.sh debug
export PYTHONPATH=$PWD/test
python3 test/ops/add_op_test.py 2>&1 | tee /tmp/tme_add.log
grep 'MUSA_KERNEL_DEBUG' /tmp/tme_add.log
```

方式二：直接进入 `test/` 目录运行，这种方式不需要设置 `PYTHONPATH`。

```bash
./build.sh debug
cd test
python3 -m ops.add_op_test 2>&1 | tee /tmp/tme_add.log
grep 'MUSA_KERNEL_DEBUG' /tmp/tme_add.log
```

如需强制保留颜色输出：

```bash
cd test
MUSA_KERNEL_DEBUG_COLOR=1 python3 -m ops.add_op_test
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

## 使用示例

详细使用示例见：

[![MUSA Playground](https://img.shields.io/badge/Gitee-TensorFlow_MUSA_Playground-C71D23?style=for-the-badge&logo=gitee&logoColor=white)](https://gitee.com/mthreadsacademy/tensorflow_musa_playground)

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
