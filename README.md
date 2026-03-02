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
    ├── musa_test_utils.py  # 测试工具
    └── *_test.py           # 各算子测试文件
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
  - TensorFlow: == 2.4.4
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

# 构建插件（Release 模式，默认）
./build.sh

# 或构建 Debug 模式（启用 Kernel 计时）
./build.sh debug

# 在 Python 中加载插件
import tensorflow as tf
tf.load_library("./build/libmusa_plugin.so")
```

## 构建指南

### 1. 编译模式选择

支持两种编译模式：

| 模式 | 命令 | 说明 |
|------|------|------|
| **Release** | `./build.sh` 或 `./build.sh release` | 优化性能，无调试开销 |
| **Debug** | `./build.sh debug` | 启用 Kernel 计时，便于性能分析 |

### 2. 算子配置

在 `CMakeLists.txt` 文件中配置需要编译的算子：

- **算子选择**：在源文件配置区域启用所需的算子实现
- **自定义内核**：如需使用 `.mu` 自定义内核实现，在 `set(MU_SOURCES "")` 中添加对应的源文件

### 3. 编译流程

执行自动化构建脚本：

```bash
# Release 模式（默认，用于生产环境）
./build.sh

# Debug 模式（用于开发和性能分析）
./build.sh debug
```

构建脚本将自动完成以下步骤：
- 配置 CMake 项目
- 编译 MUSA 内核和主机代码
- 生成动态链接库 `libmusa_plugin.so`

### 4. 加载插件

编译成功后，在 TensorFlow 应用中加载插件：

```python
import tensorflow as tf
tf.load_library("/path/to/tensorflow_musa_extension/build/libmusa_plugin.so")
```

## 环境变量

### Kernel 调试环境变量（仅 Debug 模式）

在 Debug 模式下编译后，可通过以下环境变量控制 Kernel 调试输出：

| 环境变量 | 取值 | 说明 |
|---------|------|------|
| `MUSA_KERNEL_DEBUG` | `0` | 禁用 Kernel 计时（默认） |
| | `1` | 启用基本 Kernel 计时日志 |
| | `2` | 启用详细计时（包含输入 Shape 信息） |
| `MUSA_KERNEL_DEBUG_STATS` | `0` | 禁用统计聚合（默认） |
| | `1` | 启用统计聚合，程序退出时输出汇总 |

### 使用示例

```bash
# 基本 Kernel 计时
MUSA_KERNEL_DEBUG=1 python your_script.py

# 详细计时（显示输入 Shape）
MUSA_KERNEL_DEBUG=2 python your_script.py

# 启用统计汇总
MUSA_KERNEL_DEBUG=2 MUSA_KERNEL_DEBUG_STATS=1 python your_script.py
```

### 输出示例

```
[MUSA_KERNEL] Debug level set to 2 (from MUSA_KERNEL_DEBUG=2)
[MUSA_KERNEL] Statistics aggregation enabled
[MUSA_KERNEL] GatherV2[[10000,256],[1000]] took 0.234 ms
[MUSA_KERNEL] MatMul[[1024,1024],[1024,1024]] took 2.345 ms
...
====================================================================================================
MUSA Kernel Debug Statistics
====================================================================================================
Kernel Name                              Count       Total(ms)    Avg(ms)      Min(ms)      Max(ms)
----------------------------------------------------------------------------------------------------
GatherV2[[10000,256],[1000]]             150         34.567       0.230        0.198        0.456
MatMul[[1024,1024],[1024,1024]]          150         345.234      2.301        1.890        3.456
====================================================================================================
```

## 测试

构建完成后，运行测试套件验证功能正确性。测试文件遵循 TensorFlow 官方 `python/kernel_tests` 风格，使用 `tf.test.TestCase` 作为基类。

```bash
# 运行特定算子测试
python test/ops/add_op_test.py
python test/ops/matmul_op_test.py

# 运行所有测试
./test/run_all_tests.sh

# 或者单独运行每个测试
for test_file in test/ops/*_op_test.py; do
    python "$test_file"
done
```

测试文件命名规范：
- 使用 `op_name_op_test.py` 格式
- 继承自 `tf.test.TestCase`
- 测试方法以 `test_` 开头
- 使用 `self.assert*` 系列方法进行断言

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
