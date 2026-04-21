# TensorFlow MUSA Extension

TensorFlow MUSA Extension 是一个高性能的 TensorFlow 插件，专为摩尔线程（Moore Threads）MUSA GPU 架构设计。该扩展通过原生 MUSA 内核实现，为 TensorFlow 提供完整的 GPU 加速支持，充分发挥摩尔线程全功能 GPU 的计算性能。

## 特性

- **完整的算子支持**：涵盖深度学习训练和推理所需的核心算子
- **高性能优化**：针对 MUSA 架构进行深度优化，包括内存访问模式和计算效率
- **自动图优化**：支持 Layout 自动转换、算子融合和自动混合精度（AMP）
- **无缝集成**：与 TensorFlow 生态系统完全兼容，无需修改现有代码
- **设备管理**：完整的 MUSA 设备注册、内存管理和流式处理支持
- **Kernel 调试支持**：内置 Kernel 执行时间统计功能，便于性能分析
- **Python 包支持**：提供 `tensorflow_musa` Python 包，支持 pip 安装和优化器接口

## 快速开始

### 目录结构

```
tensorflow_musa_extension/
├── CMakeLists.txt          # CMake 构建配置
├── build.sh                # 构建脚本（支持 release/debug/wheel）
├── setup.py                # Python 包构建配置
├── .clang-format           # 代码格式化配置
├── .pre-commit-config.yaml # pre-commit 钩子配置
├── .github/                # CI/CD 配置
├── python/                 # Python 包源码目录（pip 安装名为 tensorflow_musa）
│   ├── __init__.py         # 包入口，自动加载插件
│   ├── _loader.py          # 插件加载与设备查询工具
│   └── libmusa_plugin.so   # 打包进 wheel 的 MUSA 插件动态库
├── musa_ext/               # 核心源码目录
│   ├── kernels/            # MUSA 内核实现（.mu 文件）
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
  - TensorFlow: == 2.6.1（必须使用此版本）
  - NumPy: >= 1.19.0
- **开发工具**:
  - pre-commit >= 3.0.0
  - pytest >= 6.0.0

### 安装方式

#### 方式一：安装 WHL 包（推荐）

```bash
# 克隆仓库
git clone <repository-url>
cd tensorflow_musa_extension

# 确保 TensorFlow 2.6.1 已安装
pip install tensorflow==2.6.1

# 构建 WHL 包（一键构建）
./build.sh wheel

# 安装 WHL 包
pip install dist/tensorflow_musa-0.1.0-py3-none-any.whl --no-deps

# 重新构建后安装 WHL 包
pip install dist/tensorflow_musa-0.1.0-py3-none-any.whl --no-deps --force-reinstall
```

#### 方式二：开发模式

```bash
# 克隆仓库
git clone <repository-url>
cd tensorflow_musa_extension

# 构建 plugin
./build.sh release

# 在 Python 中手动加载插件进行测试
import tensorflow as tf
tf.load_library("./build/libmusa_plugin.so")
```

## 构建指南

### 1. 编译模式

支持三种构建模式：

| 模式 | 命令 | 说明 |
|------|------|------|
| **Release** | `./build.sh` 或 `./build.sh release` | 优化性能，生成 `build/libmusa_plugin.so` |
| **Debug** | `./build.sh debug` | 开启 `MUSA_KERNEL_DEBUG`，启用 kernel timing 宏 |
| **Wheel** | `./build.sh wheel` | 一键构建 WHL 包，生成 `dist/tensorflow_musa-*.whl` |

### 2. 编译流程

```bash
# Release（默认）- 仅构建 plugin
./build.sh

# Debug（计时调试）
./build.sh debug

# Wheel（构建发布包）
./build.sh wheel
```

构建脚本将自动完成以下步骤：
- 检查 TensorFlow 版本（必须为 2.6.1）
- 配置 CMake 项目
- 编译 MUSA 内核和主机代码
- 生成动态链接库 `libmusa_plugin.so` 或 WHL 包

### 3. WHL 包说明

WHL 包构建特点：
- **不自动下载 TensorFlow**：避免 pip 自动下载不兼容版本
- **版本检查**：构建前自动检查环境中 TensorFlow 版本是否为 2.6.1
- **包名映射**：源码目录为 `python/`，安装后包名为 `tensorflow_musa`

安装后使用：
```python
import tensorflow_musa as tf_musa  # 包名仍然是 tensorflow_musa
```

### 4. 调试与诊断

详细的调试指南请参阅 [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md)，包含：

- **Kernel 计时**：Debug 模式下的性能分析方法
- **遥测系统（Telemetry）**：全链路追踪和脏数据诊断
- **内存诊断**：Use-After-Free 检测和内存染色
- **环境变量**：完整的环境变量配置表

快速启用遥测系统进行诊断：

```bash
export MUSA_TELEMETRY_ENABLED=1
export MUSA_TELEMETRY_LOG_PATH=/tmp/telemetry.json
python test_runner.py
```

快速启用 Kernel 计时分析：

```bash
./build.sh debug
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1
python test_runner.py
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

### 基本用法

安装 `tensorflow_musa` 包后，导入时插件会自动加载：

```python
import tensorflow_musa as tf_musa

# 查看版本
print(f"TensorFlow MUSA 版本: {tf_musa.__version__}")

# 查看可用的 MUSA 设备
devices = tf_musa.get_musa_devices()
print(f"可用 MUSA 设备: {devices}")
```

### 设备管理

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

# 设置使用特定 MUSA 设备
with tf.device('/device:MUSA:0'):
    # 在 MUSA 设备上创建张量和计算
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print(c)
```

### 更多示例

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
