# TensorFlow MUSA Extension

TensorFlow MUSA Extension 是一个高性能的 TensorFlow 插件，专为摩尔线程（Moore Threads）MUSA GPU 架构设计。该扩展通过原生 MUSA 内核实现，为 TensorFlow 提供完整的 GPU 加速支持，充分发挥摩尔线程全功能 GPU 的计算性能。

## 特性

- **完整的算子支持**：涵盖深度学习训练和推理所需的核心算子
- **高性能优化**：针对 MUSA 架构进行深度优化，包括内存访问模式和计算效率
- **自动图优化**：支持 Layout 自动转换、算子融合和自动混合精度（AMP）
- **无缝集成**：与 TensorFlow 生态系统完全兼容，无需修改现有代码
- **设备管理**：完整的 MUSA 设备注册、内存管理和流式处理支持

## 快速开始

### 环境要求

- TensorFlow 2.x
- 摩尔线程 MUSA 驱动和 SDK
- CMake 3.15+
- GCC/G++ 7.0+

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

### 1. 算子配置

在 `CMakeLists.txt` 文件中配置需要编译的算子：

- **算子选择**：在源文件配置区域启用所需的算子实现
- **自定义内核**：如需使用 `.mu` 自定义内核实现，在 `set(MU_SOURCES "")` 中添加对应的源文件

### 2. 编译流程

执行自动化构建脚本：

```bash
./build.sh
```

构建脚本将自动完成以下步骤：
- 配置 CMake 项目
- 编译 MUSA 内核和主机代码
- 生成动态链接库 `libmusa_plugin.so`

### 3. 加载插件

编译成功后，在 TensorFlow 应用中加载插件：

```python
import tensorflow as tf
tf.load_library("/path/to/tensorflow_musa_extension/build/libmusa_plugin.so")
```

## 目录结构

```
tensorflow_musa_extension/
├── tensorflow_musa/          # 核心实现目录
│   ├── kernels/             # 算子实现
│   │   ├── musa_*_op.cc     # 各类算子的具体实现
│   │   └── utils_op.h       # 算子工具函数
│   ├── mu/                  # MUSA 设备核心组件
│   │   ├── device/          # 设备底层实现
│   │   │   ├── musa_device.h    # 设备注册和管理
│   │   │   ├── musa_allocator.h # 内存分配器
│   │   │   ├── musa_memcpy.h    # 内存拷贝操作
│   │   │   └── musa_stream.h    # 流式处理支持
│   │   ├── optimizer/       # 图优化模块
│   │   │   ├── musa_layout_optimizer.cc  # Layout 自动优化
│   │   │   ├── musa_remapper_optimizer.cc # 算子融合优化
│   │   │   └── musa_amp_optimizer.cc     # 自动混合精度优化
│   │   ├── device_register.cc   # 设备注册逻辑
│   │   └── kernel_register.cc   # 算子注册逻辑
│   └── utils/               # 工具函数
│       └── logging.h        # 日志工具
├── test/                    # 功能测试脚本
│   └── test_*.py            # 各算子的单元测试
├── build.sh                 # 自动化构建脚本
└── README.md                # 项目文档
```

## 测试

构建完成后，运行测试套件验证功能正确性。测试文件遵循 TensorFlow 官方 `python/kernel_tests` 风格，使用 `tf.test.TestCase` 作为基类。

```bash
# 运行特定算子测试
python test/add_op_test.py
python test/matmul_op_test.py

# 运行所有测试
./test/run_all_tests.sh

# 或者单独运行每个测试
for test_file in test/*_op_test.py; do
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
