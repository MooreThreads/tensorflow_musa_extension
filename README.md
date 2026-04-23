# TensorFlow MUSA Extension

[English](README.en.md) | 中文

TensorFlow MUSA Extension 是摩尔线程（Moore Threads）MUSA GPU 的 TensorFlow 插件，通过 **PluggableDevice C API** 把 MUSA 设备注册为 TensorFlow 的物理设备，并提供原生 kernel、图优化与自研内存分配器实现。

## 特性

- **PluggableDevice 集成** — 兼容 `tensorflow>=2.6, <2.17`，无需改动业务代码即可使用 MUSA 设备。
- **完整算子与图优化** — 训练 / 推理常见算子，支持 Layout 自动转换、算子融合与 AMP。
- **自研内存分配器** — Host / Device 双路缓存分配器，可选 VMM Expandable Segments；提供 OOM 结构化诊断、per-process 内存上限、H2D staging 池等运行时控制。
- **Python 诊断接口** — `tensorflow_musa.memory` / `tensorflow_musa.device` 暴露 `memory_stats`、`memory_snapshot`、`empty_cache`、Peer-Access 等 API 用于线上监控与排障。

## 安装

### 环境要求

| 组件 | 版本 |
|------|------|
| Python | ≥ 3.7 |
| TensorFlow | ≥ 2.6，< 2.17（主测试：`2.6.1`） |
| NumPy | ≥ 1.19 |
| CMake | ≥ 3.10 |
| MUSA SDK | Runtime ≥ 1.0，含 muBLAS / muDNN；默认路径 `/usr/local/musa` |

### 安装 wheel（推荐）

```bash
pip install tensorflow==2.6.1
./build.sh wheel
pip install dist/tensorflow_musa-0.1.0-py3-none-any.whl --no-deps
```

### 开发模式

```bash
./build.sh release

# 在 Python 里加载插件
python -c "
import tensorflow as tf
from tensorflow.python.framework import load_library
load_library.load_pluggable_device_library('./build/libmusa_plugin.so')
print(tf.config.list_physical_devices('MUSA'))
"
```

也可将 `libmusa_plugin.so` 放入 `<site-packages>/tensorflow/tensorflow-plugins/`，TensorFlow 启动时自动加载，无需显式调用。

## 构建

| 模式 | 命令 | 说明 |
|------|------|------|
| Release | `./build.sh` 或 `./build.sh release` | 生成 `build/libmusa_plugin.so` |
| Debug | `./build.sh debug` | 启用 `MUSA_KERNEL_DEBUG` 与 kernel 计时 |
| Wheel | `./build.sh wheel` | 生成 `dist/tensorflow_musa-*.whl` |

构建脚本会自动检查 TensorFlow 版本、配置 CMake、编译 MUSA kernel 与主机代码。

## 快速开始

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

print(tf_musa.__version__)
print(tf_musa.get_musa_devices())

with tf.device("/device:MUSA:0"):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    print(tf.matmul(a, b))
```

## 内存管理与诊断

```python
import tensorflow_musa as tf_musa

tf_musa.memory.memory_allocated()     # 当前使用字节
tf_musa.memory.memory_reserved()      # reserved（含缓存）
tf_musa.memory.memory_stats()         # 完整计数器
tf_musa.memory.memory_snapshot()      # 合成快照（stats + segments + driver + config）
tf_musa.memory.empty_cache()          # 释放空闲段回驱动
tf_musa.memory.set_per_process_memory_fraction(0.5)

if tf_musa.device.can_access_peer(0, 1):
    tf_musa.device.enable_peer_access(0, 1)
```

运行时行为（分配器后端、staging 池、VMM 扩展段、OOM 诊断等）通过环境变量控制，完整清单见 [`docs/environment-variables.md`](docs/environment-variables.md)。

## 测试

```bash
cd test
python test_runner.py                       # 所有算子测试
python test_runner.py --fusion              # 融合测试
python test_runner.py --single ops/matmul_op_test.py
```

测试按类别组织在 `test/ops/`（算子级）与 `test/fusion/`（端到端）下，用例继承 `MUSATestCase` 自动完成插件加载。

## 基准

```bash
python benchmark/bench_h2d.py                # H2D / D2H 吞吐扫频
python benchmark/bench_resnet.py             # ResNet 类训练步
python benchmark/bench_alloc_churn.py        # 分配器压力 + 不变量断言
```

## 支持的算子

- 基础：Add / Sub / Multiply / RealDiv / Maximum / Minimum
- 激活：Relu / Sigmoid / Softmax / Erf
- 矩阵：MatMul / FusedMatMul / Transpose
- 数据：Reshape / Concat / Gather / StridedSlice / ExpandDims
- 归一化：LayerNorm / FusedBatchNorm
- 其他：TensorInteraction / BiasAdd / Assign
- 优化器：ResourceApplyAdam / MusaResourceSparseApplyAdam（支持 embedding 稀疏更新）

## 目录结构

```
tensorflow_musa_extension/
├── build.sh / CMakeLists.txt / setup.py    构建入口
├── python/                                 Python 包（pip 名：tensorflow_musa）
├── musa_ext/
│   ├── kernels/                            MUSA kernel 实现
│   ├── mu/                                 设备注册、分配器、优化器
│   └── python/                             _C 扩展源码
├── benchmark/                              端到端基准
├── docs/                                   面向用户的文档
└── test/                                   测试用例
```

## 更多示例

[![MUSA Playground](https://img.shields.io/badge/Gitee-TensorFlow_MUSA_Playground-C71D23?style=for-the-badge&logo=gitee&logoColor=white)](https://gitee.com/mthreadsacademy/tensorflow_musa_playground)

## 贡献

欢迎提交新算子实现或性能优化：Fork → feature 分支 → 添加测试 → 提交 Pull Request。

## 许可证

本项目遵循 Apache License 2.0。
