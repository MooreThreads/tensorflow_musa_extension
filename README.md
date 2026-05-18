# TensorFlow MUSA Extension

TensorFlow MUSA Extension 是面向摩尔线程（Moore Threads）MUSA GPU 的 TensorFlow 插件。它将 MUSA 设备注册、算子内核和图优化能力打包为 `tensorflow_musa` Python 包。

## 主要特性

- 将 MUSA 注册为 TensorFlow 设备。
- 提供 TensorFlow 常用算子和部分融合路径的 MUSA 实现。
- 通过 `import tensorflow_musa` 自动加载运行时插件。
- 调试和环境变量说明见 [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md)。

## 环境要求

- Moore Threads MUSA SDK，默认安装路径为 `/usr/local/musa`。
- CMake 3.10 或更新版本。
- 与目标 TensorFlow wheel ABI 兼容的 GCC/G++。
- Python 3.7 或更新版本。
- 构建前需先安装 TensorFlow。
- NumPy 1.19.0 或更新版本。

生成的 wheel 需要与构建时使用的 TensorFlow 版本和 Python 环境匹配。不同 TensorFlow 环境请分别构建 wheel。

## 构建与安装

```bash
git clone <repository-url>
cd tensorflow_musa_extension

pip install tensorflow==2.6.1
./build.sh wheel
pip install --force-reinstall dist/tensorflow_musa-*.whl --no-deps
```

开发时如只需构建插件动态库：

```bash
./build.sh
```

推荐通过 wheel 安装后验证，这与用户实际加载包的方式一致。

## 快速验证

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

print(tf.__version__)
print(tf.config.list_physical_devices("MUSA"))
```

在 MUSA 设备上运行一个简单算子：

```python
import tensorflow as tf
import tensorflow_musa

with tf.device("/device:MUSA:0"):
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.matmul(x, x)

print(y)
```

## 测试

从已安装的 wheel 运行算子测试：

```bash
MUSA_VISIBLE_DEVICES=5 python test/test_runner.py
```

运行单个算子测试：

```bash
MUSA_VISIBLE_DEVICES=5 python test/test_runner.py --single matmul_op_test.py --detail
```

## 调试

日志、GraphDef dump、遥测和运行时调试环境变量请参考 [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md)。

## 参与贡献

欢迎提交 Issue 和 Pull Request。新增算子或行为变更请附带测试。

## 许可证

Apache License 2.0
