# TensorFlow MUSA Extension

面向摩尔线程（Moore Threads）MUSA GPU 的 TensorFlow 插件：通过 MUSA 内核与图优化为 TensorFlow 提供 GPU 加速。

## 特性

- 核心算子与常用融合路径的 MUSA 实现
- Grappler 图优化（布局、融合、可选混合精度等）
- Python 包 `tensorflow_musa`：自动加载插件与设备查询
- 可选遥测与调试说明见 [调试指南](docs/DEBUG_GUIDE.md)

## 环境要求

- CMake ≥ 3.10，Make，GCC/G++（与 TensorFlow 2.6.1 wheel ABI 一致）
- MUSA SDK（默认路径 `/usr/local/musa`）：Runtime、muBLAS、muDNN
- Python ≥ 3.7
- **TensorFlow == 2.6.1**（须与此版本一致）
- NumPy ≥ 1.19.0

## 安装（推荐：Wheel）

```bash
git clone <repository-url>
cd tensorflow_musa_extension

pip install tensorflow==2.6.1
./build.sh wheel
pip install dist/tensorflow_musa-*.whl --no-deps
```

重新构建后覆盖安装可加 `--force-reinstall`。

## 快速验证

```python
import tensorflow_musa as tf_musa

print(tf_musa.__version__)
print(tf_musa.get_musa_devices())
```

在计算图中使用 MUSA 设备（示例）：

```python
import tensorflow as tf
import tensorflow_musa  # 确保插件已加载

with tf.device("/device:MUSA:0"):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.matmul(a, a)
```

## 从源码构建插件（可选）

仅生成 `build/libmusa_plugin.so`（不打包 wheel）：

```bash
pip install tensorflow==2.6.1
./build.sh          # 或 ./build.sh release
```

开发时也可在 Python 中 `tf.load_library("./build/libmusa_plugin.so")` 手动加载。

## 文档与示例

- [调试与环境变量](docs/DEBUG_GUIDE.md)
- 更多示例：[TensorFlow MUSA Playground](https://gitee.com/mthreadsacademy/tensorflow_musa_playground)

## 参与贡献

欢迎提交 Issue 与 Pull Request（新算子请附带测试）。

## 许可证

Apache License 2.0

## 支持

请在仓库 Issue 中反馈问题或联系维护者。
