# TensorFlow MUSA Extension

TensorFlow MUSA Extension 是面向摩尔线程（Moore Threads）MUSA GPU 的 TensorFlow 插件。它将 MUSA 设备注册、算子内核和图优化能力打包为 `tensorflow_musa` Python 包。

## 功能

- 将 MUSA 注册为 TensorFlow 设备。
- 提供 TensorFlow 常用算子和部分融合路径的 MUSA 实现。
- 通过 `import tensorflow_musa` 自动加载 native 运行时插件。
- 可选 OpenXLA-MUSA PJRT 后端，支持 `@tf.function(jit_compile=True)` 的 XLA 编译执行。

| 模式 | 启用方式 | 运行后端 |
|---|---|---|
| native（默认） | 不加 `--xla` | TensorFlow MUSA native kernel |
| PJRT（可选） | `--xla` | OpenXLA-MUSA PJRT |

PJRT 是显式 opt-in：不加 `--xla` 时，`tensorflow_musa` 不会导入或加载 PJRT provider，原有 native 使用方式保持不变。

## 环境要求

- Moore Threads MUSA SDK，默认安装路径为 `/usr/local/musa`。
- Python、TensorFlow 与 wheel 的 ABI 必须匹配。
- 从源码构建时，需要 CMake、GCC/G++ 和目标 TensorFlow。
- 当前 PJRT provider wheel 对应 TensorFlow 2.15.1。

生成的 wheel 需要与构建时使用的 TensorFlow 版本和 Python 环境匹配。不同 TensorFlow 环境请分别构建 wheel。

## 安装

安装已有 native wheel：

```bash
python -m pip install tensorflow==2.15.1
python -m pip install tensorflow_musa-*.whl --no-deps
```

从源码构建 native wheel：

```bash
git clone <repository-url>
cd tensorflow_musa_extension

export TENSORFLOW_MUSA_TARGET_TF=2.15.1
python -m pip install tensorflow==2.15.1
./build.sh wheel
python -m pip install --force-reinstall dist/tensorflow_musa-*.whl --no-deps
```

### 启用 OpenXLA-MUSA PJRT

需要 PJRT 时，安装与当前 TensorFlow/Python ABI 匹配的两个 wheel：

```bash
python -m pip install tensorflow_musa-*.whl --no-deps
python -m pip install openxla_musa_pjrt-*.whl --no-deps
```

运行程序时只需添加一个参数：

```bash
python your_tensorflow_program.py --xla
```

PJRT provider wheel 会自动定位并加载 PJRT plugin、TF registry bridge 和编译工具链。无需复制 `.so` 文件，也无需设置 plugin、bridge 或工具链路径。

`--xla` 必须在导入 `tensorflow_musa` 的同一进程启动参数中提供；不加该参数即使用 native。

## 快速验证

```bash
# 确认 MUSA 设备可见
python -c 'import tensorflow as tf; import tensorflow_musa; print(tf.config.list_physical_devices("MUSA"))'

# 确认 PJRT 已通过 provider wheel 自动启用
python -c \
  'import tensorflow_musa as tm; print(tm.get_pjrt_status())' \
  --xla
```

PJRT 正常启用时状态为 `mode: pjrt`、`phase: ready`、`error: None`。不加 `--xla` 时为
`mode: native`；如果 PJRT 初始化失败，状态会为 `mode: native`、`phase: fallback`，同时输出
warning 并继续 native 路径。

## 测试

在仓库根目录运行：

```bash
# native（默认）
MUSA_VISIBLE_DEVICES=0 python test/test_runner.py

# PJRT
MUSA_VISIBLE_DEVICES=0 python test/pjrt/run_pjrt_tests.py --xla
```

模型 accuracy/performance 测试见 [test/multi_domain_dnn/README.md](test/multi_domain_dnn/README.md)。

## 调试与贡献

PJRT 调试、手工路径覆盖、日志和故障排查见 [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md)。

欢迎提交 Issue 和 Pull Request；新增算子或行为变更请附带测试。

## 许可证

Apache License 2.0
