# TensorFlow MUSA Extension

TensorFlow plugin for Moore Threads MUSA GPUs: MUSA kernels and graph optimizations accelerate TensorFlow on MUSA hardware.

## Features

- MUSA implementations for core ops and common fusion paths
- Grappler-based graph optimizations (layout, fusion, optional mixed precision, etc.)
- Python package `tensorflow_musa`: plugin load and device discovery
- Optional telemetry and debugging: see [Debug guide](docs/DEBUG_GUIDE.md)

## Requirements

- CMake ≥ 3.10, Make, GCC/G++ (ABI-compatible with your target TensorFlow pip wheel; see [Compatibility](docs/COMPATIBILITY.md))
- MUSA SDK (default `/usr/local/musa`): runtime, muBLAS, muDNN
- Python ≥ 3.7
- **TensorFlow**: baseline **2.6.1**; set **`TENSORFLOW_MUSA_TARGET_TF`** to a comma-separated allowlist (e.g. `2.6.1` or `2.6.1,2.8.0`). The installed `tf.__version__` must be in that set.
- NumPy ≥ 1.19.0

Build one wheel per TensorFlow version you support. Details: [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

**PluggableDevice / `SE_InitPlugin` (experimental)**: provides the StreamExecutor C API entry and `MUSA_ENABLE_SE_PLUGIN` gating, mutually exclusive with the default C++ `MusaDevice` path; set `MUSA_ENABLE_SE_PLUGIN=1` **before** loading `libmusa_plugin.so`. Full op coverage still targets the C++ path; see [COMPATIBILITY](docs/COMPATIBILITY.md).

## Install (recommended: wheel)

```bash
git clone <repository-url>
cd tensorflow_musa_extension

pip install tensorflow==2.6.1
./build.sh wheel
pip install dist/tensorflow_musa-*.whl --no-deps
```

Use `--force-reinstall` when replacing an existing install.

## Quick check

```python
import tensorflow_musa as tf_musa

print(tf_musa.__version__)
print(tf_musa.get_musa_devices())
```

Example with a MUSA device:

```python
import tensorflow as tf
import tensorflow_musa  # ensure plugin is loaded

with tf.device("/device:MUSA:0"):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.matmul(a, a)
```

Enable or disable the MUSA custom graph optimizer:

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

config = tf.compat.v1.ConfigProto()
tf_musa.set_musa_graph_optimizer_enabled(config, enabled=True)

# To disable it:
# tf_musa.set_musa_graph_optimizer_enabled(config, enabled=False)
```

Disable selected fusion patterns from Python by passing parameters to the C++
optimizer:

```python
tf_musa.disable_musa_fusion_patterns(
    config,
    patterns=["MusaGeluFusion", "MusaLayerNormFusion"],
)

# Disable all fusion patterns
tf_musa.disable_musa_fusion_patterns(config, patterns="all")

# Clear the disabled fusion pattern list
tf_musa.clear_musa_disabled_fusion_patterns(config)
```

## Build plugin from source (optional)

Produces `build/libmusa_plugin.so` only (no wheel):

```bash
pip install tensorflow==2.6.1
./build.sh          # or ./build.sh release
```

For experiments you can `tf.load_library("./build/libmusa_plugin.so")`.

## Docs and examples

- [Debugging and environment variables](docs/DEBUG_GUIDE.md)
- More examples: [TensorFlow MUSA Playground](https://gitee.com/mthreadsacademy/tensorflow_musa_playground)

## Contributing

Issues and PRs are welcome (please add tests for new ops).

## License

Apache License 2.0

## Support

Please use repository Issues or contact the maintainers.
