# TensorFlow MUSA Extension

TensorFlow MUSA Extension provides TensorFlow support for Moore Threads MUSA GPUs. It packages MUSA device registration, operator kernels, and graph optimization as the `tensorflow_musa` Python package.

## Features

- Registers MUSA as a TensorFlow device.
- Provides MUSA implementations for common TensorFlow operators and selected fusion paths.
- Loads the runtime plugin with `import tensorflow_musa`.
- Debugging and environment-variable guidance is available in [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md).

## Requirements

- Moore Threads MUSA SDK, installed at `/usr/local/musa` by default.
- CMake 3.10 or newer.
- GCC/G++ compatible with the target TensorFlow wheel ABI.
- Python 3.7 or newer.
- TensorFlow installed before building this package.
- NumPy 1.19.0 or newer.

The built wheel must match the TensorFlow version and Python environment used at build time. Build separate wheels for separate TensorFlow environments.

## Build and install

```bash
git clone <repository-url>
cd tensorflow_musa_extension

pip install tensorflow==2.6.1
./build.sh wheel
pip install --force-reinstall dist/tensorflow_musa-*.whl --no-deps
```

To build only the plugin shared library during development:

```bash
./build.sh
```

The wheel installation path is recommended for validation because it matches how users load the package.

## Quick check

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

print(tf.__version__)
print(tf.config.list_physical_devices("MUSA"))
```

Run a simple operation on MUSA:

```python
import tensorflow as tf
import tensorflow_musa

with tf.device("/device:MUSA:0"):
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.matmul(x, x)

print(y)
```

## Tests

Run the operator test suite from an installed wheel:

```bash
MUSA_VISIBLE_DEVICES=5 python test/test_runner.py
```

Run a single operator test:

```bash
MUSA_VISIBLE_DEVICES=5 python test/test_runner.py --single matmul_op_test.py --detail
```

## Debugging

See [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md) for logging, GraphDef dumps, telemetry, and runtime debugging environment variables.

## Contributing

Issues and pull requests are welcome. Please include tests for new operators or behavior changes.

## License

Apache License 2.0
