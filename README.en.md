# TensorFlow MUSA Extension

English | [中文](README.md)

TensorFlow MUSA Extension is a TensorFlow plugin for Moore Threads MUSA GPUs. It registers MUSA as a physical device through the **PluggableDevice C API** and ships native kernels, graph optimizations and a custom memory allocator.

## Features

- **PluggableDevice integration** — compatible with `tensorflow>=2.6, <2.17`; no application code changes required.
- **Operators and graph optimizations** — core ops for training and inference, plus automatic layout conversion, operator fusion and AMP.
- **Custom memory allocators** — host- and device-side caching allocators with optional VMM expandable segments; structured OOM diagnostics, per-process memory caps and an H2D staging pool are exposed as runtime knobs.
- **Python diagnostic API** — `tensorflow_musa.memory` / `tensorflow_musa.device` expose `memory_stats`, `memory_snapshot`, `empty_cache`, peer-access helpers and more for monitoring and debugging.

## Installation

### Prerequisites

| Component | Version |
|-----------|---------|
| Python | ≥ 3.7 |
| TensorFlow | ≥ 2.6, < 2.17 (primary tested: `2.6.1`) |
| NumPy | ≥ 1.19 |
| CMake | ≥ 3.10 |
| MUSA SDK | Runtime ≥ 1.0, including muBLAS / muDNN; default path `/usr/local/musa` |

### Install the wheel (recommended)

```bash
pip install tensorflow==2.6.1
./build.sh wheel
pip install dist/tensorflow_musa-0.1.0-py3-none-any.whl --no-deps
```

### Development mode

```bash
./build.sh release

# Load the plugin in Python
python -c "
import tensorflow as tf
from tensorflow.python.framework import load_library
load_library.load_pluggable_device_library('./build/libmusa_plugin.so')
print(tf.config.list_physical_devices('MUSA'))
"
```

Alternatively, copy `libmusa_plugin.so` into `<site-packages>/tensorflow/tensorflow-plugins/`; TensorFlow auto-loads every shared library in that directory at import time.

## Build

| Mode | Command | Description |
|------|---------|-------------|
| Release | `./build.sh` or `./build.sh release` | produces `build/libmusa_plugin.so` |
| Debug | `./build.sh debug` | enables `MUSA_KERNEL_DEBUG` and kernel timing |
| Wheel | `./build.sh wheel` | produces `dist/tensorflow_musa-*.whl` |

The build script automatically validates the TensorFlow version, configures CMake, and compiles MUSA kernels along with host code.

## Quick start

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

## Memory management and diagnostics

```python
import tensorflow_musa as tf_musa

tf_musa.memory.memory_allocated()     # bytes currently in use
tf_musa.memory.memory_reserved()      # reserved (including cached)
tf_musa.memory.memory_stats()         # full counter dictionary
tf_musa.memory.memory_snapshot()      # composite snapshot (stats + segments + driver + config)
tf_musa.memory.empty_cache()          # release idle segments back to the driver
tf_musa.memory.set_per_process_memory_fraction(0.5)

if tf_musa.device.can_access_peer(0, 1):
    tf_musa.device.enable_peer_access(0, 1)
```

Runtime behaviour (allocator backend, staging pool, VMM expandable segments, OOM diagnostics, and so on) is driven by environment variables. See [`docs/environment-variables.md`](docs/environment-variables.md) for the complete reference.

## Testing

```bash
cd test
python test_runner.py                       # all operator tests
python test_runner.py --fusion              # fusion tests
python test_runner.py --single ops/matmul_op_test.py
```

Tests are organized under `test/ops/` (operator-level) and `test/fusion/` (end-to-end). Test cases inherit `MUSATestCase`, which handles plugin loading automatically.

## Benchmarks

```bash
python benchmark/bench_h2d.py                # H2D / D2H throughput sweep
python benchmark/bench_resnet.py             # ResNet-like training step
python benchmark/bench_alloc_churn.py        # allocator stress + invariant assertions
```

## Supported operators

- Basic: Add / Sub / Multiply / RealDiv / Maximum / Minimum
- Activation: Relu / Sigmoid / Softmax / Erf
- Matrix: MatMul / FusedMatMul / Transpose
- Data: Reshape / Concat / Gather / StridedSlice / ExpandDims
- Normalization: LayerNorm / FusedBatchNorm
- Other: TensorInteraction / BiasAdd / Assign
- Optimizers: ResourceApplyAdam / MusaResourceSparseApplyAdam (embedding sparse update)

## Directory structure

```
tensorflow_musa_extension/
├── build.sh / CMakeLists.txt / setup.py    build entry points
├── python/                                 Python package (pip name: tensorflow_musa)
├── musa_ext/
│   ├── kernels/                            MUSA kernel implementations
│   ├── mu/                                 device registration, allocator, optimizer
│   └── python/                             _C extension source
├── benchmark/                              end-to-end benchmarks
├── docs/                                   user-facing documentation
└── test/                                   test cases
```

## More examples

[![MUSA Playground](https://img.shields.io/badge/Gitee-TensorFlow_MUSA_Playground-C71D23?style=for-the-badge&logo=gitee&logoColor=white)](https://gitee.com/mthreadsacademy/tensorflow_musa_playground)

## Contributing

Contributions are welcome — fork, create a feature branch, add tests, and open a pull request.

## License

Apache License 2.0.
