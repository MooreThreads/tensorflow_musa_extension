# TensorFlow MUSA Extension

TensorFlow MUSA Extension is a high-performance TensorFlow plugin specifically designed for Moore Threads MUSA GPU architecture. This extension provides native MUSA kernel implementations to deliver full GPU acceleration support for TensorFlow, maximizing the computational performance of Moore Threads' full-featured GPUs.

## Features

- **Comprehensive Operator Support**: Covers core operators required for deep learning training and inference
- **High-Performance Optimization**: Deeply optimized for MUSA architecture, including memory access patterns and computational efficiency
- **Automatic Graph Optimization**: Supports automatic layout conversion, operator fusion, and Automatic Mixed Precision (AMP)
- **Seamless Integration**: Fully compatible with TensorFlow ecosystem without requiring code modifications
- **Device Management**: Complete MUSA device registration, memory management, and stream processing support
- **Kernel Debugging Support**: Built-in kernel execution time statistics for performance analysis
- **Python Package Support**: Provides `tensorflow_musa` Python package with pip installation and optimizer interface

## Quick Start

### Directory Structure

```
tensorflow_musa_extension/
├── CMakeLists.txt          # CMake build configuration
├── build.sh                # Build script (supports release/debug/wheel)
├── setup.py                # Python package build configuration
├── .clang-format           # Code formatting configuration
├── .pre-commit-config.yaml # pre-commit hook configuration
├── .github/                # CI/CD configuration
├── python/                 # Python package source directory (pip name: tensorflow_musa)
│   ├── __init__.py         # Package entry, auto-loads plugin
│   ├── _loader.py          # Plugin loading utilities
│   ├── _patch.py           # tf.keras.optimizers.Adam monkey patch
│   └── optimizer/          # Optimizer module
│       ├── __init__.py
│       └── adam.py         # MUSA Adam optimizer (supports sparse update)
├── musa_ext/               # Core source directory
│   ├── kernels/            # MUSA kernel implementations (.mu files)
│   ├── mu/                 # MUSA device and optimizer implementations
│   └── utils/              # Utility functions
└── test/                   # Test cases
    ├── musa_test_utils.py  # Test utilities base class
    ├── test_runner.py      # Test runner
    ├── ops/                # Operator tests
    └── fusion/             # Fusion tests (e2e)
```

### Prerequisites

- **Build Tools**:
  - CMake (version >= 3.10)
  - Make
- **MUSA SDK**:
  - MUSA Runtime (>= 1.0)
  - muBLAS Library
  - muDNN Library
  - Default installation path: `/usr/local/musa`
- **Python Dependencies**:
  - Python: >= 3.7
  - TensorFlow: == 2.6.1 (required version)
  - NumPy: >= 1.19.0
- **Development Tools**:
  - pre-commit >= 3.0.0
  - pytest >= 6.0.0

### Installation Methods

#### Method 1: Install WHL Package (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd tensorflow_musa_extension

# Ensure TensorFlow 2.6.1 is installed
pip install tensorflow==2.6.1

# Build WHL package (one-click build)
./build.sh wheel

# Install WHL package
pip install dist/tensorflow_musa-0.1.0-py3-none-any.whl --no-deps

# Install WHL packages after rebuilding
pip install dist/tensorflow_musa-0.1.0-py3-none-any.whl --no-deps --force-reinstall
```

#### Method 2: Development Mode

```bash
# Clone the repository
git clone <repository-url>
cd tensorflow_musa_extension

# Build plugin
./build.sh release

# Load plugin in Python for testing
import tensorflow as tf
tf.load_library("./build/libmusa_plugin.so")
```

## Build Guide

### 1. Build Modes

Three build modes are supported:

| Mode | Command | Description |
|------|---------|-------------|
| **Release** | `./build.sh` or `./build.sh release` | Optimized performance, generates `build/libmusa_plugin.so` |
| **Debug** | `./build.sh debug` | Enables `MUSA_KERNEL_DEBUG` and kernel timing macros |
| **Wheel** | `./build.sh wheel` | One-click WHL package build, generates `dist/tensorflow_musa-*.whl` |

### 2. Compilation Process

```bash
# Release (default) - build plugin only
./build.sh

# Debug (timing instrumentation)
./build.sh debug

# Wheel (build release package)
./build.sh wheel
```

The build script automatically:
- Checks TensorFlow version (must be 2.6.1)
- Configures CMake project
- Compiles MUSA kernels and host code
- Generates `libmusa_plugin.so` or WHL package

### 3. WHL Package Notes

WHL package build features:
- **No auto-download TensorFlow**: Prevents pip from downloading incompatible versions
- **Version check**: Automatically checks TensorFlow version is 2.6.1 before build
- **Package name mapping**: Source directory is `python/`, but pip package name is `tensorflow_musa`

After installation:
```python
import tensorflow_musa as tf_musa  # Package name remains tensorflow_musa
```

### 4. Debugging and Diagnostics

For detailed debugging guide, see [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md), including:

- **Kernel Timing**: Performance analysis in Debug mode
- **Telemetry System**: Full-stack tracing and dirty data diagnostics
- **Memory Diagnostics**: Use-After-Free detection and memory coloring
- **Environment Variables**: Complete environment variable configuration table

Quick telemetry setup for diagnostics:

```bash
export MUSA_TELEMETRY_ENABLED=1
export MUSA_TELEMETRY_LOG_PATH=/tmp/telemetry.json
python test_runner.py
```

Quick kernel timing setup for performance analysis:

```bash
./build.sh debug
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1
python test_runner.py
```

## Testing

After building, run the test suite to verify functional correctness. Tests are divided into **operator tests** (`test/ops/`) and **fusion tests** (`test/fusion/`).

### Running Individual Tests

```bash
cd test

# Run specific operator tests
python -m ops.add_op_test
python -m ops.matmul_op_test

# Run fusion tests
python -m fusion.layernorm_gelu_fusion_test
```

### Using Test Runner

```bash
cd test

# Run all operator tests (default)
python test_runner.py

# Run all fusion tests
python test_runner.py --fusion

# Run single test file
python test_runner.py --single ops/matmul_op_test.py
python test_runner.py --single fusion/layernorm_gelu_fusion_test.py

# Detail mode (show detailed output for each test)
python test_runner.py --detail

# Quiet mode (show only progress bar and summary)
python test_runner.py --quiet
```

### Test File Naming Convention

**Operator Tests** (`test/ops/`):
- Use `op_name_op_test.py` format
- Inherit from `MUSATestCase` (wraps plugin loading)
- Test methods start with `test_`

**Fusion Tests** (`test/fusion/`):
- Use `*_fusion_test.py` format
- Inherit from `MUSATestCase`
- Test end-to-end graph optimization and operator fusion

## Supported Operators

Current version supports the following core operators:
- **Basic Operations**: Add, Sub, Multiply, RealDiv, Maximum, Minimum
- **Activation Functions**: Relu, Sigmoid, Softmax, Erf
- **Matrix Operations**: MatMul, FusedMatMul, Transpose
- **Data Manipulation**: Reshape, Concat, Gather, StridedSlice, ExpandDims
- **Normalization**: LayerNorm, FusedBatchNorm
- **Special Operators**: TensorInteraction, BiasAdd, Assign
- **Optimizers**: ResourceApplyAdam, MusaResourceSparseApplyAdam (supports embedding sparse update)

## Usage Examples

### Basic Usage

After installing the `tensorflow_musa` package, the plugin is automatically loaded on import:

```python
import tensorflow_musa as tf_musa

# Check version
print(f"TensorFlow MUSA version: {tf_musa.__version__}")

# View available MUSA devices
devices = tf_musa.get_musa_devices()
print(f"Available MUSA devices: {devices}")
```

### Auto Patch tf.keras.optimizers.Adam (Recommended)

After importing `tensorflow_musa`, `tf.keras.optimizers.Adam` is automatically patched to use MUSA fused kernels. No code changes needed:

```python
import tensorflow as tf
import tensorflow_musa as tf_musa  # Auto patches Adam

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Use standard tf.keras.optimizers.Adam (auto patched)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Embedding sparse gradients automatically use MusaResourceSparseApplyAdam kernel
```

### Explicitly Use MUSA Adam Optimizer

If you want to explicitly specify MUSA optimizer:

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Explicitly use MUSA fused Adam optimizer
optimizer = tf_musa.optimizer.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Device Management

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

# Set specific MUSA device
with tf.device('/device:MUSA:0'):
    # Create tensors and compute on MUSA device
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print(c)
```

### Embedding Sparse Update Example

MUSA Adam optimizer supports sparse gradient updates for embedding scenarios:

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

# Create embedding variable
vocab_size = 10000
embedding_dim = 128
with tf.device('/device:MUSA:0'):
    embedding = tf.Variable(tf.zeros([vocab_size, embedding_dim]))

# Use patched Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Simulate embedding lookup sparse gradient
indices = tf.constant([0, 5, 10, 15])  # Word IDs in batch
values = tf.random.normal([4, embedding_dim])  # Corresponding gradients
sparse_grad = tf.IndexedSlices(values, indices)

# Apply sparse gradient update (auto uses MusaResourceSparseApplyAdam kernel)
optimizer.apply_gradients([(sparse_grad, embedding)])
```

## Contribution Guidelines

Contributions for new operator implementations or optimizations are welcome! Contribution workflow:

1. Fork the repository and create a feature branch
2. Implement operators or optimization features
3. Add corresponding test cases
4. Update documentation (if needed)
5. Submit a Pull Request

## License

This project is licensed under Apache 2.0.

## Technical Support

For issues or questions, please submit an Issue or contact the project maintainers.
