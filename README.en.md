# TensorFlow MUSA Extension

TensorFlow MUSA Extension is a high-performance TensorFlow plugin specifically designed for Moore Threads MUSA GPU architecture. This extension provides native MUSA kernel implementations to deliver full GPU acceleration support for TensorFlow, maximizing the computational performance of Moore Threads' full-featured GPUs.

## Features

- **Comprehensive Operator Support**: Covers core operators required for deep learning training and inference
- **High-Performance Optimization**: Deeply optimized for MUSA architecture, including memory access patterns and computational efficiency
- **Automatic Graph Optimization**: Supports automatic layout conversion, operator fusion, and Automatic Mixed Precision (AMP)
- **Seamless Integration**: Fully compatible with TensorFlow ecosystem without requiring code modifications
- **Device Management**: Complete MUSA device registration, memory management, and stream processing support
- **Kernel Debugging Support**: Built-in kernel execution time statistics for performance analysis

## Quick Start

### Directory Structure

```
tensorflow_musa_extension/
├── CMakeLists.txt          # CMake build configuration
├── build.sh                # Build script
├── .clang-format           # Code formatting configuration
├── .pre-commit-config.yaml # pre-commit hook configuration
├── .gitlab-ci.yml          # CI/CD configuration
├── musa_ext/               # Core source directory
│   ├── kernels/            # MUSA kernel implementations
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
  - TensorFlow: == 2.6.1
  - protobuf: == 3.20.3
  - NumPy: >= 1.19.0
  - prettytable: >= 3.0.0
- **Development Tools**:
  - pre-commit >= 3.0.0
  - pytest >= 6.0.0

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd tensorflow_musa_extension

# Build the plugin
./build.sh

# Load the plugin in Python
import tensorflow as tf
tf.load_library("./build/libmusa_plugin.so")
```

## Build Guide

### 1. Build Type

Both Release and Debug modes are supported:

| Mode | Command | Description |
|------|---------|-------------|
| **Release** | `./build.sh` or `./build.sh release` | Optimized for performance, no debug overhead |
| **Debug** | `./build.sh debug` | Enables `MUSA_KERNEL_DEBUG` and kernel timing macros |

### 2. Compilation Process

Execute the automated build script:

```bash
# Release (default)
./build.sh

# Release (explicit)
./build.sh release

# Debug (timing instrumentation)
./build.sh debug
```

The build script automatically completes the following steps:
- Configures the CMake project
- Compiles MUSA kernels and host code
- Generates the dynamic library `libmusa_plugin.so`

### 3. Debugging and Diagnostics

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
