# TensorFlow MUSA Extension

TensorFlow MUSA Extension is a high-performance TensorFlow plugin specifically designed for Moore Threads MUSA GPU architecture. This extension provides native MUSA kernel implementations to deliver full GPU acceleration support for TensorFlow, maximizing the computational performance of Moore Threads' full-featured GPUs.

## Features

- **Comprehensive Operator Support**: Covers core operators required for deep learning training and inference
- **High-Performance Optimization**: Deeply optimized for MUSA architecture, including memory access patterns and computational efficiency
- **Automatic Graph Optimization**: Supports automatic layout conversion, operator fusion, and Automatic Mixed Precision (AMP)
- **Seamless Integration**: Fully compatible with TensorFlow ecosystem without requiring code modifications
- **Device Management**: Complete MUSA device registration, memory management, and stream processing support
- **Kernel Debugging Support**: Debug builds print operator type, input types, and input shapes, with optional terminal color highlighting

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
| **Debug** | `./build.sh debug` | Enables `MUSA_KERNEL_DEBUG` and prints kernel debug logs |

### 2. Compilation Process

Execute the automated build script:

```bash
# Release (default)
./build.sh

# Release (explicit)
./build.sh release

# Debug (kernel debug logs)
./build.sh debug
```

The build script automatically completes the following steps:
- Configures the CMake project
- Compiles MUSA kernels and host code
- Generates the dynamic library `libmusa_plugin.so`

### 3. Debugging and Diagnostics

For a more detailed debugging guide, see [docs/DEBUG_GUIDE.md](docs/DEBUG_GUIDE.md). The README has been updated to describe the current kernel debug logging flow; the old timing-macro path has been removed.

- **Kernel Debug Logs**: `MUSA_DEBUG_LOG_KERNEL(ctx)` prints `op_type`, `input_types`, and `input_shapes`
- **Telemetry System**: Full-stack tracing and dirty data diagnostics
- **Memory Diagnostics**: Use-After-Free detection and memory coloring
- **Environment Variables**: Complete environment variable configuration table

Quick telemetry setup for diagnostics:

```bash
export MUSA_TELEMETRY_ENABLED=1
export MUSA_TELEMETRY_LOG_PATH=/tmp/telemetry.json
python test_runner.py
```

### 4. Kernel Debug Change Summary

The kernel debug flow was updated as follows:

- Added a unified debug macro, `MUSA_DEBUG_LOG_KERNEL(ctx)`, to print lightweight debug metadata at the beginning of `Compute()`
- Centralized formatting and logging helpers in `musa_ext/kernels/utils_op.h` and `musa_ext/kernels/utils_op.cc`
- Kept example instrumentation in four kernels: `Add`, `AddN`, `Conv2D`, and `GELU`
- Removed the old timing macros entirely: `MUSA_KERNEL_TIMING_GUARD`, `MUSA_KERNEL_TRACE_START`, `MUSA_KERNEL_TRACE_END`, `MUSA_KERNEL_TRACE`, and `MUSA_PROFILE_OP`

The new log format looks like this:

```txt
[MUSA_KERNEL_DEBUG] op_type=AddV2 input_types=[float, float] input_shapes=[[1024,1024], [1024,1024]]
```

Notes:

- `input_types` is highlighted in cyan by default
- `input_shapes` is highlighted in yellow by default
- When output is redirected to a file, plain text is emitted by default to avoid ANSI escape codes in logs
- To force color even when using `tee` or redirection, set `MUSA_KERNEL_DEBUG_COLOR=1`
- To explicitly disable colors, set `NO_COLOR=1`

Quick setup for the new kernel debug logs:

Option 1: run from the repository root.

```bash
./build.sh debug
export PYTHONPATH=$PWD/test
python3 test/ops/add_op_test.py 2>&1 | tee /tmp/tme_add.log
grep 'MUSA_KERNEL_DEBUG' /tmp/tme_add.log
```

Option 2: enter the `test/` directory directly. In this mode you do not need to set `PYTHONPATH`.

```bash
./build.sh debug
cd test
python3 ops/add_op_test.py 2>&1 | tee /tmp/tme_add.log
grep 'MUSA_KERNEL_DEBUG' /tmp/tme_add.log
```

To force colored terminal output:

```bash
cd test
MUSA_KERNEL_DEBUG_COLOR=1 python3 ops/add_op_test.py
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
