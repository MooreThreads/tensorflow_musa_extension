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

### 3. Kernel Timing (Debug Build)

Only effective when built with `./build.sh debug` (`MUSA_KERNEL_DEBUG=ON`):

Runtime environment variables are listed in the [Environment Variables](#environment-variables) section under "Logging and Debugging".

#### 3.1 Macro Usage

```cpp
// Basic guard
MUSA_KERNEL_TIMING_GUARD(ctx);

// Section timing
MUSA_KERNEL_TRACE_START("Mem Alloc");
// ... code block ...
MUSA_KERNEL_TRACE_END("Mem Alloc");

MUSA_KERNEL_TRACE_START("Kernel");
// ... kernel launch ...
MUSA_KERNEL_TRACE_END("Kernel");

// Custom section names
MUSA_KERNEL_TRACE_START("State1");
// ... allocate / pre-process ...
MUSA_KERNEL_TRACE_END("State1");

MUSA_KERNEL_TRACE_START("State2");
// ... main kernel ...
MUSA_KERNEL_TRACE_END("State2");
```

### 4. Common Validation Commands (MatMul)

```bash
./build.sh debug

export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1

mkdir -p /tmp/musa_timing_logs
python test/test_runner.py --single matmul_op_test.py 2>&1 | tee /tmp/musa_timing_logs/matmul_l2.log
```

## Environment Variables

### Feature Control

| Variable | Description | Example |
|----------|-------------|---------|
| `MUSA_ENABLE_TF32` | Enable TF32 acceleration for MatMul/Conv | `export MUSA_ENABLE_TF32=1` |
| `MUSA_DUMP_GRAPHDEF` | Enable graph optimization debugging | `export MUSA_DUMP_GRAPHDEF=1` |
| `MUSA_DUMP_GRAPHDEF_DIR` | Specify GraphDef dump directory | `export MUSA_DUMP_GRAPHDEF_DIR=/tmp/graphs` |

### Logging and Debugging

| Variable | Description | Example |
|----------|-------------|---------|
| `MUSA_TIMING_KERNEL_LEVEL` | Timing mode (`1`=total only, `2`=total + per-section breakdown) | `export MUSA_TIMING_KERNEL_LEVEL=2` |
| `MUSA_TIMING_KERNEL_NAME` | Print only selected kernels (case-insensitive substring, `ALL` for all) | `export MUSA_TIMING_KERNEL_NAME=MatMul` |
| `MUSA_TIMING_KERNEL_STATS` | Print timing summary at process exit (`1`=on, `0`=off) | `export MUSA_TIMING_KERNEL_STATS=1` |
| `TF_CPP_MIN_LOG_LEVEL` | Global log level (0=INFO, 1=WARNING, 2=ERROR) | `export TF_CPP_MIN_LOG_LEVEL=1` |
| `TF_CPP_VMODULE` | Per-file VLOG level control | `export TF_CPP_VMODULE="musa_graph_optimizer=1,layernorm_fusion=2"` |

**Common debugging combinations:**

```bash
# 1. View detailed graph optimizer logs
export TF_CPP_VMODULE="musa_graph_optimizer=1,fusion_pattern_manager=1"
python -m fusion.layernorm_gelu_fusion_test

# 2. View operator fusion details
export TF_CPP_VMODULE="layernorm_fusion=2,gelu_fusion=1"
python -m fusion.layernorm_gelu_fusion_test

# 3. Silent mode (show errors only)
export TF_CPP_MIN_LOG_LEVEL=2
python test_runner.py

# 4. Restore default logging
unset TF_CPP_MIN_LOG_LEVEL TF_CPP_VMODULE
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
