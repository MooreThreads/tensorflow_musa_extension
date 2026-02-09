# TensorFlow MUSA Extension

TensorFlow MUSA Extension is a high-performance TensorFlow plugin specifically designed for Moore Threads MUSA GPU architecture. This extension provides native MUSA kernel implementations to deliver full GPU acceleration support for TensorFlow, maximizing the computational performance of Moore Threads' full-featured GPUs.

## Features

- **Comprehensive Operator Support**: Covers core operators required for deep learning training and inference
- **High-Performance Optimization**: Deeply optimized for MUSA architecture, including memory access patterns and computational efficiency
- **Automatic Graph Optimization**: Supports automatic layout conversion, operator fusion, and Automatic Mixed Precision (AMP)
- **Seamless Integration**: Fully compatible with TensorFlow ecosystem without requiring code modifications
- **Device Management**: Complete MUSA device registration, memory management, and stream processing support

## Quick Start

### Prerequisites

- TensorFlow 2.x
- Moore Threads MUSA Driver and SDK
- CMake 3.15+
- GCC/G++ 7.0+

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

### 1. Operator Configuration

Configure operators to be compiled in the `CMakeLists.txt` file:

- **Operator Selection**: Enable required operator implementations in the source file configuration section
- **Custom Kernels**: If using `.mu` custom kernel implementations, add corresponding source files to `set(MU_SOURCES "")`

### 2. Compilation Process

Execute the automated build script:

```bash
./build.sh
```

The build script automatically completes the following steps:
- Configures the CMake project
- Compiles MUSA kernels and host code
- Generates the dynamic library `libmusa_plugin.so`

### 3. Plugin Loading

After successful compilation, load the plugin in your TensorFlow application:

```python
import tensorflow as tf
tf.load_library("/path/to/tensorflow_musa_extension/build/libmusa_plugin.so")
```

## Directory Structure

```
tensorflow_musa_extension/
├── tensorflow_musa/          # Core implementation directory
│   ├── kernels/             # Operator implementations
│   │   ├── musa_*_op.cc     # Specific implementations of various operators
│   │   └── utils_op.h       # Operator utility functions
│   ├── mu/                  # MUSA device core components
│   │   ├── device/          # Device low-level implementations
│   │   │   ├── musa_device.h    # Device registration and management
│   │   │   ├── musa_allocator.h # Memory allocator
│   │   │   ├── musa_memcpy.h    # Memory copy operations
│   │   │   └── musa_stream.h    # Stream processing support
│   │   ├── optimizer/       # Graph optimization modules
│   │   │   ├── musa_layout_optimizer.cc  # Layout automatic optimization
│   │   │   ├── musa_remapper_optimizer.cc # Operator fusion optimization
│   │   │   └── musa_amp_optimizer.cc     # Automatic Mixed Precision optimization
│   │   ├── device_register.cc   # Device registration logic
│   │   └── kernel_register.cc   # Operator registration logic
│   └── utils/               # Utility functions
│       └── logging.h        # Logging utilities
├── test/                    # Functional test scripts
│   └── test_*.py            # Unit tests for individual operators
├── build.sh                 # Automated build script
└── README.en.md             # Project documentation
```

## Testing

After building, run the test suite to verify functional correctness. Test files follow TensorFlow's official `python/kernel_tests` style, using `tf.test.TestCase` as the base class.

```bash
# Run specific operator tests
python test/add_op_test.py
python test/matmul_op_test.py

# Run all tests
./test/run_all_tests.sh

# Or run each test individually
for test_file in test/*_op_test.py; do
    python "$test_file"
done
```

Test file naming convention:
- Use `op_name_op_test.py` format
- Inherit from `tf.test.TestCase`
- Test methods start with `test_`
- Use `self.assert*` methods for assertions

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

## Support

For issues or questions, please submit an Issue or contact the project maintainers.
