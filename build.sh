#!/bin/bash
set -e

# ============================================================================
# MUSA Plugin Build Script
# Usage:
#   ./build.sh [release|debug|wheel]
#
# Examples:
#   ./build.sh           # Default: release mode (build .so only)
#   ./build.sh release   # Release mode (optimized)
#   ./build.sh debug     # Debug mode (kernel timing enabled)
#   ./build.sh wheel     # Build wheel package directly (recommended for distribution)
# ============================================================================

# Supported TensorFlow version range.
# The plugin uses the PluggableDevice C ABI (SE_MAJOR=0) which has been
# stable from TF 2.5 onward. Keep these in sync with setup.py.
MIN_TF_VERSION="2.6"
MAX_TF_VERSION_EXCLUSIVE="2.17"
RECOMMENDED_TF_VERSION="2.6.1"

# Function to validate TensorFlow is present in a supported version range.
check_tf_version() {
    echo "Checking TensorFlow version..."
    python3 -c "
import sys
import tensorflow as tf

def _parse(v):
    out = []
    for part in v.split('+', 1)[0].split('.'):
        d = ''
        for c in part:
            if c.isdigit():
                d += c
            else:
                break
        out.append(int(d) if d else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)

version = tf.__version__
v = _parse(version)
vmin = _parse('${MIN_TF_VERSION}')
vmax = _parse('${MAX_TF_VERSION_EXCLUSIVE}')
rec = '${RECOMMENDED_TF_VERSION}'
if v < vmin or v >= vmax:
    print('ERROR: TensorFlow version out of supported range!')
    print(f'  Supported: >= ${MIN_TF_VERSION}, < ${MAX_TF_VERSION_EXCLUSIVE}')
    print(f'  Installed: {version}')
    print(f'  Recommended (fully tested): {rec}')
    sys.exit(1)
if version != rec:
    print(f'NOTE: TensorFlow {version} is within the supported range; '
          f'the primary test matrix uses {rec}.')
else:
    print(f'TensorFlow {version} found - OK')
" || exit 1
}

# Parse build type from command line argument
BUILD_TYPE="${1:-release}"
BUILD_TYPE=$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')

case "$BUILD_TYPE" in
    release)
        CMAKE_BUILD_TYPE="Release"
        MUSA_KERNEL_DEBUG="OFF"
        echo "=========================================="
        echo "Building MUSA Plugin - RELEASE Mode"
        echo "=========================================="
        echo "Features:"
        echo "  • Optimized for performance (-O3)"
        echo "  • No debug overhead"
        echo ""
        ;;
    debug)
        CMAKE_BUILD_TYPE="Debug"
        MUSA_KERNEL_DEBUG="ON"
        echo "=========================================="
        echo "Building MUSA Plugin - DEBUG Mode"
        echo "=========================================="
        echo "Features:"
        echo "  • Kernel timing instrumentation enabled"
        echo "  • TensorFlow ABI/DCHECK compatibility preserved (-DNDEBUG)"
        echo "  • Use env vars MUSA_TIMING_KERNEL_* to control output"
        echo ""
        ;;
    wheel)
        echo "=========================================="
        echo "Building tensorflow_musa Wheel Package"
        echo "=========================================="
        echo ""
        check_tf_version
        echo ""
        echo "Building wheel package..."
        echo ""

        # Clean previous wheel builds
        rm -rf build/lib build/bdist.* dist/*.whl 2>/dev/null || true

        # Build wheel using setup.py (no isolation to use existing TF)
        python3 setup.py bdist_wheel

        # Find and display the built wheel
        WHEEL_FILE=$(ls dist/*.whl 2>/dev/null | head -1)
        if [ -n "$WHEEL_FILE" ]; then
            echo ""
            echo "[SUCCESS] Wheel package built successfully!"
            ls -lh "$WHEEL_FILE"
            echo ""
            echo "=========================================="
            echo "Install with:"
            echo "  pip install $WHEEL_FILE --no-deps"
            echo "=========================================="
        else
            echo ""
            echo "[FAIL] Wheel package not found in dist/"
            exit 1
        fi
        exit 0
        ;;
    *)
        echo "Error: Unknown build type '$BUILD_TYPE'"
        echo "Usage: ./build.sh [release|debug|wheel]"
        echo ""
        echo "Options:"
        echo "  release  - Optimized release build (default)"
        echo "  debug    - Enable MUSA kernel debug/timing macros"
        echo "  wheel    - Build wheel package for distribution"
        exit 1
        ;;
esac

# Check TensorFlow version before building .so
check_tf_version

# Clean previous build if needed
rm -rf build

mkdir -p build
cd build

echo "Configuring with CMake..."
echo "  CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
echo ""

cmake .. \
    -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
    -DMUSA_KERNEL_DEBUG=$MUSA_KERNEL_DEBUG \
    -DPYTHON_EXECUTABLE=$(which python3) 2>&1 | tee cmake_output.log

echo ""
echo "Building with $(nproc) parallel jobs..."
make -j$(nproc)

# Verify build output
if [ -f "libmusa_plugin.so" ]; then
    echo ""
    echo "[SUCCESS] Build successful: libmusa_plugin.so"
    ls -lh libmusa_plugin.so
else
    echo ""
    echo "[FAIL] Build failed: libmusa_plugin.so not found"
    exit 1
fi

# Post-build information
echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Build Type: $BUILD_TYPE"
echo "Plugin: $(pwd)/libmusa_plugin.so"
echo ""
echo "To build wheel package:"
echo "  ./build.sh wheel"
echo "=========================================="
