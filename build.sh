#!/bin/bash
set -e

# ============================================================================
# MUSA Plugin Build Script
# Usage:
#   ./build.sh [release|debug]
#
# Examples:
#   ./build.sh           # Default: release mode
#   ./build.sh release   # Release mode (optimized)
#   ./build.sh debug     # Debug mode (kernel timing enabled)
# ============================================================================

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
    *)
        echo "Error: Unknown build type '$BUILD_TYPE'"
        echo "Usage: ./build.sh [release|debug]"
        echo ""
        echo "Options:"
        echo "  release  - Optimized release build (default)"
        echo "  debug    - Enable MUSA kernel debug/timing macros"
        exit 1
        ;;
esac

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
echo "=========================================="
