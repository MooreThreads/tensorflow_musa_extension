#!/bin/bash
set -e

# ============================================================================
# MUSA Plugin Build Script
# Usage:
#   ./build.sh [debug|release]
#
# Examples:
#   ./build.sh           # Default: release mode
#   ./build.sh debug     # Debug mode with automatic kernel debug info
#   ./build.sh release   # Release mode (optimized)
# ============================================================================

# Parse build type from command line argument
BUILD_TYPE="${1:-release}"
BUILD_TYPE=$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')

case "$BUILD_TYPE" in
    debug)
        CMAKE_BUILD_TYPE="Debug"
        MUSA_KERNEL_DEBUG="1"
        echo "=========================================="
        echo "Building MUSA Plugin - DEBUG Mode"
        echo "=========================================="
        echo "Features:"
        echo "  - Automatic kernel debug info enabled"
        echo "  - Debug symbols included"
        echo "  - TensorFlow ABI/DCHECK compatibility preserved (-DNDEBUG)"
        echo ""
        ;;
    release)
        CMAKE_BUILD_TYPE="Release"
        MUSA_KERNEL_DEBUG="0"
        echo "=========================================="
        echo "Building MUSA Plugin - RELEASE Mode"
        echo "=========================================="
        echo "Features:"
        echo "  - Optimized for performance (-O3)"
        echo "  - No debug overhead"
        echo ""
        ;;
    *)
        echo "Error: Unknown build type '$BUILD_TYPE'"
        echo "Usage: ./build.sh [debug|release]"
        echo ""
        echo "Options:"
        echo "  debug    - Debug build with automatic kernel debug info enabled"
        echo "  release  - Optimized release build (default)"
        exit 1
        ;;
esac

# Clean previous build if needed
# rm -rf build

mkdir -p build
cd build

echo "Configuring with CMake..."
echo "  CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
echo "  MUSA_KERNEL_DEBUG=$MUSA_KERNEL_DEBUG"
echo ""

cmake .. \
    -DMUSA_KERNEL_DEBUG=$MUSA_KERNEL_DEBUG \
    -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
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
