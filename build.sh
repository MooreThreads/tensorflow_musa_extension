#!/bin/bash
set -e

# ============================================================================
# MUSA Plugin Build Script
# Usage:
#   ./build.sh [release|wheel]
#
# Examples:
#   ./build.sh           # Default: release mode (build .so only)
#   ./build.sh release   # Release mode (optimized)
#   ./build.sh wheel     # Build wheel package directly (recommended for distribution)
# ============================================================================

# Default TensorFlow version allowlist. Override with comma-separated values, e.g.
# TENSORFLOW_MUSA_TARGET_TF=2.6.1,2.15.1 ./build.sh
TARGET_TF_VERSIONS="${TENSORFLOW_MUSA_TARGET_TF:-2.6.1,2.15.1}"
PYTHON_BIN="${PYTHON:-python3}"

# Function to check TensorFlow version
check_tf_version() {
    echo "Checking TensorFlow version..."
    "$PYTHON_BIN" -c "
import tensorflow as tf
version = tf.__version__
allowed = {v.strip() for v in '${TARGET_TF_VERSIONS}'.split(',') if v.strip()}
if version not in allowed:
    print('ERROR: TensorFlow version mismatch!')
    print(f'  Allowed: {sorted(allowed)}')
    print(f'  Installed: {version}')
    print('  Set TENSORFLOW_MUSA_TARGET_TF to include this version, or install an allowed TensorFlow version.')
    exit(1)
print(f'TensorFlow {version} found - OK (allowed: {sorted(allowed)})')
" || exit 1
}

check_source_layout() {
    local fusion_dir="musa_ext/kernels/fusion"
    local old_fusion_dir="musa_ext/mu/graph""_fusion"

    if [ ! -d "$fusion_dir" ]; then
        echo "ERROR: Fusion source directory '$fusion_dir' not found."
        exit 1
    fi

    if [ -d "$old_fusion_dir" ]; then
        echo "ERROR: Deprecated fusion source directory '$old_fusion_dir' still exists."
        echo "       Fusion sources should live under '$fusion_dir'."
        exit 1
    fi
}

# Parse build type from command line argument
BUILD_TYPE="${1:-release}"
BUILD_TYPE=$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')

case "$BUILD_TYPE" in
    release)
        CMAKE_BUILD_TYPE="Release"
        echo "=========================================="
        echo "Building MUSA Plugin - RELEASE Mode"
        echo "=========================================="
        echo "Features:"
        echo "  • Optimized for performance (-O3)"
        echo ""
        ;;
    wheel)
        echo "=========================================="
        echo "Building tensorflow_musa Wheel Package"
        echo "=========================================="
        echo ""
        check_tf_version
        check_source_layout
        echo ""
        echo "Building wheel package..."
        echo ""

        # Clean previous wheel builds
        rm -rf build/lib build/bdist.* dist/*.whl 2>/dev/null || true

        # Build wheel using setup.py (no isolation to use existing TF)
        "$PYTHON_BIN" setup.py bdist_wheel

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
        echo "Usage: ./build.sh [release|wheel]"
        echo ""
        echo "Options:"
        echo "  release  - Optimized release build (default)"
        echo "  wheel    - Build wheel package for distribution"
        exit 1
        ;;
esac

# Check TensorFlow version before building .so
check_tf_version
check_source_layout

# Clean previous build if needed
rm -rf build

mkdir -p build
cd build

echo "Configuring with CMake..."
echo "  CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
echo ""

cmake .. \
    -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
    -DPYTHON_EXECUTABLE="$PYTHON_BIN" 2>&1 | tee cmake_output.log

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
