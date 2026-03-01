#!/bin/bash
set -e

rm -rf build

mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python3)
make -j$(nproc)

echo "Build success! Plugin is located at: $(pwd)/libmusa_plugin.so"
