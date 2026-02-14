#!/bin/bash

# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

echo "Running all MUSA operator tests..."

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the plugin first if it doesn't exist
if [ ! -f "$TEST_DIR/../build/libmusa_plugin.so" ]; then
    echo "Building MUSA plugin..."
    cd "$TEST_DIR/.." && ./build.sh
fi

# Run all tests using the custom test runner in quiet mode
python3 "$TEST_DIR/test_runner.py" --quiet

echo "All tests completed!"