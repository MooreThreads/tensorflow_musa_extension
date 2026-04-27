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

# The plugin is expected to be installed as the tensorflow_musa wheel.
python3 -c "import tensorflow_musa"

# Run all tests using the custom test runner in quiet mode
python3 "$TEST_DIR/test_runner.py" --quiet

echo "All tests completed!"
