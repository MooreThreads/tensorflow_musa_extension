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
# =============================================================================
"""Tests for MUSA Equal operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class EqualOpTest(MUSATestCase):
    """Tests for MUSA Equal operator."""

    def _generate_equal_friendly_data(self, shape, dtype):
        """
        Generate data that has a higher chance of producing 'True' in equality checks.
        This helps validate the correctness of both True and False outcomes.
        """
        if dtype in [np.int32, np.int64]:
            # For integers, use a small range to increase collision probability.
            return np.random.randint(0, 3, size=shape).astype(dtype)
        else:
            # For floats, round random numbers to create exact matches.
            # Note: We generate in float64 first to avoid precision issues during rounding.
            return np.round(np.random.randn(*shape)).astype(dtype)

    def _test_equal(self, shape_x, shape_y, dtype):
        """Test equal operation with given shapes and dtype."""
        # Generate test data
        x_np = self._generate_equal_friendly_data(shape_x, dtype)
        y_np = self._generate_equal_friendly_data(shape_y, dtype)

        # Create TensorFlow constants
        # For bfloat16, we need to handle the dtype conversion carefully.
        if dtype == tf.bfloat16.as_numpy_dtype:
            x = tf.constant(x_np, dtype=tf.bfloat16)
            y = tf.constant(y_np, dtype=tf.bfloat16)
        else:
            x = tf.constant(x_np, dtype=dtype)
            y = tf.constant(y_np, dtype=dtype)

        # Use the utility method for comparison.
        # Since `tf.equal` returns bool, we don't need rtol/atol.
        # The parent class's `assertAllEqual` will be used internally.
        self._compare_cpu_musa_results(tf.equal, [x, y], dtype=tf.bool)

    def testEqualBasic(self):
        """Test basic equal operation with same shapes."""
        dtypes_to_test = [np.int32, np.int64, np.float32, np.float16, tf.bfloat16.as_numpy_dtype]
        for dtype in dtypes_to_test:
            with self.subTest(dtype=dtype):
                self._test_equal([1024], [1024], dtype)

    def testEqualMatrix(self):
        """Test equal operation on matrices (common in Wide&Deep models)."""
        self._test_equal([32, 128], [32, 128], np.float32)

    def testEqualBroadcastRow(self):
        """Test equal with row broadcasting."""
        self._test_equal([64, 1], [64, 512], np.float32)

    def testEqualBroadcastScalar(self):
        """Test equal with scalar broadcasting."""
        self._test_equal([1], [100], np.float32)

    def testEqualDataTypes(self):
        """Comprehensive test for multiple data types."""
        # This test focuses on correctness across types, using a simple shape.
        dtypes_to_test = {
            "int32": np.int32,
            "int64": np.int64,
            "float32": np.float32,
            "float16": np.float16,
            "bfloat16": tf.bfloat16.as_numpy_dtype,
        }
        for name, dtype in dtypes_to_test.items():
            with self.subTest(data_type=name):
                self._test_equal([128], [128], dtype)


if __name__ == "__main__":
    tf.test.main()
