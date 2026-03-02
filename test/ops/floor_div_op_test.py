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
"""Tests for MUSA FloorDiv operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class FloorDivOpTest(MUSATestCase):
    """Tests for MUSA FloorDiv operator."""

    def _test_floordiv(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
        """Test floordiv operation with given shapes and dtype."""
        np_dtype = dtype.as_numpy_dtype

        if dtype in [tf.int32, tf.int64]:
            # For integers: range [-100, 100], avoid division by zero
            x_np = np.random.randint(-100, 100, size=shape_x).astype(np_dtype)
            y_raw = np.random.randint(1, 10, size=shape_y).astype(np_dtype)
            sign = np.random.choice([-1, 1], size=shape_y).astype(np_dtype)
            y_np = y_raw * sign
        else:
            # For floats: uniform distribution, avoid very small divisors
            x_np = np.random.uniform(-100, 100, size=shape_x).astype(np_dtype)
            y_np = np.random.uniform(0.1, 10, size=shape_y).astype(np_dtype)
            sign = np.random.choice([-1.0, 1.0], size=shape_y).astype(np_dtype)
            y_np = y_np * sign
            y_np = np.where(np.abs(y_np) < 0.1, 0.1, y_np)

        x = tf.constant(x_np, dtype=dtype)
        y = tf.constant(y_np, dtype=dtype)

        if dtype in [tf.int32, tf.int64]:
            # Integer results must be exactly equal
            cpu_result = self._test_op_device_placement(tf.math.floordiv, [x, y], '/CPU:0')
            musa_result = self._test_op_device_placement(tf.math.floordiv, [x, y], '/device:MUSA:0')
            self.assertAllEqual(cpu_result.numpy(), musa_result.numpy())
        else:
            # Float results are compared with tolerance
            self._compare_cpu_musa_results(tf.math.floordiv, [x, y], dtype, rtol=rtol, atol=atol)

    # --- Float Tests ---
    def testFloorDivFloatBasic(self):
        """Test basic float floordiv operations."""
        for dtype in [tf.float32]:
            self._test_floordiv([1024], [1024], dtype)
            self._test_floordiv([2, 2], [2, 2], dtype)
            self._test_floordiv([10, 10], [10, 10], dtype)

    # --- Integer Tests ---
    def testFloorDivIntBasic(self):
        """Test basic integer floordiv operations."""
        for dtype in [tf.int32, tf.int64]:
            self._test_floordiv([4], [4], dtype)
            self._test_floordiv([1024], [1024], dtype)
            # Broadcast test
            self._test_floordiv([5, 5], [1, 5], dtype)

    def testFloorDivIntEdgeCases(self):
        """Test integer floordiv edge cases for correct Python semantics."""
        # Python floor division semantics: -5 // 2 == -3
        x_vals = [-5, -5, 5, 5, -10, 0]
        y_vals = [2, -2, 2, -2, 3, 5]
        expected_results = [-3, 2, 2, -3, -4, 0]

        for dtype in [tf.int32, tf.int64]:
            with self.subTest(dtype=dtype.name):
                x = tf.constant(x_vals, dtype=dtype)
                y = tf.constant(y_vals, dtype=dtype)
                expected = tf.constant(expected_results, dtype=dtype)

                # Test on MUSA
                musa_result = self._test_op_device_placement(tf.math.floordiv, [x, y], '/device:MUSA:0')

                # The result must match the expected Python semantics exactly
                self.assertAllEqual(musa_result, expected)


if __name__ == "__main__":
    tf.test.main()
