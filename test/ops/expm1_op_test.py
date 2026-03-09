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
# ==============================================================================

"""Tests for MUSA Expm1 operator.

The Expm1 operator computes exp(x) - 1 with improved numerical precision
for values of x near zero. This is the inverse operation of Log1p.

Mathematical definition: expm1(x) = e^x - 1
"""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class Expm1OpTest(MUSATestCase):
    """Test cases for MUSA Expm1 operator."""

    def _test_expm1(self, shape, dtype, rtol=1e-5, atol=1e-8):
        """Test expm1 operation with given shape and dtype.

        Args:
            shape: Shape of the input tensor.
            dtype: TensorFlow data type.
            rtol: Relative tolerance for comparison.
            atol: Absolute tolerance for comparison.
        """
        # Determine numpy dtype for data generation
        np_dtype = dtype.as_numpy_dtype
        if dtype == tf.bfloat16:
            np_dtype = np.float32

        # Generate random input data in a reasonable range for expm1
        # Use smaller range to avoid overflow and focus on precision
        if dtype in [tf.float32, tf.float64, tf.float16, tf.bfloat16]:
            x_np = np.random.uniform(-5.0, 5.0, size=shape).astype(np_dtype)
        else:
            x_np = np.random.randint(-5, 5, size=shape).astype(np_dtype)

        x = tf.constant(x_np, dtype=dtype)

        # Compare CPU and MUSA results
        self._compare_cpu_musa_results(tf.math.expm1, [x], dtype, rtol=rtol, atol=atol)

    def testExpm1Float32(self):
        """Test expm1 with float32 on a large matrix."""
        self._test_expm1([1024, 1024], tf.float32)

    def testExpm1Float16(self):
        """Test expm1 with float16."""
        self._test_expm1([4, 4], tf.float16, rtol=1e-2, atol=1e-2)

    def testExpm1BFloat16(self):
        """Test expm1 with bfloat16."""
        self._test_expm1([3, 3], tf.bfloat16, rtol=1e-2, atol=1e-2)

    def testExpm1Float64(self):
        """Test expm1 with float64."""
        self._test_expm1([64, 64], tf.float64)

    def testExpm1SmallValues(self):
        """Test expm1 with small values near zero.

        This tests the main benefit of expm1 over exp(x) - 1:
        improved numerical precision for small x values.
        """
        # Small values where expm1 provides better precision
        small_values = np.array([
            -1e-10, -1e-8, -1e-6, -1e-4,
            0.0,
            1e-10, 1e-8, 1e-6, 1e-4
        ], dtype=np.float32)
        x = tf.constant(small_values, dtype=tf.float32)

        self._compare_cpu_musa_results(tf.math.expm1, [x], tf.float32)

    def testExpm1EdgeCases(self):
        """Test expm1 with key edge case inputs."""
        # Test values including zero, negative, and positive numbers
        edge_values = np.array([
            -10.0,   # expm1(-10) ≈ -0.9999546
            -1.0,    # expm1(-1)  ≈ -0.632121
            0.0,     # expm1(0)   =  0.0
            1.0,     # expm1(1)   ≈  1.718282
            10.0,    # expm1(10)  ≈ 22025.465795
        ], dtype=np.float32)
        x = tf.constant(edge_values, dtype=tf.float32)

        self._compare_cpu_musa_results(tf.math.expm1, [x], tf.float32)

    def testExpm1EmptyTensor(self):
        """Test expm1 with empty tensor."""
        x = tf.constant([], dtype=tf.float32)
        self._compare_cpu_musa_results(tf.math.expm1, [x], tf.float32)

    def testExpm1Scalar(self):
        """Test expm1 with scalar input."""
        x = tf.constant(2.0, dtype=tf.float32)
        self._compare_cpu_musa_results(tf.math.expm1, [x], tf.float32)

    def testExpm1Vector(self):
        """Test expm1 with vector input."""
        self._test_expm1([100], tf.float32)

    def testExpm1HighDimensional(self):
        """Test expm1 with high dimensional tensor."""
        self._test_expm1([2, 3, 4, 5], tf.float32)

    def testExpm1NegativeValues(self):
        """Test expm1 with negative values."""
        x_np = np.array([-5.0, -2.0, -1.0, -0.5, -0.1], dtype=np.float32)
        x = tf.constant(x_np, dtype=tf.float32)
        self._compare_cpu_musa_results(tf.math.expm1, [x], tf.float32)

    def testExpm1PositiveValues(self):
        """Test expm1 with positive values."""
        x_np = np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float32)
        x = tf.constant(x_np, dtype=tf.float32)
        self._compare_cpu_musa_results(tf.math.expm1, [x], tf.float32)

    def testExpm1InverseOfLog1p(self):
        """Test that expm1 is the inverse of log1p.

        For valid inputs (x > -1): expm1(log1p(x)) should equal x.
        """
        # Values > -1 are valid for log1p
        x_np = np.array([0.0, 0.5, 1.0, 2.0, 10.0], dtype=np.float32)
        x = tf.constant(x_np, dtype=tf.float32)

        # Compute log1p on CPU
        with tf.device('/CPU:0'):
            log1p_result = tf.math.log1p(x)

        # Compute expm1 of log1p result on MUSA
        with tf.device('/device:MUSA:0'):
            expm1_result = tf.math.expm1(log1p_result)

        # Should recover original values
        self.assertAllClose(x.numpy(), expm1_result.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tf.test.main()