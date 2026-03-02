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
"""Tests for MUSA Exp operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class ExpOpTest(MUSATestCase):
    """Tests for MUSA Exp operator."""

    def _test_exp(self, shape, dtype, rtol=1e-5, atol=1e-8):
        """Test exp operation with given shape and dtype."""
        # Determine the corresponding numpy dtype for data generation
        if dtype == tf.bfloat16:
            np_dtype = tf.bfloat16.as_numpy_dtype
        else:
            np_dtype = dtype.as_numpy_dtype

        # Generate random input data within a reasonable range for exp
        x_np = np.random.uniform(-3.0, 3.0, size=shape).astype(np_dtype)
        x = tf.constant(x_np, dtype=dtype)

        # Compare CPU and MUSA results
        self._compare_cpu_musa_results(tf.exp, [x], dtype, rtol=rtol, atol=atol)

    def testExpBasicVectorFloat32(self):
        """Test basic vector case with float32."""
        self._test_exp([4], tf.float32)

    def testExpMatrixFloat32(self):
        """Test 2x3 matrix with float32."""
        self._test_exp([2, 3], tf.float32)

    def testExpRowColumnVectorsFloat32(self):
        """Test row and column vectors with float32."""
        self._test_exp([1, 5], tf.float32)
        self._test_exp([5, 1], tf.float32)

    def testExpScalarFloat32(self):
        """Test scalar input with float32."""
        self._test_exp([], tf.float32)

    def testExpMatrixFloat16(self):
        """Test matrix with float16."""
        self._test_exp([4, 4], tf.float16, rtol=1e-2, atol=1e-2)

    def testExpMatrixBfloat16(self):
        """Test matrix with bfloat16."""
        self._test_exp([3, 3], tf.bfloat16, rtol=1e-2, atol=1e-2)

    def testExpLargeMatrixFloat32(self):
        """Test large 1024x1024 matrix with float32."""
        self._test_exp([1024, 1024], tf.float32)

    def testExpLargeVectorFloat32(self):
        """Test large vector with float32."""
        self._test_exp([2048], tf.float32)

    def testExpEdgeCases(self):
        """Test Exp with key edge case inputs."""
        test_values = [-10.0, -1.0, 0.0, 1.0, 10.0]
        x = tf.constant(test_values, dtype=tf.float32)

        cpu_result = self._test_op_device_placement(tf.exp, [x], '/CPU:0')
        musa_result = self._test_op_device_placement(tf.exp, [x], '/device:MUSA:0')

        # Expected mathematical results
        expected = np.array([
            np.exp(-10.0), # ~4.539993e-05
            np.exp(-1.0),  # ~0.367879
            np.exp(0.0),   # 1.0
            np.exp(1.0),   # ~2.718282
            np.exp(10.0)   # ~22026.465795
        ], dtype=np.float32)

        # Validate both CPU and MUSA against expected values
        self.assertAllClose(cpu_result.numpy(), expected, rtol=1e-5, atol=1e-8)
        self.assertAllClose(musa_result.numpy(), expected, rtol=1e-5, atol=1e-8)

    def testExpDemonstration(self):
        """Test demonstration cases to ensure basic functionality."""
        demo_cases = [
            ("Positive", [1.0, 2.0, 3.0]),
            ("Negative", [-1.0, -2.0, -3.0]),
            ("Zero", [0.0, 0.0, 0.0]),
            ("Mixed", [-2.0, 0.0, 2.0]),
        ]

        for name, values in demo_cases:
            with self.subTest(case=name):
                x = tf.constant(values, dtype=tf.float32)
                # Just run it on MUSA to ensure no crash
                _ = self._test_op_device_placement(tf.exp, [x], '/device:MUSA:0')


if __name__ == "__main__":
    tf.test.main()
