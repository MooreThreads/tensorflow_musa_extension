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
"""Tests for MUSA Erf operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class ErfOpTest(MUSATestCase):
    """Tests for MUSA Erf operator."""

    def _test_erf(self, shape, dtype, rtol=1e-5, atol=1e-8):
        """Test erf operation with given shape and dtype."""
        # Determine the corresponding numpy dtype for data generation
        if dtype == tf.bfloat16:
            np_dtype = tf.bfloat16.as_numpy_dtype
        else:
            np_dtype = dtype.as_numpy_dtype

        # Generate random input data within a reasonable range for erf
        x_np = np.random.uniform(-3.0, 3.0, size=shape).astype(np_dtype)
        x = tf.constant(x_np, dtype=dtype)

        # Compare CPU and MUSA results
        self._compare_cpu_musa_results(tf.math.erf, [x], dtype, rtol=rtol, atol=atol)

    def testErfFloat32Small(self):
        """Test Erf with Float32 on a small tensor."""
        self._test_erf(shape=[20], dtype=tf.float32, rtol=1e-6, atol=1e-6)

    def testErfFloat32Large(self):
        """Test Erf with Float32 on a large tensor."""
        self._test_erf(shape=[10000], dtype=tf.float32, rtol=1e-6, atol=1e-6)

    def testErfFloat16(self):
        """Test Erf with Float16."""
        self._test_erf(shape=[1000], dtype=tf.float16, rtol=1e-3, atol=1e-3)

    def testErfFloat64(self):
        """Test Erf with Float64."""
        self._test_erf(shape=[1000], dtype=tf.float64, rtol=1e-14, atol=1e-14)

    def testErfEdgeCases(self):
        """Test Erf with edge case inputs."""
        # Create edge case inputs
        edge_vals = [0.0, float('inf'), float('-inf'), 100.0, -100.0]
        x = tf.constant(edge_vals, dtype=tf.float32)

        # Run on CPU and MUSA
        cpu_result = self._test_op_device_placement(tf.math.erf, [x], '/CPU:0')
        musa_result = self._test_op_device_placement(tf.math.erf, [x], '/device:MUSA:0')

        # Expected results for erf at these points
        expected = np.array([0.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)

        # Check both CPU and MUSA against expected values
        self.assertAllClose(cpu_result.numpy(), expected, rtol=0, atol=1e-6)
        self.assertAllClose(musa_result.numpy(), expected, rtol=0, atol=1e-6)


if __name__ == "__main__":
    tf.test.main()
