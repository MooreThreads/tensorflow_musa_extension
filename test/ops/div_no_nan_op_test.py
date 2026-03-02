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
"""Tests for MUSA DivNoNan operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class DivNoNanOpTest(MUSATestCase):
    """Tests for MUSA DivNoNan operator."""

    def _test_div_no_nan(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
        """Test div_no_nan operation with given shapes and dtype."""
        np_dtype = dtype.as_numpy_dtype
        x_np = np.random.uniform(-10, 10, size=shape_x).astype(np_dtype)
        y_np = np.random.uniform(-10, 10, size=shape_y).astype(np_dtype)
        
        # Critical: Set some elements to zero to test DivNoNan behavior (result should be 0 where y=0)
        if y_np.size > 0:
            indices = np.random.choice(y_np.size, size=min(3, y_np.size), replace=False)
            y_np.ravel()[indices] = 0.0
        
        x = tf.constant(x_np, dtype=dtype)
        y = tf.constant(y_np, dtype=dtype)
        
        # Test on CPU and MUSA
        def op_func(x, y):
            return tf.raw_ops.DivNoNan(x=x, y=y)
        
        self._compare_cpu_musa_results(op_func, [x, y], dtype, rtol=rtol, atol=atol)

    def testBasic(self):
        """Test basic div_no_nan operation with same shapes (float16/float32 only)."""
        for dtype in [tf.float32, tf.float16]:  # REMOVED tf.bfloat16 - NOT SUPPORTED
            rtol = 1e-2 if dtype == tf.float16 else 1e-5
            atol = 1e-2 if dtype == tf.float16 else 1e-8
            self._test_div_no_nan([100], [100], dtype, rtol=rtol, atol=atol)
            self._test_div_no_nan([10, 10], [10, 10], dtype, rtol=rtol, atol=atol)

    def testBroadcast(self):
        """Test div_no_nan with broadcasting (float16/float32 only)."""
        for dtype in [tf.float32, tf.float16]:  # REMOVED tf.bfloat16 - NOT SUPPORTED
            rtol = 1e-2 if dtype == tf.float16 else 1e-5
            atol = 1e-2 if dtype == tf.float16 else 1e-8
            self._test_div_no_nan([10, 10], [1], dtype, rtol=rtol, atol=atol)
            self._test_div_no_nan([10, 10], [10, 1], dtype, rtol=rtol, atol=atol)
            self._test_div_no_nan([1, 4, 8, 8], [1, 4, 1, 1], dtype, rtol=rtol, atol=atol)

    def testFloat64(self):
        """Test div_no_nan with float64 (fully supported type)."""
        # float64 has higher precision - use tighter tolerances
        self._test_div_no_nan([10], [10], tf.float64, rtol=1e-12, atol=1e-12)
        self._test_div_no_nan([5, 5], [5, 5], tf.float64, rtol=1e-12, atol=1e-12)

    def testGradient(self):
        """Test gradient of div_no_nan operation (float16/float32 only)."""
        for dtype in [tf.float32, tf.float16]:  # REMOVED tf.bfloat16 - NOT SUPPORTED
            rtol = 1e-2 if dtype == tf.float16 else 1e-5
            atol = 1e-2 if dtype == tf.float16 else 1e-8
            self._test_div_no_nan_gradient([10], [10], dtype, rtol=rtol, atol=atol)
            self._test_div_no_nan_gradient([5, 5], [5, 1], dtype, rtol=rtol, atol=atol)

    def _test_div_no_nan_gradient(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
        """Test gradient of div_no_nan operation with given shapes and dtype."""
        np_dtype = dtype.as_numpy_dtype
        x_np = np.random.uniform(-10, 10, size=shape_x).astype(np_dtype)
        y_np = np.random.uniform(-10, 10, size=shape_y).astype(np_dtype)
        
        # Ensure at least one zero in y for DivNoNan behavior test
        if y_np.size > 0:
            y_np.ravel()[0] = 0.0
        
        x = tf.constant(x_np, dtype=dtype)
        y = tf.constant(y_np, dtype=dtype)
        
        # Compute gradients on MUSA device
        with tf.GradientTape() as tape:
            tape.watch([x, y])
            z = tf.raw_ops.DivNoNan(x=x, y=y)
            loss = tf.reduce_sum(z)
        grad_musa = tape.gradient(loss, [x, y])
        
        # Compute gradients on CPU for comparison
        with tf.device('/CPU:0'):
            x_cpu = tf.constant(x_np, dtype=dtype)
            y_cpu = tf.constant(y_np, dtype=dtype)
            with tf.GradientTape() as tape_cpu:
                tape_cpu.watch([x_cpu, y_cpu])
                z_cpu = tf.raw_ops.DivNoNan(x=x_cpu, y=y_cpu)
                loss_cpu = tf.reduce_sum(z_cpu)
            grad_cpu = tape_cpu.gradient(loss_cpu, [x_cpu, y_cpu])
        
        # CRITICAL FIX: Removed 'msg' parameter (TensorFlow assertAllClose does NOT support it)
        self.assertAllClose(grad_cpu[0].numpy(), grad_musa[0].numpy(), rtol=rtol, atol=atol)
        self.assertAllClose(grad_cpu[1].numpy(), grad_musa[1].numpy(), rtol=rtol, atol=atol)

if __name__ == "__main__":
    tf.test.main()
