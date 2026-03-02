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

"""Tests for MUSA StopGradient operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class StopGradientOpTest(MUSATestCase):
  """Tests for MUSA StopGradient operator."""

  def _test_forward(self, shape, dtype, rtol=1e-5, atol=1e-8):
    """Test that stop_gradient acts as identity in forward pass."""
    # Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    if np.issubdtype(np_dtype, np.integer):
        x_np = np.random.randint(-100, 100, size=shape).astype(np_dtype)
    else:
        x_np = np.random.uniform(-10, 10, size=shape).astype(np_dtype)
        
    x = tf.constant(x_np, dtype=dtype)

    # Define Forward Operator
    def op_func(input_tensor):
        return tf.stop_gradient(input_tensor)

    # Compare Results (CPU vs MUSA)
    self._compare_cpu_musa_results(op_func, [x], dtype, rtol=rtol, atol=atol)

  def _test_backward(self, shape, dtype, rtol=1e-5, atol=1e-8):
    """Test that stop_gradient blocks gradients (backward pass)."""
    if not dtype.is_floating:
        return

    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    x_np = np.random.randn(*shape).astype(np_dtype)
    x = tf.constant(x_np, dtype=dtype)

    # Define Gradient Calculation Wrapper
    def op_grad(input_tensor):
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            # Perform some operation
            intermediate = input_tensor * 2.0
            # Block gradient
            stopped = tf.stop_gradient(intermediate)
            # Continue operation
            final = stopped + 1.0
            
        grad = tape.gradient(final, input_tensor, 
                             unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return grad

    # Compare Results
    self._compare_cpu_musa_results(op_grad, [x], dtype, rtol=rtol, atol=atol)

  def testStopGradientForwardBasic(self):
    """Test forward pass identity for various types."""
    shape = [2, 5]
    
    # Test Floating points
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_forward(shape, dtype, rtol=rtol, atol=atol)
      
    for dtype in [tf.int32, tf.int64]:
      self._test_forward(shape, dtype)

  def testStopGradientBackwardBlocking(self):
    """Test gradient blocking logic."""
    shape = [2, 5]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_backward(shape, dtype, rtol=rtol, atol=atol)

  def testStopGradientShapes(self):
    """Test with different shapes."""
    shapes = [
        [10],           # 1D
        [5, 5],         # 2D
        [2, 3, 4],      # 3D
    ]
    for shape in shapes:
        self._test_forward(shape, tf.float32)
        self._test_backward(shape, tf.float32)


if __name__ == "__main__":
  tf.test.main()