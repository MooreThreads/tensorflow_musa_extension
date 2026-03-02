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

"""Tests for MUSA Square operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class SquareOpTest(MUSATestCase):
  """Tests for MUSA Square operator."""

  def _test_square_forward(self, shape, dtype, rtol=1e-5, atol=1e-8):
    """Test forward square operation."""
    # Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    if np.issubdtype(np_dtype, np.integer):
        x_np = np.random.randint(-1000, 1000, size=shape).astype(np_dtype)
    else:
        x_np = np.array(np.random.randn(*shape)) * 10.0
        x_np = x_np.astype(np_dtype)
        
        if x_np.size > 0:
            flat_x = x_np.ravel()
            flat_x[0] = 0.0
            if x_np.size > 1: flat_x[1] = -5.5
            x_np = flat_x.reshape(shape)

    x = tf.constant(x_np, dtype=dtype)

    # Define Operator Wrapper
    def op_func(input_tensor):
        return tf.math.square(input_tensor)

    # Compare Results
    self._compare_cpu_musa_results(op_func, [x], dtype, rtol=rtol, atol=atol)

  def _test_square_grad(self, shape, dtype, rtol=1e-5, atol=1e-8):
    """Test backward (gradient) square operation."""
    if not dtype.is_floating:
        return

    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    x_np = np.array(np.random.randn(*shape)).astype(np_dtype)
    x = tf.constant(x_np, dtype=dtype)

    def op_grad(input_tensor):
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            y = tf.math.square(input_tensor)
        return tape.gradient(y, input_tensor)

    self._compare_cpu_musa_results(op_grad, [x], dtype, rtol=rtol, atol=atol)

  def testSquareForwardFloat(self):
    """Test forward pass for floating point types."""
    shapes = [(), (10,), (5, 5), (2, 3, 4, 5)]
    for shape in shapes:
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            
            self._test_square_forward(shape, dtype, rtol=rtol, atol=atol)

  def testSquareForwardInt(self):
    """Test forward pass for integer types."""
    shapes = [(), (10,), (5, 5)]
    for shape in shapes:
        for dtype in [tf.int32, tf.int64]:
            self._test_square_forward(shape, dtype)

  def testSquareForwardDouble(self):
    """Test forward pass for float64 (fallback check)."""
    self._test_square_forward((10,), tf.float64)

  def testSquareGradient(self):
    """Test gradients for floating point types."""
    grad_shapes = [(), (2, 2), (2, 3, 2)]
    for shape in grad_shapes:
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-4
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-4
            
            self._test_square_grad(shape, dtype, rtol=rtol, atol=atol)


if __name__ == "__main__":
  tf.test.main()