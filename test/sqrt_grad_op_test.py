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

"""Tests for MUSA SqrtGrad operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class SqrtGradOpTest(MUSATestCase):
  """Tests for MUSA SqrtGrad operator."""

  def _test_sqrt_grad_direct(self, shape, dtype, rtol=1e-3, atol=1e-3):
    """
    Test the raw SqrtGrad op directly: out = 0.5 * dy / y
    This matches 'test_sqrt_grad_logic' in the original script.
    """
    # Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    y_np = np.abs(np.random.randn(*shape).astype(np_dtype)) + 0.5
    dy_np = np.random.randn(*shape).astype(np_dtype)
    
    y = tf.constant(y_np, dtype=dtype)
    dy = tf.constant(dy_np, dtype=dtype)

    # Define Operator Wrapper
    def op_func(y_in, dy_in):
        return tf.raw_ops.SqrtGrad(y=y_in, dy=dy_in)

    # Compare Results
    self._compare_cpu_musa_results(op_func, [y, dy], dtype, rtol=rtol, atol=atol)

  def _test_sqrt_backprop(self, shape, dtype, rtol=1e-3, atol=1e-3):
    """
    Test the full backpropagation integration via GradientTape.
    This matches 'test_sqrt_integration' in the original script.
    """
    # Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    x_np = np.abs(np.random.randn(*shape).astype(np_dtype)) + 0.1
    x = tf.constant(x_np, dtype=dtype)

    # Define Gradient Calculation Wrapper
    def op_func(input_tensor):
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            res = tf.math.sqrt(input_tensor)
        return tape.gradient(res, input_tensor)

    # Compare Results
    self._compare_cpu_musa_results(op_func, [x], dtype, rtol=rtol, atol=atol)

  def testSqrtGradDirectBasic(self):
    """Test raw SqrtGrad op with standard types."""
    shape = [5, 5]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      # Relax tolerance for low precision types
      rtol = 3e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
      atol = 3e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
      
      self._test_sqrt_grad_direct(shape, dtype, rtol=rtol, atol=atol)

  def testSqrtGradDirectDouble(self):
    """Test raw SqrtGrad op with float64."""
    shape = [2, 3]
    self._test_sqrt_grad_direct(shape, tf.float64)

  def testSqrtIntegrationBasic(self):
    """Test Sqrt gradient integration (Tape) with standard types."""
    shape = [10, 10]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      # Relax tolerance for low precision types
      rtol = 3e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
      atol = 3e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
      
      self._test_sqrt_backprop(shape, dtype, rtol=rtol, atol=atol)

  def testSqrtIntegrationShapes(self):
    """Test Sqrt gradient integration with different shapes."""
    test_shapes = [
        [10],           # 1D
        [2, 3, 4],      # 3D
    ]
    for shape in test_shapes:
        self._test_sqrt_backprop(shape, tf.float32)


if __name__ == "__main__":
  tf.test.main()