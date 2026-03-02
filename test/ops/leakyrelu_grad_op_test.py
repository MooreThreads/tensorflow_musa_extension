#Copyright 2026 The TensorFlow MUSA Authors.All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http:  // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#== == == == == == == == == == == == == == == == == == == == == == == == == == \

"""Tests for MUSA LeakyReluGrad operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class LeakyReluGradOpTest(MUSATestCase):
  """Tests for MUSA LeakyReluGrad operator."""

  def _test_leakyrelu_grad_direct(self, shape, dtype, alpha=0.2, rtol=1e-3, atol=1e-3):
    """
    Test raw LeakyReluGrad op directly:
      dx = dy * (x > 0 ? 1 : alpha)
    Notes:
      - TF 通常把 x==0 归到 <=0 分支（乘 alpha）
    """
#Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

    x_np = np.random.randn(*shape).astype(np_dtype)
    dy_np = np.random.randn(*shape).astype(np_dtype)

    x = tf.constant(x_np, dtype=dtype)
    dy = tf.constant(dy_np, dtype=dtype)

#Define Operator Wrapper(raw op)
    def op_func(grad_in, feat_in):
      return tf.raw_ops.LeakyReluGrad(
          gradients=grad_in,
          features=feat_in,
          alpha=alpha
      )

#Compare Results
    self._compare_cpu_musa_results(op_func, [dy, x], dtype, rtol=rtol, atol=atol)

  def _test_leakyrelu_backprop(self, shape, dtype, alpha=0.2, rtol=1e-3, atol=1e-3):
    """
    Test full backprop integration via GradientTape:
      y = leaky_relu(x, alpha)
      dx = d(y)/d(x)
    """
#Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

    x_np = np.random.randn(*shape).astype(np_dtype)
    x = tf.constant(x_np, dtype=dtype)

#Define Gradient Calculation Wrapper
    def op_func(input_tensor):
      with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        res = tf.nn.leaky_relu(input_tensor, alpha=alpha)
      return tape.gradient(res, input_tensor)

#Compare Results
    self._compare_cpu_musa_results(op_func, [x], dtype, rtol=rtol, atol=atol)

  def testLeakyReluGradDirectBasic(self):
    """Test raw LeakyReluGrad op with standard types."""
    shape = [5, 5]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 3e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
      atol = 3e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
      self._test_leakyrelu_grad_direct(shape, dtype, alpha=0.2, rtol=rtol, atol=atol)

  def testLeakyReluGradDirectDouble(self):
    """Test raw LeakyReluGrad op with float64."""
    shape = [2, 3]
    self._test_leakyrelu_grad_direct(shape, tf.float64, alpha=0.2, rtol=1e-6, atol=1e-6)

  def testLeakyReluGradDirectDifferentAlpha(self):
    """Test raw LeakyReluGrad op with different alpha values."""
    shape = [2, 3, 4]
    alphas = [0.0, 0.01, 0.1, 0.2, 0.5, 1.0]
    for alpha in alphas:
      for dtype in [tf.float32, tf.float16, tf.bfloat16]:
        rtol = 5e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
        atol = 5e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
        self._test_leakyrelu_grad_direct(shape, dtype, alpha=alpha, rtol=rtol, atol=atol)

  def testLeakyReluIntegrationBasic(self):
    """Test LeakyRelu gradient integration (Tape) with standard types."""
    shape = [10, 10]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 3e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
      atol = 3e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
      self._test_leakyrelu_backprop(shape, dtype, alpha=0.2, rtol=rtol, atol=atol)

  def testLeakyReluIntegrationShapes(self):
    """Test LeakyRelu gradient integration with different shapes."""
    test_shapes = [
        [10],        # 1D
        [2, 3, 4],   # 3D
    ]
    for shape in test_shapes:
      self._test_leakyrelu_backprop(shape, tf.float32, alpha=0.2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  tf.test.main()