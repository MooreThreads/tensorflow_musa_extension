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

"""Tests for MUSA LeakyReLU operator."""

import tensorflow as tf
import numpy as np

from musa_test_utils import MUSATestCase


class LeakyReluOpTest(MUSATestCase):
  """Tests for MUSA LeakyReLU operator."""

  def _test_leaky_relu(self, input_data, dtype, alpha=0.2, rtol=1e-5, atol=1e-8):
    """Test LeakyReLU operation with given input data, dtype, and alpha."""
    if dtype == tf.bfloat16:
      input_np = np.array(input_data, dtype=np.float32)
    else:
      input_np = np.array(input_data, dtype=dtype.as_numpy_dtype)
    
    x = tf.constant(input_np, dtype=dtype)

#Test on CPU
    with tf.device('/CPU:0'):
      cpu_result = tf.nn.leaky_relu(x, alpha=alpha)

#Test on MUSA
    with tf.device('/device:MUSA:0'):
      musa_result = tf.nn.leaky_relu(x, alpha=alpha)

#Compare results
    if dtype in [tf.float16, tf.bfloat16]:
      cpu_result_f32 = tf.cast(cpu_result, tf.float32)
      musa_result_f32 = tf.cast(musa_result, tf.float32)
      self.assertAllClose(cpu_result_f32.numpy(), 
                         musa_result_f32.numpy(),
                         rtol=rtol, 
                         atol=atol)
    else:
      self.assertAllClose(cpu_result.numpy(), 
                         musa_result.numpy(),
                         rtol=rtol, 
                         atol=atol)

  def testLeakyReluBasic(self):
    """Basic LeakyReLU test with mixed positive/negative values."""
    test_data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_leaky_relu(test_data, dtype, rtol=rtol, atol=atol)

  def testLeakyRelu2D(self):
    """2D tensor LeakyReLU test."""
    test_data = [[-1.0, 2.0], [3.0, -4.0]]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_leaky_relu(test_data, dtype, rtol=rtol, atol=atol)

  def testLeakyRelu3D(self):
    """3D tensor LeakyReLU test."""
    test_data = [[[-1.0, 2.0], [3.0, -4.0]], [[5.0, -6.0], [-7.0, 8.0]]]
    for dtype in [tf.float32]:
      self._test_leaky_relu(test_data, dtype)

  def testLeakyReluAllPositive(self):
    """LeakyReLU test with all positive values."""
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    for dtype in [tf.float32, tf.int32]:
      self._test_leaky_relu(test_data, dtype)

  def testLeakyReluAllNegative(self):
    """LeakyReLU test with all negative values."""
    test_data = [-5.0, -4.0, -3.0, -2.0, -1.0]
    for dtype in [tf.float32, tf.int32]:
      self._test_leaky_relu(test_data, dtype)

  def testLeakyReluZeroInput(self):
    """LeakyReLU test with zero input."""
    test_data = [0.0, 0.0, 0.0]
    for dtype in [tf.float32, tf.int32]:
      self._test_leaky_relu(test_data, dtype)

  def testLeakyReluDifferentAlpha(self):
    """LeakyReLU test with different alpha values."""
    test_data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    alphas = [0.01, 0.1, 0.2, 0.5, 1.0]
    for alpha in alphas:
      for dtype in [tf.float32, tf.float16]:
        rtol = 1e-2 if dtype == tf.float16 else 1e-5
        atol = 1e-2 if dtype == tf.float16 else 1e-8
        self._test_leaky_relu(test_data, dtype, alpha=alpha, rtol=rtol, atol=atol)

  def testLeakyReluAlphaZero(self):
    """LeakyReLU test with alpha=0 (should behave like ReLU)."""
    test_data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_leaky_relu(test_data, dtype, alpha=0.0, rtol=rtol, atol=atol)


if __name__ == "__main__":
  tf.test.main()