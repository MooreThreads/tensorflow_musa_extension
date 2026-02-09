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

"""Tests for MUSA ReLU operator."""

import tensorflow as tf
import numpy as np

from musa_test_utils import MUSATestCase


class ReluOpTest(MUSATestCase):
  """Tests for MUSA ReLU operator."""

  def _test_relu(self, input_data, dtype, rtol=1e-5, atol=1e-8):
    """Test ReLU operation with given input data and dtype."""
    if dtype == tf.bfloat16:
      input_np = np.array(input_data, dtype=np.float32)
    else:
      input_np = np.array(input_data, dtype=dtype.as_numpy_dtype)
    
    x = tf.constant(input_np, dtype=dtype)
    
    # Test on CPU
    with tf.device('/CPU:0'):
      cpu_result = tf.nn.relu(x)
    
    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      musa_result = tf.nn.relu(x)
    
    # Compare results
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

  def testReluBasic(self):
    """Basic ReLU test with mixed positive/negative values."""
    test_data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_relu(test_data, dtype, rtol=rtol, atol=atol)

  def testRelu2D(self):
    """2D tensor ReLU test."""
    test_data = [[-1.0, 2.0], [3.0, -4.0]]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_relu(test_data, dtype, rtol=rtol, atol=atol)

  def testRelu3D(self):
    """3D tensor ReLU test."""
    test_data = [[[-1.0, 2.0], [3.0, -4.0]], [[5.0, -6.0], [-7.0, 8.0]]]
    for dtype in [tf.float32]:
      self._test_relu(test_data, dtype)

  def testReluAllPositive(self):
    """ReLU test with all positive values."""
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    for dtype in [tf.float32, tf.int32]:
      self._test_relu(test_data, dtype)

  def testReluAllNegative(self):
    """ReLU test with all negative values."""
    test_data = [-5.0, -4.0, -3.0, -2.0, -1.0]
    expected = [0.0, 0.0, 0.0, 0.0, 0.0]
    for dtype in [tf.float32, tf.int32]:
      self._test_relu(test_data, dtype)

  def testReluZeroInput(self):
    """ReLU test with zero input."""
    test_data = [0.0, 0.0, 0.0]
    for dtype in [tf.float32, tf.int32]:
      self._test_relu(test_data, dtype)


if __name__ == "__main__":
  tf.test.main()