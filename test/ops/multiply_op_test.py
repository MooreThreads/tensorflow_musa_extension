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

"""Tests for MUSA Multiply operator."""

import tensorflow as tf
import numpy as np

from musa_test_utils import MUSATestCase


class MultiplyOpTest(MUSATestCase):
  """Tests for MUSA Multiply operator."""

  def _test_multiply(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
    """Test multiply operation with given shapes and dtype."""
    if dtype == tf.bfloat16:
      x_np = np.random.uniform(-1, 1, size=shape_x).astype(np.float32)
      y_np = np.random.uniform(-1, 1, size=shape_y).astype(np.float32)
    else:
      x_np = np.random.uniform(-1, 1, size=shape_x).astype(dtype.as_numpy_dtype)
      y_np = np.random.uniform(-1, 1, size=shape_y).astype(dtype.as_numpy_dtype)
    
    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)
    
    # Test on CPU
    with tf.device('/CPU:0'):
      cpu_result = tf.multiply(x, y)
    
    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      musa_result = tf.multiply(x, y)
    
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

  def testMultiplyBasic(self):
    """Basic multiply test with same shapes."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_multiply([10, 10], [10, 10], dtype, rtol=rtol, atol=atol)

  def testMultiplyBroadcast(self):
    """Multiply with broadcasting."""
    test_cases = [
        ([10], [10, 10]),  # vector-matrix
        ([10, 1], [1, 10]),  # column-row
        ([5, 1, 3], [1, 7, 3]),  # 3D broadcasting
    ]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      for shape_x, shape_y in test_cases:
        with self.subTest(shape_x=shape_x, shape_y=shape_y, dtype=dtype):
          self._test_multiply(shape_x, shape_y, dtype, rtol=rtol, atol=atol)

  def testMultiplyScalar(self):
    """Multiply with scalar values."""
    for dtype in [tf.float32, tf.int32]:
      self._test_multiply([], [], dtype)

  def testMultiplyDifferentShapes(self):
    """Multiply with various different shapes."""
    test_cases = [
        ([1], [1]),
        ([3, 4], [3, 4]),
        ([2, 3, 4], [2, 3, 4]),
    ]
    for dtype in [tf.float32]:
      for shape_x, shape_y in test_cases:
        with self.subTest(shape_x=shape_x, shape_y=shape_y, dtype=dtype):
          self._test_multiply(shape_x, shape_y, dtype)

  def testMultiplyZeroValues(self):
    """Multiply with zero values."""
    x_data = [[0.0, 1.0], [2.0, 0.0]]
    y_data = [[1.0, 0.0], [0.0, 3.0]]
    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)
    
    expected = [[0.0, 0.0], [0.0, 0.0]]
    
    with tf.device('/device:MUSA:0'):
      result = tf.multiply(x, y)
    
    self.assertAllClose(result.numpy(), expected)


if __name__ == "__main__":
  tf.test.main()