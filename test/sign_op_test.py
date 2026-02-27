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

"""Tests for MUSA Sign operator."""

import tensorflow as tf
import numpy as np

from musa_test_utils import MUSATestCase


class SignOpTest(MUSATestCase):
  """Tests for MUSA Sign operator."""

  def _test_sign(self, input_data, dtype, rtol=1e-5, atol=1e-8):
    """Test sign operation with given input data and dtype."""
    if dtype == tf.bfloat16:
      input_np = np.array(input_data, dtype=np.float32)
    else:
      input_np = np.array(input_data, dtype=dtype.as_numpy_dtype)
    
    x = tf.constant(input_np, dtype=dtype)
    
    
    with tf.device('/CPU:0'):
      cpu_result = tf.sign(x)
    
    
    with tf.device('/device:MUSA:0'):
      musa_result = tf.sign(x)
    
    
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

  def testSignBasicFloat(self):
    """Basic sign test with float types."""
    test_data = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    for dtype in [tf.float32, tf.float64, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      with self.subTest(dtype=dtype):
        self._test_sign(test_data, dtype, rtol=rtol, atol=atol)

  def testSignBasicInt(self):
    """Basic sign test with integer types."""
    test_data = [-5, -2, -1, 0, 1, 2, 5]
    for dtype in [tf.int32, tf.int64]:
      with self.subTest(dtype=dtype):
        self._test_sign(test_data, dtype)

  def testSign2D(self):
    """2D tensor sign test."""
    test_data = [[-2.0, 0.0, 2.0], [1.0, -1.0, 3.0]]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      with self.subTest(dtype=dtype):
        self._test_sign(test_data, dtype, rtol=rtol, atol=atol)

  def testSign3D(self):
    """3D tensor sign test."""
    test_data = [[[-1.0, 0.0], [2.0, -3.0]], [[4.0, -5.0], [0.0, 6.0]]]
    for dtype in [tf.float32]:
      with self.subTest(dtype=dtype):
        self._test_sign(test_data, dtype)

  def testSignZeroInput(self):
    """Sign test with zero input - should return 0."""
    test_data = [0.0, 0.0, 0.0]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      with self.subTest(dtype=dtype):
        self._test_sign(test_data, dtype, rtol=rtol, atol=atol)

  def testSignAllPositive(self):
    """Sign test with all positive values - should return all 1s."""
    test_data = [0.1, 1.0, 10.0, 100.0, 1000.0]
    for dtype in [tf.float32, tf.float64]:
      with self.subTest(dtype=dtype):
        self._test_sign(test_data, dtype)

  def testSignAllNegative(self):
    """Sign test with all negative values - should return all -1s."""
    test_data = [-0.1, -1.0, -10.0, -100.0, -1000.0]
    for dtype in [tf.float32, tf.float64]:
      with self.subTest(dtype=dtype):
        self._test_sign(test_data, dtype)

  def testSignMixedValues(self):
    """Sign test with mixed positive, negative, and zero values."""
    test_data = [-100.0, -1.0, -0.001, 0.0, 0.001, 1.0, 100.0]
    for dtype in [tf.float32]:
      with self.subTest(dtype=dtype):
        self._test_sign(test_data, dtype)

  def testSignLargeValues(self):
    """Sign test with large positive/negative values."""
    test_data = [-1e10, -1e5, 1e5, 1e10]
    for dtype in [tf.float32, tf.float64]:
      with self.subTest(dtype=dtype):
        self._test_sign(test_data, dtype)

  def testSignSmallValues(self):
    """Sign test with very small positive/negative values."""
    test_data = [-1e-10, -1e-20, 1e-20, 1e-10]
    for dtype in [tf.float32, tf.float64]:
      with self.subTest(dtype=dtype):
        self._test_sign(test_data, dtype)

  def testSignRandomDataFloat32(self):
    """Sign test with random float32 data."""
    np.random.seed(42)
    test_data = np.random.uniform(-100, 100, size=[100]).tolist()
    self._test_sign(test_data, tf.float32)

  def testSignRandomDataInt32(self):
    """Sign test with random int32 data."""
    np.random.seed(42)
    test_data = np.random.randint(-100, 100, size=[100]).tolist()
    self._test_sign(test_data, tf.int32)

  def testSignLargeTensor(self):
    """Sign test with large tensor."""
    np.random.seed(42)
    test_data = np.random.uniform(-10, 10, size=[1000]).tolist()
    for dtype in [tf.float32]:
      with self.subTest(dtype=dtype):
        self._test_sign(test_data, dtype)

  def testSignEmptyTensor(self):
    """Sign test with empty tensor."""
    for dtype in [tf.float32, tf.int32]:
      with self.subTest(dtype=dtype):
        x = tf.constant([], dtype=dtype)
        
        with tf.device('/CPU:0'):
          cpu_result = tf.sign(x)
        
        with tf.device('/device:MUSA:0'):
          musa_result = tf.sign(x)
        
        self.assertAllClose(cpu_result.numpy(), musa_result.numpy())


if __name__ == "__main__":
  tf.test.main()