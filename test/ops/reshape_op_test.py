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

"""Tests for MUSA Reshape operator."""

import tensorflow as tf
import numpy as np

from musa_test_utils import MUSATestCase


class ReshapeOpTest(MUSATestCase):
  """Tests for MUSA Reshape operator."""

  def _test_reshape(self, input_shape, target_shape, dtype):
    """Test reshape operation with given input and target shapes."""
    if dtype == tf.bfloat16:
      input_data = np.random.uniform(-1, 1, size=input_shape).astype(np.float32)
    else:
      input_data = np.random.uniform(-1, 1, size=input_shape).astype(dtype.as_numpy_dtype)
    
    x = tf.constant(input_data, dtype=dtype)
    shape_tensor = tf.constant(target_shape, dtype=tf.int32)
    
    # Test on CPU
    with tf.device('/CPU:0'):
      cpu_result = tf.reshape(x, shape_tensor)
    
    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      musa_result = tf.reshape(x, shape_tensor)
    
    # Compare results
    if dtype in [tf.float16, tf.bfloat16]:
      cpu_result_f32 = tf.cast(cpu_result, tf.float32)
      musa_result_f32 = tf.cast(musa_result, tf.float32)
      self.assertAllClose(cpu_result_f32.numpy(), 
                         musa_result_f32.numpy(),
                         rtol=1e-2, 
                         atol=1e-2)
    else:
      self.assertAllClose(cpu_result.numpy(), 
                         musa_result.numpy())

  def testReshapeBasic(self):
    """Basic reshape test from 2D to 1D."""
    self._test_reshape([4, 5], [20], tf.float32)

  def testReshape2DTo2D(self):
    """Reshape from one 2D shape to another."""
    self._test_reshape([4, 5], [2, 10], tf.float32)

  def testReshape3DTo2D(self):
    """Reshape from 3D to 2D."""
    self._test_reshape([2, 3, 4], [6, 4], tf.float32)

  def testReshapeWithMinusOne(self):
    """Reshape with -1 dimension."""
    self._test_reshape([12], [-1, 3], tf.float32)

  def testReshapeDifferentDtypes(self):
    """Test reshape with different data types."""
    input_shape = [2, 3]
    target_shape = [6]
    dtypes = [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]
    
    for dtype in dtypes:
      with self.subTest(dtype=dtype):
        self._test_reshape(input_shape, target_shape, dtype)

  def testReshapeLargeTensor(self):
    """Test reshape with larger tensor."""
    self._test_reshape([100, 100], [10000], tf.float32)

  def testReshapeScalar(self):
    """Test reshape scalar to 1D and back."""
    # Scalar to 1D
    scalar_data = np.array(42.0, dtype=np.float32)
    x = tf.constant(scalar_data, dtype=tf.float32)
    shape_tensor = tf.constant([1], dtype=tf.int32)
    
    with tf.device('/device:MUSA:0'):
      result = tf.reshape(x, shape_tensor)
    
    self.assertAllEqual(result.shape, [1])
    self.assertAllClose(result.numpy(), [42.0])

    # 1D to scalar
    x_1d = tf.constant([42.0], dtype=tf.float32)
    shape_scalar = tf.constant([], dtype=tf.int32)
    
    with tf.device('/device:MUSA:0'):
      result_scalar = tf.reshape(x_1d, shape_scalar)
    
    self.assertAllEqual(result_scalar.shape, [])
    self.assertAllClose(result_scalar.numpy(), 42.0)


if __name__ == "__main__":
  tf.test.main()