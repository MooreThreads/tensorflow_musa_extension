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

"""Tests for MUSA Assign operator."""

import tensorflow as tf
import numpy as np

from musa_test_utils import MUSATestCase


class AssignOpTest(MUSATestCase):
  """Tests for MUSA Assign operator."""

  def _test_assign(self, shape, dtype):
    """Test assign operation with given shape and dtype."""
    if dtype == tf.bfloat16:
      init_val_np = np.random.uniform(-1, 1, size=shape).astype(np.float32)
      new_val_np = np.random.uniform(-1, 1, size=shape).astype(np.float32)
    else:
      init_val_np = np.random.uniform(-1, 1, size=shape).astype(dtype.as_numpy_dtype)
      new_val_np = np.random.uniform(-1, 1, size=shape).astype(dtype.as_numpy_dtype)
    
    init_val = tf.constant(init_val_np, dtype=dtype)
    new_val = tf.constant(new_val_np, dtype=dtype)
    
    with tf.device('/device:MUSA:0'):
      var = tf.Variable(init_val)
      var.assign(new_val)
      result = var.read_value()
    
    # Compare with expected value
    if dtype == tf.bfloat16:
      expected = tf.constant(new_val_np, dtype=tf.float32)
      actual = tf.cast(result, tf.float32)
      self.assertAllClose(expected.numpy(), actual.numpy(), rtol=1e-2, atol=1e-2)
    else:
      expected = tf.constant(new_val_np, dtype=dtype)
      self.assertAllClose(expected.numpy(), result.numpy())

  def testAssignBasicFloat32(self):
    """Basic float32 assignment test."""
    self._test_assign([3], tf.float32)

  def testAssignScalar(self):
    """Scalar assignment test."""
    self._test_assign([], tf.float32)

  def testAssign1D(self):
    """1D tensor assignment test."""
    self._test_assign([100], tf.float32)

  def testAssign2D(self):
    """2D tensor assignment test."""
    self._test_assign([32, 64], tf.float32)

  def testAssign3D(self):
    """3D tensor assignment test."""
    self._test_assign([8, 16, 32], tf.float32)

  def testAssign4D(self):
    """4D tensor assignment test."""
    self._test_assign([2, 4, 8, 16], tf.float32)

if __name__ == "__main__":
  tf.test.main()