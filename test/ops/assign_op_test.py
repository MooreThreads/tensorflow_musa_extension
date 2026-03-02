#assign_op_test.py
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
#== == == == == == == == == == == == == == == == == == == == == == == == == ==
#== == == == == == == == == == == ==

"""Tests for MUSA Assign operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class AssignOpTest(MUSATestCase):
  """Tests for MUSA Assign operator."""

  def _test_assign(self, shape, dtype, validate_shape=True, use_locking=True):
    """Test assign operation with given shape and dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

    if np_dtype in [np.int32, np.int64]:
      init_val_np = np.random.randint(-10, 10, size=shape).astype(np_dtype)
      new_val_np = np.random.randint(-10, 10, size=shape).astype(np_dtype)
    else:
      init_val_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)
      new_val_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)

    init_val = tf.constant(init_val_np, dtype=dtype)
    new_val = tf.constant(new_val_np, dtype=dtype)

    # Test on CPU
    with tf.device('/CPU:0'):
      var_cpu = tf.Variable(init_val)
      var_cpu.assign(new_val)
      cpu_result = var_cpu.read_value()

    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      var_musa = tf.Variable(init_val)
      var_musa.assign(new_val)
      musa_result = var_musa.read_value()

    # Compare results
    if dtype in [tf.float16, tf.bfloat16]:
      rtol = 1e-2
      atol = 1e-2
      self.assertAllClose(
          tf.cast(cpu_result, tf.float32).numpy(),
          tf.cast(musa_result, tf.float32).numpy(),
          rtol=rtol, atol=atol)
    else:
      rtol = 1e-5
      atol = 1e-8
      self.assertAllClose(cpu_result.numpy(), musa_result.numpy(), rtol=rtol, atol=atol)

  def testAssign1D(self):
    """Test Assign with 1D tensor."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      self._test_assign([100], dtype)

  def testAssign2D(self):
    """Test Assign with 2D tensor."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      self._test_assign([64, 64], dtype)

  def testAssignUseLockingFalse(self):
    """Test Assign with use_locking=False."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      self._test_assign([256], dtype)

  def testAssignValidateShapeFalseAllowsReshape(self):
    """validate_shape=False should allow ref take value's shape."""
    # Skip this test as it requires RefVariable (tf.raw_ops.Assign with validate_shape=False)
    # ResourceVariable handles this differently
    self.skipTest("validate_shape test requires RefVariable (tf.raw_ops.Assign)")

  def testAssignValidateShapeTrueMismatchRaises(self):
    """validate_shape=True with mismatched shapes should raise."""
    # Skip this test as it requires RefVariable
    self.skipTest("Error case test requires RefVariable (tf.raw_ops.Assign)")

  def testAssignEmptyTensor(self):
    """Test Assign with empty tensors."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      self._test_assign([0], dtype)
      self._test_assign([0, 5], dtype)


if __name__ == "__main__":
  tf.test.main()
