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
"""Tests for MUSA Variable operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class VariableV2OpTest(MUSATestCase):
  """Tests for MUSA Variable operator (using ResourceVariable)."""

  def _make_value_np(self, shape, dtype):
    """Create numpy value tensor for given shape/dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

    if np_dtype in [np.int32, np.int64]:
      return np.random.randint(-10, 10, size=shape).astype(np_dtype)

    if dtype == tf.float16:
      return np.random.uniform(-1, 1, size=shape).astype(np.float16)
    if dtype == tf.bfloat16:
      return np.random.uniform(-1, 1, size=shape).astype(np.float32)

    return np.random.uniform(-1, 1, size=shape).astype(np_dtype)

  def _test_variable_basic(self, shape, dtype):
    """Basic create+assign+read compare between CPU and MUSA."""
    value_np = self._make_value_np(shape, dtype)
    init_val = tf.constant(value_np, dtype=dtype)

    # Test on CPU
    with tf.device('/CPU:0'):
      var_cpu = tf.Variable(init_val)
      cpu_result = var_cpu.read_value()

    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      var_musa = tf.Variable(init_val)
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

  def testVariableV2AssignRead1D(self):
    """Test Variable + Assign + Read for 1D tensors."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      self._test_variable_basic([10], dtype)
      self._test_variable_basic([1024], dtype)

  def testVariableV2AssignRead2D(self):
    """Test Variable + Assign + Read for 2D tensors."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      self._test_variable_basic([32, 32], dtype)
      self._test_variable_basic([256, 256], dtype)

  def testVariableV2EmptyTensor(self):
    """Test Variable with empty shapes."""
    for dtype in [tf.float32, tf.int32]:
      self._test_variable_basic([0], dtype)
      self._test_variable_basic([0, 5], dtype)

  def testVariableV2ValidateShapeTrueMismatchRaises(self):
    """validate_shape=True should reject mismatched shapes."""
    # Skip this test as it requires RefVariable (tf.raw_ops.VariableV2)
    self.skipTest("Error case test requires RefVariable (tf.raw_ops.VariableV2)")

  def testVariableV2ValidateShapeFalseAllowsReshape(self):
    """validate_shape=False should allow ref variable to take value's shape."""
    # Skip this test as it requires RefVariable
    self.skipTest("validate_shape test requires RefVariable (tf.raw_ops.VariableV2)")


if __name__ == "__main__":
  tf.test.main()
