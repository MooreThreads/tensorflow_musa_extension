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
    == == == == == == == == == == == == ==
"""Tests for MUSA VariableV2 operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class VariableV2OpTest(MUSATestCase):
  """Tests for MUSA VariableV2 operator."""

  def _make_value_np(self, shape, dtype):
    """Create numpy value tensor for given shape/dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

    if np_dtype in [np.int32, np.int64]:
#Keep small range to avoid overflow surprises.
      return np.random.randint(-10, 10, size=shape).astype(np_dtype)

    if dtype == tf.float16:
      return np.random.uniform(-1, 1, size=shape).astype(np.float16)
    if dtype == tf.bfloat16:
#bfloat16 typically fed as float32 and cast by TF.
      return np.random.uniform(-1, 1, size=shape).astype(np.float32)

    return np.random.uniform(-1, 1, size=shape).astype(np_dtype)

  def _run_variablev2_assign_read(self,
                                 device,
                                 var_shape,
                                 value_np,
                                 dtype,
                                 validate_shape=True,
                                 use_locking=True):
    """
    Build graph on `device`:
      var = VariableV2(var_shape, dtype)
      Assign(var, value, validate_shape)
      fetch var value after assign
    Return fetched numpy array.
    """
    g = tf.Graph()
    with g.as_default():
      with tf.device(device):
        value = tf.constant(value_np, dtype=dtype)

        var = tf.raw_ops.VariableV2(
            shape=var_shape,
            dtype=dtype,
            container="",
            shared_name="")

        assign = tf.raw_ops.Assign(
            ref=var,
            value=value,
            validate_shape=validate_shape,
            use_locking=use_locking)

#Ensure assign happens before reading variable value.
        out = tf.identity(assign)

    with tf.compat.v1.Session(graph=g) as sess:
      return sess.run(out)

  def _assert_close_by_dtype(self, cpu_out, musa_out, dtype, rtol, atol):
    """Match musa_test_utils.py comparison style for fp16/bf16."""
    if dtype in [tf.float16, tf.bfloat16]:
      self.assertAllClose(np.array(cpu_out, dtype=np.float32),
                          np.array(musa_out, dtype=np.float32),
                          rtol=rtol, atol=atol)
    else:
      self.assertAllClose(cpu_out, musa_out, rtol=rtol, atol=atol)

  def _test_variablev2_basic(self, shape, dtype):
    """Basic create+assign+read compare between CPU and MUSA."""
    rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
    atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8

    value_np = self._make_value_np(shape, dtype)

    cpu_out = self._run_variablev2_assign_read(
        device="/CPU:0",
        var_shape=shape,
        value_np=value_np,
        dtype=dtype,
        validate_shape=True,
        use_locking=True)

    musa_out = self._run_variablev2_assign_read(
        device="/device:MUSA:0",
        var_shape=shape,
        value_np=value_np,
        dtype=dtype,
        validate_shape=True,
        use_locking=True)

    self._assert_close_by_dtype(cpu_out, musa_out, dtype, rtol=rtol, atol=atol)

  def testVariableV2AssignRead1D(self):
    """Test VariableV2 + Assign + Read for 1D tensors."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      self._test_variablev2_basic([10], dtype)
      self._test_variablev2_basic([1024], dtype)

  def testVariableV2AssignRead2D(self):
    """Test VariableV2 + Assign + Read for 2D tensors."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      self._test_variablev2_basic([32, 32], dtype)
      self._test_variablev2_basic([256, 256], dtype)

  def testVariableV2EmptyTensor(self):
    """Test VariableV2 with empty shapes."""
    for dtype in [tf.float32, tf.int32]:
      self._test_variablev2_basic([0], dtype)
      self._test_variablev2_basic([0, 5], dtype)

  def testVariableV2ValidateShapeTrueMismatchRaises(self):
    """validate_shape=True should reject mismatched shapes."""
    dtype = tf.float32
    var_shape = [2, 3]
    value_np = self._make_value_np([3, 2], dtype)

#CPU should raise
    with self.assertRaises((ValueError,tf.errors.InvalidArgumentError)):
      _ = self._run_variablev2_assign_read(
          device="/CPU:0",
          var_shape=var_shape,
          value_np=value_np,
          dtype=dtype,
          validate_shape=True)

#MUSA should raise
    with self.assertRaises((ValueError,tf.errors.InvalidArgumentError)):
      _ = self._run_variablev2_assign_read(
          device="/device:MUSA:0",
          var_shape=var_shape,
          value_np=value_np,
          dtype=dtype,
          validate_shape=True)

  def testVariableV2ValidateShapeFalseAllowsReshape(self):
    """validate_shape=False should allow ref variable to take value's shape."""
    dtype = tf.float32
    var_shape = [2, 3]
    value_shape = [3, 2]
    value_np = self._make_value_np(value_shape, dtype)

    cpu_out = self._run_variablev2_assign_read(
        device="/CPU:0",
        var_shape=var_shape,
        value_np=value_np,
        dtype=dtype,
        validate_shape=False)

    musa_out = self._run_variablev2_assign_read(
        device="/device:MUSA:0",
        var_shape=var_shape,
        value_np=value_np,
        dtype=dtype,
        validate_shape=False)

#Values should match and output shape should follow value.
    self.assertEqual(list(cpu_out.shape), value_shape)
    self.assertEqual(list(musa_out.shape), value_shape)
    self.assertAllClose(cpu_out, musa_out, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
  tf.test.main()