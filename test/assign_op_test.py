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
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == == ==

"""Tests for MUSA Assign operator (Ref-variable tf.raw_ops.Assign)."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class AssignOpTest(MUSATestCase):
  """Tests for MUSA Assign operator."""

  def _run_assign_in_graph(self, ref_init_np, value_np, dtype,
                           validate_shape=True, use_locking=True):
    """
    Run tf.raw_ops.Assign in TF1 graph mode to force RefVariable path,
    return numpy array result.
    """
    g = tf.Graph()
    with g.as_default():
#Put constants into the graph
      ref_init = tf.constant(ref_init_np, dtype=dtype)
      value = tf.constant(value_np, dtype=dtype)

#Create RefVariable(NOT ResourceVariable)
#This is critical : Assign(ref, value) expects ref tensor from a Variable node.
      ref_var = tf.compat.v1.Variable(ref_init, use_resource=False, trainable=False)

      assign_out = tf.raw_ops.Assign(
          ref=ref_var,
          value=value,
          validate_shape=validate_shape,
          use_locking=use_locking)

      init_op = tf.compat.v1.global_variables_initializer()

#Run
      with tf.compat.v1.Session(graph=g) as sess:
        sess.run(init_op)
        out_np = sess.run(assign_out)

    return out_np

  def _test_assign(self, ref_shape, value_shape, dtype,
                   validate_shape=True, use_locking=True,
                   rtol=1e-5, atol=1e-8):
    """Test Assign operation with given shapes and dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

#Generate ref init and value
    if np_dtype in [np.int32, np.int64]:
      ref_init_np = np.random.randint(-10, 10, size=ref_shape).astype(np_dtype)
      value_np = np.random.randint(-10, 10, size=value_shape).astype(np_dtype)
    else:
      ref_init_np = np.random.uniform(-1, 1, size=ref_shape).astype(np_dtype)
      value_np = np.random.uniform(-1, 1, size=value_shape).astype(np_dtype)

#Create TensorFlow constants for inputs(to match addn style)
    ref_init_tf = tf.constant(ref_init_np, dtype=dtype)
    value_tf = tf.constant(value_np, dtype=dtype)

#Wrapper : must accept eager tensors(from _compare_cpu_musa_results),
#run graph - mode Assign inside, and return an eager Tensor.
    def assign_wrapper(ref_init_t, value_t):
      out_np = self._run_assign_in_graph(
          ref_init_t.numpy(),
          value_t.numpy(),
          dtype=dtype,
          validate_shape=validate_shape,
          use_locking=use_locking)
      return tf.constant(out_np, dtype=dtype)

    self._compare_cpu_musa_results(
        assign_wrapper, [ref_init_tf, value_tf], dtype, rtol=rtol, atol=atol)

  def _test_assign_expect_error(self, ref_shape, value_shape, dtype,
                                validate_shape=True, use_locking=True):
    """Test Assign should raise InvalidArgumentError when shapes mismatch and validate_shape=True."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    ref_init_np = np.random.uniform(-1, 1, size=ref_shape).astype(np_dtype)
    value_np = np.random.uniform(-1, 1, size=value_shape).astype(np_dtype)

    def run_on_device(device):
      with tf.device(device):
#直接跑 wrapper（里面用 graph + session）
        _ = self._run_assign_in_graph(
            ref_init_np, value_np, dtype=dtype,
            validate_shape=validate_shape, use_locking=use_locking)

#CPU should raise
    with self.assertRaises((ValueError,tf.errors.InvalidArgumentError)):
      run_on_device('/CPU:0')

#MUSA should raise
    with self.assertRaises((ValueError,tf.errors.InvalidArgumentError)):
      run_on_device('/device:MUSA:0')

#-- -- -- -- -- -- -- -- -- -- Tests -- -- -- -- -- -- -- -- -- --

  def testAssign1D(self):
    """Test Assign with 1D tensor."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_assign([100], [100], dtype, validate_shape=True, use_locking=True,
                        rtol=rtol, atol=atol)

  def testAssign2D(self):
    """Test Assign with 2D tensor."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_assign([64, 64], [64, 64], dtype, validate_shape=True, use_locking=True,
                        rtol=rtol, atol=atol)

  def testAssignUseLockingFalse(self):
    """Test Assign with use_locking=False."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_assign([256], [256], dtype, validate_shape=True, use_locking=False,
                        rtol=rtol, atol=atol)

  def testAssignValidateShapeFalseAllowsReshape(self):
    """validate_shape=False should allow ref take value's shape."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
#ref shape != value shape
      self._test_assign([2, 3], [3, 2], dtype, validate_shape=False, use_locking=True,
                        rtol=rtol, atol=atol)

  def testAssignValidateShapeTrueMismatchRaises(self):
    """validate_shape=True with mismatched shapes should raise."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      self._test_assign_expect_error([2, 3], [3, 2], dtype,
                                     validate_shape=True, use_locking=True)

  def testAssignEmptyTensor(self):
    """Test Assign with empty tensors."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_assign([0], [0], dtype, validate_shape=True, use_locking=True,
                        rtol=rtol, atol=atol)
      self._test_assign([0, 5], [0, 5], dtype, validate_shape=True, use_locking=True,
                        rtol=rtol, atol=atol)


if __name__ == "__main__":
  tf.test.main()
