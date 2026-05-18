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
"""Tests for MUSA TensorListPushBack operator."""

import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from musa_test_utils import MUSATestCase


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _make_input(shape, dtype):
  """Return a NumPy array and matching tf.constant for the given dtype."""
  np_dtype = dtype.as_numpy_dtype
  if dtype == tf.bfloat16:
    np_dtype = np.float32
  if dtype.is_integer:
    arr = np.random.randint(-10, 10, size=shape).astype(np_dtype)
  elif dtype == tf.float16:
    arr = np.random.uniform(-5.0, 5.0, size=shape).astype(np_dtype)
  elif dtype == tf.bfloat16:
    arr = np.random.uniform(-3.0, 3.0, size=shape).astype(np_dtype)
  else:
    arr = np.random.uniform(-10.0, 10.0, size=shape).astype(np_dtype)
  return arr, tf.constant(arr, dtype=dtype)


# ---------------------------------------------------------------------------
# TensorListPushBack
# ---------------------------------------------------------------------------

class TensorListPushBackOpTest(MUSATestCase):
  """Tests for MUSA TensorListPushBack operator."""

  def _push_and_stack(self, x, element_dtype):
    """Push every row of *x* into an EmptyTensorList, then Stack."""
    num_elements = x.shape[0]
    element_shape = tf.constant(list(x.shape[1:]), dtype=tf.int32)

    handle = tf.raw_ops.EmptyTensorList(
        element_shape=element_shape,
        max_num_elements=num_elements,
        element_dtype=element_dtype,
    )
    for i in range(num_elements):
      handle = tf.raw_ops.TensorListPushBack(
          input_handle=handle,
          tensor=x[i],
      )
    return tf.raw_ops.TensorListStack(
        input_handle=handle,
        element_shape=element_shape,
        element_dtype=element_dtype,
        num_elements=num_elements,
    )

  def _run(self, shape, dtype, rtol=1e-5, atol=1e-5):
    _, x = _make_input(shape, dtype)

    def func(inp):
      return self._push_and_stack(inp, element_dtype=dtype)

    self._compare_cpu_musa_results(func, [x], dtype, rtol=rtol, atol=atol)

  def testPushBackFloat32(self):
    self._run([4, 3], tf.float32)

  def testPushBackFloat16(self):
    self._run([3, 4, 5], tf.float16, rtol=1e-2, atol=1e-2)

  def testPushBackBFloat16(self):
    self._run([4, 8], tf.bfloat16, rtol=1e-1, atol=1e-1)

  def testPushBackFloat64(self):
    self._run([5, 6], tf.float64)

  def testPushBackInt32(self):
    self._run([6, 4], tf.int32, rtol=0, atol=0)

  def testPushBackInt64(self):
    self._run([3, 7], tf.int64, rtol=0, atol=0)

  def testPushBackScalarElements(self):
    """Push scalar (rank-0) elements."""
    dtype = tf.float32
    elems = tf.constant([1.0, 2.0, 3.0], dtype=dtype)
    element_shape = tf.constant([], dtype=tf.int32)  # scalar shape

    def func(inp):
      handle = tf.raw_ops.EmptyTensorList(
          element_shape=element_shape,
          max_num_elements=3,
          element_dtype=dtype,
      )
      for i in range(3):
        handle = tf.raw_ops.TensorListPushBack(
            input_handle=handle,
            tensor=inp[i],
        )
      return tf.raw_ops.TensorListStack(
          input_handle=handle,
          element_shape=element_shape,
          element_dtype=dtype,
          num_elements=3,
      )

    self._compare_cpu_musa_results(func, [elems], dtype)

  def testPushBackPreservesOrder(self):
    """Verify that push order is preserved (FIFO stack → bottom-to-top)."""
    dtype = tf.float32
    values = np.arange(5, dtype=np.float32).reshape(5, 1)
    x = tf.constant(values, dtype=dtype)
    element_shape = tf.constant([1], dtype=tf.int32)

    with tf.device('/device:MUSA:0'):
      handle = tf.raw_ops.EmptyTensorList(
          element_shape=element_shape,
          max_num_elements=5,
          element_dtype=dtype,
      )
      for i in range(5):
        handle = tf.raw_ops.TensorListPushBack(
            input_handle=handle,
            tensor=x[i],
        )
      result = tf.raw_ops.TensorListStack(
          input_handle=handle,
          element_shape=element_shape,
          element_dtype=dtype,
          num_elements=5,
      )

    self.assertAllClose(result.numpy().flatten(), values.flatten())


if __name__ == "__main__":
  tf.test.main()
