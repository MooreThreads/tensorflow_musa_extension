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
"""Tests for MUSA TensorListPopBack operator."""

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
# TensorListPopBack
# ---------------------------------------------------------------------------

class TensorListPopBackOpTest(MUSATestCase):
  """Tests for MUSA TensorListPopBack operator."""

  def _build_list_from_tensor(self, x, element_dtype):
    """Build a TensorList by stacking all rows via PushBack."""
    element_shape = tf.constant(list(x.shape[1:]), dtype=tf.int32)
    handle = tf.raw_ops.EmptyTensorList(
        element_shape=element_shape,
        max_num_elements=x.shape[0],
        element_dtype=element_dtype,
    )
    for i in range(x.shape[0]):
      handle = tf.raw_ops.TensorListPushBack(
          input_handle=handle,
          tensor=x[i],
      )
    return handle, element_shape

  def _pop_all_and_collect(self, x, element_dtype):
    """Pop every element from the list and return them stacked."""
    n = x.shape[0]
    element_shape = tf.constant(list(x.shape[1:]), dtype=tf.int32)
    handle, _ = self._build_list_from_tensor(x, element_dtype)

    popped = []
    for _ in range(n):
      handle, elem = tf.raw_ops.TensorListPopBack(
          input_handle=handle,
          element_shape=element_shape,
          element_dtype=element_dtype,
      )
      popped.append(elem)

    # Reverse so order matches original (LIFO pop)
    popped.reverse()
    return tf.stack(popped, axis=0)

  def _run(self, shape, dtype, rtol=1e-5, atol=1e-5):
    _, x = _make_input(shape, dtype)

    def func(inp):
      return self._pop_all_and_collect(inp, element_dtype=dtype)

    self._compare_cpu_musa_results(func, [x], dtype, rtol=rtol, atol=atol)

  def testPopBackFloat32(self):
    self._run([4, 3], tf.float32)

  def testPopBackFloat16(self):
    self._run([3, 4, 5], tf.float16, rtol=1e-2, atol=1e-2)

  def testPopBackBFloat16(self):
    self._run([4, 8], tf.bfloat16, rtol=1e-1, atol=1e-1)

  def testPopBackFloat64(self):
    self._run([5, 6], tf.float64)

  def testPopBackInt32(self):
    self._run([6, 4], tf.int32, rtol=0, atol=0)

  def testPopBackInt64(self):
    self._run([3, 7], tf.int64, rtol=0, atol=0)

  def testPopBackLIFOOrder(self):
    """The last pushed element should be the first popped (LIFO)."""
    dtype = tf.float32
    values = np.arange(1, 6, dtype=np.float32).reshape(5, 1)
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
      # Pop the top element – should be values[4] == 5.0
      _, popped = tf.raw_ops.TensorListPopBack(
          input_handle=handle,
          element_shape=element_shape,
          element_dtype=dtype,
      )

    self.assertAllClose(popped.numpy().flatten(), [5.0])

  def testPopBackUninitializedElement(self):
    """Pop from a Reserved (uninitialized) list returns zeros."""
    dtype = tf.float32
    element_shape = tf.constant([3], dtype=tf.int32)

    with tf.device('/device:MUSA:0'):
      handle = tf.raw_ops.TensorListReserve(
          element_shape=element_shape,
          num_elements=2,
          element_dtype=dtype,
      )
      # Both slots are DT_INVALID; popping should give a zero tensor.
      _, popped = tf.raw_ops.TensorListPopBack(
          input_handle=handle,
          element_shape=element_shape,
          element_dtype=dtype,
      )

    self.assertAllClose(popped.numpy(), np.zeros([3], dtype=np.float32))


if __name__ == "__main__":
  tf.test.main()
