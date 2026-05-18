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
"""Tests for MUSA TensorListSetItem operator."""

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
# TensorListSetItem
# ---------------------------------------------------------------------------

class TensorListSetItemOpTest(MUSATestCase):
  """Tests for MUSA TensorListSetItem operator."""

  def _reserve_set_stack(self, x, element_dtype):
    """Reserve a list, set each slot via SetItem, then Stack."""
    num_elements = x.shape[0]
    element_shape = tf.constant(list(x.shape[1:]), dtype=tf.int32)

    handle = tf.raw_ops.TensorListReserve(
        element_shape=element_shape,
        num_elements=num_elements,
        element_dtype=element_dtype,
    )
    for i in range(num_elements):
      handle = tf.raw_ops.TensorListSetItem(
          input_handle=handle,
          index=i,
          item=x[i],
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
      return self._reserve_set_stack(inp, element_dtype=dtype)

    self._compare_cpu_musa_results(func, [x], dtype, rtol=rtol, atol=atol)

  def testSetItemFloat32(self):
    self._run([4, 3], tf.float32)

  def testSetItemFloat16(self):
    self._run([3, 4, 5], tf.float16, rtol=1e-2, atol=1e-2)

  def testSetItemBFloat16(self):
    self._run([4, 8], tf.bfloat16, rtol=1e-1, atol=1e-1)

  def testSetItemFloat64(self):
    self._run([5, 6], tf.float64)

  def testSetItemInt32(self):
    self._run([6, 4], tf.int32, rtol=0, atol=0)

  def testSetItemInt64(self):
    self._run([3, 7], tf.int64, rtol=0, atol=0)

  def testSetItemOverwrite(self):
    """SetItem should overwrite a previously written slot."""
    dtype = tf.float32
    element_shape = tf.constant([2], dtype=tf.int32)

    with tf.device('/device:MUSA:0'):
      handle = tf.raw_ops.TensorListReserve(
          element_shape=element_shape,
          num_elements=3,
          element_dtype=dtype,
      )
      # Write index 1 first, then overwrite it.
      handle = tf.raw_ops.TensorListSetItem(
          input_handle=handle,
          index=1,
          item=tf.constant([9.0, 9.0], dtype=dtype),
      )
      handle = tf.raw_ops.TensorListSetItem(
          input_handle=handle,
          index=0,
          item=tf.constant([1.0, 2.0], dtype=dtype),
      )
      handle = tf.raw_ops.TensorListSetItem(
          input_handle=handle,
          index=1,
          item=tf.constant([3.0, 4.0], dtype=dtype),
      )
      handle = tf.raw_ops.TensorListSetItem(
          input_handle=handle,
          index=2,
          item=tf.constant([5.0, 6.0], dtype=dtype),
      )
      result = tf.raw_ops.TensorListStack(
          input_handle=handle,
          element_shape=element_shape,
          element_dtype=dtype,
          num_elements=3,
      )

    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    self.assertAllClose(result.numpy(), expected)

  def testSetItemReverseOrder(self):
    """SetItem at indices in reverse order should still produce correct output."""
    dtype = tf.float32
    shape = [5, 4]
    _, x = _make_input(shape, dtype)
    element_shape = tf.constant([shape[1]], dtype=tf.int32)

    with tf.device('/device:MUSA:0'):
      handle = tf.raw_ops.TensorListReserve(
          element_shape=element_shape,
          num_elements=shape[0],
          element_dtype=dtype,
      )
      for i in reversed(range(shape[0])):
        handle = tf.raw_ops.TensorListSetItem(
            input_handle=handle,
            index=i,
            item=x[i],
        )
      result = tf.raw_ops.TensorListStack(
          input_handle=handle,
          element_shape=element_shape,
          element_dtype=dtype,
          num_elements=shape[0],
      )

    self.assertAllClose(result.numpy(), x.numpy())


if __name__ == "__main__":
  tf.test.main()
