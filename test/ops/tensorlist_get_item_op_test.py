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
"""Tests for MUSA TensorListGetItem operator."""

import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from musa_test_utils import MUSATestCase


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_input(shape, dtype):
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
# TensorListGetItem tests
# ---------------------------------------------------------------------------

class TensorListGetItemOpTest(MUSATestCase):
  """Tests for MUSA TensorListGetItem operator."""

  # ------------------------------------------------------------------ #
  # Internal helpers
  # ------------------------------------------------------------------ #

  def _build_list_via_pushback(self, x, element_dtype):
    """Build a TensorList by pushing every row of *x*."""
    n = x.shape[0]
    element_shape = tf.constant(list(x.shape[1:]), dtype=tf.int32)
    handle = tf.raw_ops.EmptyTensorList(
        element_shape=element_shape,
        max_num_elements=n,
        element_dtype=element_dtype,
    )
    for i in range(n):
      handle = tf.raw_ops.TensorListPushBack(
          input_handle=handle,
          tensor=x[i],
      )
    return handle, element_shape

  def _build_list_via_setitem(self, x, element_dtype):
    """Build a TensorList via Reserve + SetItem."""
    n = x.shape[0]
    element_shape = tf.constant(list(x.shape[1:]), dtype=tf.int32)
    handle = tf.raw_ops.TensorListReserve(
        element_shape=element_shape,
        num_elements=n,
        element_dtype=element_dtype,
    )
    for i in range(n):
      handle = tf.raw_ops.TensorListSetItem(
          input_handle=handle,
          index=i,
          item=x[i],
      )
    return handle, element_shape

  def _get_all_items(self, handle, element_shape, n, element_dtype):
    """Call GetItem for every index and stack the results."""
    items = []
    for i in range(n):
      item = tf.raw_ops.TensorListGetItem(
          input_handle=handle,
          index=i,
          element_shape=element_shape,
          element_dtype=element_dtype,
      )
      items.append(item)
    return tf.stack(items, axis=0)

  # ------------------------------------------------------------------ #
  # Parametrized dtype tests (PushBack list)
  # ------------------------------------------------------------------ #

  def _run_pushback(self, shape, dtype, rtol=1e-5, atol=1e-5):
    _, x = _make_input(shape, dtype)

    def func(inp):
      handle, element_shape = self._build_list_via_pushback(inp, dtype)
      return self._get_all_items(handle, element_shape, shape[0], dtype)

    self._compare_cpu_musa_results(func, [x], dtype, rtol=rtol, atol=atol)

  def testGetItemFloat32(self):
    self._run_pushback([4, 3], tf.float32)

  def testGetItemFloat16(self):
    self._run_pushback([3, 4, 5], tf.float16, rtol=1e-2, atol=1e-2)

  def testGetItemBFloat16(self):
    self._run_pushback([4, 8], tf.bfloat16, rtol=1e-1, atol=1e-1)

  def testGetItemFloat64(self):
    self._run_pushback([5, 6], tf.float64)

  def testGetItemInt32(self):
    self._run_pushback([6, 4], tf.int32, rtol=0, atol=0)

  def testGetItemInt64(self):
    self._run_pushback([3, 7], tf.int64, rtol=0, atol=0)

  # ------------------------------------------------------------------ #
  # Parametrized dtype tests (Reserve + SetItem list)
  # ------------------------------------------------------------------ #

  def _run_setitem(self, shape, dtype, rtol=1e-5, atol=1e-5):
    _, x = _make_input(shape, dtype)

    def func(inp):
      handle, element_shape = self._build_list_via_setitem(inp, dtype)
      return self._get_all_items(handle, element_shape, shape[0], dtype)

    self._compare_cpu_musa_results(func, [x], dtype, rtol=rtol, atol=atol)

  def testGetItemFromReservedListFloat32(self):
    self._run_setitem([4, 3], tf.float32)

  def testGetItemFromReservedListFloat16(self):
    self._run_setitem([3, 4, 5], tf.float16, rtol=1e-2, atol=1e-2)

  def testGetItemFromReservedListInt32(self):
    self._run_setitem([6, 4], tf.int32, rtol=0, atol=0)

  # ------------------------------------------------------------------ #
  # Correctness checks
  # ------------------------------------------------------------------ #

  def testGetItemCorrectIndex(self):
    """GetItem at each index should return exactly that row."""
    dtype = tf.float32
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    x = tf.constant(values, dtype=dtype)
    element_shape = tf.constant([3], dtype=tf.int32)

    with tf.device('/device:MUSA:0'):
      handle, eshape = self._build_list_via_pushback(x, dtype)
      for i in range(4):
        item = tf.raw_ops.TensorListGetItem(
            input_handle=handle,
            index=i,
            element_shape=eshape,
            element_dtype=dtype,
        )
        self.assertAllClose(item.numpy(), values[i])

  def testGetItemScalarElements(self):
    """GetItem on a list of scalar elements."""
    dtype = tf.float32
    scalars = tf.constant([1.0, 2.0, 3.0], dtype=dtype)
    element_shape = tf.constant([], dtype=tf.int32)

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
      items = [
          tf.raw_ops.TensorListGetItem(
              input_handle=handle,
              index=i,
              element_shape=element_shape,
              element_dtype=dtype,
          )
          for i in range(3)
      ]
      return tf.stack(items, axis=0)

    self._compare_cpu_musa_results(func, [scalars], dtype)

  def testGetItemUninitializedReturnsZero(self):
    """GetItem on an uninitialized (Reserved) slot should return zeros."""
    dtype = tf.float32
    element_shape = tf.constant([4], dtype=tf.int32)

    with tf.device('/device:MUSA:0'):
      handle = tf.raw_ops.TensorListReserve(
          element_shape=element_shape,
          num_elements=3,
          element_dtype=dtype,
      )
      # All three slots are DT_INVALID; read slot 1.
      item = tf.raw_ops.TensorListGetItem(
          input_handle=handle,
          index=1,
          element_shape=element_shape,
          element_dtype=dtype,
      )

    self.assertAllClose(item.numpy(), np.zeros([4], dtype=np.float32))

  def testGetItemDoesNotMutateList(self):
    """Calling GetItem must not remove or alter elements in the list."""
    dtype = tf.float32
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    x = tf.constant(values, dtype=dtype)
    element_shape = tf.constant([2], dtype=tf.int32)

    with tf.device('/device:MUSA:0'):
      handle, eshape = self._build_list_via_pushback(x, dtype)

      # Read index 2 twice; both reads should yield the same tensor.
      first = tf.raw_ops.TensorListGetItem(
          input_handle=handle,
          index=2,
          element_shape=eshape,
          element_dtype=dtype,
      )
      second = tf.raw_ops.TensorListGetItem(
          input_handle=handle,
          index=2,
          element_shape=eshape,
          element_dtype=dtype,
      )

    self.assertAllClose(first.numpy(), second.numpy())
    self.assertAllClose(first.numpy(), values[2])

  def testGetItemHighDim(self):
    """GetItem works on higher-rank element tensors (3-D slices)."""
    self._run_pushback([3, 4, 5, 2], tf.float32)


if __name__ == "__main__":
  tf.test.main()
