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

"""Tests for MUSA ReverseV2 operator."""
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class ReverseV2OpTest(MUSATestCase):
  """Tests for MUSA ReverseV2 operator."""

  @staticmethod
  def _reverse_v2_op(tensor, axis):
    return tf.raw_ops.ReverseV2(tensor=tensor, axis=axis)

  def _test_reverse_v2(self, shape, axis_values, axis_dtype, data_dtype):
    np_dtype = np.float32 if data_dtype == tf.bfloat16 else data_dtype.as_numpy_dtype
    x_np = np.random.uniform(-3.0, 3.0, size=shape).astype(np_dtype)
    x = tf.constant(x_np, dtype=data_dtype)
    axis = tf.constant(axis_values, dtype=axis_dtype)

    rtol = 1e-2 if data_dtype in [tf.float16, tf.bfloat16] else 1e-5
    atol = 1e-2 if data_dtype in [tf.float16, tf.bfloat16] else 1e-8
    self._compare_cpu_musa_results(
        self._reverse_v2_op, [x, axis], dtype=data_dtype, rtol=rtol, atol=atol
    )

  def testReverseV2MultiAxis(self):
    """Test reversing on multiple axes, e.g. axis=[3, 4]."""
    shape = [2, 3, 4, 5, 6]
    for data_dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64, tf.bool]:
      if data_dtype == tf.bool:
        x_np = np.random.randint(0, 2, size=shape).astype(np.bool_)
        x = tf.constant(x_np, dtype=data_dtype)
        axis = tf.constant([3, 4], dtype=tf.int32)
        self._compare_cpu_musa_results(
            self._reverse_v2_op, [x, axis], dtype=data_dtype, rtol=0.0, atol=0.0
        )
      else:
        self._test_reverse_v2(shape, [3, 4], tf.int32, data_dtype)

  def testReverseV2NegativeAxis(self):
    """Test reverse with negative axis indices."""
    shape = [2, 4, 6]
    for axis_dtype in [tf.int32, tf.int64]:
      self._test_reverse_v2(shape, [-1, -3], axis_dtype, tf.float32)

  def testReverseV2SingleAxis(self):
    """Test reverse on a single positive axis."""
    self._test_reverse_v2([3, 4, 5], [1], tf.int32, tf.float32)

  def testReverseV2EmptyAxis(self):
    """Test empty axis (should be identity)."""
    x_np = np.random.uniform(-1.0, 1.0, size=[8, 9]).astype(np.float32)
    x = tf.constant(x_np, dtype=tf.float32)
    axis = tf.constant([], dtype=tf.int64)
    self._compare_cpu_musa_results(
        self._reverse_v2_op, [x, axis], dtype=tf.float32, rtol=1e-5, atol=1e-8
    )

  def testReverseV2DuplicateAxisRaises(self):
    """Test duplicate axis should raise InvalidArgument."""
    x = tf.constant(np.random.uniform(-1.0, 1.0, size=[2, 3, 4]).astype(np.float32))
    axis = tf.constant([1, 1], dtype=tf.int32)

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "duplicated"):
      with tf.device("/device:MUSA:0"):
        _ = tf.raw_ops.ReverseV2(tensor=x, axis=axis)

  def testReverseV2AxisOutOfRangeRaises(self):
    """Test out-of-range axis should raise InvalidArgument."""
    x = tf.constant(np.random.uniform(-1.0, 1.0, size=[2, 3, 4]).astype(np.float32))
    axis_values = [[3], [-4]]
    for axis_value in axis_values:
      axis = tf.constant(axis_value, dtype=tf.int32)
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "out of valid range"):
        with tf.device("/device:MUSA:0"):
          _ = tf.raw_ops.ReverseV2(tensor=x, axis=axis)

  def testReverseV2ScalarInput(self):
    """Test scalar input with empty axis (rank-0 tensor)."""
    x = tf.constant(3.25, dtype=tf.float32)
    axis = tf.constant([], dtype=tf.int32)
    self._compare_cpu_musa_results(
        self._reverse_v2_op, [x, axis], dtype=tf.float32, rtol=1e-5, atol=1e-8
    )

  def testReverseV2EmptyTensor(self):
    """Test empty tensor fast path with non-empty axis."""
    x = tf.constant(np.empty([2, 0, 4], dtype=np.float32), dtype=tf.float32)
    axis = tf.constant([1], dtype=tf.int32)
    self._compare_cpu_musa_results(
        self._reverse_v2_op, [x, axis], dtype=tf.float32, rtol=1e-5, atol=1e-8
    )


if __name__ == "__main__":
  tf.test.main()
