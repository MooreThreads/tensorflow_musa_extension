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

"""Tests for MUSA SparseTensorDenseMatMul operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import load_musa_plugin


load_musa_plugin()
MUSA_DEVICES = tf.config.list_physical_devices('MUSA')


class SparseTensorDenseMatMulOpTest(tf.test.TestCase):
  """Tests for tf.raw_ops.SparseTensorDenseMatMul on MUSA."""

  def _run_sparse_dense_matmul(self,
                               a_indices,
                               a_values,
                               a_shape,
                               b,
                               adjoint_a,
                               adjoint_b,
                               device):
    with tf.device(device):
      return tf.raw_ops.SparseTensorDenseMatMul(
          a_indices=a_indices,
          a_values=a_values,
          a_shape=a_shape,
          b=b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)

  def _run_musa_sparse_dense_matmul(self, a_indices, a_values, a_shape, b,
                                    adjoint_a=False, adjoint_b=False):
    old_soft_placement = tf.config.get_soft_device_placement()
    tf.config.set_soft_device_placement(False)
    try:
      return self._run_sparse_dense_matmul(
          a_indices, a_values, a_shape, b, adjoint_a, adjoint_b,
          '/device:MUSA:0')
    finally:
      tf.config.set_soft_device_placement(old_soft_placement)

  def _test_sparse_dense_matmul(self,
                                a_indices_np,
                                a_values_np,
                                a_shape_np,
                                b_np,
                                dtype,
                                adjoint_a=False,
                                adjoint_b=False,
                                rtol=1e-5,
                                atol=1e-5):
    if not MUSA_DEVICES:
      self.skipTest("No MUSA devices found.")

    a_indices = tf.constant(a_indices_np, dtype=tf.int64)
    a_shape = tf.constant(a_shape_np, dtype=tf.int64)
    a_values = tf.constant(a_values_np, dtype=dtype)
    b = tf.constant(b_np, dtype=dtype)

    cpu_result = self._run_sparse_dense_matmul(
        a_indices, a_values, a_shape, b, adjoint_a, adjoint_b, '/CPU:0')
    musa_result = self._run_musa_sparse_dense_matmul(
        a_indices, a_values, a_shape, b, adjoint_a, adjoint_b)

    if dtype == tf.int32:
      self.assertAllEqual(cpu_result.numpy(), musa_result.numpy())
    elif dtype == tf.bfloat16:
      self.assertAllClose(
          tf.cast(cpu_result, tf.float32).numpy(),
          tf.cast(musa_result, tf.float32).numpy(),
          rtol=rtol,
          atol=atol)
    else:
      self.assertAllClose(
          cpu_result.numpy(),
          musa_result.numpy(),
          rtol=rtol,
          atol=atol)

  def testFloat32SmallEmbeddingLikeShape(self):
    a_indices = np.array(
        [[0, 1], [0, 3], [1, 0], [2, 2], [3, 4], [3, 5]], dtype=np.int64)
    a_values = np.array([1.0, 0.5, -2.0, 3.0, 4.0, -1.0], dtype=np.float32)
    b = (np.arange(6 * 11, dtype=np.float32).reshape(6, 11) / 13.0) - 2.0
    self._test_sparse_dense_matmul(
        a_indices, a_values, np.array([4, 6], dtype=np.int64), b, tf.float32)

  def testFloat32DuplicateIndicesAccumulate(self):
    a_indices = np.array([[0, 1], [0, 1], [1, 2], [1, 2]], dtype=np.int64)
    a_values = np.array([1.0, 2.0, -1.0, 3.0], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0], [-2.0, 5.0]], dtype=np.float32)
    self._test_sparse_dense_matmul(
        a_indices, a_values, np.array([2, 3], dtype=np.int64), b, tf.float32)

  def testFloat32ScalarOutputColumn(self):
    a_indices = np.array([[0, 0], [1, 2], [2, 1]], dtype=np.int64)
    a_values = np.array([2.0, -1.0, 3.0], dtype=np.float32)
    b = np.array([[4.0], [5.0], [6.0]], dtype=np.float32)
    self._test_sparse_dense_matmul(
        a_indices, a_values, np.array([3, 3], dtype=np.int64), b, tf.float32)

  def testFloat32LargeOutputColumns(self):
    a_indices = np.array(
        [[0, 1], [1, 3], [2, 5], [3, 7], [4, 9]], dtype=np.int64)
    a_values = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    b = (np.arange(10 * 65, dtype=np.float32).reshape(10, 65) % 17) / 5.0
    self._test_sparse_dense_matmul(
        a_indices, a_values, np.array([5, 10], dtype=np.int64), b, tf.float32)

  def testFloat32AdjointA(self):
    a_indices = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    a_values = np.array([1.5, -2.0, 0.25], dtype=np.float32)
    b = np.array(
        [[1.0, 2.0, 3.0], [4.0, -1.0, 5.0], [2.0, 0.5, -3.0]],
        dtype=np.float32)
    self._test_sparse_dense_matmul(
        a_indices,
        a_values,
        np.array([3, 4], dtype=np.int64),
        b,
        tf.float32,
        adjoint_a=True)

  def testFloat32AdjointB(self):
    a_indices = np.array([[0, 1], [1, 0], [1, 2]], dtype=np.int64)
    a_values = np.array([2.0, -1.0, 3.0], dtype=np.float32)
    b = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.5, 2.0]],
        dtype=np.float32)
    self._test_sparse_dense_matmul(
        a_indices,
        a_values,
        np.array([2, 3], dtype=np.int64),
        b,
        tf.float32,
        adjoint_b=True)

  def testFloat64SmallShape(self):
    a_indices = np.array([[0, 0], [1, 1], [1, 2]], dtype=np.int64)
    a_values = np.array([1.0, -1.5, 2.5], dtype=np.float64)
    b = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float64)
    self._test_sparse_dense_matmul(
        a_indices, a_values, np.array([2, 3], dtype=np.int64), b, tf.float64)

  def testInt32SmallShape(self):
    a_indices = np.array([[0, 0], [0, 1], [1, 2]], dtype=np.int64)
    a_values = np.array([2, -3, 4], dtype=np.int32)
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    self._test_sparse_dense_matmul(
        a_indices, a_values, np.array([2, 3], dtype=np.int64), b, tf.int32)

  def testBFloat16SmallShape(self):
    a_indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    a_values = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    b = np.array(
        [[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]], dtype=np.float32)
    self._test_sparse_dense_matmul(
        a_indices,
        a_values,
        np.array([2, 3], dtype=np.int64),
        b,
        tf.bfloat16,
        rtol=1e-1,
        atol=1e-1)


if __name__ == "__main__":
  tf.test.main()
