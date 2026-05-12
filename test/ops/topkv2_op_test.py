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

"""Tests for MUSA TopKV2 operator."""

import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from musa_test_utils import MUSATestCase


class TopKV2OpTest(MUSATestCase):

  def _run_topk_on_device(self, x, k_tensor, sorted, device):
    with tf.device(device):
      values, indices = tf.raw_ops.TopKV2(
          input=x,
          k=k_tensor,
          sorted=sorted)
    return values, indices

  def _test_topk(self,
                 shape,
                 k,
                 dtype,
                 sorted=True,
                 rtol=1e-5,
                 atol=1e-5):
    np_dtype = dtype.as_numpy_dtype
    if dtype == tf.bfloat16:

      np_dtype = np.float32

    if dtype == tf.float16:
      low, high = -5.0, 5.0
    elif dtype == tf.bfloat16:
      low, high = -3.0, 3.0
    else:
      low, high = -10.0, 10.0

    x_np = np.random.uniform(low, high, size=shape).astype(np_dtype)
    x = tf.constant(x_np, dtype=dtype)
    k_tensor = tf.constant(k, dtype=tf.int32)

    cpu_values, cpu_indices = self._run_topk_on_device(
        x, k_tensor, sorted, '/CPU:0')
    musa_values, musa_indices = self._run_topk_on_device(
        x, k_tensor, sorted, '/device:MUSA:0')

    if dtype in [tf.float16, tf.bfloat16]:
      self.assertAllClose(
          tf.cast(cpu_values, tf.float32).numpy(),
          tf.cast(musa_values, tf.float32).numpy(),
          rtol=rtol,
          atol=atol)
      # fp16/bf16 have limited precision: multiple float32 values may map to
      # the same low-precision value, causing tie-breaking differences between
      # CPU and MUSA.  Only values are compared; indices are skipped.
    else:
      self.assertAllClose(
          cpu_values.numpy(),
          musa_values.numpy(),
          rtol=rtol,
          atol=atol)
      self.assertAllEqual(cpu_indices.numpy(), musa_indices.numpy())

  def testTopKV2Float32(self):
    self._test_topk(
        shape=[10, 20],
        k=5,
        dtype=tf.float32,
        sorted=True,
        rtol=1e-4,
        atol=1e-4)

  def testTopKV2Float16(self):
    self._test_topk(
        shape=[2, 3, 16],
        k=4,
        dtype=tf.float16,
        sorted=True,
        rtol=1e-2,
        atol=1e-2)

  def testTopKV2BFloat16(self):
    self._test_topk(
        shape=[4, 12],
        k=3,
        dtype=tf.bfloat16,
        sorted=True,
        rtol=1e-1,
        atol=1e-1)

  def testTopKV2Float16K1(self):
    self._test_topk(
        shape=[3, 8],
        k=1,
        dtype=tf.float16,
        sorted=True,
        rtol=1e-2,
        atol=1e-2)

  def testTopKV2Float32ThreeDim(self):
    self._test_topk(
        shape=[2, 4, 10],
        k=2,
        dtype=tf.float32,
        sorted=True,
        rtol=1e-4,
        atol=1e-4)

  def testTopKV2Int32(self):
    x_np = np.array(
        [[3, 1, 4, 1, 5, 9], [2, 6, 5, 3, 5, 8]], dtype=np.int32)
    x = tf.constant(x_np, dtype=tf.int32)
    k_tensor = tf.constant(3, dtype=tf.int32)

    cpu_values, cpu_indices = self._run_topk_on_device(
        x, k_tensor, True, '/CPU:0')
    musa_values, musa_indices = self._run_topk_on_device(
        x, k_tensor, True, '/device:MUSA:0')

    self.assertAllEqual(cpu_values.numpy(), musa_values.numpy())
    self.assertAllEqual(cpu_indices.numpy(), musa_indices.numpy())

  def testSortInt32UsesTopKV2CompatiblePath(self):
    x_np = np.array(
        [[[4, 1, 3, 2], [8, 6, 7, 5]]], dtype=np.int32)
    x = tf.constant(x_np, dtype=tf.int32)

    with tf.device('/CPU:0'):
      cpu_sorted = tf.sort(x, axis=-1)
    with tf.device('/device:MUSA:0'):
      musa_sorted = tf.sort(x, axis=-1)

    self.assertAllEqual(cpu_sorted.numpy(), musa_sorted.numpy())

  # -----------------------------------------------------------------------
  # Large-k tests (k > 1024) – exercising the mudnn-based implementation
  # that has no upper limit on k.
  # -----------------------------------------------------------------------

  def testTopKV2LargeKFloat32(self):
    """k = 2048: previously would have hit the k<=1024 guard."""
    self._test_topk(
        shape=[4, 4096],
        k=2048,
        dtype=tf.float32,
        sorted=True,
        rtol=1e-4,
        atol=1e-4)

  def testTopKV2LargeKFloat16(self):
    """k = 1500 with float16 input."""
    self._test_topk(
        shape=[2, 2000],
        k=1500,
        dtype=tf.float16,
        sorted=True,
        rtol=1e-2,
        atol=1e-2)

  def testTopKV2LargeKBFloat16(self):
    """k = 1025: one beyond the old limit."""
    self._test_topk(
        shape=[3, 2048],
        k=1025,
        dtype=tf.bfloat16,
        sorted=True,
        rtol=1e-1,
        atol=1e-1)

  def testTopKV2LargeKInt32(self):
    """k = 2000 with int32 input; use a large value range to avoid ties."""
    shape = [2, 3000]
    # Use np.random.permutation per row to guarantee all values are unique,
    # which avoids tie-breaking differences between CPU and MUSA.
    rows = [
        np.random.permutation(shape[1]).astype(np.int32)
        for _ in range(shape[0])
    ]
    x_np = np.stack(rows)
    x = tf.constant(x_np, dtype=tf.int32)
    k_tensor = tf.constant(2000, dtype=tf.int32)

    cpu_values, cpu_indices = self._run_topk_on_device(
        x, k_tensor, True, '/CPU:0')
    musa_values, musa_indices = self._run_topk_on_device(
        x, k_tensor, True, '/device:MUSA:0')

    self.assertAllEqual(cpu_values.numpy(), musa_values.numpy())
    self.assertAllEqual(cpu_indices.numpy(), musa_indices.numpy())

  # int64 and float64 (double) are not supported by muDNN TopK and have no
  # MUSA kernel registration, so no large-k tests are added for those types.

  def testTopKV2LargeKUnsorted(self):
    """k > 1024 with sorted=False; values must match regardless of order."""
    k = 2000
    shape = [3, 4000]
    x_np = np.random.uniform(-10.0, 10.0, size=shape).astype(np.float32)
    x = tf.constant(x_np, dtype=tf.float32)
    k_tensor = tf.constant(k, dtype=tf.int32)

    cpu_values, _ = self._run_topk_on_device(x, k_tensor, False, '/CPU:0')
    musa_values, _ = self._run_topk_on_device(
        x, k_tensor, False, '/device:MUSA:0')

    # Values may be in any order; compare sorted copies.
    cpu_sorted = tf.sort(cpu_values, axis=-1, direction='DESCENDING').numpy()
    musa_sorted = tf.sort(musa_values, axis=-1, direction='DESCENDING').numpy()
    np.testing.assert_allclose(cpu_sorted, musa_sorted, rtol=1e-4, atol=1e-4)

  def testTopKV2KEqualsLastDimLarge(self):
    """k == last_dim (full sort) for a large last dimension."""
    n = 2048
    self._test_topk(
        shape=[4, n],
        k=n,
        dtype=tf.float32,
        sorted=True,
        rtol=1e-4,
        atol=1e-4)


if __name__ == "__main__":
  tf.test.main()
