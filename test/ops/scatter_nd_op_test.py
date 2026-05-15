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

"""Tests for MUSA ScatterNd and TensorScatterNd* operators."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


def _tolerances(dtype):
  """Return (rtol, atol) suitable for the given dtype."""
  if dtype in (tf.float16, tf.bfloat16):
    return 1e-2, 1e-2
  return 1e-5, 1e-8


class ScatterNdOpTest(MUSATestCase):
  """Tests for tf.scatter_nd (ScatterNd kernel)."""

  # ------------------------------------------------------------------
  # Helpers
  # ------------------------------------------------------------------

  def _run_scatter_nd(self, indices_np, updates_np, shape,
                      indices_dtype=tf.int32, updates_dtype=tf.float32):
    """Run tf.scatter_nd on both CPU and MUSA and compare results."""
    indices = tf.constant(indices_np, dtype=indices_dtype)
    updates = tf.constant(updates_np, dtype=updates_dtype)
    rtol, atol = _tolerances(updates_dtype)
    # Bake `shape` into the lambda so only indices/updates are device-placed.
    self._compare_cpu_musa_results(
        lambda idx, upd: tf.scatter_nd(idx, upd, shape),
        [indices, updates],
        dtype=updates_dtype,
        rtol=rtol,
        atol=atol,
    )

  # ------------------------------------------------------------------
  # Correctness tests
  # ------------------------------------------------------------------

  def testScatterNd1DBasic(self):
    """Scatter scalar updates into a 1-D output."""
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32)
    self._run_scatter_nd(indices, updates, shape=[8])

  def testScatterNd2DScalar(self):
    """Scatter scalar updates into a 2-D output (each index selects a row)."""
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    self._run_scatter_nd(indices, updates, shape=[4, 3])

  def testScatterNd2DFull(self):
    """Scatter into individual cells of a 2-D output."""
    indices = np.array([[0, 0], [1, 2], [2, 1]], dtype=np.int32)
    updates = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    self._run_scatter_nd(indices, updates, shape=[3, 3])

  def testScatterNdSlice(self):
    """Scatter 2-D slices into a 3-D output."""
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.random.uniform(-1, 1, size=(2, 4, 4)).astype(np.float32)
    self._run_scatter_nd(indices, updates, shape=[4, 4, 4])

  def testScatterNdHigherDims(self):
    """Scatter into a 4-D tensor."""
    indices = np.array([[0, 1], [2, 0]], dtype=np.int32)
    updates = np.random.uniform(-1, 1, size=(2, 5)).astype(np.float32)
    self._run_scatter_nd(indices, updates, shape=[3, 3, 5])

  def testScatterNdEmptyUpdates(self):
    """Empty indices/updates should produce a zero tensor."""
    indices = np.zeros((0, 1), dtype=np.int32)
    updates = np.zeros((0,), dtype=np.float32)
    self._run_scatter_nd(indices, updates, shape=[5])

  def testScatterNdInt64Indices(self):
    """Scatter with int64 indices."""
    indices = np.array([[0], [3], [5]], dtype=np.int64)
    updates = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    self._run_scatter_nd(indices, updates, shape=[8], indices_dtype=tf.int64)

  # ------------------------------------------------------------------
  # dtype coverage
  # ------------------------------------------------------------------

  def testScatterNdDtypes(self):
    """Verify correctness across all supported data types."""
    indices = np.array([[0], [2], [4]], dtype=np.int32)
    for dtype in [tf.float32, tf.float64, tf.float16, tf.bfloat16,
                  tf.int32, tf.int64]:
      updates_np = np.array([1, 2, 3]).astype(dtype.as_numpy_dtype)
      self._run_scatter_nd(indices, updates_np, shape=[6], updates_dtype=dtype)

  # ------------------------------------------------------------------
  # Shape / edge cases
  # ------------------------------------------------------------------

  def testScatterNdLargeShape(self):
    """Stress test: large 1-D scatter (unique indices to avoid undefined behavior
    with duplicates, which gives non-deterministic results across CPU/MUSA)."""
    n = 10000
    num_updates = n // 2
    # Use unique indices so CPU and MUSA produce identical outputs.
    chosen = np.random.choice(n, size=num_updates, replace=False).astype(np.int32)
    indices = chosen[:, np.newaxis]
    updates = np.random.uniform(-1, 1, size=(num_updates,)).astype(np.float32)
    self._run_scatter_nd(indices, updates, shape=[n])

  def testScatterNd3DOutput(self):
    """Scatter into every position of a 3-D tensor."""
    shape = [2, 3, 4]
    total = 2 * 3
    indices_np = []
    for i in range(2):
      for j in range(3):
        indices_np.append([i, j])
    indices = np.array(indices_np, dtype=np.int32)
    updates = np.random.uniform(-5, 5, size=(total, 4)).astype(np.float32)
    self._run_scatter_nd(indices, updates, shape=shape)


class TensorScatterNdUpdateOpTest(MUSATestCase):
  """Tests for tf.tensor_scatter_nd_update (TensorScatterUpdate)."""

  def _run(self, tensor_np, indices_np, updates_np,
           indices_dtype=tf.int32, dtype=tf.float32):
    tensor = tf.constant(tensor_np, dtype=dtype)
    indices = tf.constant(indices_np, dtype=indices_dtype)
    updates = tf.constant(updates_np, dtype=dtype)
    rtol, atol = _tolerances(dtype)
    self._compare_cpu_musa_results(
        tf.tensor_scatter_nd_update,
        [tensor, indices, updates],
        dtype=dtype,
        rtol=rtol,
        atol=atol,
    )

  def testBasic1D(self):
    tensor = np.zeros(8, dtype=np.float32)
    indices = np.array([[1], [4], [6]], dtype=np.int32)
    updates = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    self._run(tensor, indices, updates)

  def testBasic2D(self):
    tensor = np.ones((4, 4), dtype=np.float32)
    indices = np.array([[0, 0], [1, 2], [3, 3]], dtype=np.int32)
    updates = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    self._run(tensor, indices, updates)

  def testUpdateRows(self):
    """Update entire rows of a 2-D tensor."""
    tensor = np.zeros((5, 8), dtype=np.float32)
    indices = np.array([[0], [3]], dtype=np.int32)
    updates = np.random.uniform(-1, 1, size=(2, 8)).astype(np.float32)
    self._run(tensor, indices, updates)

  def testDtypes(self):
    tensor_np = np.arange(10, dtype=np.float32)
    indices = np.array([[0], [5], [9]], dtype=np.int32)
    for dtype in [tf.float32, tf.float64, tf.float16, tf.bfloat16]:
      updates_np = np.array([-1, -2, -3]).astype(dtype.as_numpy_dtype)
      self._run(tensor_np.astype(dtype.as_numpy_dtype),
                indices, updates_np, dtype=dtype)

  def testInt64Indices(self):
    tensor = np.zeros(10, dtype=np.float32)
    indices = np.array([[2], [7]], dtype=np.int64)
    updates = np.array([3.14, 2.71], dtype=np.float32)
    self._run(tensor, indices, updates, indices_dtype=tf.int64)


class TensorScatterNdAddOpTest(MUSATestCase):
  """Tests for tf.tensor_scatter_nd_add (TensorScatterAdd)."""

  def _run(self, tensor_np, indices_np, updates_np,
           indices_dtype=tf.int32, dtype=tf.float32):
    tensor = tf.constant(tensor_np, dtype=dtype)
    indices = tf.constant(indices_np, dtype=indices_dtype)
    updates = tf.constant(updates_np, dtype=dtype)
    rtol, atol = _tolerances(dtype)
    self._compare_cpu_musa_results(
        tf.tensor_scatter_nd_add,
        [tensor, indices, updates],
        dtype=dtype,
        rtol=rtol,
        atol=atol,
    )

  def testBasic1D(self):
    tensor = np.ones(8, dtype=np.float32)
    indices = np.array([[1], [3], [5]], dtype=np.int32)
    updates = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    self._run(tensor, indices, updates)

  def testUniqueIndices(self):
    """Unique indices: deterministic result, compare CPU vs MUSA."""
    tensor = np.zeros(5, dtype=np.float32)
    indices = np.array([[1], [2], [4]], dtype=np.int32)
    updates = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    self._run(tensor, indices, updates)

  def testAddRows2D(self):
    tensor = np.ones((4, 6), dtype=np.float32)
    indices = np.array([[0], [2], [3]], dtype=np.int32)
    updates = np.random.uniform(-1, 1, size=(3, 6)).astype(np.float32)
    self._run(tensor, indices, updates)

  def testDtypes(self):
    tensor_np = np.arange(10, dtype=np.float32)
    indices = np.array([[0], [4], [8]], dtype=np.int32)
    for dtype in [tf.float32, tf.float64, tf.float16, tf.bfloat16]:
      updates_np = np.array([1, 1, 1]).astype(dtype.as_numpy_dtype)
      self._run(tensor_np.astype(dtype.as_numpy_dtype),
                indices, updates_np, dtype=dtype)


class TensorScatterNdSubOpTest(MUSATestCase):
  """Tests for tf.tensor_scatter_nd_sub (TensorScatterSub)."""

  def _run(self, tensor_np, indices_np, updates_np,
           indices_dtype=tf.int32, dtype=tf.float32):
    tensor = tf.constant(tensor_np, dtype=dtype)
    indices = tf.constant(indices_np, dtype=indices_dtype)
    updates = tf.constant(updates_np, dtype=dtype)
    rtol, atol = _tolerances(dtype)
    self._compare_cpu_musa_results(
        tf.tensor_scatter_nd_sub,
        [tensor, indices, updates],
        dtype=dtype,
        rtol=rtol,
        atol=atol,
    )

  def testBasic1D(self):
    tensor = np.full(8, 10.0, dtype=np.float32)
    indices = np.array([[0], [3], [7]], dtype=np.int32)
    updates = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    self._run(tensor, indices, updates)

  def testSubRows2D(self):
    tensor = np.ones((4, 5), dtype=np.float32) * 5.0
    indices = np.array([[1], [3]], dtype=np.int32)
    updates = np.random.uniform(0, 1, size=(2, 5)).astype(np.float32)
    self._run(tensor, indices, updates)

  def testDtypes(self):
    tensor_np = np.full(10, 5.0, dtype=np.float32)
    indices = np.array([[2], [5], [9]], dtype=np.int32)
    for dtype in [tf.float32, tf.float64, tf.float16, tf.bfloat16]:
      updates_np = np.array([1, 1, 1]).astype(dtype.as_numpy_dtype)
      self._run(tensor_np.astype(dtype.as_numpy_dtype),
                indices, updates_np, dtype=dtype)


if __name__ == "__main__":
  tf.test.main()
