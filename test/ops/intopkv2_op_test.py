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

"""Tests for MUSA InTopKV2 operator."""

import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from musa_test_utils import MUSATestCase


class InTopKV2OpTest(MUSATestCase):
  """Tests for MUSA InTopKV2 operator."""

  def _run_in_topk_on_device(self, predictions, targets, k, device):
    """Run InTopKV2 op on specified device."""
    with tf.device(device):
      result = tf.raw_ops.InTopKV2(
          predictions=predictions,
          targets=targets,
          k=k)
    return result

  def _test_in_topk(self, batch_size, num_classes, k, dtype=tf.int32):
    """Test InTopKV2 with given parameters."""
    # Create predictable predictions: higher index = higher value
    predictions_np = np.arange(num_classes, dtype=np.float32)
    predictions_np = np.tile(predictions_np, (batch_size, 1))
    # Add some randomness per batch
    for i in range(batch_size):
      predictions_np[i] = predictions_np[i] + np.random.randn(num_classes) * 0.1
    
    predictions = tf.constant(predictions_np, dtype=tf.float32)
    
    # Create targets: mix of in-topk and not-in-topk cases
    targets_np = np.zeros(batch_size, dtype=dtype.as_numpy_dtype)
    for i in range(batch_size):
      if k == 0:
        # When k=0, nothing is in top-k, use any target
        targets_np[i] = i % num_classes
      elif k == num_classes:
        # All targets are in top-k when k equals num_classes
        targets_np[i] = i % num_classes
      elif i % 2 == 0:
        # Target in top-k (use one of the top k indices)
        targets_np[i] = num_classes - 1 - (i % k)
      else:
        # Target not in top-k (use one of the bottom indices)
        targets_np[i] = i % (num_classes - k)
    targets = tf.constant(targets_np, dtype=dtype)
    # k must match the dtype of targets (T) for InTopKV2
    k_dtype = tf.int64 if dtype == tf.int64 else tf.int32
    k_tensor = tf.constant(k, dtype=k_dtype)

    cpu_result = self._run_in_topk_on_device(
        predictions, targets, k_tensor, '/CPU:0')
    musa_result = self._run_in_topk_on_device(
        predictions, targets, k_tensor, '/device:MUSA:0')

    self.assertAllEqual(cpu_result.numpy(), musa_result.numpy())

  def testInTopKV2BasicInt32(self):
    """Test basic InTopKV2 with int32 targets."""
    self._test_in_topk(batch_size=10, num_classes=20, k=5, dtype=tf.int32)

  def testInTopKV2BasicInt64(self):
    """Test basic InTopKV2 with int64 targets."""
    self._test_in_topk(batch_size=10, num_classes=20, k=5, dtype=tf.int64)

  def testInTopKV2LargeBatch(self):
    """Test InTopKV2 with larger batch size."""
    self._test_in_topk(batch_size=100, num_classes=50, k=10, dtype=tf.int32)

  def testInTopKV2LargeClasses(self):
    """Test InTopKV2 with large number of classes."""
    self._test_in_topk(batch_size=32, num_classes=1000, k=50, dtype=tf.int32)

  def testInTopKV2KEquals1(self):
    """Test InTopKV2 with k=1 (only top-1)."""
    self._test_in_topk(batch_size=16, num_classes=100, k=1, dtype=tf.int32)

  def testInTopKV2KEqualsAll(self):
    """Test InTopKV2 with k equal to num_classes (all in top-k)."""
    self._test_in_topk(batch_size=8, num_classes=50, k=50, dtype=tf.int32)

  def testInTopKV2SmallBatch(self):
    """Test InTopKV2 with small batch size."""
    self._test_in_topk(batch_size=1, num_classes=10, k=3, dtype=tf.int32)

  def testInTopKV2ExactMatch(self):
    """Test InTopKV2 with exact prediction values."""
    # Create predictions where we know exactly what should be in top-k
    predictions_np = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],  # top-3: [4, 3, 2]
        [0.5, 0.4, 0.3, 0.2, 0.1],  # top-3: [0, 1, 2]
        [0.1, 0.5, 0.2, 0.4, 0.3],  # top-3: [1, 3, 4]
    ], dtype=np.float32)
    predictions = tf.constant(predictions_np)
    
    # Test with targets known to be in top-3
    targets_np = np.array([4, 0, 1], dtype=np.int32)  # All in top-3
    targets = tf.constant(targets_np)
    k = tf.constant(3, dtype=tf.int32)

    cpu_result = self._run_in_topk_on_device(predictions, targets, k, '/CPU:0')
    musa_result = self._run_in_topk_on_device(
        predictions, targets, k, '/device:MUSA:0')

    expected = np.array([True, True, True])
    self.assertAllEqual(expected, cpu_result.numpy())
    self.assertAllEqual(expected, musa_result.numpy())

  def testInTopKV2MixedResults(self):
    """Test InTopKV2 with mixed in/out of top-k results."""
    predictions_np = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],  # top-2: [4, 3]
        [0.5, 0.4, 0.3, 0.2, 0.1],  # top-2: [0, 1]
    ], dtype=np.float32)
    predictions = tf.constant(predictions_np)
    
    # First target (4) is in top-2, second target (2) is not in top-2
    targets_np = np.array([4, 2], dtype=np.int32)
    targets = tf.constant(targets_np)
    k = tf.constant(2, dtype=tf.int32)

    cpu_result = self._run_in_topk_on_device(predictions, targets, k, '/CPU:0')
    musa_result = self._run_in_topk_on_device(
        predictions, targets, k, '/device:MUSA:0')

    expected = np.array([True, False])
    self.assertAllEqual(expected, cpu_result.numpy())
    self.assertAllEqual(expected, musa_result.numpy())

  def testInTopKV2KZero(self):
    """Test InTopKV2 with k=0 (nothing in top-k)."""
    predictions_np = np.array([
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
    ], dtype=np.float32)
    predictions = tf.constant(predictions_np)
    targets_np = np.array([0, 1], dtype=np.int32)
    targets = tf.constant(targets_np)
    k = tf.constant(0, dtype=tf.int32)

    cpu_result = self._run_in_topk_on_device(predictions, targets, k, '/CPU:0')
    musa_result = self._run_in_topk_on_device(
        predictions, targets, k, '/device:MUSA:0')

    expected = np.array([False, False])
    self.assertAllEqual(expected, cpu_result.numpy())
    self.assertAllEqual(expected, musa_result.numpy())

  def testInTopKV2RandomData(self):
    """Test InTopKV2 with random data."""
    np.random.seed(42)
    batch_size = 50
    num_classes = 100
    top_k = 20

    for dtype in [tf.int32, tf.int64]:
      predictions_np = np.random.randn(batch_size, num_classes).astype(np.float32)
      predictions = tf.constant(predictions_np)
      targets_np = np.random.randint(0, num_classes, size=batch_size).astype(
          dtype.as_numpy_dtype)
      targets = tf.constant(targets_np, dtype=dtype)
      k_dtype = tf.int64 if dtype == tf.int64 else tf.int32
      k_tensor = tf.constant(top_k, dtype=k_dtype)

      cpu_result = self._run_in_topk_on_device(
          predictions, targets, k_tensor, '/CPU:0')
      musa_result = self._run_in_topk_on_device(
          predictions, targets, k_tensor, '/device:MUSA:0')

      self.assertAllEqual(cpu_result.numpy(), musa_result.numpy())


if __name__ == "__main__":
  tf.test.main()
