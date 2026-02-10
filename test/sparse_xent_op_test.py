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

"""Tests for MUSA SparseSoftmaxCrossEntropyWithLogits operator."""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

from musa_test_utils import MUSATestCase


class SparseXentOpTest(MUSATestCase):
  """Tests for MUSA SparseSoftmaxCrossEntropyWithLogits operator."""

  def _test_sparse_xent(self, shape, dtype, label_dtype=tf.int32, rtol=1e-4, atol=1e-4):
    """Test sparse xent operation with given shape and dtypes."""
    batch_size, num_classes = shape
    
    # Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    np_label_dtype = label_dtype.as_numpy_dtype

    logits_np = np.random.randn(batch_size, num_classes).astype(np_dtype)
    labels_np = np.random.randint(0, num_classes, size=batch_size).astype(np_label_dtype)

    logits = tf.constant(logits_np, dtype=dtype)
    labels = tf.constant(labels_np, dtype=label_dtype)

    # Define Operators for Loss and Gradient separately
    def op_loss(l_in, t_in):
        return gen_nn_ops.sparse_softmax_cross_entropy_with_logits(l_in, t_in).loss

    def op_grad(l_in, t_in):
        return gen_nn_ops.sparse_softmax_cross_entropy_with_logits(l_in, t_in).backprop

    # Compare Results
    print(f"  Testing Loss component for shape {shape}...")
    self._compare_cpu_musa_results(op_loss, [logits, labels], dtype, rtol=rtol, atol=atol)
    
    print(f"  Testing Gradient component for shape {shape}...")
    self._compare_cpu_musa_results(op_grad, [logits, labels], dtype, rtol=rtol, atol=atol)

  def testSparseXentBasic(self):
    """Test small scale basic functionality."""
    shape = [4, 8]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-4
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-4
      
      self._test_sparse_xent(shape, dtype, rtol=rtol, atol=atol)

  def testSparseXentLarge(self):
    """Test large scale (Wide & Deep scenario)."""
    shape = [1024, 1000]
    for dtype in [tf.float32, tf.float16]:
      rtol = 1e-2 if dtype == tf.float16 else 1e-4
      atol = 1e-2 if dtype == tf.float16 else 1e-4
      
      self._test_sparse_xent(shape, dtype, rtol=rtol, atol=atol)

  def testLabelInt64(self):
    """Test with int64 labels."""
    shape = [32, 100]
    self._test_sparse_xent(shape, tf.float32, label_dtype=tf.int64)


if __name__ == "__main__":
  tf.test.main()