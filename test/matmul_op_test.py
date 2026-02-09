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

"""Tests for MUSA MatMul operator."""

import tensorflow as tf
import numpy as np

from musa_test_utils import MUSATestCase


class MatMulOpTest(MUSATestCase):
  """Tests for MUSA MatMul operator."""

  def _test_matmul(self, shape_a, shape_b, transpose_a=False, transpose_b=False, 
                   dtype=tf.float32, rtol=1e-5, atol=1e-8):
    """Test matmul operation with given shapes and options."""
    if dtype == tf.bfloat16:
      a_np = np.random.uniform(-1, 1, size=shape_a).astype(np.float32)
      b_np = np.random.uniform(-1, 1, size=shape_b).astype(np.float32)
    else:
      a_np = np.random.uniform(-1, 1, size=shape_a).astype(dtype.as_numpy_dtype)
      b_np = np.random.uniform(-1, 1, size=shape_b).astype(dtype.as_numpy_dtype)
    
    a = tf.constant(a_np, dtype=dtype)
    b = tf.constant(b_np, dtype=dtype)
    
    # Test on CPU
    with tf.device('/CPU:0'):
      cpu_result = tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    
    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      musa_result = tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    
    # Compare results
    if dtype in [tf.float16, tf.bfloat16]:
      cpu_result_f32 = tf.cast(cpu_result, tf.float32)
      musa_result_f32 = tf.cast(musa_result, tf.float32)
      self.assertAllClose(cpu_result_f32.numpy(), 
                         musa_result_f32.numpy(),
                         rtol=rtol, 
                         atol=atol)
    else:
      self.assertAllClose(cpu_result.numpy(), 
                         musa_result.numpy(),
                         rtol=rtol, 
                         atol=atol)

  def testMatMulBasic(self):
    """Basic matrix multiplication test."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_matmul([10, 20], [20, 15], dtype=dtype, rtol=rtol, atol=atol)

  def testMatMulTransposeA(self):
    """Matrix multiplication with transpose_a=True."""
    for dtype in [tf.float32]:
      self._test_matmul([20, 10], [20, 15], transpose_a=True, dtype=dtype)

  def testMatMulTransposeB(self):
    """Matrix multiplication with transpose_b=True."""
    for dtype in [tf.float32]:
      self._test_matmul([10, 20], [15, 20], transpose_b=True, dtype=dtype)

  def testMatMulTransposeBoth(self):
    """Matrix multiplication with both transposes."""
    for dtype in [tf.float32]:
      self._test_matmul([20, 10], [15, 20], transpose_a=True, transpose_b=True, dtype=dtype)

  def testMatMulSquare(self):
    """Square matrix multiplication."""
    for dtype in [tf.float32, tf.float16]:
      rtol = 1e-2 if dtype == tf.float16 else 1e-5
      atol = 1e-2 if dtype == tf.float16 else 1e-8
      self._test_matmul([32, 32], [32, 32], dtype=dtype, rtol=rtol, atol=atol)

  def testMatMulVectorMatrix(self):
    """Vector-matrix multiplication."""
    for dtype in [tf.float32]:
      self._test_matmul([1, 10], [10, 5], dtype=dtype)

  def testMatMulMatrixVector(self):
    """Matrix-vector multiplication."""
    for dtype in [tf.float32]:
      self._test_matmul([5, 10], [10, 1], dtype=dtype)

  def testMatMulBatch(self):
    """Batch matrix multiplication."""
    for dtype in [tf.float32]:
      self._test_matmul([3, 4, 5], [3, 5, 6], dtype=dtype)


if __name__ == "__main__":
  tf.test.main()