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
  """Tests for MUSA MatMul operator, TF32 enabled by default."""

  def _test_matmul(self, shape_a, shape_b, transpose_a=False, transpose_b=False,
                   dtype=tf.float32, rtol=1e-3, atol=1e-3):
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
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-3
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
      self._test_matmul([32, 32], [32, 32], dtype=dtype, rtol=1e-2, atol=1e-2)

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

  def testMatMulGradEmptyInputZero(self):
    """Gradient w.r.t. weight must be all-zero when input has a 0-size dimension.

    Regression test for the bug where allocate_output returned uninitialized
    memory instead of a zero tensor when one matmul operand had NumElements==0.
    Reproduces the scenario in OneTrans block_2/mixed_ffn_2 where the S-branch
    token sequence length collapses to 0 after pyramid compression, causing the
    Dense weight gradient to contain garbage values (~0.097) on MUSA while the
    CPU correctly produces zeros.

    The upstream gradient is passed via output_gradients=tf.zeros_like(y) to
    avoid going through tf.reduce_sum whose backward cannot reshape a scalar
    gradient back to a 0-element shape in TF eager mode.
    The effective computation is: dw = x^T @ zeros(0, d_ff) = zeros(dim_in, d_ff),
    which is exactly the 0-size matmul path that was buggy.
    """
    dim_in, d_ff = 128, 512
    w_np = np.random.uniform(-1, 1, size=(dim_in, d_ff)).astype(np.float32)
    x_np = np.zeros((0, dim_in), dtype=np.float32)

    # CPU reference
    with tf.device('/CPU:0'):
      x_cpu = tf.constant(x_np)
      w_cpu = tf.Variable(w_np)
      with tf.GradientTape() as tape:
        y_cpu = tf.matmul(x_cpu, w_cpu)   # shape (0, d_ff)
      grad_cpu = tape.gradient(y_cpu, w_cpu,
                               output_gradients=tf.zeros_like(y_cpu))

    # MUSA
    with tf.device('/device:MUSA:0'):
      x_musa = tf.constant(x_np)
      w_musa = tf.Variable(w_np)
      with tf.GradientTape() as tape:
        y_musa = tf.matmul(x_musa, w_musa)   # shape (0, d_ff)
      grad_musa = tape.gradient(y_musa, w_musa,
                                output_gradients=tf.zeros_like(y_musa))

    self.assertAllClose(
        grad_musa.numpy(),
        grad_cpu.numpy(),
        rtol=0,
        atol=0,
    )

  def testBatchMatMulGradEmptySeqLen(self):
    """Batch matmul gradient is zero when the sequence-length dimension is 0.

    Mirrors the 3-D einsum path used by MixedFFN for the S-branch tokens:
      x: (batch, 0, dim_in) reshaped to (0, dim_in) @ W: (dim_in, d_ff)
    The gradient of W should be a (dim_in, d_ff) zero tensor.
    """
    batch, dim_in, d_ff = 4096, 128, 512
    w_np = np.random.uniform(-1, 1, size=(dim_in, d_ff)).astype(np.float32)
    x_np = np.zeros((batch, 0, dim_in), dtype=np.float32)

    # CPU reference
    with tf.device('/CPU:0'):
      x_cpu = tf.constant(x_np)
      w_cpu = tf.Variable(w_np)
      with tf.GradientTape() as tape:
        x2d = tf.reshape(x_cpu, (-1, dim_in))   # (0, dim_in)
        y_cpu = tf.matmul(x2d, w_cpu)            # (0, d_ff)
      grad_cpu = tape.gradient(y_cpu, w_cpu,
                               output_gradients=tf.zeros_like(y_cpu))

    # MUSA
    with tf.device('/device:MUSA:0'):
      x_musa = tf.constant(x_np)
      w_musa = tf.Variable(w_np)
      with tf.GradientTape() as tape:
        x2d = tf.reshape(x_musa, (-1, dim_in))
        y_musa = tf.matmul(x2d, w_musa)
      grad_musa = tape.gradient(y_musa, w_musa,
                                output_gradients=tf.zeros_like(y_musa))

    self.assertAllClose(
        grad_musa.numpy(),
        grad_cpu.numpy(),
        rtol=0,
        atol=0,
    )

  def testMatMulForwardEmptyInnerDim(self):
    """Forward output must be all-zero when the contracted (inner) dimension is 0.

    Regression test for the code path in MusaMatMulOp::Compute:
        if (in0.NumElements() == 0 || in1.NumElements() == 0) { flat_out.setZero(); }
    When k=0, both in0=(m,0) and in1=(0,n) have NumElements==0, but the
    output (m,n) is non-empty and must be zero-filled, not left as uninitialised
    device memory.
    """
    for m, k, n in [(8, 0, 16), (1, 0, 1), (256, 0, 512)]:
      a_np = np.empty((m, k), dtype=np.float32)
      b_np = np.empty((k, n), dtype=np.float32)

      with tf.device('/CPU:0'):
        result_cpu = tf.matmul(tf.constant(a_np), tf.constant(b_np))

      with tf.device('/device:MUSA:0'):
        result_musa = tf.matmul(tf.constant(a_np), tf.constant(b_np))

      self.assertAllClose(
          result_musa.numpy(),
          result_cpu.numpy(),
          rtol=0,
          atol=0,
      )

  def testBatchMatMulForwardEmptyInnerDim(self):
    """BatchMatMul forward output must be all-zero when inner dim is 0.

    Same regression as testMatMulForwardEmptyInnerDim but exercises the
    batch (3-D) code path in MusaMatMulOp::Compute.
    """
    for batch, m, k, n in [(4, 8, 0, 16), (4096, 1, 0, 128)]:
      a_np = np.empty((batch, m, k), dtype=np.float32)
      b_np = np.empty((batch, k, n), dtype=np.float32)

      with tf.device('/CPU:0'):
        result_cpu = tf.matmul(tf.constant(a_np), tf.constant(b_np))

      with tf.device('/device:MUSA:0'):
        result_musa = tf.matmul(tf.constant(a_np), tf.constant(b_np))

      self.assertAllClose(
          result_musa.numpy(),
          result_cpu.numpy(),
          rtol=0,
          atol=0,
      )


if __name__ == "__main__":
  tf.test.main()
