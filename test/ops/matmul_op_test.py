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

import os
os.environ.setdefault("MUSA_ENABLE_TF32", "0")

import tensorflow as tf
import numpy as np

from musa_test_utils import MUSATestCase


def is_tf32_enabled():
  return int(os.environ.get("MUSA_ENABLE_TF32", "0")) != 0


def float32_tolerance(default_rtol=1e-3, default_atol=1e-3):
  """Return (rtol, atol) appropriate for float32 matmul comparisons.

  When TF32 is enabled (MUSA_ENABLE_TF32=1), MUSA uses 10-bit mantissa
  precision internally, so errors can reach ~1e-2 for large matrices.
  When TF32 is disabled (default), full FP32 precision applies.
  """
  return (1e-2, 1e-2) if is_tf32_enabled() else (default_rtol, default_atol)


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

  def testBatchMatMulBroadcast3Dx2D(self):
    """3-D × 2-D broadcast BatchMatMul: in1 is a shared weight matrix.

    Regression test for INVALID_PARAMETER (Status: 1) from mBatchMatMul when
    the 2-D input is broadcast across the batch dimension.

    Root cause: ReshapeTo3D set shape={out_batch, rows, cols} with stride={0,...}
    for the 2-D input, but muDNN requires batch_b=1 when stride_b=0.
    Fix: use shape={1, rows, cols} for the broadcast case.

    This pattern is triggered by AFM's AttentionPooling:
        e = tf.matmul(interactions, self.W)   # (bs, num_pairs, k) x (k, t)
        e = tf.matmul(e, self.h)              # (bs, num_pairs, t) x (t, 1)

    Note on tolerance: MUSA defaults to TF32 (MUSA_ENABLE_TF32=1), which has
    FP16-level mantissa precision (10 bits). For large k (e.g. k=256), errors
    accumulate to ~1e-2, so float32 on MUSA requires the same tolerance as
    float16 rather than the full-precision 1e-3.

    For float16, only small-k shapes are tested: fp16 accumulation error scales
    as k × eps_fp16 ≈ k × 1e-3, which exceeds 1e-2 at k=256. The broadcast
    fix is dtype-agnostic, so small shapes are sufficient to cover the path.
    """
    # float32: test both small and large k (large k validates TF32 tolerance).
    # float16: small k only — fp16 precision with k=256 exceeds rtol=1e-2.
    test_shapes = {
        tf.float32: {
            "case1": [(4, 10, 16, 8), (8, 45, 256, 64)],
            "case2": [(4, 10, 8), (8, 45, 64)],
        },
        tf.float16: {
            "case1": [(4, 10, 16, 8)],
            "case2": [(4, 10, 8)],
        },
    }

    for dtype in [tf.float32, tf.float16]:
      rtol, atol = (1e-2, 1e-2) if dtype == tf.float16 else float32_tolerance()
      np_dtype = np.float16 if dtype == tf.float16 else np.float32

      # Case 1: (bs, num_pairs, k) x (k, t) — attention W matrix
      for bs, num_pairs, k, t in test_shapes[dtype]["case1"]:
        a_np = np.random.uniform(-1, 1, (bs, num_pairs, k)).astype(np_dtype)
        b_np = np.random.uniform(-1, 1, (k, t)).astype(np_dtype)
        a = tf.constant(a_np, dtype=dtype)
        b = tf.constant(b_np, dtype=dtype)

        with tf.device('/CPU:0'):
          cpu_result = tf.cast(tf.matmul(a, b), tf.float32)
        with tf.device('/device:MUSA:0'):
          musa_result = tf.cast(tf.matmul(a, b), tf.float32)

        self.assertAllClose(musa_result.numpy(), cpu_result.numpy(),
                            rtol=rtol, atol=atol)

      # Case 2: (bs, num_pairs, t) x (t, 1) — attention h vector (n=1)
      for bs, num_pairs, t in test_shapes[dtype]["case2"]:
        a_np = np.random.uniform(-1, 1, (bs, num_pairs, t)).astype(np_dtype)
        b_np = np.random.uniform(-1, 1, (t, 1)).astype(np_dtype)
        a = tf.constant(a_np, dtype=dtype)
        b = tf.constant(b_np, dtype=dtype)

        with tf.device('/CPU:0'):
          cpu_result = tf.cast(tf.matmul(a, b), tf.float32)
        with tf.device('/device:MUSA:0'):
          musa_result = tf.cast(tf.matmul(a, b), tf.float32)

        self.assertAllClose(musa_result.numpy(), cpu_result.numpy(),
                            rtol=rtol, atol=atol)

  def testBatchMatMulBroadcast3Dx2DGrad(self):
    """Backward pass of 3-D × 2-D broadcast BatchMatMul.

    Verifies that gradients flow correctly through both weight matrices
    in the AFM AttentionPooling pattern during training.
    """
    bs, num_pairs, k, t = 4, 10, 16, 8

    a_np = np.random.uniform(-1, 1, (bs, num_pairs, k)).astype(np.float32)
    W_np = np.random.uniform(-1, 1, (k, t)).astype(np.float32)
    h_np = np.random.uniform(-1, 1, (t, 1)).astype(np.float32)

    def run_attention_pooling(device):
      with tf.device(device):
        interactions = tf.constant(a_np)
        W = tf.Variable(W_np)
        h = tf.Variable(h_np)
        with tf.GradientTape() as tape:
          e = tf.nn.relu(tf.matmul(interactions, W))  # (bs, num_pairs, t)
          e = tf.matmul(e, h)                         # (bs, num_pairs, 1)
          loss = tf.reduce_sum(e)
        grads = tape.gradient(loss, [W, h])
      return grads

    cpu_grads = run_attention_pooling('/CPU:0')
    musa_grads = run_attention_pooling('/device:MUSA:0')

    rtol, atol = float32_tolerance()
    self.assertAllClose(musa_grads[0].numpy(), cpu_grads[0].numpy(),
                        rtol=rtol, atol=atol)
    self.assertAllClose(musa_grads[1].numpy(), cpu_grads[1].numpy(),
                        rtol=rtol, atol=atol)


if __name__ == "__main__":
  tf.test.main()
