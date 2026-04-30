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

"""Tests for MUSA FusedBatchNormV3 operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class FusedBatchNormOpTest(MUSATestCase):

  def testFusedBatchNormV3Forward(self):
    shape = [2, 2, 2, 4]
    dtype = tf.float32

    x_np = np.random.randn(*shape).astype(np.float32)
    scale_np = np.random.rand(shape[-1]).astype(np.float32)
    offset_np = np.random.rand(shape[-1]).astype(np.float32)
    mean_np = np.zeros(shape[-1]).astype(np.float32)
    var_np = np.ones(shape[-1]).astype(np.float32)

    def forward_op(x, scale, offset, mean, var):
      y, _, _, _, _, _ = tf.raw_ops.FusedBatchNormV3(
          x=x,
          scale=scale,
          offset=offset,
          mean=mean,
          variance=var,
          epsilon=0.001,
          exponential_avg_factor=1.0,
          data_format="NHWC",
          is_training=True)
      return y

    self._compare_cpu_musa_results(
        forward_op,
        [x_np, scale_np, offset_np, mean_np, var_np],
        dtype,
        rtol=1e-4,
        atol=1e-4
    )

  def testFusedBatchNormV3GradientDX(self):
    shape = [2, 2, 2, 4]
    dtype = tf.float32

    x_np = np.random.randn(*shape).astype(np.float32)
    scale_np = np.random.rand(shape[-1]).astype(np.float32)
    offset_np = np.random.rand(shape[-1]).astype(np.float32)
    mean_np = np.zeros(shape[-1]).astype(np.float32)
    var_np = np.ones(shape[-1]).astype(np.float32)

    def grad_dx_op(x, scale, offset, mean, var):
      # 【核心修复】强制将输入转换为 Tensor，否则 tape.watch 会报错
      x = tf.convert_to_tensor(x)

      with tf.GradientTape() as tape:
        tape.watch(x)
        y, _, _, _, _, _ = tf.raw_ops.FusedBatchNormV3(
            x=x,
            scale=scale,
            offset=offset,
            mean=mean,
            variance=var,
            epsilon=0.001,
            exponential_avg_factor=1.0,
            data_format="NHWC",
            is_training=True)
        loss = tf.reduce_sum(y)

      dx = tape.gradient(loss, x)
      return dx

    self._compare_cpu_musa_results(
        grad_dx_op,
        [x_np, scale_np, offset_np, mean_np, var_np],
        dtype,
        rtol=1e-3,
        atol=1e-3
    )

  def testFusedBatchNormGradDX2D(self):
    """BN backward dx must be correct for 2D input (Dense-layer scenario).

    Regression test for the bug where dx was allocated but not zero-initialized
    before calling mudnn RunBwd.  For 4D NHWC inputs d_scale/d_offset/d_mean/
    d_var were already memset to zero, but dx was not, leading to accumulated
    garbage when mudnn uses an add-to-output (accumulation) kernel.

    Reproduces the scenario in OneTrans MLP head:
      x_pooled (4096, 128) -> Dense(256) -> BN -> ReLU -> Dense(1)
    where dense_19's weight gradient (dW = x_pooled^T @ BN_dx) showed
    6e-3 diff on MUSA vs CPU after fixing the empty-tensor matmul bug.
    """
    batch, features = 4096, 256
    x_np = np.random.randn(batch, features).astype(np.float32)
    scale_np = np.ones(features, dtype=np.float32)
    offset_np = np.zeros(features, dtype=np.float32)

    # Use Keras BatchNormalization which internally calls FusedBatchNormV3
    # with 2D input (reshape to (N,1,1,C) path)
    def run_bn_grad(device):
      with tf.device(device):
        x = tf.constant(x_np)
        bn = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-5)
        # Build the BN layer
        _ = bn(x, training=True)
        # Set known weights to make CPU/MUSA comparable
        bn.gamma.assign(scale_np)
        bn.beta.assign(offset_np)
        with tf.GradientTape() as tape:
          tape.watch(x)
          y = bn(x, training=True)
          loss = tf.reduce_sum(y)
        dx = tape.gradient(loss, x)
      return dx

    dx_cpu = run_bn_grad('/CPU:0')
    dx_musa = run_bn_grad('/device:MUSA:0')

    self.assertAllClose(
        dx_musa.numpy(),
        dx_cpu.numpy(),
        rtol=1e-3,
        atol=1e-3,
    )

  def testFusedBatchNormGradDX2DWeightGrad(self):
    """Dense weight gradient is correct when followed by BN (training mode).

    Directly verifies that dW = input^T @ BN_dx is correct on MUSA, which
    is the exact computation for dense_19 in the OneTrans MLP head.
    """
    batch, dim_in, dim_hidden = 4096, 128, 256

    x_np = np.random.randn(batch, dim_in).astype(np.float32)
    w_np = np.random.randn(dim_in, dim_hidden).astype(np.float32)

    def run(device):
      with tf.device(device):
        x = tf.constant(x_np)
        w = tf.Variable(w_np)
        bn = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-5)
        # Build BN
        _ = bn(tf.matmul(x, w), training=True)
        with tf.GradientTape() as tape:
          h = tf.matmul(x, w)           # (batch, dim_hidden)
          h_bn = bn(h, training=True)   # (batch, dim_hidden)
          loss = tf.reduce_sum(h_bn)
        dw = tape.gradient(loss, w)
      return dw

    dw_cpu = run('/CPU:0')
    dw_musa = run('/device:MUSA:0')

    self.assertAllClose(
        dw_musa.numpy(),
        dw_cpu.numpy(),
        rtol=1e-3,
        atol=1e-3,
    )

  def testFusedBatchNormGradDXNCHW(self):
    """BN backward dx must be correct for 4D NCHW format.

    Regression test to complement testFusedBatchNormV3GradientDX (NHWC).
    Ensures the dx zero-initialization fix in MusaFusedBatchNormGradOp works
    for NCHW format as well.
    """
    N, C, H, W = 4, 8, 6, 6
    x_np = np.random.randn(N, C, H, W).astype(np.float32)
    scale_np = np.random.rand(C).astype(np.float32)
    offset_np = np.random.rand(C).astype(np.float32)
    mean_np = np.zeros(C, dtype=np.float32)
    var_np = np.ones(C, dtype=np.float32)

    def grad_dx_nchw(x, scale, offset, mean, var):
      x = tf.convert_to_tensor(x)
      with tf.GradientTape() as tape:
        tape.watch(x)
        y, _, _, _, _, _ = tf.raw_ops.FusedBatchNormV3(
            x=x,
            scale=scale,
            offset=offset,
            mean=mean,
            variance=var,
            epsilon=1e-5,
            exponential_avg_factor=1.0,
            data_format="NCHW",
            is_training=True)
        loss = tf.reduce_sum(y)
      return tape.gradient(loss, x)

    self._compare_cpu_musa_results(
        grad_dx_nchw,
        [x_np, scale_np, offset_np, mean_np, var_np],
        tf.float32,
        rtol=1e-3,
        atol=1e-3,
    )

  def testFusedBatchNormGradDXNotAffectedByPriorMemory(self):
    """BN backward dx must not inherit values from a previously freed buffer.

    Regression test for the bug where MusaFusedBatchNormGradOp allocated dx
    without zero-initialising it before passing it to mudnn RunBwd (which uses
    accumulation semantics).  If the MUSA allocator reuses a pooled buffer that
    happened to contain large non-zero values, the result would be wrong.

    Strategy: explicitly fill a same-size buffer with large sentinel values
    (999.0) on MUSA then release it, so the allocator pools that memory.  The
    subsequent BN backward is likely to get the same buffer for dx.  With the
    fix the result must still match CPU; without it the result would be
    contaminated by the sentinel values.
    """
    batch, features = 512, 64
    x_np = np.random.randn(batch, features).astype(np.float32)

    # CPU reference (no memory pollution involved)
    with tf.device('/CPU:0'):
      x_cpu = tf.constant(x_np)
      bn_cpu = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-5)
      _ = bn_cpu(x_cpu, training=True)
      with tf.GradientTape() as tape:
        tape.watch(x_cpu)
        y = bn_cpu(x_cpu, training=True)
        loss = tf.reduce_sum(y)
      dx_cpu = tape.gradient(loss, x_cpu).numpy()
    gamma_np = bn_cpu.gamma.numpy()
    beta_np = bn_cpu.beta.numpy()

    # MUSA: pollute the allocator pool, then run BN backward
    with tf.device('/device:MUSA:0'):
      # Step 1: fill a buffer with sentinel values and then release it
      sentinel = tf.Variable(
          np.full((batch, features), 999.0, dtype=np.float32))
      _ = sentinel.numpy()  # ensure the device write is materialised
      del sentinel           # return the buffer to the allocator pool

      # Step 2: BN backward -- likely reuses the sentinel buffer for dx
      x_musa = tf.constant(x_np)
      bn_musa = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-5)
      _ = bn_musa(x_musa, training=True)
      # Use the same weights as the CPU BN so results are directly comparable
      bn_musa.gamma.assign(gamma_np)
      bn_musa.beta.assign(beta_np)
      with tf.GradientTape() as tape:
        tape.watch(x_musa)
        y = bn_musa(x_musa, training=True)
        loss = tf.reduce_sum(y)
      dx_musa = tape.gradient(loss, x_musa).numpy()

    self.assertAllClose(
        dx_musa,
        dx_cpu,
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
  tf.test.main()
