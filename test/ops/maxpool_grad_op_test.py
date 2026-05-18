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

"""Tests for MUSA MaxPoolGrad operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import load_musa_plugin

load_musa_plugin()
MUSA_DEVICES = tf.config.list_physical_devices('MUSA')


class MaxPoolGradOpTest(tf.test.TestCase):
  """Tests for MUSA MaxPoolGrad operator."""

  def _test_maxpool_grad(self, input_shape, ksize, strides, padding, dtype,
                         data_format='NHWC', rtol=1e-4, atol=1e-4):
    """Compare MUSA MaxPoolGrad against CPU via tf.GradientTape."""
    if not MUSA_DEVICES:
      self.skipTest('No MUSA devices found.')

    np.random.seed(42)
    if dtype == tf.bfloat16:
      input_np = np.random.randn(*input_shape).astype(np.float32)
    else:
      input_np = np.random.randn(*input_shape).astype(dtype.as_numpy_dtype)

    def _compute_grad(device):
      is_cpu = device.startswith('/CPU')
      # CPU MaxPool only supports NHWC; for NCHW tests on CPU we transpose the
      # input to NHWC, run NHWC pool+grad, then transpose the gradient back so
      # the result shape matches the MUSA NCHW output.
      use_nchw_on_cpu = (data_format == 'NCHW' and is_cpu)
      effective_format = 'NHWC' if use_nchw_on_cpu else data_format
      if use_nchw_on_cpu:
        # NCHW ksize/strides: [N, C, H, W] -> NHWC: [N, H, W, C]
        eff_ksize = [ksize[0], ksize[2], ksize[3], ksize[1]]
        eff_strides = [strides[0], strides[2], strides[3], strides[1]]
        # Transpose input NCHW -> NHWC
        inp = np.transpose(input_np, (0, 2, 3, 1))
      else:
        eff_ksize = ksize
        eff_strides = strides
        inp = input_np

      with tf.device(device):
        x = tf.Variable(tf.constant(inp, dtype=dtype))
        with tf.GradientTape() as tape:
          y = tf.nn.max_pool2d(x, ksize=eff_ksize, strides=eff_strides,
                               padding=padding, data_format=effective_format)
          # Use sum as the scalar loss so the upstream gradient is all-ones.
          loss = tf.reduce_sum(tf.cast(y, tf.float32))
        grad = tape.gradient(loss, x)

      grad_np = tf.cast(grad, tf.float32).numpy()
      if use_nchw_on_cpu:
        # Transpose gradient back: NHWC -> NCHW so shapes match MUSA output.
        grad_np = np.transpose(grad_np, (0, 3, 1, 2))
      return grad_np

    cpu_grad = _compute_grad('/CPU:0')
    musa_grad = _compute_grad('/device:MUSA:0')

    self.assertAllClose(cpu_grad, musa_grad, rtol=rtol, atol=atol)

  # ------------------------------------------------------------------
  # NHWC tests
  # ------------------------------------------------------------------

  def testMaxPoolGradBasicSameNHWC_f32(self):
    self._test_maxpool_grad(
        input_shape=(1, 4, 4, 1),
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME', dtype=tf.float32)

  def testMaxPoolGradBasicValidNHWC_f32(self):
    self._test_maxpool_grad(
        input_shape=(1, 4, 4, 1),
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='VALID', dtype=tf.float32)

  def testMaxPoolGradStride1SameNHWC_f32(self):
    self._test_maxpool_grad(
        input_shape=(2, 6, 6, 3),
        ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
        padding='SAME', dtype=tf.float32)

  def testMaxPoolGradBatchNHWC_f32(self):
    self._test_maxpool_grad(
        input_shape=(4, 8, 8, 4),
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME', dtype=tf.float32)

  def testMaxPoolGradNHWC_f16(self):
    self._test_maxpool_grad(
        input_shape=(1, 4, 4, 2),
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME', dtype=tf.float16, rtol=1e-2, atol=1e-2)

  def testMaxPoolGradNHWC_bf16(self):
    self._test_maxpool_grad(
        input_shape=(1, 4, 4, 2),
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME', dtype=tf.bfloat16, rtol=5e-2, atol=5e-2)

  # ------------------------------------------------------------------
  # NCHW tests
  # ------------------------------------------------------------------

  def testMaxPoolGradBasicSameNCHW_f32(self):
    self._test_maxpool_grad(
        input_shape=(1, 1, 4, 4),
        ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2],
        padding='SAME', dtype=tf.float32, data_format='NCHW')

  def testMaxPoolGradBasicValidNCHW_f32(self):
    self._test_maxpool_grad(
        input_shape=(1, 2, 4, 4),
        ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2],
        padding='VALID', dtype=tf.float32, data_format='NCHW')

  def testMaxPoolGradBatchNCHW_f32(self):
    self._test_maxpool_grad(
        input_shape=(4, 4, 8, 8),
        ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2],
        padding='SAME', dtype=tf.float32, data_format='NCHW')

  def testMaxPoolGradNCHW_f16(self):
    self._test_maxpool_grad(
        input_shape=(2, 2, 4, 4),
        ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2],
        padding='SAME', dtype=tf.float16, data_format='NCHW',
        rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
  tf.test.main()
