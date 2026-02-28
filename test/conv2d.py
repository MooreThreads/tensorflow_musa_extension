"""Tests for MUSA Conv2D operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class Conv2DOpTest(MUSATestCase):
  """Tests for MUSA Conv2D operator."""

  def _tolerance_for_dtype(self, dtype):
    if dtype in (tf.float16, tf.bfloat16):
      return 1e-2, 1e-2
    return 1e-5, 1e-8

  def _make_input_and_filter(self, input_shape, filter_shape, dtype, seed=2026):
    np.random.seed(seed)
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    x_np = np.random.uniform(-1.0, 1.0, size=input_shape).astype(np_dtype)
    w_np = np.random.uniform(-1.0, 1.0, size=filter_shape).astype(np_dtype)
    return tf.constant(x_np, dtype=dtype), tf.constant(w_np, dtype=dtype)

  def _cpu_conv2d_reference(self,
                            x,
                            w,
                            strides,
                            padding,
                            data_format,
                            dilations):
    with tf.device('/CPU:0'):
      x_cpu = tf.cast(x, tf.float32)
      w_cpu = tf.cast(w, tf.float32)

      if data_format == "NHWC":
        return tf.nn.conv2d(
            x_cpu,
            w_cpu,
            strides=strides,
            padding=padding,
            data_format="NHWC",
            dilations=dilations)

      # CPU reference for NCHW via transpose to NHWC.
      x_nhwc = tf.transpose(x_cpu, [0, 2, 3, 1])
      strides_nhwc = [strides[0], strides[2], strides[3], strides[1]]
      dilations_nhwc = [dilations[0], dilations[2], dilations[3], dilations[1]]
      y_nhwc = tf.nn.conv2d(
          x_nhwc,
          w_cpu,
          strides=strides_nhwc,
          padding=padding,
          data_format="NHWC",
          dilations=dilations_nhwc)
      return tf.transpose(y_nhwc, [0, 3, 1, 2])

  def _test_conv2d(self,
                   input_shape,
                   filter_shape,
                   dtype,
                   strides,
                   padding,
                   data_format="NHWC",
                   dilations=None,
                   seed=2026):
    if dilations is None:
      dilations = [1, 1, 1, 1]

    rtol, atol = self._tolerance_for_dtype(dtype)
    x, w = self._make_input_and_filter(input_shape, filter_shape, dtype, seed)

    cpu_result = self._cpu_conv2d_reference(
        x=x,
        w=w,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations)

    with tf.device('/device:MUSA:0'):
      musa_result = tf.nn.conv2d(
          x,
          w,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)

    self.assertAllClose(
        cpu_result.numpy(),
        tf.cast(musa_result, tf.float32).numpy(),
        rtol=rtol,
        atol=atol)

  def testConv2DValidNHWC(self):
    """NHWC + VALID + stride=1."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d(
            input_shape=[1, 8, 8, 3],
            filter_shape=[3, 3, 3, 4],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="VALID",
            data_format="NHWC")

  def testConv2DSameNHWC(self):
    """NHWC + SAME + stride=1."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d(
            input_shape=[2, 15, 15, 4],
            filter_shape=[5, 5, 4, 6],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC")

  def testConv2DStride2ValidNHWC(self):
    """NHWC + VALID + stride=2."""
    for dtype in [tf.float32, tf.float16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d(
            input_shape=[1, 9, 9, 3],
            filter_shape=[3, 3, 3, 8],
            dtype=dtype,
            strides=[1, 2, 2, 1],
            padding="VALID",
            data_format="NHWC")

  def testConv2DDilationValidNHWC(self):
    """NHWC + VALID + dilation=2."""
    for dtype in [tf.float32, tf.float16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d(
            input_shape=[1, 12, 12, 3],
            filter_shape=[3, 3, 3, 5],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="VALID",
            data_format="NHWC",
            dilations=[1, 2, 2, 1])

  def testConv2DPointwise1x1NHWC(self):
    """1x1 pointwise conv."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d(
            input_shape=[4, 16, 16, 7],
            filter_shape=[1, 1, 7, 13],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC")

  def testConv2DValidNCHW(self):
    """NCHW + VALID + stride=1."""
    for dtype in [tf.float32, tf.float16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d(
            input_shape=[1, 3, 8, 8],
            filter_shape=[3, 3, 3, 4],  # TF filter layout is still HWIO
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="VALID",
            data_format="NCHW")

  def testConv2DEmptyBatch(self):
    """Empty batch should return empty output."""
    self._test_conv2d(
        input_shape=[0, 8, 8, 3],
        filter_shape=[3, 3, 3, 4],
        dtype=tf.float32,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format="NHWC")

  def testConv2DAsymmetricSamePaddingUnsupported(self):
    """Current MUSA implementation rejects asymmetric SAME padding."""
    x = tf.constant(np.random.uniform(-1, 1, [1, 4, 4, 3]).astype(np.float32))
    w = tf.constant(np.random.uniform(-1, 1, [3, 3, 3, 4]).astype(np.float32))

    # CPU should pass.
    with tf.device('/CPU:0'):
      _ = tf.nn.conv2d(
          x, w, strides=[1, 2, 2, 1], padding="SAME", data_format="NHWC")

    # MUSA current kernel throws Unimplemented/InvalidArgument.
    with self.assertRaises((tf.errors.UnimplementedError,
                            tf.errors.InvalidArgumentError)):
      with tf.device('/device:MUSA:0'):
        _ = tf.nn.conv2d(
            x, w, strides=[1, 2, 2, 1], padding="SAME", data_format="NHWC")

  def testConv2DInvalidStridesOnBatchOrChannel(self):
    """Stride on N/C dims is invalid."""
    x = tf.constant(np.random.uniform(-1, 1, [1, 8, 8, 3]).astype(np.float32))
    w = tf.constant(np.random.uniform(-1, 1, [3, 3, 3, 4]).astype(np.float32))

    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      tf.nn.conv2d(x, w, strides=[2, 1, 1, 1], padding="SAME")
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      tf.nn.conv2d(x, w, strides=[1, 1, 1, 2], padding="SAME")

  def testConv2DChannelMismatch(self):
    """Input C and filter IC mismatch should fail."""
    x = tf.constant(np.random.uniform(-1, 1, [1, 8, 8, 3]).astype(np.float32))
    w = tf.constant(np.random.uniform(-1, 1, [3, 3, 4, 8]).astype(np.float32))

    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


if __name__ == "__main__":
  tf.test.main()
