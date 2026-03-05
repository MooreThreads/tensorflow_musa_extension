"""Tests for MUSA Conv2D backward operators (Conv2DBackpropInput, Conv2DBackpropFilter)."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


def _conv2d_backprop_input(input_sizes,
                           filter_t,
                           out_backprop,
                           strides,
                           padding,
                           data_format,
                           dilations):
  return tf.raw_ops.Conv2DBackpropInput(
      input_sizes=input_sizes,
      filter=filter_t,
      out_backprop=out_backprop,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilations=dilations)


def _conv2d_backprop_filter(input_t,
                            filter_sizes,
                            out_backprop,
                            strides,
                            padding,
                            data_format,
                            dilations):
  return tf.raw_ops.Conv2DBackpropFilter(
      input=input_t,
      filter_sizes=filter_sizes,
      out_backprop=out_backprop,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilations=dilations)


class Conv2DBackpropInputOpTest(MUSATestCase):
  """Tests for MUSA Conv2DBackpropInput operator."""

  def _tolerance_for_dtype(self, dtype):
    if dtype == tf.bfloat16:
      return 5e-2, 5e-2
    if dtype == tf.float16:
      return 1e-2, 1e-2
    return 1e-4, 1e-5

  def _make_inputs(self, input_shape, filter_shape, dtype, seed=2026):
    np.random.seed(seed)
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    x_np = np.random.uniform(-1.0, 1.0, size=input_shape).astype(np_dtype)
    w_np = np.random.uniform(-1.0, 1.0, size=filter_shape).astype(np_dtype)
    return tf.constant(x_np, dtype=dtype), tf.constant(w_np, dtype=dtype)

  def _cpu_conv2d_backprop_input_reference(
      self, input_sizes, filter_t, out_backprop, strides, padding,
      data_format, dilations):
    with tf.device('/CPU:0'):
      filter_cpu = tf.cast(filter_t, tf.float32)
      out_backprop_cpu = tf.cast(out_backprop, tf.float32)

      if data_format == "NHWC":
        x_cpu = tf.zeros(input_sizes, dtype=tf.float32)
        with tf.GradientTape() as tape:
          tape.watch(x_cpu)
          y = tf.nn.conv2d(
              x_cpu,
              filter_cpu,
              strides=strides,
              padding=padding,
              data_format="NHWC",
              dilations=dilations)
          loss = tf.reduce_sum(y * out_backprop_cpu)
        return tape.gradient(loss, x_cpu)

      # CPU reference for NCHW via transpose to NHWC.
      x_cpu_nhwc = tf.zeros([input_sizes[0], input_sizes[2], input_sizes[3], input_sizes[1]],
                            dtype=tf.float32)
      out_backprop_nhwc = tf.transpose(out_backprop_cpu, [0, 2, 3, 1])
      strides_nhwc = [strides[0], strides[2], strides[3], strides[1]]
      dilations_nhwc = [dilations[0], dilations[2], dilations[3], dilations[1]]
      with tf.GradientTape() as tape:
        tape.watch(x_cpu_nhwc)
        y = tf.nn.conv2d(
            x_cpu_nhwc,
            filter_cpu,
            strides=strides_nhwc,
            padding=padding,
            data_format="NHWC",
            dilations=dilations_nhwc)
        loss = tf.reduce_sum(y * out_backprop_nhwc)
      dx_nhwc = tape.gradient(loss, x_cpu_nhwc)
      return tf.transpose(dx_nhwc, [0, 3, 1, 2])

  def _test_conv2d_backprop_input(
      self, input_shape, filter_shape, dtype, strides, padding,
      data_format="NHWC", dilations=None, seed=2026):
    if dilations is None:
      dilations = [1, 1, 1, 1]

    rtol, atol = self._tolerance_for_dtype(dtype)

    # Compute output shape for out_backprop
    if data_format == "NHWC":
      batch, in_h, in_w, in_c = input_shape
      filter_h, filter_w, _, out_c = filter_shape
    else:
      batch, in_c, in_h, in_w = input_shape
      filter_h, filter_w, _, out_c = filter_shape

    # Compute output dimensions
    effective_kh = (filter_h - 1) * dilations[1] + 1 if data_format == "NHWC" else (filter_h - 1) * dilations[2] + 1
    effective_kw = (filter_w - 1) * dilations[2] + 1 if data_format == "NHWC" else (filter_w - 1) * dilations[3] + 1
    stride_h = strides[1] if data_format == "NHWC" else strides[2]
    stride_w = strides[2] if data_format == "NHWC" else strides[3]

    if padding == "VALID":
      out_h = max(0, (in_h + stride_h - effective_kh) // stride_h)
      out_w = max(0, (in_w + stride_w - effective_kw) // stride_w)
    else:  # SAME
      out_h = (in_h + stride_h - 1) // stride_h
      out_w = (in_w + stride_w - 1) // stride_w

    if data_format == "NHWC":
      out_backprop_shape = [batch, out_h, out_w, out_c]
    else:
      out_backprop_shape = [batch, out_c, out_h, out_w]

    # Create random out_backprop
    np.random.seed(seed)
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    out_backprop_np = np.random.uniform(-1.0, 1.0, size=out_backprop_shape).astype(np_dtype)
    out_backprop = tf.constant(out_backprop_np, dtype=dtype)

    # Create filter
    _, w = self._make_inputs(input_shape, filter_shape, dtype, seed)
    input_sizes = tf.constant(input_shape, dtype=tf.int32)

    cpu_result = self._cpu_conv2d_backprop_input_reference(
        input_sizes=input_shape,
        filter_t=w,
        out_backprop=out_backprop,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations)

    with tf.device('/device:MUSA:0'):
      musa_result = _conv2d_backprop_input(
          input_sizes=input_sizes,
          filter_t=w,
          out_backprop=out_backprop,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)

    self.assertAllClose(
        cpu_result.numpy(),
        tf.cast(musa_result, tf.float32).numpy(),
        rtol=rtol,
        atol=atol)

  def testConv2DBackpropInputValidNHWC(self):
    """NHWC + VALID + stride=1."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_input(
            input_shape=[1, 8, 8, 3],
            filter_shape=[3, 3, 3, 4],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="VALID",
            data_format="NHWC")

  def testConv2DBackpropInputSameNHWC(self):
    """NHWC + SAME + stride=1."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_input(
            input_shape=[2, 15, 15, 4],
            filter_shape=[5, 5, 4, 6],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC")

  def testConv2DBackpropInputStride2ValidNHWC(self):
    """NHWC + VALID + stride=2."""
    for dtype in [tf.float32, tf.float16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_input(
            input_shape=[1, 9, 9, 3],
            filter_shape=[3, 3, 3, 8],
            dtype=dtype,
            strides=[1, 2, 2, 1],
            padding="VALID",
            data_format="NHWC")

  def testConv2DBackpropInputDilationValidNHWC(self):
    """NHWC + VALID + dilation=2."""
    self.skipTest("TF CPU backprop reference does not support dilation > 1")
    for dtype in [tf.float32, tf.float16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_input(
            input_shape=[1, 12, 12, 3],
            filter_shape=[3, 3, 3, 5],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="VALID",
            data_format="NHWC",
            dilations=[1, 2, 2, 1])

  def testConv2DBackpropInputPointwise1x1NHWC(self):
    """1x1 pointwise conv backprop input."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_input(
            input_shape=[4, 16, 16, 7],
            filter_shape=[1, 1, 7, 13],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC")

  def testConv2DBackpropInputValidNCHW(self):
    """NCHW + VALID + stride=1."""
    for dtype in [tf.float32, tf.float16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_input(
            input_shape=[1, 3, 8, 8],
            filter_shape=[3, 3, 3, 4],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="VALID",
            data_format="NCHW")


class Conv2DBackpropFilterOpTest(MUSATestCase):
  """Tests for MUSA Conv2DBackpropFilter operator."""

  def _tolerance_for_dtype(self, dtype):
    if dtype == tf.bfloat16:
      return 5e-2, 5e-2
    if dtype == tf.float16:
      return 1e-2, 1e-2
    return 1e-4, 1e-5

  def _make_inputs(self, input_shape, filter_shape, dtype, seed=2026):
    np.random.seed(seed)
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    x_np = np.random.uniform(-1.0, 1.0, size=input_shape).astype(np_dtype)
    w_np = np.random.uniform(-1.0, 1.0, size=filter_shape).astype(np_dtype)
    return tf.constant(x_np, dtype=dtype), tf.constant(w_np, dtype=dtype)

  def _cpu_conv2d_backprop_filter_reference(
      self, input_t, filter_sizes, out_backprop, strides, padding,
      data_format, dilations):
    with tf.device('/CPU:0'):
      input_cpu = tf.cast(input_t, tf.float32)
      out_backprop_cpu = tf.cast(out_backprop, tf.float32)

      if data_format == "NHWC":
        w_cpu = tf.zeros(filter_sizes, dtype=tf.float32)
        with tf.GradientTape() as tape:
          tape.watch(w_cpu)
          y = tf.nn.conv2d(
              input_cpu,
              w_cpu,
              strides=strides,
              padding=padding,
              data_format="NHWC",
              dilations=dilations)
          loss = tf.reduce_sum(y * out_backprop_cpu)
        return tape.gradient(loss, w_cpu)

      # CPU reference for NCHW via transpose to NHWC.
      input_nhwc = tf.transpose(input_cpu, [0, 2, 3, 1])
      out_backprop_nhwc = tf.transpose(out_backprop_cpu, [0, 2, 3, 1])
      strides_nhwc = [strides[0], strides[2], strides[3], strides[1]]
      dilations_nhwc = [dilations[0], dilations[2], dilations[3], dilations[1]]
      w_cpu = tf.zeros(filter_sizes, dtype=tf.float32)
      with tf.GradientTape() as tape:
        tape.watch(w_cpu)
        y = tf.nn.conv2d(
            input_nhwc,
            w_cpu,
            strides=strides_nhwc,
            padding=padding,
            data_format="NHWC",
            dilations=dilations_nhwc)
        loss = tf.reduce_sum(y * out_backprop_nhwc)
      return tape.gradient(loss, w_cpu)

  def _test_conv2d_backprop_filter(
      self, input_shape, filter_shape, dtype, strides, padding,
      data_format="NHWC", dilations=None, seed=2026):
    if dilations is None:
      dilations = [1, 1, 1, 1]

    rtol, atol = self._tolerance_for_dtype(dtype)

    # Compute output shape for out_backprop
    if data_format == "NHWC":
      batch, in_h, in_w, in_c = input_shape
      filter_h, filter_w, _, out_c = filter_shape
    else:
      batch, in_c, in_h, in_w = input_shape
      filter_h, filter_w, _, out_c = filter_shape

    # Compute output dimensions
    effective_kh = (filter_h - 1) * dilations[1] + 1 if data_format == "NHWC" else (filter_h - 1) * dilations[2] + 1
    effective_kw = (filter_w - 1) * dilations[2] + 1 if data_format == "NHWC" else (filter_w - 1) * dilations[3] + 1
    stride_h = strides[1] if data_format == "NHWC" else strides[2]
    stride_w = strides[2] if data_format == "NHWC" else strides[3]

    if padding == "VALID":
      out_h = max(0, (in_h + stride_h - effective_kh) // stride_h)
      out_w = max(0, (in_w + stride_w - effective_kw) // stride_w)
    else:  # SAME
      out_h = (in_h + stride_h - 1) // stride_h
      out_w = (in_w + stride_w - 1) // stride_w

    if data_format == "NHWC":
      out_backprop_shape = [batch, out_h, out_w, out_c]
    else:
      out_backprop_shape = [batch, out_c, out_h, out_w]

    # Create random out_backprop
    np.random.seed(seed)
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    out_backprop_np = np.random.uniform(-1.0, 1.0, size=out_backprop_shape).astype(np_dtype)
    out_backprop = tf.constant(out_backprop_np, dtype=dtype)

    # Create input
    x, _ = self._make_inputs(input_shape, filter_shape, dtype, seed)
    filter_sizes = tf.constant(filter_shape, dtype=tf.int32)

    cpu_result = self._cpu_conv2d_backprop_filter_reference(
        input_t=x,
        filter_sizes=filter_shape,
        out_backprop=out_backprop,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations)

    with tf.device('/device:MUSA:0'):
      musa_result = _conv2d_backprop_filter(
          input_t=x,
          filter_sizes=filter_sizes,
          out_backprop=out_backprop,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)

    self.assertAllClose(
        cpu_result.numpy(),
        tf.cast(musa_result, tf.float32).numpy(),
        rtol=rtol,
        atol=atol)

  def testConv2DBackpropFilterValidNHWC(self):
    """NHWC + VALID + stride=1."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_filter(
            input_shape=[1, 8, 8, 3],
            filter_shape=[3, 3, 3, 4],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="VALID",
            data_format="NHWC")

  def testConv2DBackpropFilterSameNHWC(self):
    """NHWC + SAME + stride=1."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_filter(
            input_shape=[2, 15, 15, 4],
            filter_shape=[5, 5, 4, 6],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC")

  def testConv2DBackpropFilterStride2ValidNHWC(self):
    """NHWC + VALID + stride=2."""
    for dtype in [tf.float32, tf.float16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_filter(
            input_shape=[1, 9, 9, 3],
            filter_shape=[3, 3, 3, 8],
            dtype=dtype,
            strides=[1, 2, 2, 1],
            padding="VALID",
            data_format="NHWC")

  def testConv2DBackpropFilterDilationValidNHWC(self):
    """NHWC + VALID + dilation=2."""
    self.skipTest("TF CPU backprop reference does not support dilation > 1")
    for dtype in [tf.float32, tf.float16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_filter(
            input_shape=[1, 12, 12, 3],
            filter_shape=[3, 3, 3, 5],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="VALID",
            data_format="NHWC",
            dilations=[1, 2, 2, 1])

  def testConv2DBackpropFilterPointwise1x1NHWC(self):
    """1x1 pointwise conv backprop filter."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_filter(
            input_shape=[4, 16, 16, 7],
            filter_shape=[1, 1, 7, 13],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC")

  def testConv2DBackpropFilterValidNCHW(self):
    """NCHW + VALID + stride=1."""
    for dtype in [tf.float32, tf.float16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_backprop_filter(
            input_shape=[1, 3, 8, 8],
            filter_shape=[3, 3, 3, 4],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="VALID",
            data_format="NCHW")


class Conv2DGradientTest(MUSATestCase):
  """Tests for Conv2D gradient via GradientTape."""

  def _tolerance_for_dtype(self, dtype):
    if dtype == tf.bfloat16:
      return 5e-2, 5e-2
    if dtype == tf.float16:
      return 1e-2, 1e-2
    return 1e-4, 1e-5

  def _test_conv2d_gradient(
      self, input_shape, filter_shape, dtype, strides, padding,
      data_format="NHWC", dilations=None, seed=2026):
    if dilations is None:
      dilations = [1, 1, 1, 1]

    rtol, atol = self._tolerance_for_dtype(dtype)

    np.random.seed(seed)
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    x_np = np.random.uniform(-1.0, 1.0, size=input_shape).astype(np_dtype)
    w_np = np.random.uniform(-1.0, 1.0, size=filter_shape).astype(np_dtype)

    x = tf.constant(x_np, dtype=dtype)
    w = tf.constant(w_np, dtype=dtype)

    def _to_nhwc_args(nchw_like):
      return [nchw_like[0], nchw_like[2], nchw_like[3], nchw_like[1]]

    # Compute forward pass and gradients on CPU (CPU Conv2D only supports NHWC).
    with tf.device('/CPU:0'):
      x_cpu = tf.cast(x, tf.float32)
      w_cpu = tf.cast(w, tf.float32)
      with tf.GradientTape() as tape:
        tape.watch([x_cpu, w_cpu])
        if data_format == "NHWC":
          y_cpu = tf.nn.conv2d(
              x_cpu, w_cpu, strides=strides, padding=padding,
              data_format="NHWC", dilations=dilations)
        else:
          x_cpu_nhwc = tf.transpose(x_cpu, [0, 2, 3, 1])
          y_cpu = tf.nn.conv2d(
              x_cpu_nhwc,
              w_cpu,
              strides=_to_nhwc_args(strides),
              padding=padding,
              data_format="NHWC",
              dilations=_to_nhwc_args(dilations))
        loss_cpu = tf.reduce_sum(y_cpu)

      dx_cpu, dw_cpu = tape.gradient(loss_cpu, [x_cpu, w_cpu])

    # Compute forward pass and gradients on MUSA
    with tf.device('/device:MUSA:0'):
      with tf.GradientTape() as tape:
        tape.watch([x, w])
        y_musa = tf.nn.conv2d(
            x, w, strides=strides, padding=padding,
            data_format=data_format, dilations=dilations)
        loss_musa = tf.reduce_sum(tf.cast(y_musa, tf.float32))
      dx_musa, dw_musa = tape.gradient(loss_musa, [x, w])

    # Compare input gradients
    self.assertAllClose(
        dx_cpu.numpy(),
        tf.cast(dx_musa, tf.float32).numpy(),
        rtol=rtol,
        atol=atol)

    # Compare filter gradients
    self.assertAllClose(
        dw_cpu.numpy(),
        tf.cast(dw_musa, tf.float32).numpy(),
        rtol=rtol,
        atol=atol)

  def testConv2DGradientNHWC(self):
    """Test gradient flow through Conv2D NHWC."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_gradient(
            input_shape=[2, 8, 8, 3],
            filter_shape=[3, 3, 3, 4],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC")

  def testConv2DGradientStride2NHWC(self):
    """Test gradient flow through Conv2D NHWC with stride=2."""
    for dtype in [tf.float32, tf.float16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_gradient(
            input_shape=[1, 9, 9, 3],
            filter_shape=[3, 3, 3, 8],
            dtype=dtype,
            strides=[1, 2, 2, 1],
            padding="VALID",
            data_format="NHWC")

  def testConv2DGradientNCHW(self):
    """Test gradient flow through Conv2D NCHW."""
    for dtype in [tf.float32, tf.float16]:
      with self.subTest(dtype=dtype.name):
        self._test_conv2d_gradient(
            input_shape=[2, 3, 8, 8],
            filter_shape=[3, 3, 3, 4],
            dtype=dtype,
            strides=[1, 1, 1, 1],
            padding="VALID",
            data_format="NCHW")


if __name__ == "__main__":
  tf.test.main()
