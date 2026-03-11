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

"""Tests for MUSA MaxPool operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import load_musa_plugin

# Load plugin before test discovery/runtime checks.
load_musa_plugin()
MUSA_DEVICES = tf.config.list_physical_devices('MUSA')


class MaxPoolOpTest(tf.test.TestCase):
  """Tests for MUSA MaxPool operator."""

  def _test_maxpool(self, input_data, ksize, strides, padding, dtype,
                    data_format='NHWC', rtol=1e-5, atol=1e-8):
    """Test MaxPool operation with given parameters."""
    if not MUSA_DEVICES:
      self.skipTest("No MUSA devices found.")

    if dtype == tf.bfloat16:
      input_np = np.array(input_data, dtype=np.float32)
    else:
      input_np = np.array(input_data, dtype=dtype.as_numpy_dtype)

    x = tf.constant(input_np, dtype=dtype)

    # Test on CPU
    with tf.device('/CPU:0'):
      cpu_result = tf.nn.max_pool2d(
          x, ksize=ksize, strides=strides, padding=padding, data_format=data_format)

    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      musa_result = tf.nn.max_pool2d(
          x, ksize=ksize, strides=strides, padding=padding, data_format=data_format)

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

  def testMaxPoolBasicSame(self):
    """Basic MaxPool test with SAME padding."""
    input_data = [[
        [[1.0], [2.0], [3.0], [4.0]],
        [[5.0], [6.0], [7.0], [8.0]],
        [[9.0], [10.0], [11.0], [12.0]],
        [[13.0], [14.0], [15.0], [16.0]]
    ]]
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'SAME'

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_maxpool(input_data, ksize, strides, padding, dtype,
                         rtol=rtol, atol=atol)

  def testMaxPoolBasicValid(self):
    """Basic MaxPool test with VALID padding."""
    input_data = [[
        [[1.0], [2.0], [3.0], [4.0]],
        [[5.0], [6.0], [7.0], [8.0]],
        [[9.0], [10.0], [11.0], [12.0]],
        [[13.0], [14.0], [15.0], [16.0]]
    ]]
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_maxpool(input_data, ksize, strides, padding, dtype,
                         rtol=rtol, atol=atol)

  def testMaxPoolMultipleChannels(self):
    """MaxPool test with multiple channels."""
    input_data = [[
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ]]
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = 'VALID'

    for dtype in [tf.float32, tf.float16]:
      rtol = 1e-2 if dtype == tf.float16 else 1e-5
      atol = 1e-2 if dtype == tf.float16 else 1e-8
      self._test_maxpool(input_data, ksize, strides, padding, dtype,
                         rtol=rtol, atol=atol)

  def testMaxPoolNCHW(self):
    """MaxPool test with NCHW data format."""
    self.skipTest("CPU MaxPool2D NCHW is unsupported in TF2.6 test baseline.")

  def testMaxPoolDifferentStrides(self):
    """MaxPool test with different stride values."""
    input_data = [[[ [float(i * 6 + j + 1) for j in range(6)] ] for i in range(6)]]
    ksize = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'

    for dtype in [tf.float32, tf.float16]:
      rtol = 1e-2 if dtype == tf.float16 else 1e-5
      atol = 1e-2 if dtype == tf.float16 else 1e-8
      self._test_maxpool(input_data, ksize, strides, padding, dtype,
                         rtol=rtol, atol=atol)

  def testMaxPoolLargeWindow(self):
    """MaxPool test with large pooling window."""
    input_data = [[
        [[1.0], [2.0], [3.0], [4.0]],
        [[5.0], [6.0], [7.0], [8.0]],
        [[9.0], [10.0], [11.0], [12.0]],
        [[13.0], [14.0], [15.0], [16.0]]
    ]]
    ksize = [1, 4, 4, 1]
    strides = [1, 1, 1, 1]
    padding = 'VALID'

    for dtype in [tf.float32]:
      self._test_maxpool(input_data, ksize, strides, padding, dtype)

  def testMaxPoolNegativeValues(self):
    """MaxPool test with negative values."""
    input_data = [[
        [[-1.0], [-2.0]],
        [[-3.0], [-4.0]]
    ]]
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = 'VALID'

    for dtype in [tf.float32]:
      self._test_maxpool(input_data, ksize, strides, padding, dtype)

  def testMaxPoolMixedValues(self):
    """MaxPool test with mixed positive and negative values."""
    input_data = [[
        [[-1.0], [2.0], [-3.0]],
        [[4.0], [-5.0], [6.0]],
        [[-7.0], [8.0], [-9.0]]
    ]]
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = 'VALID'

    for dtype in [tf.float32]:
      self._test_maxpool(input_data, ksize, strides, padding, dtype)

  def testMaxPoolBatch(self):
    """MaxPool test with multiple batches."""
    input_data = [
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]]
        ],
        [
            [[5.0], [6.0]],
            [[7.0], [8.0]]
        ]
    ]
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = 'VALID'

    for dtype in [tf.float32]:
      self._test_maxpool(input_data, ksize, strides, padding, dtype)


if __name__ == "__main__":
  tf.test.main()
