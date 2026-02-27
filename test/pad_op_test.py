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

"""Tests for the MUSA Pad operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class PadOpTest(MUSATestCase):
  """Tests for the MUSA Pad operator."""

  def _test_pad(self,
                shape,
                paddings,
                dtype,
                paddings_dtype=tf.int32,
                constant_values=None,
                rtol=1e-5,
                atol=1e-8):
    """Helper that compares CPU and MUSA pad results."""
    if dtype in (tf.int32, tf.int64):
      np_dtype = dtype.as_numpy_dtype
      x_np = np.random.randint(-5, 6, size=shape).astype(np_dtype)
    else:
      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
      x_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)

    x = tf.constant(x_np, dtype=dtype)
    paddings_tensor = tf.constant(paddings, dtype=paddings_dtype)

    def pad_fn(input_tensor, paddings_tensor):
      kwargs = {}
      if constant_values is not None:
        kwargs["constant_values"] = constant_values
      return tf.pad(input_tensor, paddings_tensor, **kwargs)

    self._compare_cpu_musa_results(pad_fn, [x, paddings_tensor], dtype,
                                   rtol=rtol, atol=atol)

  def testPadConstantDefault(self):
    """Pad with default (zero) constant value across dtypes."""
    paddings = [[1, 3], [0, 2]]
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      if dtype in (tf.int32, tf.int64):
        atol = 0
        rtol = 0
      self._test_pad([8, 10], paddings, dtype, rtol=rtol, atol=atol)

  def testPadConstantValueScalar(self):
    """Pad while supplying a non-zero constant value."""
    self._test_pad([4, 4], [[2, 1], [1, 2]], tf.float32,
                   paddings_dtype=tf.int64,
                   constant_values=3.5)

  def testPadNoOp(self):
    """Pad with zero paddings to trigger the forward-input fast path."""
    self._test_pad([12, 6], [[0, 0], [0, 0]], tf.float32)

  def testPadThreeDimensional(self):
    """Pad a 3D tensor and verify behavior across dimensions."""
    paddings = [[0, 0], [1, 1], [2, 0]]
    self._test_pad([2, 3, 4], paddings, tf.float32,
                   paddings_dtype=tf.int64,
                   constant_values=7.0)


if __name__ == "__main__":
  tf.test.main()
