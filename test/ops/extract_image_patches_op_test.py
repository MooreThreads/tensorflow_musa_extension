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

"""Tests for MUSA ExtractImagePatches operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import load_musa_plugin


load_musa_plugin()
MUSA_DEVICES = tf.config.list_physical_devices('MUSA')


class ExtractImagePatchesOpTest(tf.test.TestCase):
  """Tests for MUSA ExtractImagePatches operator."""

  def _run_extract_image_patches(self, x, ksizes, strides, rates, padding,
                                 device):
    with tf.device(device):
      return tf.raw_ops.ExtractImagePatches(
          images=x,
          ksizes=ksizes,
          strides=strides,
          rates=rates,
          padding=padding)

  def _run_musa_extract_image_patches(self, x, ksizes, strides, rates, padding):
    old_soft_placement = tf.config.get_soft_device_placement()
    tf.config.set_soft_device_placement(False)
    try:
      return self._run_extract_image_patches(
          x, ksizes, strides, rates, padding, '/device:MUSA:0')
    finally:
      tf.config.set_soft_device_placement(old_soft_placement)

  def _test_extract_image_patches(self,
                                  shape,
                                  ksizes,
                                  strides,
                                  rates,
                                  padding,
                                  dtype,
                                  rtol=1e-5,
                                  atol=1e-5):
    if not MUSA_DEVICES:
      self.skipTest("No MUSA devices found.")

    if dtype in (tf.int32, tf.int64):
      values = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
      values = (values % 23 - 11).astype(dtype.as_numpy_dtype)
    else:
      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
      values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
      values = (values / 7.0 - 3.0).astype(np_dtype)
    x = tf.constant(values, dtype=dtype)

    cpu_result = self._run_extract_image_patches(
        x, ksizes, strides, rates, padding, '/CPU:0')
    musa_result = self._run_musa_extract_image_patches(
        x, ksizes, strides, rates, padding)

    if dtype in (tf.int32, tf.int64):
      self.assertAllEqual(cpu_result.numpy(), musa_result.numpy())
    elif dtype in (tf.float16, tf.bfloat16):
      self.assertAllClose(
          tf.cast(cpu_result, tf.float32).numpy(),
          tf.cast(musa_result, tf.float32).numpy(),
          rtol=rtol,
          atol=atol)
    else:
      self.assertAllClose(
          cpu_result.numpy(),
          musa_result.numpy(),
          rtol=rtol,
          atol=atol)

  def testFloat32ValidPadding(self):
    self._test_extract_image_patches(
        shape=[1, 4, 4, 1],
        ksizes=[1, 2, 2, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
        dtype=tf.float32)

  def testFloat32UnitPatchDirectCopy(self):
    self._test_extract_image_patches(
        shape=[2, 4, 5, 3],
        ksizes=[1, 1, 1, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
        dtype=tf.float32)

  def testFloat32UnitPatchStrided(self):
    self._test_extract_image_patches(
        shape=[1, 6, 7, 2],
        ksizes=[1, 1, 1, 1],
        strides=[1, 2, 3, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
        dtype=tf.float32)

  def testFloat32RowOnlySamePadding(self):
    self._test_extract_image_patches(
        shape=[1, 5, 4, 2],
        ksizes=[1, 3, 1, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME',
        dtype=tf.float32)

  def testFloat32ColumnOnlySamePadding(self):
    self._test_extract_image_patches(
        shape=[1, 4, 5, 2],
        ksizes=[1, 1, 3, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME',
        dtype=tf.float32)

  def testFloat32LargePatchDepthFallback(self):
    self._test_extract_image_patches(
        shape=[1, 4, 4, 33],
        ksizes=[1, 3, 3, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
        dtype=tf.float32)

  def testFloat64ValidPadding(self):
    self._test_extract_image_patches(
        shape=[1, 4, 4, 1],
        ksizes=[1, 2, 2, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
        dtype=tf.float64)

  def testFloat32SamePaddingMultiChannel(self):
    self._test_extract_image_patches(
        shape=[1, 3, 4, 2],
        ksizes=[1, 3, 2, 1],
        strides=[1, 2, 2, 1],
        rates=[1, 1, 1, 1],
        padding='SAME',
        dtype=tf.float32)

  def testFloat32Rates(self):
    self._test_extract_image_patches(
        shape=[1, 5, 5, 1],
        ksizes=[1, 2, 2, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 2, 2, 1],
        padding='VALID',
        dtype=tf.float32)

  def testFloat16Batch(self):
    self._test_extract_image_patches(
        shape=[2, 4, 5, 3],
        ksizes=[1, 2, 3, 1],
        strides=[1, 2, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
        dtype=tf.float16,
        rtol=1e-2,
        atol=1e-2)

  def testBFloat16SamePadding(self):
    self._test_extract_image_patches(
        shape=[1, 4, 4, 2],
        ksizes=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        rates=[1, 1, 1, 1],
        padding='SAME',
        dtype=tf.bfloat16,
        rtol=1e-1,
        atol=1e-1)

  def testInt32SamePaddingGradientIndexShape(self):
    self._test_extract_image_patches(
        shape=[1, 3, 4, 2],
        ksizes=[1, 3, 2, 1],
        strides=[1, 1, 2, 1],
        rates=[1, 1, 1, 1],
        padding='SAME',
        dtype=tf.int32)

  def testInt64Rates(self):
    self._test_extract_image_patches(
        shape=[1, 5, 5, 1],
        ksizes=[1, 2, 2, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 2, 2, 1],
        padding='VALID',
        dtype=tf.int64)


if __name__ == "__main__":
  tf.test.main()
