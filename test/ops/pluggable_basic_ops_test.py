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
"""Installed-package PluggableDevice smoke tests for basic stateless MUSA ops."""

import os
import subprocess
import sys

import tensorflow as tf


class PluggableBasicOpsTest(tf.test.TestCase):

  def test_basic_ops_with_installed_tensorflow_musa_package(self):
    script = r"""
import os
import sys

os.environ.pop('TENSORFLOW_MUSA_USE_LEGACY_DEVICE', None)
os.environ.pop('MUSA_ENABLE_SE_PLUGIN', None)

import numpy as np
import tensorflow as tf
import tensorflow_musa as tf_musa

assert tf_musa is not None

musa_devs = [d for d in tf.config.list_physical_devices()
             if getattr(d, 'name', '').upper().find('MUSA') >= 0]
if not musa_devs:
  print('SKIP_NO_MUSA_DEVICE')
  sys.exit(0)

with tf.device('/device:MUSA:0'):
  cast_bool = tf.cast(tf.constant([True, False]), tf.float32)
  np.testing.assert_allclose(cast_bool.numpy(), [1.0, 0.0], rtol=1e-5)

  cast_int = tf.cast(tf.constant([1.2, -2.8], dtype=tf.float32), tf.int32)
  np.testing.assert_array_equal(cast_int.numpy(), np.array([1, -2], dtype=np.int32))

  identity_cast = tf.cast(tf.constant([3.0, 4.0], dtype=tf.float32), tf.float32)
  np.testing.assert_allclose(identity_cast.numpy(), [3.0, 4.0], rtol=1e-5)

  empty_cast = tf.cast(tf.constant([], dtype=tf.float32), tf.int32)
  np.testing.assert_array_equal(empty_cast.numpy(), np.array([], dtype=np.int32))

  mul_same = tf.multiply(tf.constant([1.0, 2.0], dtype=tf.float32),
                         tf.constant([3.0, 4.0], dtype=tf.float32))
  np.testing.assert_allclose(mul_same.numpy(), [3.0, 8.0], rtol=1e-5)

  mul_scalar = tf.multiply(tf.constant([1.0, 2.0, 3.0], dtype=tf.float32),
                           tf.constant(2.0, dtype=tf.float32))
  np.testing.assert_allclose(mul_scalar.numpy(), [2.0, 4.0, 6.0], rtol=1e-5)

  mul_tail = tf.multiply(tf.ones([2, 3], dtype=tf.float32),
                         tf.constant([1.0, 2.0, 3.0], dtype=tf.float32))
  np.testing.assert_allclose(mul_tail.numpy(), [[1.0, 2.0, 3.0],
                                                [1.0, 2.0, 3.0]], rtol=1e-5)

  mul_empty = tf.multiply(tf.constant([], dtype=tf.float32),
                          tf.constant([], dtype=tf.float32))
  np.testing.assert_array_equal(mul_empty.numpy(), np.array([], dtype=np.float32))

  sub_same = tf.subtract(tf.constant([5.0, 7.0], dtype=tf.float32),
                         tf.constant([2.0, 3.0], dtype=tf.float32))
  np.testing.assert_allclose(sub_same.numpy(), [3.0, 4.0], rtol=1e-5)

  sub_bcast = tf.subtract(tf.constant([[5.0, 7.0]], dtype=tf.float32),
                          tf.constant([2.0, 3.0], dtype=tf.float32))
  np.testing.assert_allclose(sub_bcast.numpy(), [[3.0, 4.0]], rtol=1e-5)

  zeros_float = tf.zeros_like(tf.constant([1.0, -1.0], dtype=tf.float32))
  np.testing.assert_allclose(zeros_float.numpy(), [0.0, 0.0], rtol=1e-5)

  zeros_int = tf.zeros_like(tf.constant([1, -1], dtype=tf.int32))
  np.testing.assert_array_equal(zeros_int.numpy(), np.array([0, 0], dtype=np.int32))

  zeros_bool = tf.zeros_like(tf.constant([True, False], dtype=tf.bool))
  np.testing.assert_array_equal(zeros_bool.numpy(), np.array([False, False]))

  zeros_empty = tf.zeros_like(tf.constant([], dtype=tf.float32))
  np.testing.assert_array_equal(zeros_empty.numpy(), np.array([], dtype=np.float32))

  identity = tf.identity(tf.constant([9.0, 10.0], dtype=tf.float32))
  np.testing.assert_allclose(identity.numpy(), [9.0, 10.0], rtol=1e-5)

  identity_n = tf.identity_n([
      tf.constant([1, 2], dtype=tf.int32),
      tf.constant([3.0, 4.0], dtype=tf.float32),
  ])
  np.testing.assert_array_equal(identity_n[0].numpy(), np.array([1, 2], dtype=np.int32))
  np.testing.assert_allclose(identity_n[1].numpy(), [3.0, 4.0], rtol=1e-5)

  reshaped = tf.reshape(tf.constant([1, 2, 3, 4], dtype=tf.int32), [2, 2])
  np.testing.assert_array_equal(reshaped.numpy(), np.array([[1, 2], [3, 4]], dtype=np.int32))

  shape = tf.shape(tf.ones([2, 3], dtype=tf.float32), out_type=tf.int32)
  np.testing.assert_array_equal(shape.numpy(), np.array([2, 3], dtype=np.int32))

  shape_int64 = tf.shape(tf.ones([2, 3], dtype=tf.float32), out_type=tf.int64)
  np.testing.assert_array_equal(shape_int64.numpy(), np.array([2, 3], dtype=np.int64))

  size = tf.size(tf.ones([2, 3], dtype=tf.float32), out_type=tf.int32)
  np.testing.assert_array_equal(size.numpy(), np.array(6, dtype=np.int32))

  expanded = tf.expand_dims(tf.constant([1.0, 2.0], dtype=tf.float32), 0)
  np.testing.assert_allclose(expanded.numpy(), [[1.0, 2.0]], rtol=1e-5)

  squeezed = tf.squeeze(tf.ones([1, 2, 1], dtype=tf.float32), axis=[0, 2])
  np.testing.assert_allclose(squeezed.numpy(), [1.0, 1.0], rtol=1e-5)

  stopped = tf.stop_gradient(tf.constant([5.0, 6.0], dtype=tf.float32))
  np.testing.assert_allclose(stopped.numpy(), [5.0, 6.0], rtol=1e-5)

print('OK')
"""
    env = os.environ.copy()
    env.pop("TENSORFLOW_MUSA_USE_LEGACY_DEVICE", None)
    env.pop("MUSA_ENABLE_SE_PLUGIN", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if "SKIP_NO_MUSA_DEVICE" in out:
      self.skipTest(
          "no MUSA device visible after importing installed tensorflow_musa")
    self.assertEqual(proc.returncode, 0, out)
    self.assertIn("OK", proc.stdout or "")


if __name__ == "__main__":
  tf.test.main()
