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
"""Installed-package PluggableDevice smoke tests for stream-launch MUSA ops."""

import os
import subprocess
import sys

import tensorflow as tf


class PluggableStreamOpsTest(tf.test.TestCase):

  def test_stream_ops_with_installed_tensorflow_musa_package(self):
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
  isnan = tf.math.is_nan(tf.constant([1.0, np.nan, -2.0], dtype=tf.float32))
  np.testing.assert_array_equal(isnan.numpy(), np.array([False, True, False]))

  softplus = tf.nn.softplus(tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32))
  np.testing.assert_allclose(softplus.numpy(),
                             np.log1p(np.exp(np.array([-1.0, 0.0, 1.0], dtype=np.float32))),
                             rtol=1e-4)

  top_values, top_indices = tf.math.top_k(
      tf.constant([[1.0, 5.0, 3.0], [2.0, 4.0, 0.0]], dtype=tf.float32), k=2)
  np.testing.assert_allclose(top_values.numpy(), [[5.0, 3.0], [4.0, 2.0]], rtol=1e-5)
  np.testing.assert_array_equal(top_indices.numpy(), np.array([[1, 2], [1, 0]], dtype=np.int32))

  uniform = tf.random.uniform([4], minval=0.0, maxval=1.0, dtype=tf.float32)
  uniform_np = uniform.numpy()
  assert uniform_np.shape == (4,)
  assert np.all(np.isfinite(uniform_np))
  assert np.all(uniform_np >= 0.0) and np.all(uniform_np < 1.0)

  uniform_int = tf.random.uniform([4], minval=1, maxval=7, dtype=tf.int32)
  uniform_int_np = uniform_int.numpy()
  assert uniform_int_np.shape == (4,)
  assert np.all(uniform_int_np >= 1) and np.all(uniform_int_np < 7)

  normal = tf.random.normal([4], dtype=tf.float32)
  normal_np = normal.numpy()
  assert normal_np.shape == (4,)
  assert np.all(np.isfinite(normal_np))

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
