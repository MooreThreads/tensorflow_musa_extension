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
"""Installed-package PluggableDevice smoke tests for workspace-backed MUSA ops."""

import os
import subprocess
import sys

import tensorflow as tf


class PluggableWorkspaceOpsTest(tf.test.TestCase):

  def test_workspace_ops_with_installed_tensorflow_musa_package(self):
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
  where_2d = tf.where(tf.constant([[True, False, True],
                                   [False, True, False]], dtype=tf.bool))
  np.testing.assert_array_equal(where_2d.numpy(),
                                np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64))

  where_empty = tf.where(tf.constant([[False, False], [False, False]], dtype=tf.bool))
  np.testing.assert_array_equal(where_empty.numpy(), np.empty((0, 2), dtype=np.int64))

  sparse = tf.SparseTensor(
      indices=tf.constant([[0, 0], [0, 3], [1, 1], [2, 2]], dtype=tf.int64),
      values=tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32),
      dense_shape=tf.constant([4, 5], dtype=tf.int64))
  sparse_slice = tf.sparse.slice(
      sparse,
      start=tf.constant([0, 1], dtype=tf.int64),
      size=tf.constant([3, 3], dtype=tf.int64))
  np.testing.assert_array_equal(
      sparse_slice.indices.numpy(), np.array([[0, 2], [1, 0], [2, 1]], dtype=np.int64))
  np.testing.assert_allclose(sparse_slice.values.numpy(), [2.0, 3.0, 4.0], rtol=1e-5)
  np.testing.assert_array_equal(sparse_slice.dense_shape.numpy(), np.array([3, 3], dtype=np.int64))

  unique_values, unique_indices = tf.unique(
      tf.constant([3, 1, 3, 2, 1], dtype=tf.int32))
  np.testing.assert_array_equal(unique_values.numpy(), np.array([3, 1, 2], dtype=np.int32))
  np.testing.assert_array_equal(unique_indices.numpy(), np.array([0, 1, 0, 2, 1], dtype=np.int32))

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
