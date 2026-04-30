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
"""Eager float Add on SE-only path (subprocess; env before first plugin load)."""

import ctypes
import os
import subprocess
import sys

import tensorflow as tf


def _plugin_path():
  here = os.path.dirname(os.path.abspath(__file__))
  candidates = [
      os.path.normpath(os.path.join(here, "..", "..", "build", "libmusa_plugin.so")),
      os.path.normpath(os.path.join(here, "..", "build", "libmusa_plugin.so")),
      os.path.normpath(os.path.join(os.getcwd(), "build", "libmusa_plugin.so")),
  ]
  for p in candidates:
    if os.path.isfile(p):
      return p
  return None


class PluggableSeEagerAddTest(tf.test.TestCase):
  """Subprocess loads Pluggable SE + kernels; validates Add on MUSA when GPU exists.

  **Hardware-gated**: driver count 0 or enum error ⇒ skip — does not validate
  the full SE-only op path in CPU-only CI. Use FAIL_SE mismatch to catch broken
  `load_pluggable_device_library` when devices exist but TF sees none.

  With **drv_n > 0**, ``UnimplementedError`` fails the test (migration regression).
  """

  def test_float_addv2_se_only_in_fresh_subprocess(self):
    path = _plugin_path()
    if not path:
      self.skipTest("libmusa_plugin.so not found")
    script = r"""
import ctypes, os, sys
os.environ['MUSA_ENABLE_SE_PLUGIN'] = '1'
import numpy as np
import tensorflow as tf

def _driver_device_count_so(so_path):
  tf_fw_path = os.path.join(os.path.dirname(tf.__file__), 'libtensorflow_framework.so.2')
  tf_fw = ctypes.CDLL(tf_fw_path)
  tf_fw.TF_NewStatus.argtypes = []
  tf_fw.TF_NewStatus.restype = ctypes.c_void_p
  tf_fw.TF_GetCode.argtypes = [ctypes.c_void_p]
  tf_fw.TF_GetCode.restype = ctypes.c_int
  tf_fw.TF_DeleteStatus.argtypes = [ctypes.c_void_p]
  lib = ctypes.cdll.LoadLibrary(so_path)
  fn = lib.MUSA_TestPluginGetDeviceCount
  fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p]
  fn.restype = None
  n = ctypes.c_int(-1)
  st = tf_fw.TF_NewStatus()
  try:
    fn(ctypes.byref(n), st)
    code = tf_fw.TF_GetCode(st)
  finally:
    tf_fw.TF_DeleteStatus(st)
  if code != 0:
    return -1, code
  return n.value, 0

drv_n, drv_code = _driver_device_count_so(sys.argv[1])

if drv_code != 0:
  print('SKIP_DRIVER_ENUM_ERROR')
  sys.exit(0)

from tensorflow.python.framework.load_library import load_pluggable_device_library
plugin = sys.argv[1]
load_pluggable_device_library(plugin)
lib = tf.load_op_library(plugin)
assert lib is not None

musa_devs = [d for d in tf.config.list_physical_devices()
             if getattr(d, 'name', '').upper().find('MUSA') >= 0]
if drv_n > 0 and not musa_devs:
  print('FAIL_SE_ENUM_MISMATCH')
  sys.exit(2)

if drv_n <= 0:
  print('SKIP_NO_MUSA_DEVICE')
  sys.exit(0)

# Hardware present (drv reports devices): require AddV2 — no masking Unimplemented.

with tf.device('/device:MUSA:0'):
  a = tf.constant([1.0, 2.0], dtype=tf.float32)
  b = tf.constant([3.0, 4.0], dtype=tf.float32)
  c = tf.raw_ops.AddV2(x=a, y=b)
  np.testing.assert_allclose(c.numpy(), [4.0, 6.0], rtol=1e-5)
print('OK')
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, path],
        capture_output=True,
        text=True,
        timeout=180,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if "SKIP_NO_MUSA_DEVICE" in out or "SKIP_DRIVER_ENUM_ERROR" in out:
      self.skipTest(
          "no MUSA device / driver enum issue "
          "(SE-only eager Add skipped in CPU-only environment)")
    if proc.returncode == 2 and "FAIL_SE_ENUM_MISMATCH" in out:
      self.fail(
          "driver reports devices but TensorFlow sees no MUSA physical device "
          "after load_pluggable_device_library (%s)" % out)
    self.assertEqual(proc.returncode, 0, out)
    self.assertIn("OK", proc.stdout or "")


if __name__ == "__main__":
  tf.test.main()
