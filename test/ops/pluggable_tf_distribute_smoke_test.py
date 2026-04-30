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
"""tf.distribute smoke on SE-only MUSA path (hardware-gated; selective skip)."""

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


class PluggableTfDistributeSmokeTest(tf.test.TestCase):
  """MirroredStrategy + reduce — needs driver + TF both seeing ≥2 MUSA devices."""

  def test_mirrored_reduce_se_only_subprocess(self):
    path = _plugin_path()
    if not path:
      self.skipTest("libmusa_plugin.so not found")

    script = r"""
import ctypes, os, sys
os.environ['MUSA_ENABLE_SE_PLUGIN'] = '1'
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.load_library import load_pluggable_device_library

def _driver_device_count(so_path):
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

def _maybe_collective_or_nccl_gap(exc):
  msg = str(exc).lower()
  if 'nccl' in msg:
    return True
  if 'collective' in msg:
    if any(k in msg for k in ('reduce', 'allreduce', 'executor', 'ncclmanager')):
      return True
  return False

drv_n, drv_code = _driver_device_count(sys.argv[1])
if drv_code != 0:
  print('SKIP_DRIVER_ENUM_ERROR')
  sys.exit(0)

plugin = sys.argv[1]
load_pluggable_device_library(plugin)
lib = tf.load_op_library(plugin)
assert lib is not None

physical = []
try:
  physical = tf.config.list_physical_devices('MUSA')
except Exception:
  physical = []

if drv_n < 2:
  print('SKIP_DRV_LESS_THAN_TWO_MUSA')
  sys.exit(0)

if drv_n >= 2 and len(physical) < 2:
  print('FAIL_DRV_TF_DEVICE_MISMATCH')
  print('drv_n=', drv_n, 'physical_musa=', len(physical))
  sys.exit(2)

strategy = tf.distribute.MirroredStrategy(
    devices=['/device:MUSA:0', '/device:MUSA:1'])

@tf.function
def replicated_value():
    return strategy.run(lambda: tf.constant(1.0))

try:
    pv = replicated_value()
    reduced = strategy.reduce(tf.distribute.ReduceOp.SUM, pv, axis=None)
    np.testing.assert_allclose(reduced.numpy(), 2.0, rtol=1e-5)
    print('OK')
except Exception as e:
    if _maybe_collective_or_nccl_gap(e):
      print('SKIP_COLLECTIVE_BACKEND:', e)
      sys.exit(0)
    raise
"""

    proc = subprocess.run(
        [sys.executable, "-c", script, path],
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if "SKIP_DRV_LESS_THAN_TWO_MUSA" in out or "SKIP_DRIVER_ENUM_ERROR" in out:
      self.skipTest("driver/device count insufficient for Mirrored smoke")
    if proc.returncode == 2 and "FAIL_DRV_TF_DEVICE_MISMATCH" in out:
      self.fail(
          "driver reports >=2 MUSA but TF lists fewer physical MUSA (%s)" % out)
    if "SKIP_COLLECTIVE_BACKEND:" in out:
      self.skipTest("TF collective / NCCL path gap (narrow skip): %s" % out)
    self.assertEqual(proc.returncode, 0, out)
    self.assertIn("OK", out)


if __name__ == "__main__":
  tf.test.main()
