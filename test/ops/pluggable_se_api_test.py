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
"""StreamExecutor C API tests that do NOT import musa_test_utils (no early plugin load)."""

import ctypes
import os
import subprocess
import sys

import tensorflow as tf

_TF_UNIMPLEMENTED = 12


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


class PluggableSeApiTest(tf.test.TestCase):
  """SE_InitPlugin without pre-loading via tensorflow_musa (CPU-only)."""

  def _tf_framework_library(self):
    tf_lib = os.path.join(os.path.dirname(tf.__file__),
                          "libtensorflow_framework.so.2")
    if not os.path.isfile(tf_lib):
      self.skipTest("libtensorflow_framework.so.2 not found")
    return tf_lib

  def test_se_init_plugin_returns_unimplemented_when_pluggable_env_disabled(self):
    tf_fw = ctypes.CDLL(self._tf_framework_library())
    tf_fw.TF_NewStatus.argtypes = []
    tf_fw.TF_NewStatus.restype = ctypes.c_void_p
    tf_fw.TF_GetCode.argtypes = [ctypes.c_void_p]
    tf_fw.TF_GetCode.restype = ctypes.c_int
    tf_fw.TF_DeleteStatus.argtypes = [ctypes.c_void_p]
    tf_fw.TF_DeleteStatus.restype = None

    path = _plugin_path()
    if not path:
      self.skipTest("libmusa_plugin.so not found")
    prev = os.environ.pop("MUSA_ENABLE_SE_PLUGIN", None)
    try:
      lib = ctypes.cdll.LoadLibrary(path)
      se_init = lib.SE_InitPlugin
      se_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
      se_init.restype = None
      params = (ctypes.c_ubyte * 256)()
      st = tf_fw.TF_NewStatus()
      try:
        se_init(ctypes.addressof(params), st)
        self.assertEqual(tf_fw.TF_GetCode(st), _TF_UNIMPLEMENTED)
      finally:
        tf_fw.TF_DeleteStatus(st)
    finally:
      if prev is not None:
        os.environ["MUSA_ENABLE_SE_PLUGIN"] = prev

  def test_se_init_plugin_succeeds_in_fresh_subprocess_with_env(self):
    """MUSA_ENABLE_SE_PLUGIN must be set before dlopen; use isolated process."""
    path = _plugin_path()
    if not path:
      self.skipTest("libmusa_plugin.so not found")
    script = r"""
import ctypes, os, sys
os.environ['MUSA_ENABLE_SE_PLUGIN'] = '1'
import tensorflow as tf
code_ok = 0
tf_lib = os.path.join(os.path.dirname(tf.__file__), 'libtensorflow_framework.so.2')
tf_fw = ctypes.CDLL(tf_lib)
tf_fw.TF_NewStatus.argtypes = []
tf_fw.TF_NewStatus.restype = ctypes.c_void_p
tf_fw.TF_GetCode.argtypes = [ctypes.c_void_p]
tf_fw.TF_GetCode.restype = ctypes.c_int
tf_fw.TF_DeleteStatus.argtypes = [ctypes.c_void_p]
lib = ctypes.cdll.LoadLibrary(sys.argv[1])
se_init = lib.SE_InitPlugin
se_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
se_init.restype = None
params = (ctypes.c_ubyte * 256)()
st = tf_fw.TF_NewStatus()
try:
  se_init(ctypes.addressof(params), st)
  c = tf_fw.TF_GetCode(st)
  if c != 0:
    print('SE_InitPlugin status code', c)
    sys.exit(1)
finally:
  tf_fw.TF_DeleteStatus(st)
sys.exit(0)
"""
    env = os.environ.copy()
    env["MUSA_ENABLE_SE_PLUGIN"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script, path],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    self.assertEqual(
        proc.returncode, 0,
        "subprocess failed: " + (proc.stdout or "") + (proc.stderr or ""))


if __name__ == "__main__":
  tf.test.main()
