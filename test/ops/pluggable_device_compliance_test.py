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
"""Minimal compliance checks for MUSA plugin load, device visibility, and dispatch.

SE_InitPlugin-only tests live in pluggable_se_api_test.py (no early plugin load).
"""

import ctypes
import os
import subprocess
import sys

import musa_test_utils
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


class PluggableDeviceComplianceTest(musa_test_utils.MUSATestCase):
  """Runs in CI or dev when MUSA hardware + plugin are available (see base)."""

  def test_physical_devices_name_contains_musa(self):
    for d in tf.config.list_physical_devices():
      name = d.name
      if "MUSA" in name or "musa" in name.lower():
        return
    self.fail("No physical device with MUSA in name")

  def test_se_init_plugin_symbol_exported(self):
    path = _plugin_path()
    if not path:
      self.skipTest("libmusa_plugin.so not found next to test/build")
    # Run in a fresh subprocess: loading the plugin in the current process may
    # re-trigger op registration and abort when TensorFlow has already loaded it.
    script = r"""
import ctypes, sys
lib = ctypes.cdll.LoadLibrary(sys.argv[1])
if not hasattr(lib, "SE_InitPlugin"):
  raise SystemExit(2)
print("OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, path],
        capture_output=True,
        text=True,
        timeout=60,
    )
    self.assertEqual(
        proc.returncode,
        0,
        "SE_InitPlugin not exported from %s (stdout=%s stderr=%s)"
        % (path, proc.stdout, proc.stderr),
    )

  def test_minimal_musa_eager_add(self):
    with tf.device("/device:MUSA:0"):
      a = tf.constant([1.0, 2.0, 3.0])
      b = tf.constant([4.0, 5.0, 6.0])
      c = a + b
    self.assertAllClose(c.numpy(), [5.0, 7.0, 9.0])


if __name__ == "__main__":
  tf.test.main()
