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

import os
import shutil
import subprocess
import sys
import tempfile

import tensorflow as tf


def _repo_root():
  return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", ".."))


def _plugin_path():
  root = _repo_root()
  here = os.path.dirname(os.path.abspath(__file__))
  candidates = [
      os.path.join(root, "build", "libmusa_plugin.so"),
      os.path.normpath(os.path.join(here, "..", "build", "libmusa_plugin.so")),
  ]
  for p in candidates:
    if os.path.isfile(p):
      return p
  return None


def _run_with_repo_package(script, timeout=180):
  plugin_path = _plugin_path()
  if not plugin_path:
    return None
  root = _repo_root()
  with tempfile.TemporaryDirectory() as tmpdir:
    package_dir = os.path.join(tmpdir, "tensorflow_musa")
    shutil.copytree(os.path.join(root, "python"), package_dir, symlinks=True)
    shutil.copy2(plugin_path, os.path.join(package_dir, "libmusa_plugin.so"))
    env = os.environ.copy()
    env.pop("MUSA_ENABLE_SE_PLUGIN", None)
    env.pop("TENSORFLOW_MUSA_USE_LEGACY_DEVICE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [tmpdir, os.path.join(root, "test"), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class PluggableDeviceComplianceTest(tf.test.TestCase):
  """Runs in CI or dev when MUSA hardware + plugin are available."""

  def test_physical_devices_name_contains_musa(self):
    script = r"""
import tensorflow as tf
import tensorflow_musa
for d in tf.config.list_physical_devices():
  if "MUSA" in d.name or "musa" in d.name.lower():
    print("OK", d.name)
    raise SystemExit(0)
print("SKIP_NO_MUSA_DEVICE")
"""
    proc = _run_with_repo_package(script)
    if proc is None:
      self.skipTest("libmusa_plugin.so not found next to test/build")
    out = (proc.stdout or "") + (proc.stderr or "")
    if "SKIP_NO_MUSA_DEVICE" in out:
      self.skipTest("No MUSA devices found")
    self.assertEqual(proc.returncode, 0, out)
    self.assertIn("OK", proc.stdout or "")

  def test_se_init_plugin_symbol_exported(self):
    path = _plugin_path()
    if not path:
      self.skipTest("libmusa_plugin.so not found next to test/build")
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
    script = r"""
import tensorflow as tf
import tensorflow_musa
if not tf.config.list_physical_devices('MUSA'):
  print('SKIP_NO_MUSA_DEVICE')
  raise SystemExit(0)
with tf.device('/device:MUSA:0'):
  a = tf.constant([1.0, 2.0, 3.0])
  b = tf.constant([4.0, 5.0, 6.0])
  c = a + b
values = c.numpy().tolist()
print('OK', values)
if values != [5.0, 7.0, 9.0]:
  raise SystemExit(3)
"""
    proc = _run_with_repo_package(script)
    if proc is None:
      self.skipTest("libmusa_plugin.so not found next to test/build")
    out = (proc.stdout or "") + (proc.stderr or "")
    if "SKIP_NO_MUSA_DEVICE" in out:
      self.skipTest("No MUSA devices found")
    self.assertEqual(proc.returncode, 0, out)
    self.assertIn("OK", proc.stdout or "")


if __name__ == "__main__":
  tf.test.main()
