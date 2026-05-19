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
"""Default import path should load MUSA as a PluggableDevice."""

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
  for path in candidates:
    if os.path.isfile(path):
      return path
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


class PluggableDefaultImportTest(tf.test.TestCase):

  def test_import_tensorflow_musa_uses_default_pluggable_path(self):
    if not _plugin_path():
      self.skipTest("libmusa_plugin.so not found")
    script = r"""
import tensorflow as tf
import tensorflow_musa
musa_devs = tf.config.list_physical_devices('MUSA')
if not musa_devs:
  print('SKIP_NO_MUSA_DEVICE')
  raise SystemExit(0)
print('OK', len(musa_devs))
"""
    proc = _run_with_repo_package(script)
    out = (proc.stdout or "") + (proc.stderr or "")
    if "SKIP_NO_MUSA_DEVICE" in out:
      self.skipTest("TensorFlow did not list MUSA physical devices")
    self.assertEqual(proc.returncode, 0, out)
    self.assertIn("OK", proc.stdout or "")

  def test_repeated_import_default_path(self):
    if not _plugin_path():
      self.skipTest("libmusa_plugin.so not found")
    script = r"""
import importlib
import tensorflow as tf
import tensorflow_musa
first = tensorflow_musa.load_plugin()
second = tensorflow_musa.load_plugin()
importlib.reload(tensorflow_musa)
third = tensorflow_musa.load_plugin()
print('OK', bool(first), first == second == third,
      len(tf.config.list_physical_devices('MUSA')))
"""
    proc = _run_with_repo_package(script)
    out = (proc.stdout or "") + (proc.stderr or "")
    self.assertEqual(proc.returncode, 0, out)
    self.assertIn("OK", proc.stdout or "")


if __name__ == "__main__":
  tf.test.main()
