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
# ============================================================================

"""Regression tests for the TensorFlow MUSA gradient import contract.

``tensorflow_musa.gradients`` imports TensorFlow in its module body.  PJRT
therefore requires that module to be deferred until the provider has prepared
and registered the PJRT runtime.  These tests observe import order in a fresh
process and check the registration table; they intentionally do not execute
PJRT GELU/LayerNorm backward kernels.
"""

import json
import os
import subprocess
import sys
import unittest


_PJRT_ENV_KEYS = (
    "TENSORFLOW_MUSA_ENABLE_PJRT",
    "TENSORFLOW_MUSA_PJRT_PROVIDER_MODULE",
    "TENSORFLOW_MUSA_PJRT_LIBRARY_PATH",
    "TENSORFLOW_MUSA_PJRT_BRIDGE_PATH",
    "TF_PLUGGABLE_DEVICE_LIBRARY_PATH",
    "PJRT_NAMES_AND_LIBRARY_PATHS",
    "NEXT_PLUGGABLE_DEVICE_USE_C_API",
    "MUSA_TFNPD_IS_PLUGGABLE_DEVICE",
    "MUSA_TFNPD_COMPILATION_DEVICE_NAME",
    "MTCC20X_TOOLS_DIR",
    "TF_XLA_FLAGS",
)


_CHILD_SCRIPT = r'''
import importlib.abc
import json
import sys

if "tensorflow" in sys.modules or "tensorflow_musa.gradients" in sys.modules:
    raise SystemExit("TensorFlow modules were preloaded before the import probe")

events = []


class ImportProbe(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("tensorflow", "tensorflow_musa.gradients"):
            events.append(fullname)
        return None


sys.meta_path.insert(0, ImportProbe())
import tensorflow_musa

status = tensorflow_musa.get_pjrt_status()
from tensorflow.python.framework import ops as tf_ops

registry = getattr(tf_ops, "_gradient_registry", None)
registered = {}
for op_name in ("MusaGelu", "MusaLayerNorm"):
    try:
        registered[op_name] = callable(registry.lookup(op_name))
    except (AttributeError, LookupError):
        registered[op_name] = False

print("IMPORT_CONTRACT=" + json.dumps({
    "status": status,
    "events": events,
    "registered": registered,
}))
'''


def _run_child(xla: bool) -> dict:
    env = os.environ.copy()
    for key in _PJRT_ENV_KEYS:
        env.pop(key, None)
    command = [sys.executable, "-c", _CHILD_SCRIPT]
    if xla:
        command.append("--xla")
    process = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = process.stdout + process.stderr
    if process.returncode != 0:
        raise AssertionError(
            f"gradient import probe failed ({process.returncode}):\n{output}"
        )
    marker = next(
        (line for line in process.stdout.splitlines() if line.startswith("IMPORT_CONTRACT=")),
        None,
    )
    if marker is None:
        raise AssertionError(f"gradient import probe produced no result:\n{output}")
    return json.loads(marker.split("=", 1)[1])


class GradientImportContractTest(unittest.TestCase):
    def test_pjrt_defers_gradients_until_after_tensorflow_import(self):
        result = _run_child(xla=True)
        self.assertEqual(result["status"]["mode"], "pjrt", result)
        self.assertEqual(result["status"]["phase"], "ready", result)
        self.assertLess(
            result["events"].index("tensorflow"),
            result["events"].index("tensorflow_musa.gradients"),
            result,
        )
        self.assertEqual(result["registered"], {
            "MusaGelu": True,
            "MusaLayerNorm": True,
        }, result)

    def test_native_keeps_gradients_before_tensorflow(self):
        result = _run_child(xla=False)
        self.assertEqual(result["status"]["mode"], "native", result)
        self.assertLess(
            result["events"].index("tensorflow_musa.gradients"),
            result["events"].index("tensorflow"),
            result,
        )
        self.assertEqual(result["registered"], {
            "MusaGelu": True,
            "MusaLayerNorm": True,
        }, result)


if __name__ == "__main__":
    sys.argv[1:] = [arg for arg in sys.argv[1:] if arg not in ("--xla", "--no-xla")]
    unittest.main()
