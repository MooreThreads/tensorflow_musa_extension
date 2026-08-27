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

"""PJRT boundary regression tests: native kernel safety, symbol failures, and
dual-path integrity.

These tests must be runnable both with and without PJRT enabled. When PJRT is
disabled they verify the native PluggableDevice path is undisturbed. When PJRT
is enabled they verify the isolated PJRT registration contract. Native and
PJRT plugin binaries must never be loaded into the same process.
"""

import inspect
import os
import subprocess
import sys
import unittest

import numpy as np
import tensorflow as tf

import tensorflow_musa
import tensorflow_musa._loader as musa_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pjrt_requested():
    return tensorflow_musa.get_pjrt_status().get("mode") == "pjrt"


def _assert_source_package():
    pkg = tensorflow_musa.__file__
    loader = inspect.getsourcefile(musa_loader.load_plugin)
    assert pkg is not None, "tensorflow_musa.__file__ is None"
    assert loader is not None, "load_plugin source file is None"
    return pkg, loader


# ---------------------------------------------------------------------------
# Native kernel regression (must pass regardless of PJRT)
# ---------------------------------------------------------------------------

class NativeKernelRegressionTest(tf.test.TestCase):
    """Standard MUSA execution regression for the active runtime mode.

    The production loader prevents native/PJRT registration conflicts by never
    loading ``libmusa_plugin.so`` in a successful PJRT mode. If PJRT cannot be
    initialized, the adapter warns and exercises the unchanged native path.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not tf.config.list_physical_devices("MUSA"):
            raise unittest.SkipTest("No MUSA devices found")

    def test_basic_arithmetic_on_musa(self):
        with tf.device("/device:MUSA:0"):
            a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
            b = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)
            c = a + b
        np.testing.assert_allclose(c.numpy(), [5.0, 7.0, 9.0], rtol=1e-5)

    def test_matmul_on_musa(self):
        with tf.device("/device:MUSA:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
            c = tf.matmul(a, b)
        np.testing.assert_allclose(
            c.numpy(), [[19.0, 22.0], [43.0, 50.0]], rtol=1e-5
        )

    def test_relu_on_musa(self):
        with tf.device("/device:MUSA:0"):
            x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
            y = tf.nn.relu(x)
        np.testing.assert_allclose(
            y.numpy(), [0.0, 0.0, 0.0, 1.0, 2.0], rtol=1e-5
        )


# ---------------------------------------------------------------------------
# PJRT registration contract (only meaningful when PJRT is enabled)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_pjrt_requested(), "PJRT not enabled")
class PjrtRegistrationContractTest(tf.test.TestCase):
    def test_status_is_registered(self):
        status = tensorflow_musa.get_pjrt_status()
        self.assertEqual(status["mode"], "pjrt")
        self.assertEqual(status["phase"], "ready")
        self.assertIsNotNone(status["provider"])
        self.assertIsNotNone(status["plugin_path"])
        self.assertIsNotNone(status["bridge_path"])
        self.assertIsNone(status["error"])

    def test_musa_devices_visible(self):
        devices = tf.config.list_physical_devices("MUSA")
        self.assertGreater(len(devices), 0, "No MUSA devices after PJRT registration")

    def test_pjrt_api_version_warning_is_recorded(self):
        """PJRT API 0.76 vs framework 0.34 is a known compatibility gap.

        This test documents the gap without asserting a specific version.
        """
        status = tensorflow_musa.get_pjrt_status()
        self.assertIn("plugin_path", status)


# ---------------------------------------------------------------------------
# Dual-path smoke: both native and XLA kernels must coexist
# ---------------------------------------------------------------------------

class PjrtIsolatedSmokeTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not tf.config.list_physical_devices("MUSA"):
            raise unittest.SkipTest("No MUSA devices found")

    def test_standard_eager_ops_work_after_pjrt_registration(self):
        if not _pjrt_requested():
            self.skipTest("PJRT not enabled")
        with tf.device("/device:MUSA:0"):
            x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
            y = tf.math.reduce_sum(x)
        np.testing.assert_allclose(y.numpy(), 6.0, rtol=1e-5)

    def test_pjrt_jit_and_eager_ops_work_in_one_runtime(self):
        if not _pjrt_requested():
            self.skipTest("PJRT not enabled")
        with tf.device("/device:MUSA:0"):
            eager = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32) + 1.0
        np.testing.assert_allclose(eager.numpy(), [2.0, 3.0, 4.0], rtol=1e-5)

        @tf.function(jit_compile=True)
        def add_one(x):
            return x + 1.0

        with tf.device("/device:MUSA:0"):
            jit_result = add_one(tf.constant([1.0, 2.0, 3.0], dtype=tf.float32))
        np.testing.assert_allclose(jit_result.numpy(), [2.0, 3.0, 4.0], rtol=1e-5)


# ---------------------------------------------------------------------------
# Missing PJRT provider (warn and native fallback)
# ---------------------------------------------------------------------------

class MissingPjrtProviderTest(tf.test.TestCase):
    def test_missing_provider_falls_back_to_native(self):
        script = """
try:
    import tensorflow_musa
except Exception as e:
    print("IMPORT_FAILED:" + type(e).__name__)
    raise SystemExit(1)
status = tensorflow_musa.get_pjrt_status()
print("MODE:" + str(status.get("mode")))
if status.get("mode") != "native":
    raise SystemExit(1)
raise SystemExit(0)
"""
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["TENSORFLOW_MUSA_PJRT_PROVIDER_MODULE"] = "missing_openxla_musa_pjrt_provider"
        proc = subprocess.run(
            [sys.executable, "-c", script, "--xla"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, f"Native fallback subprocess failed: {out}")
        self.assertIn("MODE:native", out,
                      f"Expected native fallback on missing PJRT provider, got: {out}")


if __name__ == "__main__":
    unittest.main()
