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

"""Unit tests for the thin TensorFlow MUSA PJRT provider adapter."""

import importlib.util
import os
import unittest
from unittest import mock


_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "python",
    "_pjrt_bootstrap.py",
)
_SPEC = importlib.util.spec_from_file_location(
    "tensorflow_musa_pjrt_bootstrap_test_module", _MODULE_PATH
)
_pjrt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_pjrt)


class _FakeProvider:
    __name__ = "fake_openxla_musa_pjrt"

    def __init__(self):
        self.prepare_calls = 0
        self.bootstrap_calls = 0
        self.status = {
            "mode": "pjrt",
            "phase": "idle",
            "provider": self.__name__,
            "plugin_path": "/provider/libmusa_pjrt_plugin_zy.so",
            "bridge_path": "/provider/libmusa_tf215_registry_bridge.so",
            "error": None,
        }

    def prepare(self, environ):
        self.prepare_calls += 1
        environ["PROVIDER_PREPARED"] = "1"
        self.status["phase"] = "import_tensorflow"

    def bootstrap(self, tf_module):
        self.bootstrap_calls += 1
        self.status["phase"] = "ready"

    def get_pjrt_status(self):
        return dict(self.status)


class PjrtBootstrapTest(unittest.TestCase):

    def setUp(self):
        _pjrt._provider = None
        _pjrt._status = _pjrt._new_status()

    def test_disabled_by_default_does_not_import_provider_or_mutate_environment(self):
        env = {"EXISTING": "value"}
        with mock.patch.object(_pjrt.importlib, "import_module") as importer:
            self.assertFalse(_pjrt.prepare_pjrt_bootstrap(env))
        importer.assert_not_called()
        self.assertEqual(env, {"EXISTING": "value"})
        self.assertEqual(_pjrt.get_pjrt_bootstrap_status()["mode"], "native")
        self.assertEqual(_pjrt.get_pjrt_bootstrap_status()["phase"], "idle")

    def test_invalid_switch_is_rejected_and_recorded(self):
        env = {_pjrt.PJRT_ENABLE_ENV: "sometimes"}
        with self.assertRaisesRegex(ValueError, _pjrt.PJRT_ENABLE_ENV):
            _pjrt.prepare_pjrt_bootstrap(env)

        status = _pjrt.get_pjrt_bootstrap_status()
        self.assertEqual(status["mode"], "pjrt")
        self.assertEqual(status["phase"], "failed")
        self.assertIn(_pjrt.PJRT_ENABLE_ENV, status["error"])

    def test_requested_mode_delegates_all_runtime_work_to_provider(self):
        provider = _FakeProvider()
        env = {
            _pjrt.PJRT_ENABLE_ENV: "true",
            _pjrt.PJRT_PROVIDER_MODULE_ENV: provider.__name__,
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(_pjrt.importlib, "import_module", return_value=provider) as importer:
                self.assertTrue(_pjrt.prepare_pjrt_bootstrap(env))
                _pjrt.load_pjrt_plugin(object())

        importer.assert_called_once_with(provider.__name__)
        self.assertEqual(provider.prepare_calls, 1)
        self.assertEqual(provider.bootstrap_calls, 1)
        self.assertEqual(env["PROVIDER_PREPARED"], "1")
        status = _pjrt.get_pjrt_bootstrap_status()
        self.assertEqual(status["mode"], "pjrt")
        self.assertEqual(status["phase"], "ready")
        self.assertEqual(status["provider"], provider.__name__)
        self.assertIsNone(status["error"])

    def test_missing_provider_is_fail_fast(self):
        env = {_pjrt.PJRT_ENABLE_ENV: "1"}
        missing = ModuleNotFoundError("not installed", name="openxla_musa_pjrt")
        with mock.patch.object(_pjrt.importlib, "import_module", side_effect=missing):
            with self.assertRaisesRegex(ModuleNotFoundError, "provider is not installed"):
                _pjrt.prepare_pjrt_bootstrap(env)
        self.assertEqual(_pjrt.get_pjrt_bootstrap_status()["phase"], "failed")


if __name__ == "__main__":
    unittest.main()
