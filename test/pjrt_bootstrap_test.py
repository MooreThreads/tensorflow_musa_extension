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

"""Unit tests for stage-1 optional PJRT configuration and discovery."""

import importlib.util
import os
import tempfile
import unittest


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


class PjrtBootstrapTest(unittest.TestCase):

    def test_disabled_by_default_does_not_mutate_environment(self):
        env = {"EXISTING": "value"}
        status = _pjrt.prepare_pjrt_bootstrap(env)

        self.assertFalse(status.requested)
        self.assertEqual(status.state, "disabled")
        self.assertEqual(env, {"EXISTING": "value"})

    def test_invalid_switch_is_rejected_and_recorded(self):
        env = {_pjrt.PJRT_ENABLE_ENV: "sometimes"}
        with self.assertRaisesRegex(ValueError, _pjrt.PJRT_ENABLE_ENV):
            _pjrt.prepare_pjrt_bootstrap(env)

        status = _pjrt.get_pjrt_bootstrap_status()
        self.assertEqual(status["state"], "invalid_configuration")
        self.assertIn(_pjrt.PJRT_ENABLE_ENV, status["error"])

    def test_explicit_file_is_resolved_and_environment_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            library = os.path.join(directory, _pjrt.PJRT_LIBRARY)
            with open(library, "wb"):
                pass
            tools = os.path.join(directory, "tools")
            os.mkdir(tools)
            for tool in _pjrt._REQUIRED_MTGPU_TOOLS:
                path = os.path.join(tools, tool)
                with open(path, "wb"):
                    pass
                os.chmod(path, 0o755)
            env = {
                _pjrt.PJRT_ENABLE_ENV: "true",
                _pjrt.PJRT_LIBRARY_ENV: library,
                _pjrt.MTGPU_TOOLS_ENV: tools,
                "TF_PLUGGABLE_DEVICE_LIBRARY_PATH": "/user/plugin.so",
                "PJRT_NAMES_AND_LIBRARY_PATHS": "CPU:/user/cpu.so",
                "NEXT_PLUGGABLE_DEVICE_USE_C_API": "user-choice",
            }

            first = _pjrt.prepare_pjrt_bootstrap(env)
            second = _pjrt.prepare_pjrt_bootstrap(env)

            self.assertEqual(first.state, "prepared")
            self.assertEqual(second.library_path, library)
            self.assertEqual(env[_pjrt.MTGPU_TOOLS_ENV], tools)
            self.assertEqual(
                env["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"].split(","),
                ["/user/plugin.so", library],
            )
            self.assertEqual(
                env["PJRT_NAMES_AND_LIBRARY_PATHS"].split(","),
                ["CPU:/user/cpu.so", f"MUSA:{library}"],
            )
            self.assertEqual(
                env["NEXT_PLUGGABLE_DEVICE_USE_C_API"], "user-choice"
            )

    def test_explicit_directory_resolves_standard_library_name(self):
        with tempfile.TemporaryDirectory() as directory:
            library = os.path.join(directory, _pjrt.PJRT_LIBRARY)
            with open(library, "wb"):
                pass
            env = {
                _pjrt.PJRT_ENABLE_ENV: "1",
                _pjrt.PJRT_LIBRARY_ENV: directory,
            }

            self.assertEqual(_pjrt.find_pjrt_library(env), library)

    def test_requested_incomplete_toolchain_is_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            library = os.path.join(directory, _pjrt.PJRT_LIBRARY)
            with open(library, "wb"):
                pass
            env = {
                _pjrt.PJRT_ENABLE_ENV: "1",
                _pjrt.PJRT_LIBRARY_ENV: library,
                _pjrt.MTGPU_TOOLS_ENV: os.path.join(directory, "missing-tools"),
            }
            with self.assertRaisesRegex(FileNotFoundError, _pjrt.MTGPU_TOOLS_ENV):
                _pjrt.prepare_pjrt_bootstrap(env)

    def test_requested_missing_library_is_fail_fast(self):
        env = {
            _pjrt.PJRT_ENABLE_ENV: "1",
            _pjrt.PJRT_LIBRARY_ENV: "/does/not/exist/libpjrt.so",
            "LD_LIBRARY_PATH": "",
        }
        original_isfile = os.path.isfile
        try:
            os.path.isfile = lambda _path: False
            with self.assertRaisesRegex(FileNotFoundError, _pjrt.PJRT_LIBRARY_ENV):
                _pjrt.prepare_pjrt_bootstrap(env)
        finally:
            os.path.isfile = original_isfile

        status = _pjrt.get_pjrt_bootstrap_status()
        self.assertTrue(status["requested"])
        self.assertEqual(status["state"], "failed")


if __name__ == "__main__":
    unittest.main()
