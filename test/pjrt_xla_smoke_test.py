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

"""Minimal TensorFlow 2.15 MUSA PJRT XLA end-to-end smoke test."""

import inspect
import os

import numpy as np
import tensorflow as tf

import tensorflow_musa
import tensorflow_musa._loader as musa_loader


EXPECTED = np.asarray([3.0, 5.0, 7.0, 9.0], dtype=np.float32)


def main():
    package_file = tensorflow_musa.__file__
    loader_file = inspect.getsourcefile(musa_loader.load_plugin)
    status = tensorflow_musa.get_pjrt_status()
    musa_devices = tf.config.list_physical_devices("MUSA")

    print("PACKAGE", package_file)
    print("LOADER", loader_file)
    print("STATUS", status)
    print("MUSA_DEVICES", musa_devices)
    print("MTCC20X_TOOLS_DIR", os.environ.get("MTCC20X_TOOLS_DIR"))
    print(
        "MUSA_TFNPD_IS_PLUGGABLE_DEVICE",
        os.environ.get("MUSA_TFNPD_IS_PLUGGABLE_DEVICE"),
    )
    print(
        "MUSA_TFNPD_COMPILATION_DEVICE_NAME",
        os.environ.get("MUSA_TFNPD_COMPILATION_DEVICE_NAME"),
    )
    print("TF_XLA_FLAGS", os.environ.get("TF_XLA_FLAGS"))

    assert package_file is not None
    assert loader_file is not None
    assert status["state"] == "registered", status
    assert status["registered"] is True, status
    assert musa_devices, "No MUSA devices were registered"
    assert os.environ.get("MTCC20X_TOOLS_DIR"), "MTCC20X_TOOLS_DIR was not auto-set"
    assert os.environ.get("MUSA_TFNPD_IS_PLUGGABLE_DEVICE") == "1"
    assert os.environ.get("MUSA_TFNPD_COMPILATION_DEVICE_NAME") == "XLA_GPU_JIT"
    assert "--tf_xla_use_device_api=true" in os.environ.get("TF_XLA_FLAGS", "")

    @tf.function(jit_compile=True)
    def compiled_add(x):
        return x + tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float32)

    with tf.device("/device:MUSA:0"):
        result = compiled_add(tf.constant([2.0, 4.0, 6.0, 8.0], dtype=tf.float32))

    value = result.numpy()
    print("RESULT", value)
    print("RESULT_DEVICE", result.device)
    print("CONCRETE_GRAPH", compiled_add.get_concrete_function(tf.TensorSpec([4], tf.float32)).graph)

    np.testing.assert_allclose(value, EXPECTED, rtol=1e-5, atol=1e-5)
    assert "MUSA" in result.device.upper(), result.device
    print("XLA_SMOKE=PASS")


if __name__ == "__main__":
    main()
