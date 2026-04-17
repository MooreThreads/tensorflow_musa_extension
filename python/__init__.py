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

"""
TensorFlow MUSA Extension - High-performance TensorFlow plugin for Moore Threads GPUs.

This package provides:
- Automatic plugin loading on import
- Optimized optimizer implementations (Adam, etc.) using fused MUSA kernels
- Device management utilities
- Monkey patching of tf.keras.optimizers.Adam for transparent MUSA acceleration

Example usage:
    import tensorflow_musa as tf_musa

    # Plugin is automatically loaded on import
    # tf.keras.optimizers.Adam is automatically patched to use MUSA kernels

    # Use MUSA-accelerated Adam optimizer (no changes needed!)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Or use explicit MUSA optimizer
    optimizer = tf_musa.optimizer.Adam(learning_rate=0.001)

    # Check available MUSA devices
    devices = tf_musa.get_musa_devices()
"""

import logging

from ._loader import load_plugin, is_plugin_loaded, get_musa_devices, get_musa_ops_module

# Package version
__version__ = "0.1.0"

# Load plugin automatically on import
_plugin_loaded = False
_plugin_path = None

try:
    _plugin_path = load_plugin()
    _plugin_loaded = True
except Exception as e:
    logging.warning(f"Failed to load MUSA plugin: {e}")
    logging.warning(
        "MUSA functionality will not be available. "
        "Please ensure the plugin is built and MUSA SDK is installed."
    )


# Import optimizer module after plugin is loaded
from . import optimizer

# Import patch utilities
from ._patch import patch_keras_adam, unpatch_keras_adam, is_adam_patched

# Auto-patch tf.keras.optimizers.Adam when MUSA devices are available
if _plugin_loaded and get_musa_devices():
    try:
        patch_keras_adam()
    except Exception as e:
        logging.warning(f"Failed to patch tf.keras.optimizers.Adam: {e}")

# Public API
__all__ = [
    "__version__",
    "load_plugin",
    "is_plugin_loaded",
    "get_musa_devices",
    "get_musa_ops_module",
    "optimizer",
    "patch_keras_adam",
    "unpatch_keras_adam",
    "is_adam_patched",
]
