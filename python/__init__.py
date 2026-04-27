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

"""TensorFlow MUSA plugin package.

This package provides:
- Automatic plugin loading on import
- Device discovery utilities for available MUSA devices

Example usage:
    import tensorflow_musa as tf_musa

    # Plugin is automatically loaded on import
    devices = tf_musa.get_musa_devices()
"""

import logging

from ._graph_optimizer import (
    MUSA_GRAPH_OPTIMIZER_NAME,
    disable_musa_graph_optimizer,
    enable_musa_graph_optimizer,
    is_musa_graph_optimizer_enabled,
    set_musa_graph_optimizer_enabled,
)
from ._loader import get_musa_devices, get_musa_ops, is_plugin_loaded, load_plugin

# Package version
__version__ = "0.1.0"

# Load plugin automatically on import
_plugin_loaded = False

try:
    load_plugin()
    _plugin_loaded = True
except Exception as e:
    logging.warning(f"Failed to load MUSA plugin: {e}")
    logging.warning(
        "MUSA functionality will not be available. "
        "Please ensure the plugin is built and MUSA SDK is installed."
    )

# Public API
__all__ = [
    "__version__",
    "load_plugin",
    "get_musa_ops",
    "is_plugin_loaded",
    "get_musa_devices",
    "MUSA_GRAPH_OPTIMIZER_NAME",
    "set_musa_graph_optimizer_enabled",
    "enable_musa_graph_optimizer",
    "disable_musa_graph_optimizer",
    "is_musa_graph_optimizer_enabled",
]
