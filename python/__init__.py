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

On import the package:

* Loads ``libmusa_plugin.so`` into TensorFlow as a PluggableDevice
  (see ``_loader.py``).
* Exposes the ``tensorflow_musa.memory`` and ``tensorflow_musa.device``
  submodules, giving users direct access to the caching allocator's
  stats, manual cache drain, per-process memory cap, and the driver's
  raw free/total view. Nothing in this file touches the native
  extension; that load happens lazily when ``memory.*`` or
  ``device.*`` are first used (see ``_ext.py``), so importing the
  package on a host without MUSA hardware still succeeds.

Example usage:

    import tensorflow_musa as tf_musa

    if tf_musa.is_available():
        print("bytes in use:", tf_musa.memory.memory_allocated())
        tf_musa.memory.set_per_process_memory_fraction(0.5)
"""

import logging

from . import device, memory
from ._loader import get_musa_devices, is_plugin_loaded, load_plugin
from .device import (
    current_device,
    device_count,
    get_device_name,
    is_available,
)

__version__ = "0.1.0"

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

__all__ = [
    "__version__",
    "load_plugin",
    "is_plugin_loaded",
    "get_musa_devices",
    "memory",
    "device",
    "device_count",
    "current_device",
    "get_device_name",
    "is_available",
]
