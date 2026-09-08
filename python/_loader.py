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

"""MUSA plugin loading utilities."""

import logging
import os
import sys

from ._pjrt_bootstrap import (
    fallback_to_native,
    get_pjrt_bootstrap_status,
    is_pjrt_requested,
    load_pjrt_plugin,
    prepare_pjrt_bootstrap,
)

logger = logging.getLogger(__name__)

# Plugin library name
PLUGIN_LIBRARY = "libmusa_plugin.so"
_plugin_path = None
_op_module = None
_device_library_loaded = False
_native_plugin_load_attempted = False


def _find_plugin_library():
    """Find the MUSA plugin shared library.

    Search order:
    1. Package installation directory (next to __init__.py)
    2. Project build directory (for development)
    3. System paths (LD_LIBRARY_PATH, /usr/local/musa/lib)
    """
    # Get the directory where this package is installed
    package_dir = os.path.dirname(os.path.abspath(__file__))

    # Candidate paths to search
    candidate_paths = [
        # Package installation directory
        os.path.join(package_dir, PLUGIN_LIBRARY),
        # Build directory relative to package (development mode)
        os.path.join(package_dir, "..", "build", PLUGIN_LIBRARY),
        # Build directory relative to project root (when running from project)
        os.path.join(os.getcwd(), "build", PLUGIN_LIBRARY),
        # System MUSA library path
        os.path.join("/usr/local/musa", "lib", PLUGIN_LIBRARY),
        os.path.join("/usr/local/musa", "lib64", PLUGIN_LIBRARY),
    ]

    # Also check LD_LIBRARY_PATH
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    for path in ld_library_path.split(os.pathsep):
        if path:
            candidate_paths.append(os.path.join(path, PLUGIN_LIBRARY))

    # Search for the library
    for path in candidate_paths:
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            return normalized_path

    # If not found, raise an error with helpful message
    searched = "\n".join(f"  - {os.path.normpath(p)}" for p in candidate_paths)
    raise FileNotFoundError(
        f"MUSA plugin library '{PLUGIN_LIBRARY}' not found.\n"
        f"Searched locations:\n{searched}\n"
        f"Please ensure the plugin has been built (run './build.sh' or 'pip install')."
    )


def _use_legacy_loader():
    return os.environ.get("TENSORFLOW_MUSA_USE_LEGACY_DEVICE") == "1"


def _load_pluggable_device_library(plugin_path):
    try:
        from tensorflow.python.framework.load_library import (
            load_pluggable_device_library,
        )
    except ImportError as e:
        raise RuntimeError(
            "TensorFlow does not expose load_pluggable_device_library; "
            "PluggableDevice loading requires a TensorFlow build with the "
            "StreamExecutor C API loader. To use the transitional legacy path, "
            "set TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1 before importing "
            "tensorflow_musa."
        ) from e

    load_pluggable_device_library(plugin_path)


def load_plugin():
    """Load the MUSA plugin library into TensorFlow.

    This must be called before using any MUSA-specific operations.
    The plugin registers MUSA device and kernels with TensorFlow.

    Returns:
        str: Path to the loaded plugin library

    Raises:
        FileNotFoundError: If the plugin library cannot be found
        RuntimeError: If TensorFlow cannot load the plugin
    """
    global _plugin_path, _op_module, _device_library_loaded
    global _native_plugin_load_attempted

    if _plugin_path is not None:
        return _plugin_path

    # PJRT preparation occurs before importing TensorFlow so the provider can
    # publish TensorFlow's plug-in discovery variables. A failed optional
    # attempt is rolled back and falls through to the native path.
    pjrt_requested = False
    if is_pjrt_requested():
        if "tensorflow" in sys.modules:
            raise RuntimeError(
                "PJRT requires importing tensorflow_musa before tensorflow; "
                "start a new process and move 'import tensorflow_musa' ahead "
                "of 'import tensorflow'."
            )
        try:
            pjrt_requested = prepare_pjrt_bootstrap()
        except Exception as e:
            fallback_to_native(e)
            logger.warning(
                "OpenXLA-MUSA PJRT could not be initialized; falling back to "
                "native MUSA: %s",
                e,
            )

    import tensorflow as tf

    if pjrt_requested:
        # Never resolve or load libmusa_plugin.so in PJRT mode. Native and PJRT
        # kernels overlap on the MUSA device type and must not coexist.
        try:
            load_pjrt_plugin(tf)
            pjrt_status = get_pjrt_bootstrap_status()
            plugin_path = pjrt_status["plugin_path"]
            if not plugin_path:
                raise RuntimeError("PJRT provider completed without a plugin path")
            _op_module = None
            _plugin_path = plugin_path
            logger.info("MUSA PJRT provider ready: %s", plugin_path)
            return plugin_path
        except Exception as e:
            raise RuntimeError(
                "OpenXLA-MUSA PJRT failed after registration began. Native "
                "fallback is unsafe in this process because PJRT may already "
                "have registered MUSA kernels; start a new process without "
                f"--xla. Original error: {e}"
            ) from e

    # Native default path: preserve the existing TF MUSA loading sequence.
    plugin_path = _find_plugin_library()
    try:
        _native_plugin_load_attempted = True
        _op_module = tf.load_op_library(plugin_path)
        if not _use_legacy_loader() and not _device_library_loaded:
            _load_pluggable_device_library(plugin_path)
            _device_library_loaded = True
        _plugin_path = plugin_path
        logger.info(f"MUSA plugin loaded successfully from: {plugin_path}")
        return plugin_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MUSA plugin from {plugin_path}: {e}\n"
            f"Please ensure TensorFlow and MUSA SDK are properly installed."
        ) from e


def get_musa_ops():
    """Get the generated Python wrappers for MUSA custom ops.

    Importing tensorflow_musa loads and registers the plugin. Tests that need
    to call custom ops directly can use this helper instead of resolving and
    loading the shared library themselves.
    """
    if _op_module is None:
        load_plugin()
    return _op_module


def get_pjrt_status():
    """Return the current optional PJRT bootstrap status."""
    return get_pjrt_bootstrap_status()


def is_plugin_loaded():
    """Check if the MUSA plugin has been loaded.

    Returns:
        bool: True if plugin is loaded, False otherwise
    """
    import tensorflow as tf

    try:
        return bool(tf.config.list_physical_devices("MUSA"))
    except Exception:
        return False


def get_musa_devices():
    """Get list of available MUSA devices.

    Returns:
        list: List of MUSA physical devices
    """
    import tensorflow as tf

    try:
        return tf.config.list_physical_devices("MUSA")
    except Exception:
        return []
