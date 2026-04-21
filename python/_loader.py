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

import os
import logging

logger = logging.getLogger(__name__)

# Plugin library name
PLUGIN_LIBRARY = "libmusa_plugin.so"


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
    import tensorflow as tf

    plugin_path = _find_plugin_library()

    try:
        tf.load_op_library(plugin_path)
        logger.info(f"MUSA plugin loaded successfully from: {plugin_path}")
        return plugin_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MUSA plugin from {plugin_path}: {e}\n"
            f"Please ensure TensorFlow and MUSA SDK are properly installed."
        )


def is_plugin_loaded():
    """Check if the MUSA plugin has been loaded.

    Returns:
        bool: True if plugin is loaded, False otherwise
    """
    import tensorflow as tf

    # Check for MUSA device availability
    try:
        devices = tf.config.list_physical_devices()
        for device in devices:
            if "MUSA" in device:
                return True
    except Exception:
        pass

    return False


def get_musa_devices():
    """Get list of available MUSA devices.

    Returns:
        list: List of MUSA device names
    """
    import tensorflow as tf

    musa_devices = []
    try:
        devices = tf.config.list_physical_devices()
        for device in devices:
            if "MUSA" in device:
                musa_devices.append(device)
    except Exception:
        pass

    return musa_devices
