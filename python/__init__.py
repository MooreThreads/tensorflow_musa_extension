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
- Runtime configuration helpers for MUSA memory growth

Example usage:
    import tensorflow_musa as tf_musa

    # Plugin is automatically loaded on import
    devices = tf_musa.get_musa_devices()
"""

import logging

from ._pjrt_bootstrap import is_pjrt_requested, find_pjrt_library, prepare_pjrt_environment
from ._graph_optimizer import (
    DISABLED_FUSION_PATTERNS_PARAM,
    MUSA_GRAPH_OPTIMIZER_NAME,
    clear_musa_disabled_fusion_patterns,
    clear_musa_graph_dump_config,
    disable_musa_graph_optimizer,
    disable_musa_fusion_patterns,
    disable_musa_graph_dump,
    enable_musa_graph_optimizer,
    enable_musa_graph_dump,
    get_musa_graph_dump_directory,
    get_musa_disabled_fusion_patterns,
    is_musa_graph_dump_enabled,
    is_musa_graph_dump_slim_enabled,
    is_musa_graph_dump_text_enabled,
    is_musa_graph_optimizer_enabled,
    set_musa_graph_dump_config,
    set_musa_disabled_fusion_patterns,
    set_musa_graph_optimizer_enabled,
)
from ._loader import (
    get_musa_devices,
    get_musa_ops,
    get_pjrt_status,
    is_plugin_loaded,
    load_plugin,
)
from ._optimizers import apply_adam_mixed, apply_sparse_adam_mixed
from ._runtime_config import (
    disable_musa_telemetry,
    enable_musa_telemetry,
    get_musa_telemetry_health,
    is_musa_telemetry_enabled,
    set_musa_allow_growth,
    set_musa_telemetry_config,
)
from . import gradients, ops, raw_ops

# Package version
__version__ = "1.5.0"


def enable_xla(library_path=None):
    """Enable OpenXLA-MUSA PJRT backend.

    Sets ``TENSORFLOW_MUSA_ENABLE_PJRT=1`` and
    ``TENSORFLOW_MUSA_PJRT_LIBRARY_PATH``.

    **Important**: PJRT mode requires ``LD_LIBRARY_PATH`` set at the
    **shell level** before the Python process starts.  Use::

        export LD_LIBRARY_PATH=/path/to/tensorflow:$LD_LIBRARY_PATH
        export TENSORFLOW_MUSA_ENABLE_PJRT=1
        export TENSORFLOW_MUSA_PJRT_LIBRARY_PATH=/path/to/libmusa_pjrt_plugin_zy.so

    Then ``import tensorflow_musa`` in a new Python process.

    Args:
        library_path: Optional path to ``libmusa_pjrt_plugin_zy.so``.
            If ``None``, auto-searches the package directory and
            ``openxla_musa/bazel-bin/pjrt_plugin/``.
    """
    import os
    if library_path is None:
        library_path = find_pjrt_library()
    os.environ["TENSORFLOW_MUSA_ENABLE_PJRT"] = "1"
    if library_path:
        os.environ["TENSORFLOW_MUSA_PJRT_LIBRARY_PATH"] = library_path
    print("PJRT mode enabled.  Start a new Python process to take effect.")
    print("  TENSORFLOW_MUSA_ENABLE_PJRT=1")
    print(f"  TENSORFLOW_MUSA_PJRT_LIBRARY_PATH={library_path}")
    return get_pjrt_status()


def disable_xla():
    """Disable PJRT and return to native MUSA kernel mode."""
    import os
    os.environ.pop("TENSORFLOW_MUSA_ENABLE_PJRT", None)
    os.environ.pop("TENSORFLOW_MUSA_PJRT_LIBRARY_PATH", None)


# Load plugin automatically on import
_plugin_loaded = False

try:
    load_plugin()
    _plugin_loaded = True
except Exception as e:
    if is_pjrt_requested():
        # PJRT is explicit opt-in. Do not silently downgrade to the native-only
        # runtime when the requested backend cannot be initialized.
        raise
    logging.warning(f"Failed to load MUSA plugin: {e}")
    logging.warning(
        "MUSA functionality will not be available. "
        "Please ensure the plugin is built and MUSA SDK is installed."
    )

# Public API
__all__ = [
    "__version__",
    "gradients",
    "ops",
    "raw_ops",
    "enable_xla",
    "disable_xla",
    "load_plugin",
    "get_musa_ops",
    "get_pjrt_status",
    "is_plugin_loaded",
    "get_musa_devices",
    "MUSA_GRAPH_OPTIMIZER_NAME",
    "DISABLED_FUSION_PATTERNS_PARAM",
    "set_musa_graph_optimizer_enabled",
    "enable_musa_graph_optimizer",
    "disable_musa_graph_optimizer",
    "is_musa_graph_optimizer_enabled",
    "set_musa_disabled_fusion_patterns",
    "disable_musa_fusion_patterns",
    "clear_musa_disabled_fusion_patterns",
    "get_musa_disabled_fusion_patterns",
    "set_musa_graph_dump_config",
    "enable_musa_graph_dump",
    "disable_musa_graph_dump",
    "clear_musa_graph_dump_config",
    "is_musa_graph_dump_enabled",
    "get_musa_graph_dump_directory",
    "is_musa_graph_dump_text_enabled",
    "is_musa_graph_dump_slim_enabled",
    "set_musa_allow_growth",
    "set_musa_telemetry_config",
    "enable_musa_telemetry",
    "disable_musa_telemetry",
    "is_musa_telemetry_enabled",
    "get_musa_telemetry_health",
    "apply_adam_mixed",
    "apply_sparse_adam_mixed",
    "MusaAdam",
]


def __getattr__(name):
    if name == "MusaAdam":
        from ._optimizers import _make_musa_adam_class

        cls = _make_musa_adam_class()
        globals()["MusaAdam"] = cls
        return cls
    raise AttributeError(f"module 'tensorflow_musa' has no attribute {name!r}")
