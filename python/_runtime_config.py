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

"""Helpers for configuring TensorFlow MUSA runtime options."""

def _runtime_config_bindings():
    from ._loader import load_plugin

    load_plugin()

    from . import _runtime_config_bindings as bindings

    return bindings


def set_musa_allow_growth(enabled=True):
    """Set process-wide MUSA BFC allocator allow_growth.

    This setting is applied to subsequently created MUSA devices. The
    `TF_FORCE_GPU_ALLOW_GROWTH` environment variable, when set to `true` or
    `false`, takes precedence over this Python setting.

    Args:
        enabled: Whether the MUSA device allocator should grow on demand.
    """
    _runtime_config_bindings().set_musa_allow_growth(bool(enabled))
