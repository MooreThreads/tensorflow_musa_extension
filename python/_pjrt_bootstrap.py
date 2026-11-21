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

"""Thin TensorFlow MUSA adapter for the optional OpenXLA-MUSA PJRT provider.

All PJRT artifact discovery, environment preparation, dynamic loading, and
registration live in the provider wheel. This module only owns the explicit
opt-in switch, provider selection, and a stable diagnostic forwarding surface.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, MutableMapping, Optional, Tuple

PJRT_PROVIDER_MODULE_ENV = "TENSORFLOW_MUSA_PJRT_PROVIDER_MODULE"
PJRT_ENABLE_ARG = "--xla"
_DEFAULT_PROVIDER_MODULES = ("openxla_musa_pjrt",)

# Environment entries published by the provider before TensorFlow is imported.
# They are snapshotted so a failed optional PJRT attempt can be rolled back
# without disturbing values supplied by the caller.
_PJRT_ENV_KEYS = (
    "LD_LIBRARY_PATH",
    "TF_PLUGGABLE_DEVICE_LIBRARY_PATH",
    "PJRT_NAMES_AND_LIBRARY_PATHS",
    "NEXT_PLUGGABLE_DEVICE_USE_C_API",
    "MUSA_TFNPD_IS_PLUGGABLE_DEVICE",
    "MUSA_TFNPD_COMPILATION_DEVICE_NAME",
    "MTCC20X_TOOLS_DIR",
    "TF_XLA_FLAGS",
)

_STATUS_KEYS = ("mode", "phase", "provider", "plugin_path", "bridge_path", "error")

_provider = None
_status: Dict[str, Optional[str]] = {}
_prepared_env_snapshot: Optional[Dict[str, Optional[str]]] = None


def _new_status(
    mode: str = "native",
    phase: str = "idle",
    provider: Optional[str] = None,
    plugin_path: Optional[str] = None,
    bridge_path: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    return {
        "mode": mode,
        "phase": phase,
        "provider": provider,
        "plugin_path": plugin_path,
        "bridge_path": bridge_path,
        "error": error,
    }


_status = _new_status()


def is_pjrt_requested(environ=None, argv=None) -> bool:
    """Return whether PJRT was requested by the public ``--xla`` switch.

    TensorFlow MUSA is imported by user programs before their own argument
    parser usually runs, so the adapter must inspect the process arguments at
    import/bootstrap time. Only the exact public switch is recognized;
    options such as ``--xla_jit_scope`` are intentionally unrelated.
    """
    del environ  # Kept for source compatibility; mode selection is CLI-only.
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" in arguments:
        arguments = arguments[:arguments.index("--")]
    return PJRT_ENABLE_ARG in arguments


def _provider_module_names(environ=None) -> Tuple[str, ...]:
    env = os.environ if environ is None else environ
    explicit = env.get(PJRT_PROVIDER_MODULE_ENV, "").strip()
    if explicit:
        names = tuple(name.strip() for name in explicit.split(",") if name.strip())
        if names:
            return names
    return _DEFAULT_PROVIDER_MODULES


def _load_provider(environ=None):
    """Import the configured provider only after the user enables PJRT."""
    global _provider
    if _provider is not None:
        return _provider

    names = _provider_module_names(environ)
    unavailable = []
    for module_name in names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Do not hide a missing dependency inside an otherwise found provider.
            if exc.name != module_name:
                raise
            unavailable.append(module_name)
            continue
        for attribute in ("prepare", "bootstrap", "get_pjrt_status"):
            if not callable(getattr(module, attribute, None)):
                raise RuntimeError(
                    f"PJRT provider {module_name!r} does not implement {attribute}()"
                )
        _provider = module
        return module

    tried = ", ".join(repr(name) for name in unavailable or names)
    raise ModuleNotFoundError(
        "OpenXLA-MUSA PJRT provider is not installed. Install the "
        f"openxla_musa_pjrt wheel (tried {tried})."
    )


def _provider_status(provider) -> Dict[str, Optional[str]]:
    status = provider.get_pjrt_status()
    if not isinstance(status, dict):
        raise TypeError(
            f"PJRT provider {provider.__name__!r} returned a non-dict status"
        )
    return {key: status.get(key) for key in _STATUS_KEYS}


def _record_failure(error: Exception, provider: Optional[str] = None) -> None:
    global _status
    _status = _new_status(
        mode="pjrt",
        phase="failed",
        provider=provider,
        error=str(error)[:600],
    )


def prepare_pjrt_bootstrap(environ: Optional[MutableMapping[str, str]] = None) -> bool:
    """Delegate pre-import PJRT preparation to the selected provider.

    Returns ``True`` only when PJRT is enabled. Success is otherwise reported
    through :func:`get_pjrt_bootstrap_status`; provider failures are raised
    immediately because PJRT is an explicit user choice.
    """
    global _status, _prepared_env_snapshot
    env = os.environ if environ is None else environ
    requested = is_pjrt_requested()
    if not requested:
        _prepared_env_snapshot = None
        _status = _new_status()
        return False

    _prepared_env_snapshot = {
        key: env.get(key) for key in _PJRT_ENV_KEYS
    }
    provider = None
    try:
        provider = _load_provider(env)
        provider.prepare(environ=env)
        _status = _provider_status(provider)
        return True
    except Exception as exc:
        if provider is not None:
            try:
                _status = _provider_status(provider)
            except Exception:
                _record_failure(exc, getattr(provider, "__name__", None))
        else:
            _record_failure(exc)
        raise


def fallback_to_native(
    error: Exception,
    environ: Optional[MutableMapping[str, str]] = None,
) -> None:
    """Undo provider-published pre-import state before native fallback.

    This helper is only used when ``prepare()`` fails, before PJRT has begun
    loading or registering its shared libraries. A failure during
    ``bootstrap()`` cannot safely fall back in the same process because PJRT
    may already have registered MUSA kernels.
    """
    global _status, _prepared_env_snapshot
    env = os.environ if environ is None else environ
    if _prepared_env_snapshot is not None:
        for key, previous in _prepared_env_snapshot.items():
            if previous is None:
                env.pop(key, None)
            else:
                env[key] = previous
    _prepared_env_snapshot = None
    _status = _new_status(
        mode="native",
        phase="fallback",
        error=str(error)[:600],
    )


def load_pjrt_plugin(tf_module) -> None:
    """Delegate post-import PJRT loading and registration to the provider."""
    global _status, _prepared_env_snapshot
    if not is_pjrt_requested():
        return
    provider = _load_provider()
    try:
        provider.bootstrap(tf_module)
        _status = _provider_status(provider)
        _prepared_env_snapshot = None
    except Exception as exc:
        try:
            _status = _provider_status(provider)
        except Exception:
            _record_failure(exc, getattr(provider, "__name__", None))
        raise


def get_pjrt_bootstrap_status() -> Dict[str, Optional[str]]:
    """Return the stable PJRT status without importing the provider eagerly."""
    return dict(_status)
