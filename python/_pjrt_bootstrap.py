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

"""Configuration, discovery, and loading for the optional MUSA PJRT runtime.

The bootstrap is explicitly opt-in and preserves the required order: prepare
non-destructive discovery variables, import TensorFlow, load PJRT globally,
register it through TensorFlow's pluggable-device loader, then invoke the MUSA
PJRT client registration hook.
"""

from dataclasses import asdict, dataclass
import ctypes
import logging
import os
from typing import Dict, Optional, Tuple


PJRT_ENABLE_ENV = "TENSORFLOW_MUSA_ENABLE_PJRT"
PJRT_LIBRARY_ENV = "TENSORFLOW_MUSA_PJRT_LIBRARY_PATH"
PJRT_LIBRARY = "libmusa_pjrt_plugin_zy.so"
PJRT_FORCE_REGISTER_SYMBOL = "ForceRegisterMusa"
MTGPU_TOOLS_ENV = "MTCC20X_TOOLS_DIR"
MTGPU_TOOLS_DIRECTORY = "mtcc20x_tools"
_REQUIRED_MTGPU_TOOLS = ("llvm-link", "opt", "llc", "lld")

_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("", "0", "false", "no", "off"))


@dataclass(frozen=True)
class PjrtBootstrapStatus:
    """A stable, serializable snapshot of the PJRT bootstrap state."""

    requested: bool = False
    state: str = "disabled"
    library_path: Optional[str] = None
    error: Optional[str] = None
    loaded: bool = False
    registered: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_status = PjrtBootstrapStatus()
_pjrt_library_handle = None
_logger = logging.getLogger(__name__)


def is_pjrt_requested(environ=None) -> bool:
    """Return whether optional PJRT integration was explicitly requested.

    PJRT is disabled by default.  Invalid values fail early instead of silently
    selecting a backend that differs from the user's intent.
    """
    env = os.environ if environ is None else environ
    raw = env.get(PJRT_ENABLE_ENV, "0").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid {PJRT_ENABLE_ENV}={raw!r}; expected one of "
        "0/1, false/true, no/yes, or off/on."
    )


def _candidate_library_paths(environ=None) -> Tuple[str, ...]:
    env = os.environ if environ is None else environ
    package_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    explicit = env.get(PJRT_LIBRARY_ENV, "").strip()
    if explicit:
        explicit = os.path.abspath(os.path.expanduser(explicit))
        if os.path.isdir(explicit):
            explicit = os.path.join(explicit, PJRT_LIBRARY)
        candidates.append(explicit)

    candidates.extend(
        (
            os.path.join(package_dir, PJRT_LIBRARY),
            os.path.join(package_dir, "..", "build", PJRT_LIBRARY),
            os.path.join(os.getcwd(), "build", PJRT_LIBRARY),
            os.path.join("/usr/local/musa", "lib", PJRT_LIBRARY),
            os.path.join("/usr/local/musa", "lib64", PJRT_LIBRARY),
            # Auto-search openxla_musa Bazel output (sibling directory)
            os.path.join(package_dir, "..", "openxla_musa", "bazel-bin",
                         "pjrt_plugin", PJRT_LIBRARY),
            os.path.join(os.getcwd(), "..", "openxla_musa", "bazel-bin",
                         "pjrt_plugin", PJRT_LIBRARY),
        )
    )
    for directory in env.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if directory:
            candidates.append(os.path.join(directory, PJRT_LIBRARY))

    # Preserve priority while removing normalized duplicates.
    result = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def find_pjrt_library(environ=None) -> str:
    """Resolve the optional PJRT plugin using deterministic search order."""
    env = os.environ if environ is None else environ
    explicit = env.get(PJRT_LIBRARY_ENV, "").strip()
    candidates = _candidate_library_paths(env)
    if explicit:
        explicit_candidate = candidates[0]
        if os.path.isfile(explicit_candidate):
            return explicit_candidate
        raise FileNotFoundError(
            f"MUSA PJRT plugin specified by {PJRT_LIBRARY_ENV} does not exist: "
            f"{explicit_candidate}"
        )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"MUSA PJRT plugin library '{PJRT_LIBRARY}' not found.\n"
        f"Searched locations:\n{searched}\n"
        f"Set {PJRT_LIBRARY_ENV} to the plugin file or containing directory."
    )


def _find_mtgpu_tools_directory(environ=None) -> str:
    """Resolve and validate the OpenXLA-MUSA compiler tool directory."""
    env = os.environ if environ is None else environ
    package_dir = os.path.dirname(os.path.abspath(__file__))
    explicit = env.get(MTGPU_TOOLS_ENV, "").strip()

    def is_complete(candidate):
        return all(
            os.path.isfile(os.path.join(candidate, tool))
            and os.access(os.path.join(candidate, tool), os.X_OK)
            for tool in _REQUIRED_MTGPU_TOOLS
        )

    if explicit:
        explicit_candidate = os.path.abspath(os.path.expanduser(explicit))
        if is_complete(explicit_candidate):
            return explicit_candidate
        raise FileNotFoundError(
            f"MUSA PJRT compiler tools specified by {MTGPU_TOOLS_ENV} are "
            f"incomplete or not executable: {explicit_candidate}. Expected: "
            f"{', '.join(_REQUIRED_MTGPU_TOOLS)}"
        )

    candidates = (
        os.path.join(package_dir, MTGPU_TOOLS_DIRECTORY),
        os.path.normpath(
            os.path.join(package_dir, "..", "openxla_musa", MTGPU_TOOLS_DIRECTORY)
        ),
        # Search from cwd: ../openxla_musa/mtcc20x_tools (dev layout)
        os.path.normpath(
            os.path.join(os.getcwd(), "..", "openxla_musa", MTGPU_TOOLS_DIRECTORY)
        ),
        # Search from cwd: openxla_musa/mtcc20x_tools (sibling)
        os.path.normpath(
            os.path.join(os.getcwd(), "openxla_musa", MTGPU_TOOLS_DIRECTORY)
        ),
    )
    for candidate in candidates:
        if is_complete(candidate):
            return candidate
    details = ", ".join(candidates)
    raise FileNotFoundError(
        "MUSA PJRT compiler tools are incomplete or not executable. "
        f"Set {MTGPU_TOOLS_ENV} to a directory containing "
        f"{', '.join(_REQUIRED_MTGPU_TOOLS)}. Searched: {details}"
    )


def _append_unique_entry(environ, name: str, value: str) -> None:
    # TensorFlow's plugin-list variables use comma-separated entries.  Do not
    # use os.pathsep here: PJRT entries themselves contain ``MUSA:/path``.
    current = environ.get(name, "")
    entries = [entry.strip() for entry in current.split(",") if entry.strip()]
    if value not in entries:
        entries.append(value)
    if entries:
        environ[name] = ",".join(entries)


def _append_unique_flag(environ, name: str, value: str) -> None:
    flags = environ.get(name, "").split()
    if value not in flags:
        flags.append(value)
    environ[name] = " ".join(flags)


def _ensure_tf_library_path(environ):
    """Ensure the TF library directory is on LD_LIBRARY_PATH.

    The PJRT plugin's RPATH cannot resolve ``libtensorflow_framework.so.2``
    in isolation.  ``LD_LIBRARY_PATH`` must be set **before the Python
    process starts** so the dynamic linker can find the framework when
    NPD ``dlopen``\s the PJRT plugin during ``import tensorflow``.

    This function records the path in ``os.environ`` as a hint for
    subprocesses and for the shell wrapper to prepend.
    """
    import site
    ld_path = environ.get("LD_LIBRARY_PATH", "")
    for site_dir in site.getsitepackages():
        tf_dir = os.path.join(site_dir, "tensorflow")
        if os.path.isdir(tf_dir):
            if tf_dir not in ld_path.split(os.pathsep):
                environ["LD_LIBRARY_PATH"] = (
                    tf_dir + os.pathsep + ld_path if ld_path else tf_dir
                )
            return


def prepare_pjrt_environment(library_path: str, environ=None) -> None:
    """Set discovery entries for PJRT plugin loading.

    ``TF_PLUGGABLE_DEVICE_LIBRARY_PATH`` triggers NPD auto-discovery
    during ``import tensorflow``, which calls ``SetPjrtApi`` (required
    for JIT) and registers delegating OpKernel instances.

    **Native kernels are not loaded in PJRT mode** to avoid
    ``Multiple OpKernel registrations`` conflicts.  Both eager and JIT
    execution use the PJRT backend.

    Requires ``LD_LIBRARY_PATH`` to include the TF library directory at
    the **shell level** so the PJRT plugin's RPATH can resolve
    ``libtensorflow_framework.so.2``.
    """
    env = os.environ if environ is None else environ
    normalized = os.path.abspath(library_path)
    tools_directory = _find_mtgpu_tools_directory(env)

    _ensure_tf_library_path(env)

    _append_unique_entry(
        env, "TF_PLUGGABLE_DEVICE_LIBRARY_PATH", normalized
    )
    _append_unique_entry(
        env, "PJRT_NAMES_AND_LIBRARY_PATHS", f"MUSA:{normalized}"
    )
    env.setdefault("NEXT_PLUGGABLE_DEVICE_USE_C_API", "true")
    env.setdefault("MUSA_TFNPD_IS_PLUGGABLE_DEVICE", "1")
    env.setdefault("MUSA_TFNPD_COMPILATION_DEVICE_NAME", "XLA_GPU_JIT")
    _append_unique_flag(env, "TF_XLA_FLAGS", "--tf_xla_use_device_api=true")
    env.setdefault(MTGPU_TOOLS_ENV, tools_directory)


def prepare_pjrt_bootstrap(environ=None) -> PjrtBootstrapStatus:
    """Prepare the requested PJRT backend and record a diagnostic snapshot.

    This function is idempotent for a stable environment.  Discovery failures
    are recorded and re-raised because the user explicitly requested PJRT.
    """
    global _status
    env = os.environ if environ is None else environ
    try:
        requested = is_pjrt_requested(env)
    except ValueError as exc:
        _status = PjrtBootstrapStatus(
            requested=False, state="invalid_configuration", error=str(exc)
        )
        raise

    if not requested:
        _status = PjrtBootstrapStatus()
        return _status

    try:
        library_path = find_pjrt_library(env)
        prepare_pjrt_environment(library_path, env)
    except Exception as exc:
        _status = PjrtBootstrapStatus(
            requested=True, state="failed", error=str(exc)
        )
        raise

    _status = PjrtBootstrapStatus(
        requested=True,
        state="prepared",
        library_path=library_path,
    )
    return _status


def _load_global_library(library_path: str):
    global _pjrt_library_handle
    if _pjrt_library_handle is not None:
        return _pjrt_library_handle
    mode = getattr(ctypes, "RTLD_GLOBAL", None)
    if mode is None:
        _pjrt_library_handle = ctypes.CDLL(library_path)
    else:
        _pjrt_library_handle = ctypes.CDLL(library_path, mode=mode)
    return _pjrt_library_handle


def load_pjrt_plugin(tf) -> PjrtBootstrapStatus:
    """Load and register PJRT after native kernel and TensorFlow import.

    Uses ``load_pluggable_device_library`` on the PJRT plugin, which:
    1. Calls ``SetPjrtApi`` via the plugin's ``GetPjrtApi`` export —
       required for ``jit_compile=True`` XLA compilation.
    2. Calls ``RegisterPluggableDevicePlugin`` which registers delegating
       OpKernel instances on the MUSA device type.

    Native kernels from ``libmusa_plugin.so`` are loaded FIRST (by
    ``_loader.py``), so they take priority for eager execution.
    Delegating kernels registered here are a fallback for ops that
    don't have native implementations.

    ``ForceRegisterMusa`` creates the PJRT client for XLA JIT.
    """
    global _status
    if not is_pjrt_requested():
        return _status
    if _status.registered:
        return _status
    if not _status.library_path:
        prepare_pjrt_bootstrap()

    try:
        from tensorflow.python.framework.load_library import (
            load_pluggable_device_library,
        )
    except ImportError as e:
        _status = PjrtBootstrapStatus(
            requested=True, state="failed",
            library_path=_status.library_path,
            error="TensorFlow does not expose load_pluggable_device_library",
        )
        raise RuntimeError(
            "TensorFlow does not expose load_pluggable_device_library; "
            "PJRT registration requires a TF build with the StreamExecutor "
            "C API loader."
        ) from e

    try:
        load_pluggable_device_library(_status.library_path)

        # Also call ForceRegisterMusa for the XLA JIT client
        handle = _load_global_library(_status.library_path)
        register = getattr(handle, PJRT_FORCE_REGISTER_SYMBOL, None)
        if register is not None:
            register.argtypes = []
            register.restype = None
            register()

        devices = tf.config.list_physical_devices("MUSA")
        if not devices:
            raise RuntimeError(
                "PJRT registration succeeded but TensorFlow exposes no MUSA devices"
            )
        _status = PjrtBootstrapStatus(
            requested=True, state="registered",
            library_path=_status.library_path,
            loaded=True, registered=True,
        )
        _logger.info("MUSA PJRT registered with %d devices", len(devices))
        return _status
    except Exception as exc:
        _status = PjrtBootstrapStatus(
            requested=True,
            state="failed",
            library_path=_status.library_path,
            loaded=_pjrt_library_handle is not None,
            error=str(exc),
        )
        raise RuntimeError(f"Failed to bootstrap MUSA PJRT: {exc}") from exc


def get_pjrt_bootstrap_status() -> Dict[str, object]:
    """Return a copy of the current PJRT bootstrap status."""
    return _status.to_dict()
