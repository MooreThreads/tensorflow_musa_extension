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

"""Robust loader for the native ``tensorflow_musa._C`` extension.

The extension is always shipped inside the installed
``tensorflow_musa`` package directory alongside ``libmusa_plugin.so``
(see ``setup.py``). During development, however, the same Python
files are imported directly from the source tree while the compiled
``_C.<extsuffix>.so`` lives under ``build/``; this helper bridges the
two cases so ``memory.py`` / ``device.py`` can simply do
``from ._ext import _C`` and get a valid module in both modes.

Loading strategy, in order:

  1. ``from tensorflow_musa import _C`` — the normal installed case.
     Works as soon as ``_C.<extsuffix>.so`` is in the package dir and
     Python's import machinery picked us up as a package.
  2. Filesystem lookup for ``_C.<extsuffix>.so`` next to this file
     (same as #1 but explicit, useful when the caller invoked us via
     a sys.path tweak rather than the installed package).
  3. Filesystem lookup under ``<package_dir>/../build/``, matching
     the layout that ``./build.sh`` produces in the repo.
  4. Filesystem lookup under ``<cwd>/build/``, matching the layout
     when tests are run from the repo root.

A clean ``RuntimeError`` is raised if every attempt fails. The error
message lists every path we inspected so users can tell whether their
build step ran.
"""

import importlib
import importlib.util
import os
import sysconfig
from typing import Optional

__all__ = ["_C"]


def _ext_suffix() -> str:
    # Keep in sync with setup.py / CMakeLists.txt.
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _load_from_file(path: str):
    """Import ``path`` as ``tensorflow_musa._C``."""
    spec = importlib.util.spec_from_file_location("tensorflow_musa._C", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _try_import_package_submodule():
    # Only attempt the standard import path when this module was
    # itself imported as part of a package (i.e. our ``__package__``
    # is non-empty). Otherwise the underlying ``importlib.import_module``
    # call would search ``sys.path`` and could pull in an unrelated
    # ``_C`` from the user's environment.
    if not __package__:
        return None
    try:
        return importlib.import_module(f"{__package__}._C")
    except ImportError:
        return None


def _load_c_extension():
    """Return the loaded ``tensorflow_musa._C`` module.

    Raises:
        RuntimeError: when the extension cannot be located anywhere.
    """
    # (1) Standard package-relative import.
    mod = _try_import_package_submodule()
    if mod is not None:
        return mod

    # (2) / (3) / (4) Filesystem candidates, in order of likelihood.
    here = os.path.dirname(os.path.abspath(__file__))
    name = "_C" + _ext_suffix()
    candidates = [
        os.path.join(here, name),
        os.path.normpath(os.path.join(here, os.pardir, "build", name)),
        os.path.join(os.getcwd(), "build", name),
    ]
    seen = set()
    for path in candidates:
        path = os.path.normpath(path)
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return _load_from_file(path)

    # Report every searched path so the failure is self-diagnosing.
    formatted = "\n".join(f"  - {p}" for p in candidates)
    raise RuntimeError(
        "tensorflow_musa._C extension not found. Searched:\n" + formatted +
        "\nRun ./build.sh (or `pip install .`) to produce the binary."
    )


# Cache the loaded module so repeated imports across submodules pay
# the dlopen cost exactly once.
_C_cached: Optional[object] = None


def __getattr__(name: str):
    # Lazily resolve `_C` on first access. Importing the extension at
    # module load time would force every package consumer (even those
    # that only want `_loader` utilities) to pay the dlopen cost.
    global _C_cached
    if name == "_C":
        if _C_cached is None:
            _C_cached = _load_c_extension()
        return _C_cached
    raise AttributeError(
        f"module 'tensorflow_musa._ext' has no attribute {name!r}"
    )
