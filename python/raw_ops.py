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

"""Raw generated wrappers for MUSA extension ops."""

from ._loader import get_musa_ops


def __getattr__(name):
    op_module = get_musa_ops()
    try:
        return getattr(op_module, name)
    except AttributeError as exc:
        raise AttributeError(
            f"MUSA raw op {name!r} is not available. "
            "Check that the MUSA plugin is built and loaded, or inspect "
            "tensorflow_musa.get_musa_ops() for generated wrapper names."
        ) from exc


def __dir__():
    names = set(globals())
    try:
        names.update(dir(get_musa_ops()))
    except Exception:
        pass
    return sorted(names)
