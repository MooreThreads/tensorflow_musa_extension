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

"""CI backstop for the allocator churn benchmark (plan §5.5, task 7-b).

The plan singles out ``bench_alloc_churn.py`` as the allocator
regression guard that should run in CI: if a future change makes the
device caching allocator leak segments, refuse cache hits, or fail
``empty_cache``, this test goes red. We invoke the benchmark with a
small (and cheap) iteration count so the whole thing completes in a
handful of seconds on CI, while still giving the allocator enough
traffic to see its steady-state behavior.
"""

import os
import subprocess
import sys
import sysconfig

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BUILD_DIR = os.path.join(REPO_ROOT, "build")
BENCH = os.path.join(REPO_ROOT, "benchmark", "bench_alloc_churn.py")


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _require_built_artifacts():
    plugin = os.path.join(BUILD_DIR, "libmusa_plugin.so")
    ext = os.path.join(BUILD_DIR, f"_C{_ext_suffix()}")
    if not (os.path.isfile(plugin) and os.path.isfile(ext)):
        pytest.skip("plugin / _C extension not built; skipping churn bench")


def test_alloc_churn_passes_invariants():
    _require_built_artifacts()
    # Use a small workload so the CI lane finishes quickly. The
    # benchmark asserts its own invariants (no leaks, no spurious OOM,
    # empty_cache releases ≥95% of reserved, cache actually hits), so
    # we only need to check the process exit code here.
    r = subprocess.run(
        [sys.executable, BENCH, "--iters", "20", "--batch", "8"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
    )
    assert r.returncode == 0, (
        f"bench_alloc_churn failed (rc={r.returncode})\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    assert "PASS" in r.stdout, r.stdout
