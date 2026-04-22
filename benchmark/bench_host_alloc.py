"""Microbenchmark for the Host Pinned Caching Allocator (commit C1).

Measures the wall-clock cost of many small H2D transfers, which exercise the
`HostCachingAllocator` via both:
  - `host_memory_allocate` (TF's pinned host allocations),
  - `PinnedStagingPool` (staging buffers for pageable sources above threshold).

A/B is driven by `TF_MUSA_DISABLE_HOST_CACHING`:
  cached:  allocator on  (default)
  fresh:   allocator off (every request goes straight to musaHostAlloc)

Usage (run from the repo root, with the plugin already built in ./build/):

  # Run the full A/B comparison in one invocation:
  python benchmark/bench_host_alloc.py

  # Or run a single mode (useful for profiling):
  TF_MUSA_DISABLE_HOST_CACHING=1 python benchmark/bench_host_alloc.py --mode fresh
  TF_MUSA_DISABLE_HOST_CACHING=0 python benchmark/bench_host_alloc.py --mode cached

Target: cached mode should be measurably faster than fresh mode. On the
TF-eager path most of the per-iteration cost is Python + op dispatch, so this
harness sets a modest 1.10x PASS gate here. A cleaner driver-level
microbenchmark will land with the pybind extension (plan §5.4 / commit C6),
which will exercise the plan §5.1 ">50% improvement" target more directly.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np


DEFAULT_PLUGIN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    "build",
    "libmusa_plugin.so",
)


def _run_one_mode(mode: str, iters: int, size_bytes: int, plugin_path: str) -> float:
    """Run the H2D loop in a fresh subprocess so the env toggle takes effect.

    Returns average microseconds per H2D copy.
    """
    env = os.environ.copy()
    env["TF_MUSA_DISABLE_HOST_CACHING"] = "1" if mode == "fresh" else "0"
    env.setdefault("TF_MUSA_H2D_STAGING_THRESHOLD_BYTES", "4096")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    script = (
        "import os, time, numpy as np\n"
        "import tensorflow as tf\n"
        "from tensorflow.python.framework import load_library\n"
        f"load_library.load_pluggable_device_library({plugin_path!r})\n"
        "n = int(os.environ['_BENCH_N'])\n"
        "size_elems = int(os.environ['_BENCH_SIZE']) // 4\n"
        "# Pageable source, above the 4 KiB staging threshold, so every\n"
        "# iteration goes through PinnedStagingPool -> HostCachingAllocator.\n"
        "hosts = [np.random.rand(size_elems).astype(np.float32) for _ in range(32)]\n"
        "with tf.device('/MUSA:0'):\n"
        "    acc = tf.Variable(tf.zeros([size_elems], dtype=tf.float32))\n"
        "    # Warm up: allocator priming.\n"
        "    for i in range(min(64, n)):\n"
        "        acc.assign_add(tf.constant(hosts[i % len(hosts)]))\n"
        "    _ = acc.numpy()  # one sync\n"
        "    t0 = time.perf_counter()\n"
        "    for i in range(n):\n"
        "        acc.assign_add(tf.constant(hosts[i % len(hosts)]))\n"
        "    _ = acc.numpy()  # single sync at end\n"
        "    t1 = time.perf_counter()\n"
        "print(f'{(t1-t0)*1e6/n:.3f}')\n"
    )

    env["_BENCH_N"] = str(iters)
    env["_BENCH_SIZE"] = str(size_bytes)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    last = proc.stdout.strip().splitlines()[-1]
    return float(last)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--size", type=int, default=64 * 1024)
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN_PATH)
    ap.add_argument("--mode", choices=["both", "cached", "fresh"], default="both")
    args = ap.parse_args()

    plugin = os.path.abspath(args.plugin)
    if not os.path.isfile(plugin):
        print(f"ERROR: plugin not found: {plugin}", file=sys.stderr)
        return 2

    modes = ["fresh", "cached"] if args.mode == "both" else [args.mode]
    results = {}
    for mode in modes:
        t0 = time.perf_counter()
        avg_us = _run_one_mode(mode, args.iters, args.size, plugin)
        results[mode] = avg_us
        total = time.perf_counter() - t0
        print(
            f"[{mode:>6}] iters={args.iters} size={args.size}B  "
            f"avg={avg_us:.2f} us  (wall {total:.1f}s)"
        )

    if "cached" in results and "fresh" in results:
        speedup = results["fresh"] / results["cached"] if results["cached"] > 0 else 0.0
        print(f"\nspeedup(cached vs fresh) = {speedup:.2f}x")
        # TF-eager Variable.assign_add path is dominated by op dispatch; we just
        # want to confirm the allocator is not a regression and that cached mode
        # wins. The sharper driver-level bench comes with the _C pybind (C6).
        if speedup >= 1.10:
            print("PASS: cached >= 1.10x fresh (TF-eager harness; see docstring)")
        else:
            print("WARN: speedup below 1.10x; host caching allocator may have a")
            print("      regression or the workload is not allocator-bound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
