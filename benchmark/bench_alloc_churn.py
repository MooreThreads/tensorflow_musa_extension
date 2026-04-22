"""Allocator churn stress test (plan §5.5, task 7-a / CI guard per 7-b).

Hammers the device caching allocator with a repeated pattern of small
and medium allocations, mixing lifetimes so that split/merge paths
in ``DeviceCachingAllocator`` stay busy. The goal isn't a throughput
number — it's a **regression backstop**: if the allocator starts
leaking segments, grows OOM events on a previously-clean workload,
or stops honoring ``empty_cache``, this bench will catch it and exit
non-zero so CI can reject the change.

Key invariants asserted at the end of the run:

* ``in_use_bytes`` is 0 after the final barrier (no leaks).
* ``empty_cache`` releases ≥ 95 % of ``reserved_bytes`` — a proxy for
  "no segment stuck in a reserved-but-unusable state".
* ``oom_events`` stayed at 0 throughout (the workload is sized to fit).
* ``cache_hits`` is strictly greater than 0 for the steady-state phase
  (otherwise the cache isn't actually caching).

Usage::

    python benchmark/bench_alloc_churn.py
    python benchmark/bench_alloc_churn.py --iters 500 --batch 64

Subprocess isolation follows the same rationale as the other benches.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_PLUGIN = os.path.join(REPO_ROOT, "build", "libmusa_plugin.so")
BUILD_DIR = os.path.join(REPO_ROOT, "build")
PKG_DIR = os.path.join(REPO_ROOT, "python")


def _run_churn(*, iters: int, batch: int, plugin: str) -> dict:
    """Run the churn workload in a fresh subprocess; return stats dict."""
    script = textwrap.dedent(
        f"""
        import json, os, sys, gc, random, types, importlib.util
        import numpy as np

        # Bring in _C + memory.py by hand (mirrors test helpers).
        sys.path.insert(0, {REPO_ROOT!r})
        sys.path.insert(0, {BUILD_DIR!r})
        pkg = types.ModuleType('tensorflow_musa')
        pkg.__path__ = [{PKG_DIR!r}]
        sys.modules['tensorflow_musa'] = pkg
        def load(name):
            spec = importlib.util.spec_from_file_location(
                'tensorflow_musa.' + name,
                os.path.join({PKG_DIR!r}, name + '.py'))
            m = importlib.util.module_from_spec(spec)
            sys.modules['tensorflow_musa.' + name] = m
            spec.loader.exec_module(m)
            return m
        load('_ext')
        memory = load('memory')

        import tensorflow as tf
        from tensorflow.python.framework import load_library
        load_library.load_pluggable_device_library({plugin!r})

        ITERS = {iters}
        BATCH = {batch}

        # Mixed-size allocations, in elements of float32. The mix
        # intentionally brackets the allocator's split threshold so
        # both split and merge paths see traffic.
        sizes = [256, 1024, 4096, 16_384, 65_536, 262_144, 1_048_576]
        rng = random.Random(0xC0FFEE)

        with tf.device('/MUSA:0'):
            # Warmup so the allocator reaches steady state before we
            # sample the cache_hits floor.
            for _ in range(64):
                _ = tf.random.uniform([sizes[rng.randrange(len(sizes))]])
            _ = _.numpy()
            memory.reset_peak_memory_stats()

            warm_stats = memory.memory_stats()

            hit_floor = warm_stats['cache_hits']
            miss_floor = warm_stats['cache_misses']

            for _ in range(ITERS):
                batch_tensors = []
                for _ in range(BATCH):
                    n = sizes[rng.randrange(len(sizes))]
                    batch_tensors.append(tf.random.uniform([n]))
                # Kick a sync so the stream drains between batches;
                # otherwise the allocator just keeps extending.
                _ = batch_tensors[-1].numpy()
                # Randomly drop a portion of the batch to exercise
                # free-list coalescing mid-stream.
                keep = rng.sample(batch_tensors, len(batch_tensors) // 2)
                del batch_tensors
                del keep
                gc.collect()

        gc.collect()
        final = memory.memory_stats()
        released = memory.empty_cache()
        after_empty = memory.memory_stats()

        out = {{
            'warm': warm_stats,
            'final': final,
            'released': released,
            'after_empty': after_empty,
            'hit_floor': hit_floor,
            'miss_floor': miss_floor,
        }}
        print(json.dumps(out))
        """
    )
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(
        "bench_alloc_churn subprocess produced no JSON.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    plugin = os.path.abspath(args.plugin)
    if not os.path.isfile(plugin):
        print(f"ERROR: plugin not found at {plugin}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    out = _run_churn(iters=args.iters, batch=args.batch, plugin=plugin)
    wall = time.perf_counter() - t0

    final = out["final"]
    after = out["after_empty"]
    released = out["released"]

    print(f"wall {wall:.1f}s  iters={args.iters} batch={args.batch}")
    print(
        f"  steady:  in_use={final['in_use_bytes']:>12}  "
        f"reserved={final['reserved_bytes']:>12}  segments={final['segments']}"
    )
    print(
        f"  peak:    peak_in_use={final['peak_in_use_bytes']:>12}  "
        f"oom_events={final['oom_events']}"
    )
    print(
        f"  cache:   hits={final['cache_hits']-out['hit_floor']:>10}  "
        f"misses={final['cache_misses']-out['miss_floor']:>10}"
    )
    print(
        f"  empty:   released={released:>12}  "
        f"after.reserved={after['reserved_bytes']:>12}  "
        f"after.in_use={after['in_use_bytes']:>12}"
    )

    failures = []
    if after["in_use_bytes"] != 0:
        failures.append(
            f"in_use_bytes leaked after empty_cache: {after['in_use_bytes']}"
        )
    if final["oom_events"] != 0:
        failures.append(f"oom_events non-zero during churn: {final['oom_events']}")
    if final["cache_hits"] - out["hit_floor"] == 0:
        failures.append("no cache_hits recorded in steady state (cache disabled?)")
    if final["reserved_bytes"] > 0:
        release_ratio = released / final["reserved_bytes"]
        if release_ratio < 0.95:
            failures.append(
                f"empty_cache only released {release_ratio:.1%} of reserved "
                f"({released} / {final['reserved_bytes']})"
            )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    if failures:
        print("\nFAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nPASS: allocator churn invariants held.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
