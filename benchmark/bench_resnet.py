"""ResNet50 training-step benchmark (plan §5.5, task 7-a).

Drives 100 forward+backward+optimizer steps of a ResNet-50-like stack
on synthetic data and reports the mean step time plus the allocator's
peak footprint. Matches the "ResNet50 train 单卡 100 step" bullet in
plan §5.5 without pulling in a real dataset — for regression tracking
the synthetic timing is the relevant number, and it keeps the
benchmark runnable on any host with a MUSA device.

Usage (from the repo root, plugin built in ``./build/``)::

    python benchmark/bench_resnet.py
    python benchmark/bench_resnet.py --steps 200 --batch 32
    python benchmark/bench_resnet.py --json out.json

If ``tensorflow.keras.applications.ResNet50`` is not available in the
installed TF version, the benchmark falls back to a small conv stack
so it still exercises the allocator; the output clearly labels which
model was used.
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


def _run_training(*, steps: int, batch: int, plugin: str) -> dict:
    """Drive the training loop in a fresh subprocess."""
    script = textwrap.dedent(
        f"""
        import json, os, sys, time, statistics, types, importlib.util

        # _C + memory.py hand-load (same pattern as the churn bench).
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

        STEPS = {steps}
        BATCH = {batch}

        # Try keras first; some TF 2.6 installs have a broken stand-
        # alone keras wheel ("dtensor" ImportError) that we can't fix
        # here. Fall back to a hand-rolled low-level TF model built
        # from tf.Variable + tf.nn.* so the bench still measures the
        # allocator on an equivalently varied workload.
        model_name = 'resnet50'
        use_keras = True
        model = None
        try:
            # `tf.keras` is a lazy loader; touching .applications may
            # trigger an ImportError from a broken stand-alone keras
            # wheel (common on TF 2.6 where pip-installed keras can
            # drift ahead of the bundled version).
            _ = tf.keras.applications
            model = tf.keras.applications.ResNet50(
                weights=None, input_shape=(224, 224, 3), classes=1000)
        except Exception as exc:
            use_keras = False
            model_name = f'fallback_convnet ({{type(exc).__name__}})'

        if not use_keras:
            # Hand-written stack: five 3x3 convs + global-pool + dense.
            # Mirrors the mix of short-lived activations and long-
            # lived weights that ResNet50 exercises, which is the
            # property the caching allocator actually cares about.
            def _glorot(shape):
                lim = (6.0 / (shape[-2] + shape[-1])) ** 0.5
                return tf.random.uniform(shape, -lim, lim)

            w = {{
                'c1': tf.Variable(_glorot([3, 3, 3, 64])),
                'c2': tf.Variable(_glorot([3, 3, 64, 128])),
                'c3': tf.Variable(_glorot([3, 3, 128, 256])),
                'c4': tf.Variable(_glorot([3, 3, 256, 512])),
                'c5': tf.Variable(_glorot([3, 3, 512, 512])),
                'fc': tf.Variable(_glorot([512, 1000])),
                'fb': tf.Variable(tf.zeros([1000])),
            }}
            variables = list(w.values())

        # Hand-rolled SGD so we don't pull in tf.optimizers / keras.
        # Plain SGD is enough allocator traffic for this benchmark
        # (the interesting alloc patterns come from conv activations,
        # not optimizer state), and it side-steps keras's broken
        # dtensor import on some TF 2.6 installs.
        LEARNING_RATE = 0.01

        with tf.device('/MUSA:0'):
            x = tf.random.uniform((BATCH, 224, 224, 3), dtype=tf.float32)
            y = tf.random.uniform((BATCH,), maxval=1000, dtype=tf.int32)

            if use_keras:
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True)

                @tf.function
                def train_step(x, y):
                    with tf.GradientTape() as tape:
                        logits = model(x, training=True)
                        loss = loss_fn(y, logits)
                    grads = tape.gradient(loss, model.trainable_variables)
                    for v, g in zip(model.trainable_variables, grads):
                        v.assign_sub(LEARNING_RATE * g)
                    return loss

            else:
                # Kernel / stride combo chosen so the implicit SAME
                # padding stays symmetric — the MUSA conv kernel
                # rejects asymmetric padding. 3x3 stride 1 SAME pads
                # (1,1,1,1) which is fine; 2x2 stride 2 VALID
                # downsamples without padding.
                @tf.function
                def train_step(x, y):
                    with tf.GradientTape() as tape:
                        h = tf.nn.relu(tf.nn.conv2d(x, w['c1'], 1, 'SAME'))
                        h = tf.nn.max_pool2d(h, 2, 2, 'VALID')
                        h = tf.nn.relu(tf.nn.conv2d(h, w['c2'], 1, 'SAME'))
                        h = tf.nn.max_pool2d(h, 2, 2, 'VALID')
                        h = tf.nn.relu(tf.nn.conv2d(h, w['c3'], 1, 'SAME'))
                        h = tf.nn.max_pool2d(h, 2, 2, 'VALID')
                        h = tf.nn.relu(tf.nn.conv2d(h, w['c4'], 1, 'SAME'))
                        h = tf.nn.max_pool2d(h, 2, 2, 'VALID')
                        h = tf.nn.relu(tf.nn.conv2d(h, w['c5'], 1, 'SAME'))
                        h = tf.reduce_mean(h, axis=[1, 2])
                        logits = tf.matmul(h, w['fc']) + w['fb']
                        loss = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=y, logits=logits))
                    grads = tape.gradient(loss, variables)
                    for v, g in zip(variables, grads):
                        v.assign_sub(LEARNING_RATE * g)
                    return loss

            for _ in range(5):
                _ = train_step(x, y)
            _ = _.numpy()

            memory.reset_peak_memory_stats()
            samples = []
            for _ in range(STEPS):
                t0 = time.perf_counter()
                loss = train_step(x, y)
                _ = loss.numpy()
                samples.append(time.perf_counter() - t0)

        samples.sort()
        mean = sum(samples) / len(samples)
        p50 = samples[len(samples) // 2]
        p95 = samples[int(len(samples) * 0.95)]
        stats = memory.memory_stats()
        out = {{
            'model': model_name,
            'steps': STEPS,
            'batch': BATCH,
            'mean_s': mean,
            'p50_s': p50,
            'p95_s': p95,
            'min_s': samples[0],
            'max_s': samples[-1],
            'peak_in_use_bytes': stats['peak_in_use_bytes'],
            'reserved_bytes': stats['reserved_bytes'],
            'oom_events': stats['oom_events'],
        }}
        print(json.dumps(out))
        """
    )
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench_resnet subprocess failed (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(
        "bench_resnet subprocess produced no JSON.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    plugin = os.path.abspath(args.plugin)
    if not os.path.isfile(plugin):
        print(f"ERROR: plugin not found at {plugin}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    out = _run_training(steps=args.steps, batch=args.batch, plugin=plugin)
    wall = time.perf_counter() - t0

    print(f"model:  {out['model']}")
    print(
        f"steps:  {out['steps']}  batch: {out['batch']}  wall: {wall:.1f}s"
    )
    print(
        f"time:   mean={out['mean_s']*1e3:.2f} ms  "
        f"p50={out['p50_s']*1e3:.2f} ms  "
        f"p95={out['p95_s']*1e3:.2f} ms  "
        f"min={out['min_s']*1e3:.2f}  max={out['max_s']*1e3:.2f}"
    )
    print(
        f"alloc:  peak_in_use={out['peak_in_use_bytes']:>12}  "
        f"reserved={out['reserved_bytes']:>12}  "
        f"oom={out['oom_events']}"
    )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote: {args.json}")

    if out["oom_events"] != 0:
        print("\nFAIL: allocator reported OOM during steady-state training.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
