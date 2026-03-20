# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Simple warmup-aware benchmark for the MUSA Abs operator."""

import argparse
import time

import numpy as np
import tensorflow as tf

from musa_test_utils import load_musa_plugin


def _parse_shape(shape_text):
  if not shape_text:
    return []
  return [int(dim) for dim in shape_text.split(",") if dim]


def _dtype_from_name(name):
  mapping = {
      "float32": tf.float32,
      "float16": tf.float16,
      "bfloat16": tf.bfloat16,
      "double": tf.float64,
      "int32": tf.int32,
      "int64": tf.int64,
  }
  return mapping[name]


def _make_input(shape, dtype):
  np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
  if dtype.is_integer:
    data = np.random.randint(-1024, 1024, size=shape, dtype=np_dtype)
  else:
    data = np.random.uniform(-10.0, 10.0, size=shape).astype(np_dtype)
  return tf.constant(data, dtype=dtype)


def _run_once(x):
  start = time.perf_counter()
  with tf.device("/device:MUSA:0"):
    y = tf.abs(x)
  # Materialize on host to synchronize device work before timing ends.
  _ = y.numpy()
  return (time.perf_counter() - start) * 1000.0


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--shape", default="1024,1024",
                      help="Comma-separated tensor shape, e.g. 1024,1024")
  parser.add_argument("--dtype", default="float32",
                      choices=["float32", "float16", "bfloat16", "double",
                               "int32", "int64"])
  parser.add_argument("--warmup", type=int, default=3,
                      help="Number of warmup iterations")
  parser.add_argument("--iters", type=int, default=10,
                      help="Number of timed iterations after warmup")
  args = parser.parse_args()

  load_musa_plugin()

  musa_devices = tf.config.list_physical_devices("MUSA")
  if not musa_devices:
    raise RuntimeError("No MUSA devices found.")

  shape = _parse_shape(args.shape)
  dtype = _dtype_from_name(args.dtype)
  x = _make_input(shape, dtype)

  print(f"Benchmarking tf.abs on MUSA: shape={shape}, dtype={dtype.name}")

  warmup_ms = []
  for _ in range(args.warmup):
    warmup_ms.append(_run_once(x))

  timed_ms = []
  for _ in range(args.iters):
    timed_ms.append(_run_once(x))

  print("Warmup iterations (ms):", ", ".join(f"{v:.3f}" for v in warmup_ms))
  print("Timed iterations (ms):", ", ".join(f"{v:.3f}" for v in timed_ms))
  print(
      "Steady-state summary (ms): "
      f"avg={np.mean(timed_ms):.3f}, min={np.min(timed_ms):.3f}, "
      f"max={np.max(timed_ms):.3f}"
  )


if __name__ == "__main__":
  main()
