#!/usr/bin/env python3
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

"""Benchmark TensorFlow InTopKV2 on MUSA.

Default mode builds an op-only graph with predictions and targets resident on
MUSA so timings focus on InTopKV2 instead of host-to-device copies. The optional
`dense` mode builds a small classifier-like graph and reports end-to-end latency
for MatMul + BiasAdd + InTopKV2.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2


tf.disable_eager_execution()

TARGET_DTYPE_MAP = {
    "int32": tf.int32,
    "int64": tf.int64,
}


class InTopKV2Model:
  def __init__(self,
               batch_size: int,
               num_classes: int,
               k: int,
               target_dtype: tf.DType,
               mode: str,
               hidden_size: int,
               seed: int):
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.k = k
    self.target_dtype = target_dtype
    self.mode = mode
    self.hidden_size = hidden_size
    self.seed = seed

  def build(self, device: str) -> Dict[str, object]:
    graph = tf.Graph()
    rng = np.random.RandomState(self.seed)
    target_np = rng.randint(
        0, self.num_classes, size=self.batch_size).astype(
            self.target_dtype.as_numpy_dtype)

    with graph.as_default():
      with tf.device(device):
        targets = tf.Variable(
            tf.constant(target_np, dtype=self.target_dtype), name="targets")
        k = tf.constant(self.k, dtype=self.target_dtype, name="k")

        if self.mode == "dense":
          features_np = rng.standard_normal(
              [self.batch_size, self.hidden_size]).astype(np.float32)
          weights_np = rng.standard_normal(
              [self.hidden_size, self.num_classes]).astype(np.float32)
          bias_np = rng.standard_normal([self.num_classes]).astype(np.float32)
          features = tf.Variable(tf.constant(features_np), name="features")
          weights = tf.Variable(tf.constant(weights_np), name="weights")
          bias = tf.Variable(tf.constant(bias_np), name="bias")
          predictions = tf.nn.bias_add(
              tf.matmul(features, weights, name="dense_matmul"),
              bias,
              name="logits")
        else:
          predictions_np = rng.standard_normal(
              [self.batch_size, self.num_classes]).astype(np.float32)
          predictions = tf.Variable(
              tf.constant(predictions_np), name="predictions")

        output = tf.raw_ops.InTopKV2(
            predictions=predictions, targets=targets, k=k, name="intopkv2")
        true_count = tf.reduce_sum(tf.cast(output, tf.int32), name="true_count")
        with tf.control_dependencies([true_count]):
          benchmark_op = tf.no_op(name="benchmark_step")

      init_op = tf.global_variables_initializer()

    return {
        "graph": graph,
        "init_op": init_op,
        "output": output,
        "true_count": true_count,
        "benchmark_op": benchmark_op,
    }


def load_musa_plugin() -> None:
  import tensorflow_musa

  tensorflow_musa.load_plugin()


def create_config() -> config_pb2.ConfigProto:
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = False
  rw = config.graph_options.rewrite_options
  rw.min_graph_nodes = -1
  rw.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
  rw.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
  rw.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
  rw.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
  rw.shape_optimization = rewriter_config_pb2.RewriterConfig.OFF
  return config


def percentile(values: List[float], q: float) -> float:
  return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_one_case(batch_size: int,
                 num_classes: int,
                 k: int,
                 target_dtype: tf.DType,
                 mode: str,
                 hidden_size: int,
                 warmup_iters: int,
                 measure_iters: int,
                 seed: int,
                 validate_cpu: bool) -> Dict[str, object]:
  if k < 0 or k > num_classes:
    raise ValueError(f"k must be in [0, num_classes], got k={k}, num_classes={num_classes}")

  musa_model = InTopKV2Model(
      batch_size=batch_size,
      num_classes=num_classes,
      k=k,
      target_dtype=target_dtype,
      mode=mode,
      hidden_size=hidden_size,
      seed=seed)
  musa_graph = musa_model.build("/device:MUSA:0")

  config = create_config()
  with tf.Session(graph=musa_graph["graph"], config=config) as sess:
    sess.run(musa_graph["init_op"])
    true_count = int(sess.run(musa_graph["true_count"]))

    for _ in range(warmup_iters):
      sess.run(musa_graph["benchmark_op"])

    timings_ms = []
    for _ in range(measure_iters):
      start = time.perf_counter()
      sess.run(musa_graph["benchmark_op"])
      timings_ms.append((time.perf_counter() - start) * 1000.0)

  result = {
      "mode": mode,
      "batch_size": batch_size,
      "num_classes": num_classes,
      "k": k,
      "target_dtype": target_dtype.name,
      "hidden_size": hidden_size if mode == "dense" else None,
      "warmup_iters": warmup_iters,
      "measure_iters": measure_iters,
      "true_count": true_count,
      "avg_ms": float(np.mean(timings_ms)),
      "min_ms": float(np.min(timings_ms)),
      "max_ms": float(np.max(timings_ms)),
      "p50_ms": percentile(timings_ms, 50),
      "p90_ms": percentile(timings_ms, 90),
      "p99_ms": percentile(timings_ms, 99),
      "items_per_second": float(batch_size / (np.mean(timings_ms) / 1000.0)),
  }

  if validate_cpu:
    cpu_model = InTopKV2Model(
        batch_size=batch_size,
        num_classes=num_classes,
        k=k,
        target_dtype=target_dtype,
        mode=mode,
        hidden_size=hidden_size,
        seed=seed)
    cpu_graph = cpu_model.build("/CPU:0")
    with tf.Session(graph=cpu_graph["graph"]) as sess:
      sess.run(cpu_graph["init_op"])
      cpu_output = sess.run(cpu_graph["output"])

    musa_model = InTopKV2Model(
        batch_size=batch_size,
        num_classes=num_classes,
        k=k,
        target_dtype=target_dtype,
        mode=mode,
        hidden_size=hidden_size,
        seed=seed)
    musa_graph = musa_model.build("/device:MUSA:0")
    with tf.Session(graph=musa_graph["graph"], config=config) as sess:
      sess.run(musa_graph["init_op"])
      musa_output = sess.run(musa_graph["output"])

    result["cpu_validation_passed"] = bool(np.array_equal(cpu_output, musa_output))

  return result


def parse_csv_ints(value: str) -> List[int]:
  return [int(part.strip()) for part in value.split(",") if part.strip()]


def print_result(result: Dict[str, object]) -> None:
  hidden = "" if result["hidden_size"] is None else f" hidden={result['hidden_size']}"
  validation = ""
  if "cpu_validation_passed" in result:
    validation = f" cpu_valid={result['cpu_validation_passed']}"
  print(
      f"mode={result['mode']} batch={result['batch_size']} "
      f"classes={result['num_classes']} k={result['k']} "
      f"target={result['target_dtype']}{hidden} "
      f"avg={result['avg_ms']:.4f} ms min={result['min_ms']:.4f} ms "
      f"p50={result['p50_ms']:.4f} ms p90={result['p90_ms']:.4f} ms "
      f"p99={result['p99_ms']:.4f} ms throughput={result['items_per_second']:.2f} items/s"
      f" true_count={result['true_count']}{validation}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=["op", "dense"], default="op")
  parser.add_argument("--batch-sizes", default="32,128,1024")
  parser.add_argument("--num-classes", default="1000,10000")
  parser.add_argument("--k-values", default="1,5,50")
  parser.add_argument("--target-dtype", choices=sorted(TARGET_DTYPE_MAP), default="int32")
  parser.add_argument("--hidden-size", type=int, default=1024)
  parser.add_argument("--warmup-iters", type=int, default=50)
  parser.add_argument("--measure-iters", type=int, default=200)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--validate-cpu", action="store_true")
  parser.add_argument("--output-json", type=Path, default=None)
  args = parser.parse_args()

  load_musa_plugin()

  target_dtype = TARGET_DTYPE_MAP[args.target_dtype]
  results = []
  for batch_size in parse_csv_ints(args.batch_sizes):
    for num_classes in parse_csv_ints(args.num_classes):
      for k in parse_csv_ints(args.k_values):
        if k > num_classes:
          continue
        result = run_one_case(
            batch_size=batch_size,
            num_classes=num_classes,
            k=k,
            target_dtype=target_dtype,
            mode=args.mode,
            hidden_size=args.hidden_size,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            seed=args.seed,
            validate_cpu=args.validate_cpu)
        print_result(result)
        results.append(result)

  payload = {
      "created_at": datetime.now().isoformat(timespec="seconds"),
      "benchmark": "intopkv2",
      "results": results,
  }
  if args.output_json:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
  main()
