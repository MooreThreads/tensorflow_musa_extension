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

"""Extract real GELU entry shapes from a fused graph and benchmark them.

Typical workflow:
1. Run the whole model once with GELU fusion enabled and dump after_fusion pb.
2. Run this script with GELU fusion enabled to benchmark the fused path.
3. Re-run this script in a fresh process with `MUSA_DISABLE_GELU_FUSION=1`
   to benchmark the fallback path over the exact same shape set.
4. Compare the generated JSON result files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2

tf.disable_eager_execution()

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = ROOT_DIR.parent
DEFAULT_FUSED_GRAPH = ROOT_DIR / "gelu_big_graphs_pb" / "musa_optimizer_0002_after_fusion.pb"
DEFAULT_MODEL_GRAPH = WORKSPACE_DIR / "tf_test_model" / "prunedGraph" / "graph_def.pb"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "test" / "fusion" / "benchmark_results"
DEFAULT_PLUGIN_PATH = ROOT_DIR / "build" / "libmusa_plugin.so"


def is_truthy_env(value: Optional[str]) -> bool:
  if value is None:
    return False
  return value in ("1", "true", "TRUE", "yes", "YES", "on", "ON")


def clean_input_name(input_name: str) -> str:
  name = input_name[1:] if input_name.startswith("^") else input_name
  return name.split(":")[0]


def load_graph_def(path: Path) -> graph_pb2.GraphDef:
  graph_def = graph_pb2.GraphDef()
  with tf.io.gfile.GFile(str(path), "rb") as handle:
    graph_def.ParseFromString(handle.read())
  return graph_def


def infer_placeholder_shape_from_usage(
    graph_def: graph_pb2.GraphDef, placeholder_name: str
) -> Optional[List[int]]:
  for node in graph_def.node:
    for input_name in node.input:
      if clean_input_name(input_name) != placeholder_name:
        continue

      if node.op in ("MatMul", "Tensordot") and "_output_shapes" in node.attr:
        output_shapes = node.attr["_output_shapes"].list.shape
        if output_shapes and len(output_shapes[0].dim) == 2:
          return [64, 32]

      if node.op == "BiasAdd" and "_output_shapes" in node.attr:
        output_shapes = node.attr["_output_shapes"].list.shape
        if output_shapes and output_shapes[0].dim:
          return [output_shapes[0].dim[-1].size]

  return None


def load_placeholders(graph_def: graph_pb2.GraphDef) -> Dict[str, Dict[str, object]]:
  placeholders: Dict[str, Dict[str, object]] = {}
  dtype_map = {
      tf.float32.as_datatype_enum: np.float32,
      tf.int32.as_datatype_enum: np.int32,
      tf.int64.as_datatype_enum: np.int64,
      tf.bool.as_datatype_enum: np.bool_,
      tf.string.as_datatype_enum: np.str_,
  }

  for node in graph_def.node:
    if node.op != "Placeholder":
      continue

    dtype = dtype_map.get(node.attr["dtype"].type, np.float32)
    shape: List[Optional[int]] = []
    shape_found = False

    if "shape" in node.attr:
      shape_proto = node.attr["shape"].shape
      if not shape_proto.unknown_rank:
        for dim in shape_proto.dim:
          shape.append(dim.size if dim.size != -1 else None)
        shape_found = True

    if not shape_found and "_output_shapes" in node.attr:
      output_shapes = node.attr["_output_shapes"].list.shape
      if output_shapes and not output_shapes[0].unknown_rank:
        for dim in output_shapes[0].dim:
          shape.append(dim.size if dim.size != -1 else None)
        shape_found = True

    if not shape_found:
      inferred = infer_placeholder_shape_from_usage(graph_def, node.name)
      shape = inferred if inferred else []

    placeholders[node.name] = {"dtype": dtype, "shape": shape}

  return placeholders


def create_mock_data(
    placeholders: Dict[str, Dict[str, object]], batch_size: int, seed: int
) -> Dict[str, np.ndarray]:
  rng = np.random.RandomState(seed)
  feed_dict: Dict[str, np.ndarray] = {}

  for name, info in placeholders.items():
    shape_template = info["shape"]
    dtype = info["dtype"]
    mock_shape: List[int] = []

    for dim in shape_template:
      if dim is None:
        mock_shape.append(batch_size)
      elif dim == 0:
        mock_shape.append(0)
      else:
        mock_shape.append(dim)

    if not mock_shape:
      mock_shape = []

    if dtype == np.float32:
      mock_data = rng.normal(0.0, 1.0, mock_shape).astype(dtype)
    elif dtype == np.int32:
      mock_data = rng.randint(0, 100, mock_shape).astype(dtype)
    elif dtype == np.int64:
      mock_data = rng.randint(0, 100, mock_shape).astype(dtype)
    elif dtype == np.bool_:
      mock_data = rng.choice([True, False], mock_shape).astype(dtype)
    else:
      mock_data = rng.normal(0.0, 1.0, mock_shape).astype(np.float32)

    feed_dict[name + ":0"] = mock_data

  return feed_dict


def load_musa_plugin(plugin_path: Optional[str] = None) -> str:
  candidates: List[Path] = []
  if plugin_path:
    candidates.append(Path(plugin_path).expanduser())

  env_plugin = os.environ.get("MUSA_PLUGIN_PATH")
  if env_plugin:
    candidates.append(Path(env_plugin).expanduser())

  candidates.extend(
      [
          DEFAULT_PLUGIN_PATH,
          Path.cwd() / "build" / "libmusa_plugin.so",
          ROOT_DIR / "build" / "libmusa_plugin.so",
      ]
  )

  seen = set()
  for candidate in candidates:
    resolved = candidate.resolve()
    if resolved in seen:
      continue
    seen.add(resolved)
    if not resolved.exists():
      continue
    tf.load_op_library(str(resolved))
    return str(resolved)

  searched = "\n".join(f"  - {path}" for path in candidates)
  raise FileNotFoundError(
      "MUSA plugin not found. Searched locations:\n" + searched
  )


def create_config_with_musa_optimizer() -> config_pb2.ConfigProto:
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True
  rewriter_config = config.graph_options.rewrite_options
  custom_optimizer = rewriter_config.custom_optimizers.add()
  custom_optimizer.name = "musa_graph_optimizer"
  rewriter_config.min_graph_nodes = -1
  rewriter_config.optimizers.extend(["musa_graph_optimizer"])
  return config


def extract_gelu_entries(
    fused_graph_def: graph_pb2.GraphDef,
) -> List[Dict[str, object]]:
  entries: List[Dict[str, object]] = []
  for node in fused_graph_def.node:
    if node.op != "MusaGelu":
      continue
    entries.append(
        {
            "node_name": node.name,
            "input_name": clean_input_name(node.input[0]),
            "approximate": bool(node.attr["approximate"].b),
        }
    )

  if not entries:
    raise ValueError(
        "No MusaGelu nodes found in fused graph. Please pass an after_fusion "
        "graph dumped with GELU fusion enabled."
    )

  return entries


def resolve_runtime_shapes(
    model_graph_def: graph_pb2.GraphDef,
    input_names: Sequence[str],
    batch_size: int,
    seed: int,
) -> Dict[str, List[int]]:
  placeholders = load_placeholders(model_graph_def)
  feed_dict_np = create_mock_data(placeholders, batch_size=batch_size, seed=seed)

  runtime_shapes: Dict[str, List[int]] = {}
  unique_inputs = list(OrderedDict.fromkeys(input_names))

  with tf.Graph().as_default() as graph:
    tf.import_graph_def(model_graph_def, name="")

    session_feed_dict = {}
    for tensor_name, value in feed_dict_np.items():
      try:
        session_feed_dict[graph.get_tensor_by_name(tensor_name)] = value
      except KeyError:
        continue

    fetch_tensors = [graph.get_tensor_by_name(name + ":0") for name in unique_inputs]
    with tf.Session(graph=graph) as sess:
      outputs = sess.run(fetch_tensors, feed_dict=session_feed_dict)

  for name, value in zip(unique_inputs, outputs):
    runtime_shapes[name] = list(value.shape)

  return runtime_shapes


def build_case_specs(
    entries: Sequence[Dict[str, object]],
    runtime_shapes: Dict[str, List[int]],
) -> List[Dict[str, object]]:
  grouped: "OrderedDict[Tuple[Tuple[int, ...], bool], Dict[str, object]]" = OrderedDict()

  for entry in entries:
    input_name = entry["input_name"]
    if input_name not in runtime_shapes:
      raise KeyError(f"Missing runtime shape for GELU input '{input_name}'")

    shape = runtime_shapes[input_name]
    approximate = bool(entry["approximate"])
    key = (tuple(shape), approximate)

    if key not in grouped:
      grouped[key] = {
          "shape": shape,
          "approximate": approximate,
          "count": 0,
          "node_names": [],
          "input_names": [],
      }

    grouped[key]["count"] += 1
    grouped[key]["node_names"].append(entry["node_name"])
    grouped[key]["input_names"].append(input_name)

  return list(grouped.values())


def build_exact_gelu_graph(shape: Sequence[int], approximate: bool) -> Tuple[tf.Graph, tf.Tensor, tf.Tensor]:
  graph = tf.Graph()
  with graph.as_default():
    with tf.device("/device:MUSA:0"):
      x = tf.placeholder(tf.float32, shape=list(shape), name="gelu_input")

      if approximate:
        half_x = tf.math.multiply(x, tf.constant(0.5, dtype=tf.float32), name="half_x")
        x_pow3 = tf.math.pow(x, tf.constant(3.0, dtype=tf.float32), name="x_pow3")
        cubic_term = tf.math.multiply(
            tf.constant(0.044715, dtype=tf.float32), x_pow3, name="cubic_term"
        )
        inner = tf.math.add(x, cubic_term, name="approx_inner")
        scaled_inner = tf.math.multiply(
            tf.constant(0.7978845608, dtype=tf.float32),
            inner,
            name="approx_scaled_inner",
        )
        tanh = tf.math.tanh(scaled_inner, name="approx_tanh")
        one_plus_tanh = tf.math.add(
            tf.constant(1.0, dtype=tf.float32), tanh, name="one_plus_tanh"
        )
        y = tf.math.multiply(one_plus_tanh, half_x, name="gelu_output")
      else:
        rsqrt_two = tf.constant(np.array([0.70710678118], dtype=np.float32), name="rsqrt_two")
        div = tf.math.multiply(rsqrt_two, x, name="div_sqrt2")
        erf = tf.math.erf(div, name="erf")
        one_plus_erf = tf.math.add(
            erf, tf.constant(1.0, dtype=tf.float32), name="one_plus_erf"
        )
        half_factor = tf.math.multiply(
            tf.constant(0.5, dtype=tf.float32), one_plus_erf, name="half_factor"
        )
        y = tf.math.multiply(x, half_factor, name="gelu_output")

  return graph, x, y


def benchmark_case(
    shape: Sequence[int],
    approximate: bool,
    warmup_rounds: int,
    benchmark_rounds: int,
    seed: int,
) -> Dict[str, object]:
  graph, x, output = build_exact_gelu_graph(shape, approximate)
  rng = np.random.RandomState(seed)
  x_np = rng.standard_normal(shape).astype(np.float32)

  config = create_config_with_musa_optimizer()
  times: List[float] = []
  with tf.Session(graph=graph, config=config) as sess:
    for _ in range(warmup_rounds):
      sess.run(output, feed_dict={x: x_np})

    for _ in range(benchmark_rounds):
      start = time.perf_counter()
      sess.run(output, feed_dict={x: x_np})
      times.append(time.perf_counter() - start)

  avg_s = float(np.mean(times))
  min_s = float(np.min(times))
  max_s = float(np.max(times))
  std_s = float(np.std(times))
  num_elements = int(np.prod(shape)) if shape else 1
  sample_count = shape[0] if shape else 1

  return {
      "shape": list(shape),
      "approximate": approximate,
      "warmup_rounds": warmup_rounds,
      "benchmark_rounds": benchmark_rounds,
      "average_s": avg_s,
      "min_s": min_s,
      "max_s": max_s,
      "std_s": std_s,
      "samples_per_s": float(sample_count / avg_s) if avg_s > 0 else 0.0,
      "elements_per_s": float(num_elements / avg_s) if avg_s > 0 else 0.0,
  }


def summarize_results(
    case_specs: Sequence[Dict[str, object]], fusion_enabled: bool
) -> Dict[str, object]:
  summary_cases: List[Dict[str, object]] = []
  weighted_total_s = 0.0
  total_invocations = 0

  for spec in case_specs:
    result = spec["benchmark"]
    count = spec["count"]
    weighted_total_s += result["average_s"] * count
    total_invocations += count

    summary_cases.append(
        {
            "shape": spec["shape"],
            "approximate": spec["approximate"],
            "count": count,
            "average_ms": result["average_s"] * 1000.0,
            "min_ms": result["min_s"] * 1000.0,
            "max_ms": result["max_s"] * 1000.0,
            "std_ms": result["std_s"] * 1000.0,
            "samples_per_s": result["samples_per_s"],
            "elements_per_s": result["elements_per_s"],
            "node_names": spec["node_names"],
            "input_names": spec["input_names"],
        }
    )

  return {
      "fusion_enabled": fusion_enabled,
      "fusion_mode": "fusion_on" if fusion_enabled else "fusion_off",
      "estimated_total_gelu_ms_per_step": weighted_total_s * 1000.0,
      "estimated_average_gelu_ms_per_invocation": (
          weighted_total_s * 1000.0 / total_invocations if total_invocations else 0.0
      ),
      "total_gelu_invocations": total_invocations,
      "cases": summary_cases,
  }


def save_results(payload: Dict[str, object], output_dir: Path) -> Path:
  output_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
  mode = payload["fusion_mode"]
  output_path = output_dir / f"gelu_fusion_benchmark_{mode}_{timestamp}.json"
  with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
  return output_path


def print_case_specs(case_specs: Sequence[Dict[str, object]]) -> None:
  print("\nExtracted GELU benchmark cases:")
  for idx, spec in enumerate(case_specs, 1):
    print(
        f"  [{idx}] shape={spec['shape']} approximate={spec['approximate']} "
        f"count={spec['count']}"
    )
    print(f"      sample nodes: {spec['node_names'][:3]}")


def print_benchmark_summary(summary: Dict[str, object]) -> None:
  print("\nGELU benchmark summary:")
  print(f"  fusion_mode: {summary['fusion_mode']}")
  print(f"  total_gelu_invocations: {summary['total_gelu_invocations']}")
  print(
      "  estimated_total_gelu_ms_per_step: "
      f"{summary['estimated_total_gelu_ms_per_step']:.4f}"
  )
  print(
      "  estimated_average_gelu_ms_per_invocation: "
      f"{summary['estimated_average_gelu_ms_per_invocation']:.4f}"
  )
  print("  cases:")
  for case in summary["cases"]:
    print(
        f"    shape={case['shape']} approximate={case['approximate']} "
        f"count={case['count']} avg_ms={case['average_ms']:.4f} "
        f"elements/s={case['elements_per_s']:.2f}"
    )


def compare_payloads(lhs: Dict[str, object], rhs: Dict[str, object]) -> None:
  payloads = {lhs["fusion_mode"]: lhs, rhs["fusion_mode"]: rhs}
  if "fusion_on" not in payloads or "fusion_off" not in payloads:
    raise ValueError("Comparison expects one fusion_on result and one fusion_off result")

  fusion_on = payloads["fusion_on"]
  fusion_off = payloads["fusion_off"]

  on_total = fusion_on["estimated_total_gelu_ms_per_step"]
  off_total = fusion_off["estimated_total_gelu_ms_per_step"]
  delta_pct = ((off_total - on_total) / off_total * 100.0) if off_total else 0.0

  print("\nGELU fusion comparison:")
  print(f"  fusion_on total gelu ms/step:  {on_total:.4f}")
  print(f"  fusion_off total gelu ms/step: {off_total:.4f}")
  print(f"  relative improvement:          {delta_pct:.2f}%")

  on_cases = {
      (tuple(case["shape"]), case["approximate"], case["count"]): case
      for case in fusion_on["cases"]
  }
  off_cases = {
      (tuple(case["shape"]), case["approximate"], case["count"]): case
      for case in fusion_off["cases"]
  }

  print("  per-case:")
  for key in sorted(on_cases):
    if key not in off_cases:
      continue
    on_case = on_cases[key]
    off_case = off_cases[key]
    off_ms = off_case["average_ms"]
    on_ms = on_case["average_ms"]
    case_delta = ((off_ms - on_ms) / off_ms * 100.0) if off_ms else 0.0
    print(
        f"    shape={list(key[0])} approximate={key[1]} count={key[2]} "
        f"on_ms={on_ms:.4f} off_ms={off_ms:.4f} improvement={case_delta:.2f}%"
    )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Benchmark GELU fusion using shapes extracted from a whole-model dump."
  )
  parser.add_argument(
      "--fused-graph",
      default=str(DEFAULT_FUSED_GRAPH),
      help="after_fusion graph containing MusaGelu nodes (typically fusion-on dump).",
  )
  parser.add_argument(
      "--model-graph",
      default=str(DEFAULT_MODEL_GRAPH),
      help="Original whole-model GraphDef used to resolve runtime GELU input shapes.",
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=100,
      help="Concrete batch size used when materializing dynamic dimensions.",
  )
  parser.add_argument(
      "--warmup-rounds",
      type=int,
      default=10,
      help="Warmup iterations per GELU benchmark case.",
  )
  parser.add_argument(
      "--benchmark-rounds",
      type=int,
      default=50,
      help="Measured iterations per GELU benchmark case.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for mock inputs and benchmark tensors.",
  )
  parser.add_argument(
      "--output-dir",
      default=str(DEFAULT_OUTPUT_DIR),
      help="Directory to save benchmark JSON outputs.",
  )
  parser.add_argument(
      "--plugin-path",
      default=None,
      help="Optional explicit path to libmusa_plugin.so.",
  )
  parser.add_argument(
      "--extract-only",
      action="store_true",
      help="Only extract the whole-model GELU cases; skip benchmarking.",
  )
  parser.add_argument(
      "--compare-json",
      nargs=2,
      metavar=("FUSION_ON_JSON", "FUSION_OFF_JSON"),
      help="Compare two previously generated benchmark JSON files.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  if args.compare_json:
    lhs = json.load(open(args.compare_json[0], "r", encoding="utf-8"))
    rhs = json.load(open(args.compare_json[1], "r", encoding="utf-8"))
    compare_payloads(lhs, rhs)
    return

  fused_graph_path = Path(args.fused_graph).expanduser().resolve()
  model_graph_path = Path(args.model_graph).expanduser().resolve()

  print(f"Using fused graph: {fused_graph_path}")
  print(f"Using model graph: {model_graph_path}")
  print(f"Batch size for runtime shape materialization: {args.batch_size}")
  print(
      "Fusion mode for this benchmark process: "
      + ("fusion_off" if is_truthy_env(os.environ.get("MUSA_DISABLE_GELU_FUSION")) else "fusion_on")
  )

  fused_graph_def = load_graph_def(fused_graph_path)
  model_graph_def = load_graph_def(model_graph_path)
  gelu_entries = extract_gelu_entries(fused_graph_def)
  runtime_shapes = resolve_runtime_shapes(
      model_graph_def,
      [entry["input_name"] for entry in gelu_entries],
      batch_size=args.batch_size,
      seed=args.seed,
  )
  case_specs = build_case_specs(gelu_entries, runtime_shapes)
  print_case_specs(case_specs)

  if args.extract_only:
    return

  plugin_path = load_musa_plugin(args.plugin_path)
  print(f"Loaded MUSA plugin from: {plugin_path}")

  physical_devices = tf.config.list_physical_devices("MUSA")
  if not physical_devices:
    print(
        "Warning: tf.config.list_physical_devices('MUSA') returned no devices. "
        "Continuing anyway because the plugin-backed session may still execute "
        "on /device:MUSA:0 in this environment."
    )

  for idx, spec in enumerate(case_specs):
    print(
        f"Benchmarking case {idx + 1}/{len(case_specs)}: "
        f"shape={spec['shape']} approximate={spec['approximate']} count={spec['count']}"
    )
    spec["benchmark"] = benchmark_case(
        shape=spec["shape"],
        approximate=spec["approximate"],
        warmup_rounds=args.warmup_rounds,
        benchmark_rounds=args.benchmark_rounds,
        seed=args.seed + idx,
    )

  summary = summarize_results(
      case_specs,
      fusion_enabled=not is_truthy_env(os.environ.get("MUSA_DISABLE_GELU_FUSION")),
  )
  print_benchmark_summary(summary)

  output_path = save_results(summary, Path(args.output_dir).expanduser().resolve())
  print(f"\nSaved benchmark summary to: {output_path}")


if __name__ == "__main__":
  main()
