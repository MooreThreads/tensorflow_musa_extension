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
"""Benchmark MusaShiftedAffineMap against the unfused elementwise chain.

This script compares three execution modes on MUSA:

1. primitive_chain:
   add(data_left, sliced_var_left) -> mul(mask) -> add(sliced_var_right)
   with the MUSA graph optimizer disabled.

2. fused_graph:
   the exact same graph, but with musa_graph_optimizer enabled so the pattern
   can be rewritten into MusaShiftedAffineMap.

3. custom_op:
   directly calls musa_shifted_affine_map with the same surrounding
   StridedSlice/Select inputs.

`primitive_chain` is a practical approximation of the older multi-pass path.
`fused_graph` and `custom_op` represent the current single-op path.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2

tf.disable_eager_execution()

DTYPE_MAP = {
    "float16": tf.float16,
    "float32": tf.float32,
    "float64": tf.float64,
    "bfloat16": tf.bfloat16,
}


def parse_shape(shape_text: str) -> List[int]:
    cleaned = shape_text.strip()
    if not cleaned:
        return []
    return [int(part.strip()) for part in cleaned.split(",") if part.strip()]


def numpy_dtype_for_tf(dtype: tf.DType) -> np.dtype:
    if dtype == tf.bfloat16:
        return np.float32
    return dtype.as_numpy_dtype


def load_musa_ops():
    import tensorflow_musa

    tensorflow_musa.load_plugin()
    return tensorflow_musa.get_musa_ops()


def create_config(enable_musa_optimizer: bool) -> config_pb2.ConfigProto:
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rw = config.graph_options.rewrite_options
    rw.min_graph_nodes = -1
    rw.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
    rw.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rw.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rw.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rw.shape_optimization = rewriter_config_pb2.RewriterConfig.OFF

    if enable_musa_optimizer:
        custom_opt = rw.custom_optimizers.add()
        custom_opt.name = "musa_graph_optimizer"
        rw.optimizers.extend(["musa_graph_optimizer"])

    return config


def has_fused_op(partition_graphs, op_name: str = "MusaShiftedAffineMap") -> bool:
    return any(node.op == op_name for pg in partition_graphs for node in pg.node)


def summarize_ops(partition_graphs) -> List[str]:
    return sorted({node.op for pg in partition_graphs for node in pg.node})


def make_inputs(dtype: tf.DType,
                data_shape: List[int],
                left_shape: List[int],
                mask_shape: List[int],
                right_shape: List[int],
                seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    np_dtype = numpy_dtype_for_tf(dtype)

    data_left = rng.standard_normal(data_shape).astype(np_dtype)
    sliced_var_left = (rng.standard_normal(left_shape) * 0.1).astype(np_dtype)
    mask_bool = rng.random(mask_shape) > 0.5
    sliced_var_right = (rng.standard_normal(right_shape) * 0.1).astype(np_dtype)

    return {
        "data_left": data_left,
        "sliced_var_left": sliced_var_left,
        "mask_bool": mask_bool.astype(np.bool_),
        "sliced_var_right": sliced_var_right,
    }


def expected_output(inputs: Dict[str, np.ndarray], dtype: tf.DType) -> np.ndarray:
    np_dtype = numpy_dtype_for_tf(dtype)
    mask = inputs["mask_bool"].astype(np_dtype)
    return (
        mask
        * (inputs["data_left"] + inputs["sliced_var_left"])
        + inputs["sliced_var_right"]
    )


def create_variable(name: str, value: np.ndarray, dtype: tf.DType) -> tf.Tensor:
    return tf.Variable(tf.constant(value, dtype=dtype), name=name)


def build_common_inputs(inputs: Dict[str, np.ndarray],
                        dtype: tf.DType) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    data_left = create_variable("data_left", inputs["data_left"], dtype)

    var_left = create_variable("var_left", inputs["sliced_var_left"], dtype)
    sliced_var_left = tf.strided_slice(
        var_left,
        [0] * len(inputs["sliced_var_left"].shape),
        list(inputs["sliced_var_left"].shape),
        [1] * len(inputs["sliced_var_left"].shape),
        name="strided_slice_left",
    )

    var_right = create_variable("var_right", inputs["sliced_var_right"], dtype)
    sliced_var_right = tf.strided_slice(
        var_right,
        [0] * len(inputs["sliced_var_right"].shape),
        list(inputs["sliced_var_right"].shape),
        [1] * len(inputs["sliced_var_right"].shape),
        name="strided_slice_right",
    )

    mask_cond = tf.Variable(
        tf.constant(inputs["mask_bool"], dtype=tf.bool), name="mask_cond")
    ones = tf.ones(inputs["mask_bool"].shape, dtype=dtype)
    zeros = tf.zeros(inputs["mask_bool"].shape, dtype=dtype)
    mask = tf.where(mask_cond, ones, zeros, name="mask_select")

    return data_left, sliced_var_left, mask, sliced_var_right


def build_primitive_graph(inputs: Dict[str, np.ndarray], dtype: tf.DType) -> Tuple[tf.Graph, tf.Tensor, tf.Operation]:
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            data_left, sliced_var_left, mask, sliced_var_right = (
                build_common_inputs(inputs, dtype)
            )

            add_left = tf.math.add(data_left, sliced_var_left, name="add_left")
            mul_gated = tf.math.multiply(add_left, mask, name="mul_gated")
            output = tf.math.add(mul_gated, sliced_var_right, name="output")
            benchmark_output = tf.identity(output, name="benchmark_output")
            with tf.control_dependencies([benchmark_output]):
                benchmark_op = tf.no_op(name="benchmark_step")

    return graph, benchmark_output, benchmark_op


def build_custom_op_graph(musa_ops,
                          inputs: Dict[str, np.ndarray],
                          dtype: tf.DType) -> Tuple[tf.Graph, tf.Tensor, tf.Operation]:
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            data_left, sliced_var_left, mask, sliced_var_right = (
                build_common_inputs(inputs, dtype)
            )

            output = musa_ops.musa_shifted_affine_map(
                data_left=data_left,
                sliced_var_left=sliced_var_left,
                mask=mask,
                sliced_var_right=sliced_var_right,
                name="musa_shifted_affine_map_benchmark",
            )
            benchmark_output = tf.identity(output, name="benchmark_output")
            with tf.control_dependencies([benchmark_output]):
                benchmark_op = tf.no_op(name="benchmark_step")

    return graph, benchmark_output, benchmark_op


def fetch_tensor_for_validation(output: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    if dtype in (tf.float16, tf.bfloat16):
        return tf.cast(output, tf.float32, name="validation_output")
    return output


def run_benchmark_mode(name: str,
                       graph: tf.Graph,
                       output_tensor: tf.Tensor,
                       benchmark_op: tf.Operation,
                       config: config_pb2.ConfigProto,
                       dtype: tf.DType,
                       warmup_iters: int,
                       measure_iters: int,
                       capture_partition_graph: bool = False) -> Dict[str, object]:
    validation_tensor = fetch_tensor_for_validation(output_tensor, dtype)
    run_meta = tf.compat.v1.RunMetadata() if capture_partition_graph else None
    run_opts = (
        tf.compat.v1.RunOptions(output_partition_graphs=True)
        if capture_partition_graph else None
    )

    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        validation_output = sess.run(
            validation_tensor, options=run_opts, run_metadata=run_meta)

        for _ in range(warmup_iters):
            sess.run(benchmark_op)

        times = []
        for _ in range(measure_iters):
            start = time.perf_counter()
            sess.run(benchmark_op)
            times.append(time.perf_counter() - start)

    timings_ms = [elapsed * 1000.0 for elapsed in times]
    result = {
        "mode": name,
        "avg_ms": float(np.mean(timings_ms)),
        "min_ms": float(np.min(timings_ms)),
        "max_ms": float(np.max(timings_ms)),
        "p50_ms": float(np.percentile(timings_ms, 50)),
        "p90_ms": float(np.percentile(timings_ms, 90)),
        "measure_iters": measure_iters,
        "warmup_iters": warmup_iters,
        "validation_output": validation_output,
    }

    if capture_partition_graph and run_meta is not None:
        result["fused"] = has_fused_op(run_meta.partition_graphs)
        result["ops"] = summarize_ops(run_meta.partition_graphs)

    return result


def print_mode_summary(result: Dict[str, object]) -> None:
    fused_suffix = ""
    if "fused" in result:
        fused_suffix = f", fused={result['fused']}"

    print(
        f"{result['mode']:>16}: "
        f"avg={result['avg_ms']:.3f} ms, "
        f"p50={result['p50_ms']:.3f} ms, "
        f"p90={result['p90_ms']:.3f} ms, "
        f"min={result['min_ms']:.3f} ms, "
        f"max={result['max_ms']:.3f} ms"
        f"{fused_suffix}"
    )


def save_json_summary(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ShiftedAffineMap primitive/fused/custom paths."
    )
    parser.add_argument("--data-shape", default="16,32,64")
    parser.add_argument("--left-shape", default="64")
    parser.add_argument("--mask-shape", default="")
    parser.add_argument("--right-shape", default="64")
    parser.add_argument("--dtype", default="float32", choices=sorted(DTYPE_MAP))
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measure-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    musa_ops = load_musa_ops()
    if not tf.config.list_physical_devices("MUSA"):
        raise RuntimeError("No MUSA devices found.")

    dtype = DTYPE_MAP[args.dtype]
    data_shape = parse_shape(args.data_shape)
    left_shape = parse_shape(args.left_shape)
    right_shape = parse_shape(args.right_shape)
    mask_shape = parse_shape(args.mask_shape) if args.mask_shape else list(data_shape)

    inputs = make_inputs(
        dtype=dtype,
        data_shape=data_shape,
        left_shape=left_shape,
        mask_shape=mask_shape,
        right_shape=right_shape,
        seed=args.seed,
    )

    expected = expected_output(inputs, dtype)
    if dtype in (tf.float16, tf.bfloat16):
        expected = expected.astype(np.float32)

    primitive_graph, primitive_output, primitive_step = build_primitive_graph(
        inputs, dtype)
    fused_graph, fused_output, fused_step = build_primitive_graph(inputs, dtype)
    custom_graph, custom_output, custom_step = build_custom_op_graph(
        musa_ops, inputs, dtype)

    primitive_result = run_benchmark_mode(
        "primitive_chain",
        primitive_graph,
        primitive_output,
        primitive_step,
        create_config(enable_musa_optimizer=False),
        dtype,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        capture_partition_graph=False,
    )
    fused_result = run_benchmark_mode(
        "fused_graph",
        fused_graph,
        fused_output,
        fused_step,
        create_config(enable_musa_optimizer=True),
        dtype,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        capture_partition_graph=True,
    )
    custom_result = run_benchmark_mode(
        "custom_op",
        custom_graph,
        custom_output,
        custom_step,
        create_config(enable_musa_optimizer=False),
        dtype,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        capture_partition_graph=False,
    )

    np.testing.assert_allclose(
        primitive_result["validation_output"],
        expected,
        rtol=5e-3 if dtype in (tf.float16, tf.bfloat16) else 1e-5,
        atol=5e-3 if dtype in (tf.float16, tf.bfloat16) else 1e-6,
    )
    np.testing.assert_allclose(
        fused_result["validation_output"],
        expected,
        rtol=5e-3 if dtype in (tf.float16, tf.bfloat16) else 1e-5,
        atol=5e-3 if dtype in (tf.float16, tf.bfloat16) else 1e-6,
    )
    np.testing.assert_allclose(
        custom_result["validation_output"],
        expected,
        rtol=5e-3 if dtype in (tf.float16, tf.bfloat16) else 1e-5,
        atol=5e-3 if dtype in (tf.float16, tf.bfloat16) else 1e-6,
    )

    if not fused_result.get("fused", False):
        raise AssertionError(
            "fused_graph mode did not produce MusaShiftedAffineMap. "
            f"Observed ops: {fused_result.get('ops', [])}"
        )

    print("\nShiftedAffineMap benchmark")
    print(f"  dtype={args.dtype}")
    print(f"  data_shape={data_shape}")
    print(f"  left_shape={left_shape}")
    print(f"  mask_shape={mask_shape}")
    print(f"  right_shape={right_shape}")
    if args.label:
        print(f"  label={args.label}")
    print()

    print_mode_summary(primitive_result)
    print_mode_summary(fused_result)
    print_mode_summary(custom_result)

    fused_speedup = primitive_result["avg_ms"] / fused_result["avg_ms"]
    custom_speedup = primitive_result["avg_ms"] / custom_result["avg_ms"]
    fused_vs_custom = fused_result["avg_ms"] / custom_result["avg_ms"]

    print("\nSpeedups")
    print(f"  fused_graph vs primitive_chain : {fused_speedup:.3f}x")
    print(f"  custom_op  vs primitive_chain : {custom_speedup:.3f}x")
    print(f"  fused_graph vs custom_op      : {fused_vs_custom:.3f}x")

    output_payload = {
        "timestamp": datetime.now().isoformat(),
        "label": args.label,
        "dtype": args.dtype,
        "data_shape": data_shape,
        "left_shape": left_shape,
        "mask_shape": mask_shape,
        "right_shape": right_shape,
        "results": {
            "primitive_chain": {
                key: value for key, value in primitive_result.items()
                if key != "validation_output"
            },
            "fused_graph": {
                key: value for key, value in fused_result.items()
                if key != "validation_output"
            },
            "custom_op": {
                key: value for key, value in custom_result.items()
                if key != "validation_output"
            },
        },
        "speedups": {
            "fused_graph_vs_primitive_chain": fused_speedup,
            "custom_op_vs_primitive_chain": custom_speedup,
            "fused_graph_vs_custom_op": fused_vs_custom,
        },
    }

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        save_json_summary(output_path, output_payload)
        print(f"\nSaved benchmark summary to: {output_path}")


if __name__ == "__main__":
    main()
