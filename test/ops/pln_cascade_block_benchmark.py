#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PLUGIN_PATH = ROOT_DIR / "build" / "libmusa_plugin.so"


def parse_shape(text: str):
    return [int(v.strip()) for v in text.split(",") if v.strip()]


from typing import Optional

def resolve_plugin_path(cli_path: Optional[str]) -> Path:
    if cli_path:
        p = Path(cli_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"plugin not found: {p}")
        return p

    candidates = [
        DEFAULT_PLUGIN_PATH,
        ROOT_DIR.parent / "build" / "libmusa_plugin.so",
        Path.cwd() / "build" / "libmusa_plugin.so",
        Path.cwd().parent / "build" / "libmusa_plugin.so",
    ]
    for p in candidates:
        r = p.resolve()
        if r.exists():
            return r

    raise FileNotFoundError("libmusa_plugin.so not found in default locations")


def _broadcast_gate(gate: np.ndarray, out_shape):
    if gate.ndim == 1 and len(out_shape) >= 2 and gate.shape[0] == out_shape[0]:
        shape = [out_shape[0]] + [1] * (len(out_shape) - 1)
        gate = gate.reshape(shape)
    return np.broadcast_to(gate, out_shape)


def ref_pln_cascade_block(norm_out, add_table, bias_table, gates,
                          table_indices, select_on_true):
    out = norm_out.copy()
    out_shape = out.shape
    rank = len(out_shape)
    width = out_shape[-1]
    row_shape = [1] * (rank - 1) + [width]

    for step, gate in enumerate(gates):
        row_idx = table_indices[step]
        add_row = add_table[row_idx].reshape(row_shape)
        bias_row = bias_table[row_idx].reshape(row_shape)
        candidate = out * add_row + bias_row

        gate_b = _broadcast_gate(gate.astype(np.bool_), out_shape)
        take_candidate = gate_b if select_on_true[step] else np.logical_not(gate_b)
        out = np.where(take_candidate, candidate, out)

    return out


def build_inputs(shape, table_rows, steps, gate_mode, seed):
    rng = np.random.RandomState(seed)
    width = shape[-1]

    norm_out = rng.standard_normal(shape).astype(np.float32)
    add_table = (1.0 + 0.1 * rng.standard_normal((table_rows, width))).astype(np.float32)
    bias_table = (0.1 * rng.standard_normal((table_rows, width))).astype(np.float32)

    table_indices = [int((i * 3) % table_rows) for i in range(steps)]
    select_on_true = [bool(i % 2 == 0) for i in range(steps)]

    gates = []
    for _ in range(steps):
        if gate_mode == "batch":
            gate = (rng.random((shape[0],)) > 0.5).astype(np.bool_)
        else:
            gate = (rng.random(shape) > 0.5).astype(np.bool_)
        gates.append(gate)

    return norm_out, add_table, bias_table, gates, table_indices, select_on_true


def run(shape, table_rows, steps, gate_mode, warmup, iters, seed, plugin_path):
    op_module = tf.load_op_library(str(plugin_path))

    (norm_out, add_table, bias_table,
     gates, table_indices, select_on_true) = build_inputs(
        shape, table_rows, steps, gate_mode, seed)

    expected = ref_pln_cascade_block(
        norm_out, add_table, bias_table, gates, table_indices, select_on_true
    )

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            norm_t = tf.constant(norm_out)
            add_t = tf.constant(add_table)
            bias_t = tf.constant(bias_table)
            gate_ts = [tf.constant(g, dtype=tf.bool) for g in gates]

            out_t = op_module.musa_pln_cascade_block(
                norm_out=norm_t,
                add_input=add_t,
                bias_input=bias_t,
                gates=gate_ts,
                table_indices=table_indices,
                select_on_true=select_on_true,
            )
            out_t = tf.identity(out_t, name="benchmark_output")
            with tf.control_dependencies([out_t]):
                step_op = tf.no_op(name="benchmark_step")

    cfg = tf.ConfigProto()
    cfg.allow_soft_placement = True

    with tf.Session(graph=graph, config=cfg) as sess:
        got = sess.run(out_t)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

        for _ in range(warmup):
            sess.run(step_op)

        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            sess.run(step_op)
            ts.append((time.perf_counter() - t0) * 1000.0)

    result = {
        "shape": shape,
        "table_rows": table_rows,
        "steps": steps,
        "gate_mode": gate_mode,
        "warmup": warmup,
        "iters": iters,
        "mean_ms": float(np.mean(ts)),
        "p50_ms": float(np.percentile(ts, 50)),
        "p90_ms": float(np.percentile(ts, 90)),
        "min_ms": float(np.min(ts)),
        "max_ms": float(np.max(ts)),
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Benchmark MusaPlnCascadeBlock")
    parser.add_argument("--shape", type=str, default="100,36,1024")
    parser.add_argument("--table-rows", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--gate-mode", choices=["full", "batch"], default="full")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--musa-plugin", type=str, default="")
    args = parser.parse_args()

    shape = parse_shape(args.shape)
    if len(shape) < 2:
        raise ValueError("shape rank must be >= 2")
    if args.steps < 1 or args.steps > 16:
        raise ValueError("steps must be within [1, 16]")

    plugin_path = resolve_plugin_path(args.musa_plugin or None)
    run(shape, args.table_rows, args.steps, args.gate_mode,
        args.warmup, args.iters, args.seed, plugin_path)


if __name__ == "__main__":
    main()
