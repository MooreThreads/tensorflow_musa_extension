#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare result_AMP.txt vs result_AMP_fix.txt

Assumption:
- Each file contains multiple result arrays.
- Each result array is stored in a bracket block like:
  [0.1 0.2 0.3 ...]
- Typically there should be 10 runs in each file.

What this script does:
1. Parse all bracketed arrays from both txt files
2. Check run count and shape consistency
3. Compare each paired run:
   - max abs error
   - mean abs error
   - RMSE
   - max/mean relative error
   - allclose / pass rate
4. Aggregate all runs
5. Give a simple conclusion on whether there is a large precision deviation
"""

import re
import sys
import argparse
from pathlib import Path

import numpy as np


def parse_result_file(path: str):
    """
    Parse all bracketed float arrays from a txt file.

    Supports content like:
    [0.1 0.2 0.3]
    [0.4 0.5 0.6]

    Returns:
        runs: list[np.ndarray]
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    # Match every [...] block, including multiline blocks
    blocks = re.findall(r"\[(.*?)\]", text, flags=re.S)
    if not blocks:
        raise ValueError(f"No bracketed result blocks found in file: {path}")

    runs = []
    for i, block in enumerate(blocks):
        # Normalize whitespace, then parse floats
        normalized = " ".join(block.replace("\n", " ").split())
        arr = np.fromstring(normalized, sep=" ", dtype=np.float64)

        if arr.size == 0:
            raise ValueError(f"Failed to parse block #{i} in file: {path}")

        runs.append(arr)

    return runs


def compare_two_arrays(ref: np.ndarray, test: np.ndarray, rtol=1e-5, atol=1e-7):
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {test.shape}")

    diff = test - ref
    abs_err = np.abs(diff)
    rel_err = abs_err / np.maximum(np.abs(ref), 1e-12)

    metrics = {
        "shape": ref.shape,
        "max_abs_err": float(abs_err.max()),
        "mean_abs_err": float(abs_err.mean()),
        "median_abs_err": float(np.median(abs_err)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_rel_err": float(rel_err.max()),
        "mean_rel_err": float(rel_err.mean()),
        "relative_l2": float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12)),
        "pass_rate": float(np.isclose(test, ref, rtol=rtol, atol=atol).mean()),
        "all_close": bool(np.allclose(test, ref, rtol=rtol, atol=atol)),
        "ref_mean": float(ref.mean()),
        "test_mean": float(test.mean()),
        "ref_std": float(ref.std()),
        "test_std": float(test.std()),
        "has_nan_ref": bool(np.isnan(ref).any()),
        "has_nan_test": bool(np.isnan(test).any()),
        "has_inf_ref": bool(np.isinf(ref).any()),
        "has_inf_test": bool(np.isinf(test).any()),
    }

    worst_idx = int(np.argmax(abs_err))
    metrics["worst_index_flat"] = worst_idx
    metrics["worst_ref"] = float(ref.reshape(-1)[worst_idx])
    metrics["worst_test"] = float(test.reshape(-1)[worst_idx])
    metrics["worst_abs_err"] = float(abs_err.reshape(-1)[worst_idx])
    metrics["worst_rel_err"] = float(rel_err.reshape(-1)[worst_idx])

    return metrics


def print_run_report(run_id: int, m: dict):
    print("=" * 90)
    print(f"Run #{run_id}")
    print("=" * 90)
    print(f"shape            : {m['shape']}")
    print(f"ref mean/std     : {m['ref_mean']:.10f} / {m['ref_std']:.10f}")
    print(f"test mean/std    : {m['test_mean']:.10f} / {m['test_std']:.10f}")
    print(f"max abs err      : {m['max_abs_err']:.10e}")
    print(f"mean abs err     : {m['mean_abs_err']:.10e}")
    print(f"median abs err   : {m['median_abs_err']:.10e}")
    print(f"rmse             : {m['rmse']:.10e}")
    print(f"max rel err      : {m['max_rel_err']:.10e}")
    print(f"mean rel err     : {m['mean_rel_err']:.10e}")
    print(f"relative L2      : {m['relative_l2']:.10e}")
    print(f"pass rate        : {m['pass_rate']:.6f}")
    print(f"all close        : {m['all_close']}")
    print(f"nan(ref/test)    : {m['has_nan_ref']} / {m['has_nan_test']}")
    print(f"inf(ref/test)    : {m['has_inf_ref']} / {m['has_inf_test']}")
    print(f"worst flat index : {m['worst_index_flat']}")
    print(f"worst ref/test   : {m['worst_ref']:.10f} / {m['worst_test']:.10f}")
    print(f"worst abs err    : {m['worst_abs_err']:.10e}")
    print(f"worst rel err    : {m['worst_rel_err']:.10e}")


def summarize_all(metrics_list):
    keys = [
        "max_abs_err", "mean_abs_err", "median_abs_err", "rmse",
        "max_rel_err", "mean_rel_err", "relative_l2", "pass_rate"
    ]

    print("\n" + "#" * 90)
    print("Aggregate summary across paired runs")
    print("#" * 90)

    for k in keys:
        vals = np.array([m[k] for m in metrics_list], dtype=np.float64)
        print(
            f"{k:16s}: "
            f"min={vals.min():.10e}, "
            f"mean={vals.mean():.10e}, "
            f"max={vals.max():.10e}"
        )

    all_allclose = all(m["all_close"] for m in metrics_list)
    print(f"\nall runs allclose : {all_allclose}")

    # Simple engineering conclusion
    max_abs = max(m["max_abs_err"] for m in metrics_list)
    max_rel_l2 = max(m["relative_l2"] for m in metrics_list)
    min_pass_rate = min(m["pass_rate"] for m in metrics_list)

    print("\nConclusion:")
    if max_abs < 1e-8 and max_rel_l2 < 1e-8:
        print("-> The two result files are essentially identical; no obvious precision deviation.")
    elif min_pass_rate == 1.0 and max_rel_l2 < 1e-5:
        print("-> The difference is extremely small; no large precision deviation.")
    elif min_pass_rate > 0.999 and max_rel_l2 < 1e-3:
        print("-> The difference is small and likely acceptable, but worth checking worst-case elements.")
    else:
        print("-> There may be noticeable precision deviation; inspect the per-run worst errors.")


def main():
    print("=============")
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp", type=str, default="result_AMP.txt",
                        help="Path to original AMP result txt")
    parser.add_argument("--fix", type=str, default="result_AMP_fix.txt",
                        help="Path to optimized AMP result txt")
    parser.add_argument("--rtol", type=float, default=1e-5,
                        help="Relative tolerance for allclose/isclose")
    parser.add_argument("--atol", type=float, default=1e-7,
                        help="Absolute tolerance for allclose/isclose")
    parser.add_argument("--expected-runs", type=int, default=10,
                        help="Expected number of runs in each file")
    args = parser.parse_args()

    amp_runs = parse_result_file(args.amp)
    fix_runs = parse_result_file(args.fix)

    print(f"Parsed {len(amp_runs)} runs from {args.amp}")
    print(f"Parsed {len(fix_runs)} runs from {args.fix}")

    if len(amp_runs) != len(fix_runs):
        raise ValueError(
            f"Run count mismatch: {len(amp_runs)} (AMP) vs {len(fix_runs)} (FIX)"
        )

    if args.expected_runs is not None:
        if len(amp_runs) != args.expected_runs:
            print(
                f"[Warning] Expected {args.expected_runs} runs, "
                f"but parsed {len(amp_runs)} from AMP file."
            )
        if len(fix_runs) != args.expected_runs:
            print(
                f"[Warning] Expected {args.expected_runs} runs, "
                f"but parsed {len(fix_runs)} from FIX file."
            )

    metrics_list = []
    for i, (ref, test) in enumerate(zip(amp_runs, fix_runs)):
        if ref.shape != test.shape:
            raise ValueError(
                f"Shape mismatch at run #{i}: {ref.shape} vs {test.shape}"
            )

        m = compare_two_arrays(ref, test, rtol=args.rtol, atol=args.atol)
        metrics_list.append(m)
        print_run_report(i, m)

    # Also compare all runs flattened together
    ref_all = np.concatenate([x.reshape(-1) for x in amp_runs], axis=0)
    test_all = np.concatenate([x.reshape(-1) for x in fix_runs], axis=0)
    all_metrics = compare_two_arrays(ref_all, test_all, rtol=args.rtol, atol=args.atol)

    print("\n" + "#" * 90)
    print("Global comparison over all runs concatenated")
    print("#" * 90)
    print_run_report(-1, all_metrics)

    summarize_all(metrics_list)


if __name__ == "__main__":
    main()
