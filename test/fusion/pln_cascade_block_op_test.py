"""Unit tests for MusaPlnCascadeBlock op correctness."""

import numpy as np
import tensorflow as tf
from pathlib import Path

from musa_test_utils import MUSATestCase


def _broadcast_gate(gate: np.ndarray, out_shape):
    if gate.ndim == 1 and len(out_shape) >= 2 and gate.shape[0] == out_shape[0]:
        gate = gate.reshape([out_shape[0]] + [1] * (len(out_shape) - 1))
    return np.broadcast_to(gate, out_shape)


def _ref_pln_cascade_block(norm_out, add_table, bias_table, gates,
                           table_indices, select_on_true):
    out = norm_out.copy()
    rank = len(out.shape)
    width = out.shape[-1]
    row_shape = [1] * (rank - 1) + [width]

    for i, gate in enumerate(gates):
        row = table_indices[i]
        add_row = add_table[row].reshape(row_shape)
        bias_row = bias_table[row].reshape(row_shape)
        candidate = out * add_row + bias_row

        gate_mask = _broadcast_gate(gate.astype(np.bool_), out.shape)
        choose_candidate = gate_mask if select_on_true[i] else np.logical_not(gate_mask)
        out = np.where(choose_candidate, candidate, out)

    return out


class PlnCascadeBlockOpTest(MUSATestCase):
    def _run_case(self, norm_out_np, add_table_np, bias_table_np,
                  gates_np, table_indices, select_on_true):
        plugin_path = Path(__file__).resolve().parents[2] / "build" / "libmusa_plugin.so"
        if not plugin_path.exists():
            self.skipTest(f"plugin not found: {plugin_path}")
        op_module = tf.load_op_library(str(plugin_path))
        if not hasattr(op_module, "musa_pln_cascade_block"):
            self.skipTest("musa_pln_cascade_block is unavailable")

        expected = _ref_pln_cascade_block(
            norm_out_np, add_table_np, bias_table_np,
            gates_np, table_indices, select_on_true,
        )

        with tf.device("/device:MUSA:0"):
            out = op_module.musa_pln_cascade_block(
                norm_out=tf.constant(norm_out_np, dtype=tf.float32),
                add_input=tf.constant(add_table_np, dtype=tf.float32),
                bias_input=tf.constant(bias_table_np, dtype=tf.float32),
                gates=[tf.constant(g, dtype=tf.bool) for g in gates_np],
                table_indices=table_indices,
                select_on_true=select_on_true,
            )

        self.assertAllClose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_full_gate_shape(self):
        rng = np.random.RandomState(7)
        shape = [4, 6, 16]
        norm_out_np = rng.standard_normal(shape).astype(np.float32)
        add_table_np = (1.0 + 0.1 * rng.standard_normal((8, 16))).astype(np.float32)
        bias_table_np = (0.1 * rng.standard_normal((8, 16))).astype(np.float32)
        gates_np = [
            (rng.random(shape) > 0.5).astype(np.bool_),
            (rng.random(shape) > 0.3).astype(np.bool_),
        ]
        self._run_case(
            norm_out_np, add_table_np, bias_table_np, gates_np,
            table_indices=[1, 5],
            select_on_true=[True, False],
        )

    def test_left_aligned_batch_gate(self):
        rng = np.random.RandomState(11)
        shape = [5, 4, 8]
        norm_out_np = rng.standard_normal(shape).astype(np.float32)
        add_table_np = (1.0 + 0.05 * rng.standard_normal((6, 8))).astype(np.float32)
        bias_table_np = (0.05 * rng.standard_normal((6, 8))).astype(np.float32)
        gates_np = [
            (rng.random((shape[0],)) > 0.4).astype(np.bool_),
            (rng.random((shape[0],)) > 0.6).astype(np.bool_),
        ]
        self._run_case(
            norm_out_np, add_table_np, bias_table_np, gates_np,
            table_indices=[2, 3],
            select_on_true=[True, True],
        )


if __name__ == "__main__":
    tf.test.main()
