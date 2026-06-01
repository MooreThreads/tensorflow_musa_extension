"""XLA tests for remaining custom op families."""

import numpy as np
import tensorflow as tf

from xla_test_utils import (
    MUSA_OPS,
    XlaOpTestCase,
    pln_cascade_block_ref,
    pln_cascade_ref,
)


class XlaMiscOpsTest(XlaOpTestCase):
    def test_musa_shifted_affine_map_xla(self):
        rng = np.random.RandomState(11)
        left_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
        mask_np = (rng.random((2, 1, 4)) > 0.5).astype(np.float32)
        right_np = rng.standard_normal((4,)).astype(np.float32)
        left = tf.constant(left_np)
        mask = tf.constant(mask_np)
        right = tf.constant(right_np)

        expected = mask_np * left_np + right_np
        actual = self.run_xla(
            lambda a, b, c: MUSA_OPS.musa_shifted_affine_map(
                data_left=a, mask=b, sliced_var_right=c
            ),
            left,
            mask,
            right,
        )
        self.assertTensorClose(actual, expected)

    def test_musa_pln_cascade_xla(self):
        rng = np.random.RandomState(12)
        norm_out_np = rng.standard_normal((3, 4, 8)).astype(np.float32)
        mask_np = (rng.random((3,)) > 0.4).astype(np.bool_)
        add_input_np = (1.0 + 0.1 * rng.standard_normal((2, 8))).astype(np.float32)
        bias_input_np = (0.1 * rng.standard_normal((2, 8))).astype(np.float32)

        expected = pln_cascade_ref(
            norm_out_np,
            mask_np,
            add_input_np[1],
            bias_input_np[1],
            select_on_true=True,
        )
        actual = self.run_xla(
            lambda x, mask, add, bias: MUSA_OPS.musa_pln_cascade(
                norm_out=x,
                adpos=mask,
                add_input=add,
                bias_input=bias,
                use_table=True,
                table_index=1,
                select_on_true=True,
            ),
            tf.constant(norm_out_np),
            tf.constant(mask_np),
            tf.constant(add_input_np),
            tf.constant(bias_input_np),
        )
        self.assertTensorClose(actual, expected)

    def test_musa_pln_cascade_block_xla(self):
        rng = np.random.RandomState(13)
        norm_out_np = rng.standard_normal((3, 5, 8)).astype(np.float32)
        add_table_np = (1.0 + 0.1 * rng.standard_normal((4, 8))).astype(np.float32)
        bias_table_np = (0.1 * rng.standard_normal((4, 8))).astype(np.float32)
        gates_np = [
            (rng.random((3,)) > 0.5).astype(np.bool_),
            (rng.random((3, 5, 8)) > 0.4).astype(np.bool_),
        ]
        table_indices = [1, 3]
        select_on_true = [True, False]

        expected = pln_cascade_block_ref(
            norm_out_np, add_table_np, bias_table_np, gates_np, table_indices, select_on_true
        )
        actual = self.run_xla(
            lambda x, add, bias, gate0, gate1: MUSA_OPS.musa_pln_cascade_block(
                norm_out=x,
                add_input=add,
                bias_input=bias,
                gates=[gate0, gate1],
                table_indices=table_indices,
                select_on_true=select_on_true,
            ),
            tf.constant(norm_out_np),
            tf.constant(add_table_np),
            tf.constant(bias_table_np),
            tf.constant(gates_np[0]),
            tf.constant(gates_np[1]),
        )
        self.assertTensorClose(actual, expected)

    def test_musa_tensor_dot_xla(self):
        rng = np.random.RandomState(14)
        a = tf.constant(rng.standard_normal((2, 3, 4)).astype(np.float32))
        b = tf.constant(rng.standard_normal((4, 5, 6)).astype(np.float32))

        expected = tf.tensordot(a, b, axes=([2], [0]))
        actual = self.run_xla(
            lambda x, y: MUSA_OPS.musa_tensor_dot(a=x, b=y, axes_a=[2], axes_b=[0]),
            a,
            b,
        )
        self.assertTensorClose(actual, expected)

    def test_musa_tensor_dot_bias_xla(self):
        rng = np.random.RandomState(15)
        a_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
        b_np = rng.standard_normal((4, 5, 6)).astype(np.float32)
        bias_np = rng.standard_normal((3, 5, 6)).astype(np.float32)
        a = tf.constant(a_np)
        b = tf.constant(b_np)
        bias = tf.constant(bias_np)

        expected = tf.tensordot(a, b, axes=([2], [0])) + bias
        actual = self.run_xla(
            lambda x, y, z: MUSA_OPS.musa_tensor_dot_bias(
                a=x, b=y, bias=z, axes_a=[2], axes_b=[0]
            ),
            a,
            b,
            bias,
        )
        self.assertTensorClose(actual, expected)

    def test_musa_token_mixer_xla(self):
        rng = np.random.RandomState(16)
        x_np = rng.standard_normal((2, 3, 8)).astype(np.float32)
        x = tf.constant(x_np)

        expected = x_np.reshape(2, 3, 2, 4).transpose(0, 2, 1, 3).reshape(2, 2, 12)
        actual = self.run_xla(
            lambda a: MUSA_OPS.musa_token_mixer(x=a, num_T=3, num_H=2, d_k=4),
            x,
        )
        self.assertTensorClose(actual, expected)


if __name__ == "__main__":
    tf.test.main()
