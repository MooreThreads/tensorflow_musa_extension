"""XLA tests for matmul-related custom ops."""

import numpy as np
import tensorflow as tf

from xla_test_utils import MUSA_OPS, XlaOpTestCase, run_on_cpu


class XlaMatmulOpsTest(XlaOpTestCase):
    def test_musa_matmul_bias_add_xla(self):
        rng = np.random.RandomState(8)
        a = tf.constant(rng.standard_normal((6, 10)).astype(np.float32))
        b = tf.constant(rng.standard_normal((10, 7)).astype(np.float32))
        bias = tf.constant(rng.standard_normal((7,)).astype(np.float32))

        expected = run_on_cpu(lambda x, y, z: tf.matmul(x, y) + z, a, b, bias)
        actual = self.run_xla(
            lambda x, y, z: MUSA_OPS.musa_mat_mul_bias_add(a=x, b=y, bias=z),
            a,
            b,
            bias,
        )
        self.assertTensorClose(actual, expected)

    def test_musa_linear_activation_xla(self):
        rng = np.random.RandomState(9)
        a = tf.constant(rng.standard_normal((6, 8)).astype(np.float32))
        b = tf.constant(rng.standard_normal((8, 5)).astype(np.float32))
        bias = tf.constant(rng.standard_normal((5,)).astype(np.float32))

        expected = run_on_cpu(lambda x, y, z: tf.nn.relu(tf.matmul(x, y) + z), a, b, bias)
        actual = self.run_xla(
            lambda x, y, z: MUSA_OPS.musa_linear_activation(
                a=x,
                b=y,
                bias=z,
                activation="relu",
                alpha=0.0,
                transpose_a=False,
                transpose_b=False,
            ),
            a,
            b,
            bias,
        )
        self.assertTensorClose(actual, expected)

    def test_musa_reshape_mat_mul_xla(self):
        rng = np.random.RandomState(10)
        x_np = rng.standard_normal((2, 3, 4, 6)).astype(np.float32)
        w_np = rng.standard_normal((6, 5)).astype(np.float32)
        x = tf.constant(x_np)
        w = tf.constant(w_np)

        expected = np.matmul(x_np.reshape(-1, 6), w_np).reshape(2, 3, 4, 5)
        actual = self.run_xla(
            lambda a, b: MUSA_OPS.musa_reshape_mat_mul(x=a, w=b, transpose_b=False),
            x,
            w,
        )
        self.assertTensorClose(actual, expected)

    def test_musa_concat_mat_mul_xla(self):
        rng = np.random.RandomState(17)
        a0_np = rng.standard_normal((2, 3)).astype(np.float32)
        other_np = rng.standard_normal((3, 4)).astype(np.float32)
        axis = tf.constant(1, dtype=tf.int32)

        expected = run_on_cpu(
            lambda x0, y: tf.matmul(tf.concat([x0], axis=1), y),
            tf.constant(a0_np),
            tf.constant(other_np),
        )
        actual = self.run_xla(
            lambda x0, concat_axis, other: MUSA_OPS.musa_concat_mat_mul(
                inputs=[x0],
                axis=concat_axis,
                other=other,
                concat_input_idx=0,
                transpose_a=False,
                transpose_b=False,
            ),
            tf.constant(a0_np),
            axis,
            tf.constant(other_np),
        )
        self.assertTensorClose(actual, expected)

    def test_musa_bias_add_relu_mat_mul_xla(self):
        rng = np.random.RandomState(18)
        x_np = rng.standard_normal((4, 6)).astype(np.float32)
        bias_np = rng.standard_normal((6,)).astype(np.float32)
        other_np = rng.standard_normal((6, 5)).astype(np.float32)

        expected = run_on_cpu(
            lambda x, bias, other: tf.matmul(tf.nn.relu(x + bias), other),
            tf.constant(x_np),
            tf.constant(bias_np),
            tf.constant(other_np),
        )
        actual = self.run_xla(
            lambda x, bias, other: MUSA_OPS.musa_bias_add_relu_mat_mul(
                input=x,
                bias=bias,
                other=other,
                relu_input_slot=0,
                transpose_a=False,
                transpose_b=False,
            ),
            tf.constant(x_np),
            tf.constant(bias_np),
            tf.constant(other_np),
        )
        self.assertTensorClose(actual, expected)


if __name__ == "__main__":
    tf.test.main()
