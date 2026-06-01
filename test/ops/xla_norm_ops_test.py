"""XLA tests for normalization custom ops."""

import numpy as np
import tensorflow as tf

from xla_test_utils import (
    MUSA_OPS,
    XlaOpTestCase,
    layer_norm_ref,
    normalize_ref,
    run_on_cpu,
)


class XlaNormOpsTest(XlaOpTestCase):
    def test_musa_layer_norm_xla(self):
        rng = np.random.RandomState(5)
        x = tf.constant(rng.standard_normal((4, 6, 8)).astype(np.float32))
        gamma = tf.constant(rng.uniform(0.5, 1.5, size=(8,)).astype(np.float32))
        beta = tf.constant(rng.uniform(-0.3, 0.3, size=(8,)).astype(np.float32))
        epsilon = 1e-5

        expected = run_on_cpu(layer_norm_ref, x, gamma, beta, epsilon)
        actual = self.run_xla(
            lambda a, b, c: MUSA_OPS.musa_layer_norm(x=a, gamma=b, beta=c, epsilon=epsilon),
            x,
            gamma,
            beta,
        )
        self.assertTensorClose(actual, expected)

    def test_musa_layer_norm_grad_xla(self):
        rng = np.random.RandomState(6)
        x = tf.constant(rng.standard_normal((3, 5, 8)).astype(np.float32))
        dy = tf.constant(rng.standard_normal((3, 5, 8)).astype(np.float32))
        gamma = tf.constant(rng.uniform(0.5, 1.5, size=(8,)).astype(np.float32))
        beta = tf.constant(rng.uniform(-0.3, 0.3, size=(8,)).astype(np.float32))
        epsilon = 1e-5

        with tf.device("/CPU:0"):
            with tf.GradientTape() as tape:
                tape.watch([x, gamma, beta])
                y = layer_norm_ref(x, gamma, beta, epsilon)
            expected = tape.gradient(y, [x, gamma, beta], output_gradients=dy)
        actual = self.run_xla(
            lambda grad, inp, scale, shift: MUSA_OPS.musa_layer_norm_grad(
                dy=grad, x=inp, gamma=scale, beta=shift, epsilon=epsilon
            ),
            dy,
            x,
            gamma,
            beta,
        )
        self.assertTensorClose(actual, expected, rtol=2e-4, atol=2e-4)

    def test_musa_normalize_xla(self):
        rng = np.random.RandomState(7)
        x_np = rng.standard_normal((4, 5, 8)).astype(np.float32)
        x = tf.constant(x_np)
        gamma = tf.ones((8,), dtype=tf.float32)
        beta = tf.zeros((8,), dtype=tf.float32)
        epsilon = 1e-4
        max_std = 2.0

        expected = normalize_ref(x_np, epsilon=epsilon, max_std=max_std)
        actual = self.run_xla(
            lambda a, b, c: MUSA_OPS.musa_normalize(
                x=a, gamma=b, beta=c, epsilon=epsilon, max_std=max_std
            ),
            x,
            gamma,
            beta,
        )
        self.assertTensorClose(actual, expected)


if __name__ == "__main__":
    tf.test.main()
