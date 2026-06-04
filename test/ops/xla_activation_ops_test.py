"""XLA tests for activation-like custom ops."""

import numpy as np
import tensorflow as tf

from xla_test_utils import MUSA_OPS, XlaOpTestCase


class XlaActivationOpsTest(XlaOpTestCase):
    def test_musa_clip_xla(self):
        x = tf.constant([[-2.0, -0.5, 4.0], [1.5, 6.2, 9.0]], dtype=tf.float32)
        lo = tf.constant([[0.0], [1.0]], dtype=tf.float32)
        hi = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)

        expected = tf.clip_by_value(x, lo, hi)
        actual = self.run_xla(lambda a, b, c: MUSA_OPS.musa_clip(x=a, lo=b, hi=c), x, lo, hi)
        self.assertTensorClose(actual, expected)

    def test_musa_prelu_xla(self):
        x = tf.constant(np.random.RandomState(1).standard_normal((2, 3, 4)).astype(np.float32))
        alpha = tf.constant(np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32))

        expected = tf.maximum(x, 0.0) + tf.minimum(x, 0.0) * alpha
        actual = self.run_xla(lambda a, b: MUSA_OPS.MusaPRelu(x=a, alpha=b), x, alpha)
        self.assertTensorClose(actual, expected)

    def test_musa_gelu_xla(self):
        x = tf.constant(np.random.RandomState(2).standard_normal((8, 16)).astype(np.float32))
        expected = tf.nn.gelu(x, approximate=True)
        actual = self.run_xla(
            lambda a: MUSA_OPS.musa_gelu(x=a, approximate=True),
            x,
        )
        self.assertTensorClose(actual, expected)

    def test_musa_gelu_grad_xla(self):
        x = tf.constant(np.random.RandomState(3).standard_normal((8, 16)).astype(np.float32))
        dy = tf.constant(np.random.RandomState(4).standard_normal((8, 16)).astype(np.float32))

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = tf.nn.gelu(x, approximate=False)
        expected = tape.gradient(y, x, output_gradients=dy)
        actual = self.run_xla(
            lambda grad, inp: MUSA_OPS.musa_gelu_grad(dy=grad, x=inp, approximate=False),
            dy,
            x,
        )
        self.assertTensorClose(actual, expected)


if __name__ == "__main__":
    tf.test.main()
