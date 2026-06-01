"""Correctness tests for MusaOneTrans3DEinsum against CPU einsum."""

import numpy as np
import tensorflow as tf

from xla_test_utils import MUSA_OPS, XlaOpTestCase


class MusaOneTrans3DEinsumTest(XlaOpTestCase):
    def test_musa_one_trans3d_einsum(self):
        rng = np.random.RandomState(19)
        cases = [
            (
                "btd,tde->bte",
                rng.standard_normal((2, 3, 4)).astype(np.float32),
                rng.standard_normal((3, 4, 5)).astype(np.float32),
            ),
            (
                "bte,tde->btd",
                rng.standard_normal((2, 3, 5)).astype(np.float32),
                rng.standard_normal((3, 4, 5)).astype(np.float32),
            ),
            (
                "bte,btd->tde",
                rng.standard_normal((2, 3, 5)).astype(np.float32),
                rng.standard_normal((2, 3, 4)).astype(np.float32),
            ),
        ]

        for equation, a_np, b_np in cases:
            with self.subTest(equation=equation):
                with tf.device("/CPU:0"):
                    expected = tf.einsum(equation, a_np, b_np)
                with tf.device("/device:MUSA:0"):
                    actual = MUSA_OPS.musa_one_trans3d_einsum(
                        a=tf.constant(a_np), b=tf.constant(b_np), equation=equation
                    )
                self.assertTensorClose(actual, expected)


if __name__ == "__main__":
    tf.test.main()
