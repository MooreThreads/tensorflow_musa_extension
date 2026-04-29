# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA Dropout operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase, load_musa_ops


class DropoutOpTest(MUSATestCase):
    """Tests for MUSA Dropout forward and backward operators."""

    @classmethod
    def setUpClass(cls):
        """Get the MUSA custom ops module (plugin already loaded by musa_test_utils)."""
        super(DropoutOpTest, cls).setUpClass()
        cls._musa_ops = load_musa_ops()

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------
    def _run_dropout(self, x, rate=0.5, seed=42, offset=0):
        """Run MusaDropout on MUSA device and return (y, mask)."""
        with tf.device('/device:MUSA:0'):
            y, mask = self._musa_ops.musa_dropout(
                x=x, rate=rate, seed=seed, offset=offset)
        return y, mask

    def _run_dropout_grad(self, grad, mask, rate=0.5):
        """Run MusaDropoutGrad on MUSA device."""
        with tf.device('/device:MUSA:0'):
            grad_input = self._musa_ops.musa_dropout_grad(
                grad=grad, mask=mask, rate=rate)
        return grad_input

    # -------------------------------------------------------------------------
    # Forward tests
    # -------------------------------------------------------------------------
    def testOutputShapeAndDtype(self):
        """Output tensor and mask must have the same shape as input."""

        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            with self.subTest(dtype=dtype):
                shape = [4, 128]
                x = tf.ones(shape, dtype=dtype)
                y, mask = self._run_dropout(x, rate=0.5)

                self.assertEqual(y.shape.as_list(), shape)
                self.assertEqual(mask.shape.as_list(), shape)
                self.assertEqual(y.dtype, dtype)
                self.assertEqual(mask.dtype, tf.bool)

    def testZeroRateIsPassthrough(self):
        """rate=0 must preserve all elements exactly."""

        np_x = np.random.uniform(1.0, 2.0, size=[64, 64]).astype(np.float32)
        x = tf.constant(np_x, dtype=tf.float32)
        y, mask = self._run_dropout(x, rate=0.0)

        # All mask values should be True (nothing dropped)
        mask_np = mask.numpy()
        self.assertTrue(np.all(mask_np == True),
                        "rate=0 should keep all elements (mask all True)")

        # Output values should match input (scale=1/(1-0)=1)
        self.assertAllClose(y.numpy(), np_x, rtol=1e-5, atol=1e-5)

    def testMaskIsBinary(self):
        """Mask must contain only 0 or 1."""

        x = tf.ones([256, 256], dtype=tf.float32)
        _, mask = self._run_dropout(x, rate=0.3)
        mask_np = mask.numpy()

        unique_vals = np.unique(mask_np)
        self.assertTrue(set(unique_vals).issubset({False, True}),
                        f"Mask contains non-binary values: {unique_vals}")

    def testDropRateStatistics(self):
        """Empirical drop rate should be close to the requested rate."""

        rate = 0.4
        x = tf.ones([1024, 1024], dtype=tf.float32)
        _, mask = self._run_dropout(x, rate=rate, seed=0)

        mask_np = mask.numpy().astype(np.float32)  # bool -> float for mean
        actual_drop_rate = 1.0 - mask_np.mean()
        # Allow ±5% tolerance for statistical fluctuation
        self.assertAlmostEqual(actual_drop_rate, rate, delta=0.05,
                               msg=f"Drop rate {actual_drop_rate:.4f} too far from {rate}")

    def testDroppedElementsAreZero(self):
        """Elements where mask==0 must be zero in the output."""

        np_x = np.random.uniform(0.5, 1.5, size=[128, 128]).astype(np.float32)
        x = tf.constant(np_x, dtype=tf.float32)
        y, mask = self._run_dropout(x, rate=0.5)

        y_np = y.numpy()
        mask_np = mask.numpy()
        dropped = mask_np == False
        self.assertTrue(np.all(y_np[dropped] == 0.0),
                        "Dropped elements (mask=False) must be 0 in output")

    def testKeptElementsAreScaled(self):
        """Elements where mask==1 must equal input * scale."""

        rate = 0.5
        scale = 1.0 / (1.0 - rate)
        np_x = np.random.uniform(0.5, 1.5, size=[128, 128]).astype(np.float32)
        x = tf.constant(np_x, dtype=tf.float32)
        y, mask = self._run_dropout(x, rate=rate)

        y_np = y.numpy()
        mask_np = mask.numpy()
        kept = mask_np == True
        self.assertAllClose(y_np[kept], (np_x * scale)[kept],
                            rtol=1e-5, atol=1e-5)

    def testDifferentShapes(self):
        """Forward pass should work for various tensor shapes."""

        shapes = [
            [256],
            [32, 64],
            [8, 16, 32],
            [2, 4, 8, 16],
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                x = tf.ones(shape, dtype=tf.float32)
                y, mask = self._run_dropout(x, rate=0.3)
                self.assertEqual(y.shape.as_list(), shape)
                self.assertEqual(mask.shape.as_list(), shape)

    # -------------------------------------------------------------------------
    # Backward tests
    # -------------------------------------------------------------------------
    def testGradOutputShape(self):
        """Gradient output must have the same shape as grad input."""

        shape = [64, 128]
        grad = tf.ones(shape, dtype=tf.float32)
        # Create a plausible binary mask (bool)
        mask_np = np.random.randint(0, 2, size=shape).astype(bool)
        mask = tf.constant(mask_np, dtype=tf.bool)

        with tf.device('/device:MUSA:0'):
            mask = tf.identity(mask)

        grad_input = self._run_dropout_grad(grad, mask, rate=0.5)
        self.assertEqual(grad_input.shape.as_list(), shape)

    def testGradDroppedElementsAreZero(self):
        """Gradient should be zero where mask==0."""

        shape = [128, 128]
        grad_np = np.random.uniform(0.5, 1.5, size=shape).astype(np.float32)
        mask_np = np.random.randint(0, 2, size=shape).astype(bool)

        grad = tf.constant(grad_np, dtype=tf.float32)
        with tf.device('/device:MUSA:0'):
            mask = tf.constant(mask_np, dtype=tf.bool)

        grad_input = self._run_dropout_grad(grad, mask, rate=0.5)
        grad_input_np = grad_input.numpy()

        dropped = mask_np == False
        self.assertTrue(np.all(grad_input_np[dropped] == 0.0),
                        "Gradient for dropped elements (mask=False) must be 0")

    def testGradKeptElementsAreScaled(self):
        """Gradient for kept elements must equal upstream_grad * scale."""

        rate = 0.3
        scale = 1.0 / (1.0 - rate)
        shape = [128, 128]
        grad_np = np.random.uniform(0.5, 1.5, size=shape).astype(np.float32)
        mask_np = np.random.randint(0, 2, size=shape).astype(bool)

        grad = tf.constant(grad_np, dtype=tf.float32)
        with tf.device('/device:MUSA:0'):
            mask = tf.constant(mask_np, dtype=tf.bool)

        grad_input = self._run_dropout_grad(grad, mask, rate=rate)
        grad_input_np = grad_input.numpy()

        kept = mask_np == True
        self.assertAllClose(grad_input_np[kept], (grad_np * scale)[kept],
                            rtol=1e-5, atol=1e-5)

    def testForwardBackwardConsistency(self):
        """Use the mask from forward pass in backward pass."""

        rate = 0.5
        scale = 1.0 / (1.0 - rate)
        shape = [64, 64]
        np_x = np.random.uniform(1.0, 2.0, size=shape).astype(np.float32)
        x = tf.constant(np_x, dtype=tf.float32)

        y, mask = self._run_dropout(x, rate=rate)
        grad_input = self._run_dropout_grad(y, mask, rate=rate)
        grad_np = grad_input.numpy()
        mask_np = mask.numpy()

        # Where mask==1: grad_input = y * scale = x * scale * scale
        # Where mask==0: grad_input = 0
        kept = mask_np == True
        expected_kept = np_x[kept] * scale * scale
        self.assertAllClose(grad_np[kept], expected_kept, rtol=1e-4, atol=1e-4)
        self.assertTrue(np.all(grad_np[mask_np == False] == 0.0))


if __name__ == "__main__":
    tf.test.main()
