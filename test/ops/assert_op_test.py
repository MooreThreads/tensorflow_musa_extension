# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA Assert operator."""

import tensorflow as tf
from musa_test_utils import MUSATestCase


class AssertOpTest(MUSATestCase):
    """Tests for MUSA Assert operator."""

    def testAssertSuccess(self):
        """Test Assert op when condition is true."""
        with tf.device("/device:MUSA:0"):
            condition = tf.constant(True)
            data = [tf.constant(1.0), tf.constant("test message")]
            # In Eager mode, tf.debugging.Assert or tf.raw_ops.Assert 
            # executes immediately or creates a node that is executed.
            tf.raw_ops.Assert(condition=condition, data=data)

    def testAssertFailure(self):
        """Test Assert op when condition is false."""
        with tf.device("/device:MUSA:0"):
            condition = tf.constant(False)
            data = [tf.constant(42), tf.constant("error occurred")]
            
            with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "assertion failed: .*42.*error occurred"):
                tf.raw_ops.Assert(condition=condition, data=data)

    def testAssertInvalidCondition(self):
        """Test Assert op with non-scalar condition."""
        with tf.device("/device:MUSA:0"):
            # MusaAssertOp expects a scalar condition
            condition = tf.constant([True, True])
            data = [tf.constant("invalid condition")]
            
            with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "In\\[0\\] should be a scalar"):
                tf.raw_ops.Assert(condition=condition, data=data)

    def testAssertSummarize(self):
        """Test Assert op with summarize attribute."""
        with tf.device("/device:MUSA:0"):
            condition = tf.constant(False)
            # Create a large tensor to test summarization
            data = [tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
            
            # summarize=3 means only show 3 elements
            with self.assertRaises(tf.errors.InvalidArgumentError) as cm:
                tf.raw_ops.Assert(condition=condition, data=data, summarize=3)
            
            error_msg = str(cm.exception)
            self.assertIn("assertion failed:", error_msg)
            self.assertIn("1 2 3", error_msg)

if __name__ == "__main__":
    tf.test.main()
