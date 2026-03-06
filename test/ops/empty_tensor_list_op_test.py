# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA EmptyTensorList operator."""

import tensorflow as tf
from musa_test_utils import MUSATestCase


class EmptyTensorListOpTest(MUSATestCase):
  """Tests for MUSA EmptyTensorList operator."""

  def testEmptyTensorListBasic(self):
    """Test basic EmptyTensorList operation on MUSA."""
    element_shape = [2, 3]
    max_num_elements = 10
    element_dtype = tf.float32

    def list_ops():
        # Using tf.raw_ops to ensure we call the specific MUSA kernel if available
        # or at least test the behavior on MUSA device context.
        handle = tf.raw_ops.EmptyTensorList(
            element_shape=element_shape,
            max_num_elements=max_num_elements,
            element_dtype=element_dtype,
            name="empty_list"
        )
        return handle

    # Test on MUSA
    with tf.device('/device:MUSA:0'):
        musa_handle = list_ops()
        # To verify it's a valid list handle, we can use other list ops
        # but since we only care about if EmptyTensorList works:
        self.assertIsNotNone(musa_handle)
    
    # Test on CPU
    with tf.device('/CPU:0'):
        cpu_handle = list_ops()
        self.assertIsNotNone(cpu_handle)

  def testEmptyTensorListUnknownShape(self):
    """Test EmptyTensorList with unknown element shape."""
    # -1 scalar represents unknown shape in some contexts, 
    # but usually element_shape is a vector. 
    # TensorFlow's EmptyTensorList expects element_shape to be a 1D tensor.
    element_shape = tf.constant(-1, dtype=tf.int32, shape=[])
    max_num_elements = 5
    element_dtype = tf.int64

    with tf.device('/device:MUSA:0'):
        handle = tf.raw_ops.EmptyTensorList(
            element_shape=element_shape,
            max_num_elements=max_num_elements,
            element_dtype=element_dtype
        )
        self.assertIsNotNone(handle)

if __name__ == "__main__":
  tf.test.main()
