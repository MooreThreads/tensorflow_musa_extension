import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class BroadcastToOpTest(MUSATestCase):

  def _test_broadcast_to(self, input_data, target_shape, dtype):
    x = tf.constant(input_data, dtype=dtype)
    
    def op_wrapper(input_tensor):
        return tf.broadcast_to(input_tensor, target_shape)

    self._compare_cpu_musa_results(
        op_wrapper,
        [x],
        dtype=dtype
    )

  def testBroadcastToBasic(self):
    x_np = np.array([1, 2, 3])
    target_shape = [3, 3]

    for dtype in [tf.float32, tf.int32, tf.int64]:
      self._test_broadcast_to(x_np, target_shape, dtype)

  def testBroadcastToBool(self):
    x_np = np.array([True, False, True])
    target_shape = [3, 3]
    self._test_broadcast_to(x_np, target_shape, tf.bool)

  def testBroadcastToScalar(self):
    x_np = np.array(1)
    target_shape = [3, 3]
    self._test_broadcast_to(x_np, target_shape, tf.int32)

  def testBroadcastToVariousShapes(self):
    for input_dim in range(1, 6):
        for output_dim in range(input_dim, 6):
            input_shape = [2] * input_dim
            output_shape = [2] * output_dim
            
            x_np = np.random.randint(5, size=input_shape).astype(np.int32)
            
            self._test_broadcast_to(x_np, output_shape, tf.int32)

  def testBroadcastToComplexShapes(self):
    self._test_broadcast_to(
        np.random.randint(5, size=[2, 1, 3]),
        [2, 5, 3],
        tf.int32
    )

    input_shape = [2, 1, 3, 2, 2, 2]
    output_shape = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 15, 3, 2, 2, 2]
    
    x_np = np.random.randint(5, size=input_shape).astype(np.int32)
    self._test_broadcast_to(x_np, output_shape, tf.int32)

  def testGradient(self):
    def grad_wrapper(input_tensor, target_shape):
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            broadcasted = tf.broadcast_to(input_tensor, target_shape)
            loss = 2 * broadcasted
        return tape.gradient(loss, input_tensor)

    x_scalar = tf.constant(1.0, dtype=tf.float32)
    target_scalar = [2, 4, 3]
    
    self._compare_cpu_musa_results(
        lambda t: grad_wrapper(t, target_scalar),
        [x_scalar],
        dtype=tf.float32
    )

    x_rank = tf.constant([[1.0], [2.0]], dtype=tf.float32)
    target_rank = [5, 2, 3]
    
    self._compare_cpu_musa_results(
        lambda t: grad_wrapper(t, target_rank),
        [x_rank],
        dtype=tf.float32
    )

if __name__ == "__main__":
  tf.test.main()
