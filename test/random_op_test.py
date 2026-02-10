import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class RandomOpTest(MUSATestCase):

  def testRandomUniformFloat(self):
    shape = [4, 4]
    
    with tf.device('/device:MUSA:0'):
      u_float = tf.random.uniform(shape, minval=0, maxval=1.0, dtype=tf.float32)
    
    val = u_float.numpy()
    self.assertEqual(val.shape, tuple(shape))
    self.assertEqual(u_float.dtype, tf.float32)
    self.assertTrue(np.all(val >= 0))
    self.assertTrue(np.all(val < 1.0))

  def testRandomNormalFloat(self):
    shape = [4, 4]
    
    with tf.device('/device:MUSA:0'):
      n_float = tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)
    
    val = n_float.numpy()
    self.assertEqual(val.shape, tuple(shape))
    self.assertEqual(n_float.dtype, tf.float32)

  def testRandomUniformInt(self):
    shape = [4, 4]
    
    with tf.device('/device:MUSA:0'):
      u_int = tf.random.uniform(shape, minval=0, maxval=100, dtype=tf.int32)
    
    val = u_int.numpy()
    self.assertEqual(val.shape, tuple(shape))
    self.assertEqual(u_int.dtype, tf.int32)
    self.assertTrue(np.all(val >= 0))
    self.assertTrue(np.all(val < 100))

  def testStatelessRandomUniform(self):
    shape = [4, 4]
    
    with tf.device('/device:MUSA:0'):
      result = tf.random.stateless_uniform(shape, seed=[1, 2], dtype=tf.float32)
    
    val = result.numpy()
    self.assertEqual(val.shape, tuple(shape))
    self.assertEqual(result.dtype, tf.float32)
    self.assertTrue(np.all(val >= 0))
    self.assertTrue(np.all(val < 1.0))

  def testRandomness(self):
    shape = [2, 2]
    
    with tf.device('/device:MUSA:0'):
      u1 = tf.random.uniform(shape).numpy()
      u2 = tf.random.uniform(shape).numpy()
    
    self.assertFalse(np.array_equal(u1, u2))

  def testRandomUniformDifferentShapes(self):
    shapes = [[], [1], [2, 3], [2, 3, 4], [2, 3, 4, 5]]
    
    for shape in shapes:
      with tf.device('/device:MUSA:0'):
        result = tf.random.uniform(shape, dtype=tf.float32)
      
      self.assertEqual(result.shape, tuple(shape))
      self.assertEqual(result.dtype, tf.float32)

  def testRandomNormalDifferentShapes(self):
    shapes = [[], [1], [2, 3], [2, 3, 4]]
    
    for shape in shapes:
      with tf.device('/device:MUSA:0'):
        result = tf.random.normal(shape, dtype=tf.float32)
      
      self.assertEqual(result.shape, tuple(shape))
      self.assertEqual(result.dtype, tf.float32)

  def testRandomUniformFloat16(self):
    shape = [4, 4]
    
    with tf.device('/device:MUSA:0'):
      result = tf.random.uniform(shape, dtype=tf.float16)
    
    self.assertEqual(result.shape, tuple(shape))
    self.assertEqual(result.dtype, tf.float16)


if __name__ == "__main__":
  tf.test.main()
