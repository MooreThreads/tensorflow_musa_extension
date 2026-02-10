import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class RangeOpTest(MUSATestCase):

  def _test_range(self, start, limit, delta, dtype, rtol=1e-5, atol=1e-8):
    def range_func(start_tensor, limit_tensor, delta_tensor):
      return tf.range(start_tensor, limit_tensor, delta_tensor, dtype=dtype)
    
    if dtype.is_floating:
      start_tensor = tf.constant(float(start), dtype=dtype)
      limit_tensor = tf.constant(float(limit), dtype=dtype)
      delta_tensor = tf.constant(float(delta), dtype=dtype)
    else:
      start_tensor = tf.constant(int(start), dtype=dtype)
      limit_tensor = tf.constant(int(limit), dtype=dtype)
      delta_tensor = tf.constant(int(delta), dtype=dtype)
    
    self._compare_cpu_musa_results(range_func, [start_tensor, limit_tensor, delta_tensor], dtype, rtol=rtol, atol=atol)

  def testRangeFloat32(self):
    self._test_range(0.0, 10.0, 1.0, tf.float32)
    self._test_range(0.0, 5.0, 0.5, tf.float32)

  def testRangeFloat64(self):
    self._test_range(0.0, 1.0, 0.1, tf.float64)

  def testRangeInt32(self):
    self._test_range(0, 10, 2, tf.int32, rtol=0, atol=0)
    self._test_range(10, 0, -1, tf.int32, rtol=0, atol=0)

  def testRangeInt64(self):
    self._test_range(0, 100, 20, tf.int64, rtol=0, atol=0)

  def testRangeDifferentSteps(self):
    for step in [1, 2, 3]:
      self._test_range(0, 20, step, tf.int32, rtol=0, atol=0)
    for step in [-1, -2]:
      self._test_range(20, 0, step, tf.int32, rtol=0, atol=0)

  def testRangeDifferentLimits(self):
    test_cases = [
        (0, 10, 1),
        (5, 15, 1),
        (10, 0, -1),
        (0, 100, 10),
        (100, 0, -10),
    ]
    for start, limit, delta in test_cases:
      self._test_range(start, limit, delta, tf.int32, rtol=0, atol=0)

  def testRangeEdgeCases(self):
    self._test_range(0, 1, 1, tf.int32, rtol=0, atol=0)
    self._test_range(0, 0, 1, tf.int32, rtol=0, atol=0)
    self._test_range(5, 5, 1, tf.int32, rtol=0, atol=0)

  def testRangeLargeRange(self):
    self._test_range(0, 10000, 1, tf.int32, rtol=0, atol=0)
    self._test_range(0, 1000, 0.1, tf.float32)


if __name__ == "__main__":
  tf.test.main()
