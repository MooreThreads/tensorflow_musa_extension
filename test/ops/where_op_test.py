# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================

"""Tests for MUSA Where operator that mirror TensorFlow behavior."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase

_RNG = np.random.default_rng(42)


def _np_dtype_for_tf(dtype):
  if dtype == tf.bfloat16:
    return np.float32
  return dtype.as_numpy_dtype


def _random_values(shape, dtype):
  np_dtype = _np_dtype_for_tf(dtype)
  samples = _RNG.uniform(-4.5, 4.5, size=shape).astype(np_dtype)
  return samples


def _random_bool(shape):
  mask = _RNG.random(shape) > 0.4
  if mask.size:
    mask.flat[0] = True
  return mask


def _tolerances(dtype):
  if dtype in (tf.float16, tf.bfloat16):
    return 1e-2, 1e-2
  return 1e-5, 1e-8


class WhereOpTest(MUSATestCase):
  """Compare `tf.where` on CPU and MUSA devices."""

  def _compare_where_indices(self, condition_shape):
    condition = tf.constant(_random_bool(condition_shape), dtype=tf.bool)
    self._compare_cpu_musa_results(lambda cond: tf.where(cond), [condition],
                                   tf.int64)

  def _compare_where_selection(self,
                               condition_shape,
                               x_shape,
                               y_shape,
                               dtype):
    condition = tf.constant(_random_bool(condition_shape), dtype=tf.bool)
    x = tf.constant(_random_values(x_shape, dtype), dtype=dtype)
    y = tf.constant(_random_values(y_shape, dtype), dtype=dtype)
    rtol, atol = _tolerances(dtype)
    self._compare_cpu_musa_results(lambda cond, x_val, y_val: tf.where(
        cond, x_val, y_val), [condition, x, y], dtype, rtol=rtol, atol=atol)

  def testWhereIndicesCoverRanks(self):
    """`tf.where(condition)` should match CPU results across ranks."""
    for shape in ([4, 4], [2, 3, 4], [1, 5, 1]):
      self._compare_where_indices(shape)

  def testWhereSelectionSameShapes(self):
    """`tf.where(condition, x, y)` should work when all inputs share a shape."""
    for dtype in (tf.float32, tf.float16, tf.bfloat16):
      self._compare_where_selection([2, 3, 4], [2, 3, 4], [2, 3, 4], dtype)

  def testWhereSelectionBroadcasting(self):
    """Validate broadcasting semantics for condition, x, and y."""
    broadcast_cases = [
        ([2, 1, 4], [2, 3, 4], [1, 3, 1]),
        ([2, 3, 1], [2, 3, 4], [2, 1, 4]),
        ([3, 5], (), [3, 5]),
        ([2, 3, 4], [4], [1, 1, 4]),
    ]
    for dtype in (tf.float32, tf.float16, tf.bfloat16):
      for condition_shape, x_shape, y_shape in broadcast_cases:
        self._compare_where_selection(condition_shape, x_shape, y_shape, dtype)


if __name__ == "__main__":
  tf.test.main()