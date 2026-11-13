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
# ==============================================================================

"""Tests for MUSA Sub (Subtract) operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class SubOpTest(MUSATestCase):
  """Tests for MUSA Sub operator."""

  _boundary_dtypes = (
      tf.float32, tf.float16, tf.bfloat16, tf.float64, tf.int32, tf.int64)

  def _check_runtime_broadcast(self, cases, invalid=False):
    # Unknown shapes and fed inputs force runtime broadcasting instead of
    # static shape rejection or constant folding. Check the executed partition
    # as well as disabling soft placement to guard against CPU fallback.
    for dtype in self._boundary_dtypes:
      graph = tf.Graph()
      with graph.as_default():
        with tf.device('/CPU:0'):
          x = tf.compat.v1.placeholder(dtype, shape=None, name='x')
          y = tf.compat.v1.placeholder(dtype, shape=None, name='y')
          cpu = tf.raw_ops.Sub(x=x, y=y, name='cpu_sub')
        with tf.device('/device:MUSA:0'):
          musa = tf.raw_ops.Sub(x=x, y=y, name='musa_sub')
      config = tf.compat.v1.ConfigProto(allow_soft_placement=False)
      config.graph_options.rewrite_options.disable_meta_optimizer = True
      config.graph_options.optimizer_options.opt_level = -1
      with tf.compat.v1.Session(graph=graph, config=config) as session:
        for shape_x, shape_y, expected_shape in cases:
          with self.subTest(dtype=dtype.name, x=shape_x, y=shape_y):
            # Small exact values expose operand reversal without introducing
            # rounding differences in half/bfloat16 or integer arithmetic.
            x_np = (np.arange(np.prod(shape_x, dtype=int)) % 7 + 1)
            y_np = (np.arange(np.prod(shape_y, dtype=int)) % 5 - 2)
            x_np = x_np.reshape(shape_x).astype(dtype.as_numpy_dtype)
            y_np = y_np.reshape(shape_y).astype(dtype.as_numpy_dtype)
            feed = {x: x_np, y: y_np}
            if invalid:
              for result in (cpu, musa):
                with self.assertRaisesRegex(
                    tf.errors.InvalidArgumentError, '[Ii]ncompatible shapes'):
                  session.run(result, feed_dict=feed)
              continue
            metadata = tf.compat.v1.RunMetadata()
            expected, actual = session.run(
                [cpu, musa], feed_dict=feed,
                options=tf.compat.v1.RunOptions(output_partition_graphs=True),
                run_metadata=metadata)
            self.assertEqual(expected.shape, expected_shape)
            self.assertEqual(actual.shape, expected_shape)
            self.assertEqual(actual.dtype, np.dtype(dtype.as_numpy_dtype))
            self.assertAllEqual(expected, actual)
            executed = [node for partition in metadata.partition_graphs
                        for node in partition.node
                        if node.name == musa.op.name and node.op == 'Sub']
            self.assertLen(executed, 1)
            self.assertIn('device:MUSA:0', executed[0].device)

  def testSubEmptyRuntimeBroadcast(self):
    self._check_runtime_broadcast([
        ((0,), (1,), (0,)),
        ((1,), (0,), (0,)),
        ((0,), (0,), (0,)),
        ((0, 4), (1, 4), (0, 4)),
        ((1, 4), (0, 4), (0, 4)),
        ((2, 0, 4), (1, 1, 4), (2, 0, 4)),
        ((0, 1), (1, 0), (0, 0)),
        ((0,), (), (0,)),
        ((), (0,), (0,)),
        ((0, 4), (4,), (0, 4)),
    ])

  def testSubScalarRuntimeBroadcast(self):
    self._check_runtime_broadcast([
        ((), (), ()),
        ((), (5,), (5,)),
        ((5,), (), (5,)),
        ((), (2, 3), (2, 3)),
        ((2, 3), (), (2, 3)),
        ((1,), (1,), (1,)),
        ((2, 1), (1, 3), (2, 3)),
        ((2, 1, 3, 1), (1, 4, 1, 5), (2, 4, 3, 5)),
    ])

  def testSubInvalidEmptyRuntimeBroadcast(self):
    self._check_runtime_broadcast([
        ((0,), (2,), None),
        ((2,), (0,), None),
        ((0, 3), (1, 4), None),
        ((2, 0, 3), (2, 2, 1), None),
    ], invalid=True)

  def _test_sub(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
    """Test subtract operation with given shapes and dtype."""
    np_dtype = dtype.as_numpy_dtype

    if np.issubdtype(np_dtype, np.integer):
        x_np = np.random.randint(-100, 100, size=shape_x).astype(np_dtype)
        y_np = np.random.randint(-100, 100, size=shape_y).astype(np_dtype)
    else:
        x_np = np.random.uniform(-1, 1, size=shape_x).astype(np_dtype)
        y_np = np.random.uniform(-1, 1, size=shape_y).astype(np_dtype)

    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)

    self._compare_cpu_musa_results(tf.math.subtract, [x, y], dtype, rtol=rtol, atol=atol)

  def testSubBasic(self):
    """Test basic subtract operation with same shapes."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_sub([1024, 1024], [1024, 1024], dtype, rtol=rtol, atol=atol)

  def testSubIsolationLegacy(self):
    """Test the specific small/unaligned case from the original test_sub.py."""
    shape = [3]
    for dtype in [tf.float32, tf.int32]:
        self._test_sub(shape, shape, dtype)

  def testSubBroadcastVectorMatrix(self):
    """Test subtract with vector-matrix broadcasting."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_sub([1024], [1024, 1024], dtype, rtol=rtol, atol=atol)

  def testSubBroadcastColumnRow(self):
    """Test subtract with column-row broadcasting."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_sub([1024, 1], [1, 1024], dtype, rtol=rtol, atol=atol)

  def testSubScalar(self):
    """Test subtract with scalar values."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_sub([], [], dtype, rtol=rtol, atol=atol)

  def testSubDifferentShapes(self):
    """Test subtract with various different shapes."""
    test_cases = [
        ([1], [1]),
        ([5], [5]),
        ([3, 4], [3, 4]),
        ([2, 3, 4], [2, 3, 4]),
        ([1, 1, 10], [5, 3, 10]),
    ]
    for dtype in [tf.float32]:
      for shape_x, shape_y in test_cases:
        self._test_sub(shape_x, shape_y, dtype)


if __name__ == "__main__":
  tf.test.main()
