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

"""Empty state, TensorList, control-flow and checkpoint round trips."""
import os
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class EmptyStateOpsTest(MUSATestCase):
  def testEmptyResourceVariable(self):
    for shape in [(0,), (2, 0, 3)]:
      with self.subTest(shape=shape), tf.device('/device:MUSA:0'):
        var = tf.Variable(np.ones(shape), dtype=tf.float32)
        var.assign(tf.zeros(shape))
        var.assign_add(tf.zeros(shape))
        var.assign_sub(tf.zeros(shape))
        self.assertEqual(var.read_value().numpy().shape, shape)
        self.assertIn('device:MUSA:0', var.device)
        self.assertEqual(tf.gather(var, tf.constant([], tf.int32)).numpy().shape,
                         (0,) + shape[1:])

  def testEmptyTensorLists(self):
    for shape in [(0, 3), (2, 0, 3)]:
      with self.subTest(shape=shape), tf.device('/device:MUSA:0'):
        values = tf.zeros(shape)
        handle = tf.raw_ops.TensorListFromTensor(tensor=values,
                                                element_shape=list(shape[1:]))
        result = tf.raw_ops.TensorListStack(input_handle=handle,
            element_shape=list(shape[1:]), element_dtype=tf.float32,
            num_elements=shape[0])
        self.assertEqual(result.numpy().shape, shape)
        reserved = tf.raw_ops.TensorListReserve(element_shape=[3],
            num_elements=0, element_dtype=tf.float32)
        result = tf.raw_ops.TensorListStack(input_handle=reserved,
            element_shape=[3], element_dtype=tf.float32, num_elements=0)
        self.assertEqual(result.numpy().shape, (0, 3))

  def testEmptyControlFlow(self):
    graph = tf.Graph()
    with graph.as_default(), tf.device('/device:MUSA:0'):
      x = tf.compat.v1.placeholder(tf.float32, None)
      switched = tf.raw_ops.Switch(data=x, pred=True)
      merged, _ = tf.raw_ops.Merge(inputs=list(switched))
      result = merged
    with tf.compat.v1.Session(graph=graph,
        config=tf.compat.v1.ConfigProto(allow_soft_placement=False)) as session:
      value = session.run(result, {x: np.ones((0, 3), np.float32)})
      self.assertEqual(value.shape, (0, 3))

  def testEmptyWhile(self):
    signature = [tf.TensorSpec([], tf.int32), tf.TensorSpec([None, 3], tf.float32)]
    @tf.function(input_signature=signature)
    def cond(i, x):
      with tf.device('/CPU:0'):
        return i < 2
    @tf.function(input_signature=signature)
    def body(i, x):
      with tf.device('/CPU:0'):
        return i + 1, tf.identity(x)
    # Call the registered functional kernel directly; tf.while_loop lowering
    # tests Switch/Merge/Enter/Exit graph rewriting instead of this kernel.
    with tf.device('/device:MUSA:0'):
      result = tf.raw_ops.StatelessWhile(
          input=[tf.constant(0), tf.zeros([0, 3])],
          cond=cond.get_concrete_function(), body=body.get_concrete_function(),
          output_shapes=[[], [None, 3]])
    self.assertEqual(result[0].numpy(), 2)
    self.assertEqual(result[1].numpy().shape, (0, 3))

  def testEmptyCheckpoint(self):
    prefix = os.path.join(self.get_temp_dir(), 'empty')
    with tf.device('/device:MUSA:0'):
      tf.raw_ops.SaveV2(prefix=prefix, tensor_names=['empty'], shape_and_slices=[''],
                       tensors=[tf.zeros([0, 3])])
      restored = tf.raw_ops.RestoreV2(prefix=prefix, tensor_names=['empty'],
                                    shape_and_slices=[''], dtypes=[tf.float32])[0]
      self.assertEqual(restored.numpy().shape, (0, 3))

if __name__ == '__main__':
  tf.test.main()
