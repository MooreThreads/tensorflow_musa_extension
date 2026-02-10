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

"""Tests for MUSA Gather operator on Variables."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class GatherOpTest(MUSATestCase):

  def testGatherResourceVariable(self):
    embedding_matrix = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4]
    ], dtype=np.float32)
    ids_to_lookup = np.array([1, 3], dtype=np.int32)

    def gather_on_variable_op(params, indices):
      v = tf.Variable(params)
      return tf.gather(v, indices)

    self._compare_cpu_musa_results(
        gather_on_variable_op,
        [embedding_matrix, ids_to_lookup],
        tf.float32,
        rtol=1e-5,
        atol=1e-5
    )

  def testAssignAddVariable(self):
    embedding_matrix = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4]
    ], dtype=np.float32)
    update_val = np.ones_like(embedding_matrix, dtype=np.float32) * 0.1

    def assign_add_op(params, update):
      v = tf.Variable(params)
      v.assign_add(update)
      return v.read_value()

    self._compare_cpu_musa_results(
        assign_add_op,
        [embedding_matrix, update_val],
        tf.float32,
        rtol=1e-5,
        atol=1e-5
    )

  def testGatherRandomHighRank(self):
    shape = (10, 20, 5)
    params = np.random.rand(*shape).astype(np.float32)
    indices = np.random.randint(0, 10, size=(5,)).astype(np.int32)

    def gather_op(p, i):
      v = tf.Variable(p)
      return tf.gather(v, i)

    self._compare_cpu_musa_results(
        gather_op,
        [params, indices],
        tf.float32,
        rtol=1e-5,
        atol=1e-5
    )


if __name__ == "__main__":
  tf.test.main()
