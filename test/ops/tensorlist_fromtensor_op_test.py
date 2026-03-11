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
"""Tests for MUSA TensorListFromTensor operator."""

import sys
import numpy as np
import tensorflow as tf

sys.path.append("..")
from musa_test_utils import MUSATestCase


class TensorListFromTensorOpTest(MUSATestCase):

  def _tensor_list_from_tensor_and_stack(self, x, element_dtype):
    element_shape = tf.constant(x.shape[1:], dtype=tf.int32)

    handle = tf.raw_ops.TensorListFromTensor(
        tensor=x,
        element_shape=element_shape
    )

    return tf.raw_ops.TensorListStack(
        input_handle=handle,
        element_shape=element_shape,
        element_dtype=element_dtype,
        num_elements=x.shape[0]
    )

  def _test_tensor_list_from_tensor(self, shape, dtype, rtol=1e-5, atol=1e-5):
    np_dtype = dtype.as_numpy_dtype
    if dtype == tf.bfloat16:
      np_dtype = np.float32

    if dtype == tf.bool:
      x_np = np.random.choice([True, False], size=shape)
    elif dtype.is_integer:
      x_np = np.random.randint(-10, 10, size=shape).astype(np_dtype)
    elif dtype == tf.float16:
      x_np = np.random.uniform(-5.0, 5.0, size=shape).astype(np_dtype)
    elif dtype == tf.bfloat16:
      x_np = np.random.uniform(-3.0, 3.0, size=shape).astype(np_dtype)
    else:
      x_np = np.random.uniform(-10.0, 10.0, size=shape).astype(np_dtype)

    x = tf.constant(x_np, dtype=dtype)

    def func(inp):
      return self._tensor_list_from_tensor_and_stack(
          inp,
          element_dtype=dtype
      )

    self._compare_cpu_musa_results(
        func,
        [x],
        dtype,
        rtol=rtol,
        atol=atol
    )

  def testTensorListFromTensorFloat32(self):
    self._test_tensor_list_from_tensor(
        [10, 10],
        tf.float32,
        rtol=1e-4,
        atol=1e-4
    )

  def testTensorListFromTensorFloat16(self):
    self._test_tensor_list_from_tensor(
        [2, 3, 4],
        tf.float16,
        rtol=1e-2,
        atol=1e-2
    )

  def testTensorListFromTensorBFloat16(self):
    self._test_tensor_list_from_tensor(
        [10, 10],
        tf.bfloat16,
        rtol=1e-1,
        atol=1e-1
    )

  def testTensorListFromTensorFloat64(self):
    self._test_tensor_list_from_tensor(
        [4, 5],
        tf.float64,
        rtol=1e-5,
        atol=1e-5
    )

  def testTensorListFromTensorInt32(self):
    self._test_tensor_list_from_tensor(
        [6, 7],
        tf.int32,
        rtol=0,
        atol=0
    )

  def testTensorListFromTensorInt64(self):
    self._test_tensor_list_from_tensor(
        [3, 4, 5],
        tf.int64,
        rtol=0,
        atol=0
    )

  


if __name__ == "__main__":
  tf.test.main()