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

"""Empty variable updates must be valid no-ops, including optimizer slots."""
import inspect
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class EmptyTrainingOpsTest(MUSATestCase):
  def testEmptyResourceUpdates(self):
    names = ['ResourceApplyAdam', 'ResourceApplyAdaMax',
             'ResourceApplyAdagradV2', 'ResourceApplyGradientDescent',
             'ResourceApplyRMSProp', 'ResourceApplyCenteredRMSProp',
             'ResourceApplyFtrl', 'ResourceApplyFtrlV2']
    scalars = dict(lr=.01, alpha=.01, beta1_power=.9, beta2_power=.99,
                   beta1=.9, beta2=.99, epsilon=1e-6, rho=.9, momentum=.1,
                   l1=0., l2=0., l2_shrinkage=0., lr_power=-.5)
    slot_names = {'var', 'm', 'v', 'accum', 'ms', 'mg', 'mom', 'linear'}
    for name in names:
      op = getattr(tf.raw_ops, name)
      for dtype in [tf.float32, tf.float16, tf.bfloat16]:
        for shape in [(0,), (2, 0, 3)]:
          with self.subTest(op=name, dtype=dtype, shape=shape):
            with tf.device('/device:MUSA:0'):
              variables = {}
              kwargs = {}
              for key, param in inspect.signature(op).parameters.items():
                if key in slot_names:
                  variables[key] = tf.Variable(np.ones(shape), dtype=dtype)
                  kwargs[key] = variables[key].handle
                elif key in scalars:
                  kwargs[key] = tf.constant(scalars[key], dtype=dtype)
                elif key in ('grad', 'delta'):
                  kwargs[key] = tf.constant(np.ones(shape), dtype=dtype)
                elif key == 'use_locking':
                  kwargs[key] = False
              op(**kwargs)
              for var in variables.values():
                self.assertIn('device:MUSA:0', var.device)
                self.assertEqual(var.numpy().shape, shape)
                self.assertEqual(var.numpy().dtype, np.dtype(dtype.as_numpy_dtype))

  def testEmptyRefUpdates(self):
    names = ['ApplyAdam', 'ApplyAdaMax', 'ApplyAdagradV2',
             'ApplyGradientDescent', 'ApplyRMSProp', 'ApplyCenteredRMSProp',
             'ApplyFtrl', 'ApplyFtrlV2']
    scalars = dict(lr=.01, alpha=.01, beta1_power=.9, beta2_power=.99,
                   beta1=.9, beta2=.99, epsilon=1e-6, rho=.9, momentum=.1,
                   l1=0., l2=0., l2_shrinkage=0., lr_power=-.5)
    slot_names = {'var', 'm', 'v', 'accum', 'ms', 'mg', 'mom', 'linear'}
    for name in names:
      for dtype in [tf.float32, tf.float16, tf.bfloat16]:
        with self.subTest(op=name, dtype=dtype):
          graph = tf.Graph()
          with graph.as_default(), tf.device('/device:MUSA:0'):
            kwargs, variables = {}, []
            op = getattr(tf.raw_ops, name)
            for key in inspect.signature(op).parameters:
              if key in slot_names:
                var = tf.compat.v1.get_variable(key, initializer=tf.zeros([0, 3], dtype), use_resource=False)
                variables.append(var)
                kwargs[key] = var
              elif key in scalars:
                kwargs[key] = tf.constant(scalars[key], dtype)
              elif key in ('grad', 'delta'):
                kwargs[key] = tf.zeros([0, 3], dtype)
              elif key == 'use_locking':
                kwargs[key] = False
            update = op(**kwargs)
            with tf.control_dependencies([update]):
              results = [tf.identity(v) for v in variables]
            init = tf.compat.v1.global_variables_initializer()
          config = tf.compat.v1.ConfigProto(allow_soft_placement=False)
          config.graph_options.rewrite_options.disable_meta_optimizer = True
          with tf.compat.v1.Session(graph=graph, config=config) as session:
            session.run(init)
            for result in session.run(results):
              self.assertEqual(result.shape, (0, 3))
              self.assertEqual(result.dtype, np.dtype(dtype.as_numpy_dtype))


  def testRefUpdatesStillChangeNonemptyVariables(self):
    def run(device, adam):
      graph = tf.Graph()
      with graph.as_default(), tf.device(device):
        var = tf.compat.v1.get_variable('var', initializer=[1., 2.], use_resource=False)
        if adam:
          m = tf.compat.v1.get_variable('m', initializer=[0., 0.], use_resource=False)
          v = tf.compat.v1.get_variable('v', initializer=[0., 0.], use_resource=False)
          update = tf.raw_ops.ApplyAdam(var=var, m=m, v=v,
              beta1_power=.9, beta2_power=.99, lr=.1, beta1=.9, beta2=.99,
              epsilon=1e-6, grad=[1., 2.], use_locking=True)
          variables = [var, m, v]
        else:
          update = tf.raw_ops.ApplyGradientDescent(var=var, alpha=.1,
                                                  delta=[1., 2.], use_locking=True)
          variables = [var]
        with tf.control_dependencies([update]):
          results = [tf.identity(t) for t in variables]
        init = tf.compat.v1.global_variables_initializer()
      with tf.compat.v1.Session(graph=graph,
          config=tf.compat.v1.ConfigProto(allow_soft_placement=False)) as session:
        session.run(init)
        return session.run(results)
    for adam in [False, True]:
      with self.subTest(adam=adam):
        cpu, musa = run('/CPU:0', adam), run('/device:MUSA:0', adam)
        for a, b in zip(cpu, musa):
          self.assertAllClose(a, b)


if __name__ == '__main__':
  tf.test.main()
