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

"""Runtime empty-dimension contracts shared across MUSA operators.

Inputs are fed to unknown-shape placeholders, so neither constant folding nor
static shape inference can hide an empty-input kernel bug. CPU is the oracle;
shape and dtype are checked separately because allclose can broadcast empties.
"""
import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class EmptyDimensionsOpTest(MUSATestCase):
  def check_case(self, op, inputs, attrs=None, invalid=False, reference=None):
    with tf.device('/CPU:0'):
      cpu_inputs = tf.nest.map_structure(tf.convert_to_tensor, inputs)
      if invalid:
        with self.assertRaises(tf.errors.InvalidArgumentError):
          (reference or op)(**cpu_inputs, **(attrs or {}))
      else:
        cpu = tf.nest.map_structure(
            lambda t: t.numpy(),
            (reference or op)(**cpu_inputs, **(attrs or {}), name='cpu_target'))
    graph = tf.Graph()
    with graph.as_default():
      with tf.device('/CPU:0'):
        feeds = tf.nest.map_structure(
            lambda value: tf.compat.v1.placeholder(tf.as_dtype(value.dtype), None), inputs)
      with tf.device('/device:MUSA:0'):
        actual = op(**feeds, **(attrs or {}), name='musa_target')
    config = tf.compat.v1.ConfigProto(allow_soft_placement=False)
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    config.graph_options.optimizer_options.opt_level = -1
    with tf.compat.v1.Session(graph=graph, config=config) as session:
      feed = dict(zip(tf.nest.flatten(feeds), tf.nest.flatten(inputs)))
      if invalid:
        with self.assertRaises(tf.errors.InvalidArgumentError):
          session.run(actual, feed)
        return
      metadata = tf.compat.v1.RunMetadata()
      musa = session.run(
          actual, feed,
          options=tf.compat.v1.RunOptions(output_partition_graphs=True),
          run_metadata=metadata)
      for index, (a, b) in enumerate(zip(tf.nest.flatten(cpu), tf.nest.flatten(musa))):
        self.assertEqual(a.dtype, b.dtype)
        # BatchNorm reserves are opaque and backend dependent; CPU leaves
        # empty-input reserves uninitialized. Compare public outputs only.
        if op.__name__.startswith('fused_batch_norm') and index >= 3:
          continue
        self.assertEqual(a.shape, b.shape)
        if a.dtype.kind in 'OSUbiu' or a.size == 0:
          self.assertAllEqual(a, b)
        else:
          if a.dtype == tf.bfloat16.as_numpy_dtype:
            a, b = a.astype(np.float32), b.astype(np.float32)
          np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True)
      targets = [n for g in metadata.partition_graphs for n in g.node
                 if n.name == 'musa_target']
      self.assertLen(targets, 1)
      self.assertIn('device:MUSA:0', targets[0].device)


def array(shape, dtype=np.float32):
  return np.ones(shape, dtype=dtype)


def add_cases(name, cases):
  def test(self):
    for label, inputs, attrs, invalid in cases:
      with self.subTest(case=label):
        self.check_case(getattr(tf.raw_ops, name), inputs, attrs, invalid,
                        reference=CPU_REFERENCES.get(name))
  setattr(EmptyDimensionsOpTest, 'testEmpty' + name, test)


# oneDNN in TF 2.15 returns the input shape for a zero-batch Conv2D.
# Convolve a dummy row and slice it away to obtain a reliable CPU shape oracle.
def _empty_conv_reference(input, filter, strides, padding, name):
  dummy = tf.ones(tf.concat([[1], tf.shape(input)[1:]], 0), input.dtype)
  return tf.raw_ops.Conv2D(input=dummy, filter=filter, strides=strides,
                          padding=padding, name=name)[:0]

CPU_REFERENCES = {'Conv2D': _empty_conv_reference}

for name in ('Abs Neg Exp Expm1 Log Log1p Square Sqrt Rsqrt Reciprocal Round '
             'Sign Erf IsNan Relu LeakyRelu Sigmoid Tanh Softplus Identity '
             'Snapshot StopGradient ZerosLike').split():
  add_cases(name, [(str(s), {('features' if name in ('Relu', 'LeakyRelu', 'Softplus') else
       'x' if name == 'ZerosLike' else
       'input' if name in ('Identity', 'Snapshot', 'StopGradient') else 'x'): array(s)}, {}, False)
      for s in [(0,), (0, 3), (2, 0, 3)]])

for name in ('Add AddV2 Sub Mul RealDiv DivNoNan FloorDiv FloorMod Pow '
             'SquaredDifference Maximum Minimum Equal NotEqual Less LessEqual '
             'Greater GreaterEqual LogicalAnd LogicalOr BitwiseAnd').split():
  dtype = np.bool_ if name.startswith('Logical') else (
      np.int32 if name == 'BitwiseAnd' else np.float32)
  cases = []
  for x, y in [((0,), (1,)), ((1,), (0,)), ((0,), (0,)), ((0,), ()),
               ((), (0,)), ((2, 0, 3), (1, 1, 3)), ((0, 1), (1, 0))]:
    cases.append((str((x, y)), {'x': array(x, dtype), 'y': array(y, dtype)}, {}, False))
  for x, y in [((0,), (2,)), ((2,), (0,)), ((0, 3), (1, 4))]:
    cases.append((str((x, y)), {'x': array(x, dtype), 'y': array(y, dtype)}, {}, True))
  add_cases(name, cases)

for name in ['Sum', 'Prod', 'Mean', 'Min', 'Max', 'All']:
  cases = []
  for dtype in ([np.bool_] if name == 'All' else [np.float32, np.float16, tf.bfloat16.as_numpy_dtype, np.float64, np.int32, np.int64] if name != 'Mean' else [np.float32, np.float16, tf.bfloat16.as_numpy_dtype]):
    for shape in [(0,), (0, 3), (2, 0, 3)]:
      for axes in [[], [0], [-1], list(range(len(shape))), [len(shape)]]:
        for keep in [False, True]:
          cases.append((str((shape, axes, keep, dtype)),
                        {'input': array(shape, dtype),
                         'axis': np.array(axes, np.int32)},
                        {'keep_dims': keep}, axes == [len(shape)]))
  add_cases(name, cases)

# Shape-changing and indexed operations: empty output, and empty input that
# must still produce nonempty values (Pad, MatMul, reductions, segment sum).
CASES = {
    'LogicalNot': [({'x': array((0, 3), np.bool_)}, {})],
    'Cast': [({'x': array((0, 3))}, {'DstT': tf.float16})],
    'Reshape': [({'tensor': array((0, 3)), 'shape': np.array([3, 0], np.int32)}, {})],
    'Transpose': [({'x': array((2, 0, 3)), 'perm': np.array([2, 0, 1], np.int32)}, {})],
    'ExpandDims': [({'input': array((0, 3)), 'axis': np.array(1, np.int32)}, {})],
    'Squeeze': [({'input': array((1, 0, 3))}, {'axis': [0]})],
    'Shape': [({'input': array((0, 3))}, {})],
    'Size': [({'input': array((0, 3))}, {})],
    'Rank': [({'input': array((0, 3))}, {})],
    'Fill': [({'dims': np.array([2, 0, 3], np.int32), 'value': np.array(2., np.float32)}, {})],
    'Tile': [({'input': array((2, 3)), 'multiples': np.array([0, 2], np.int32)}, {}),
             ({'input': array((0, 3)), 'multiples': np.array([2, 2], np.int32)}, {})],
    'BroadcastTo': [({'input': array((1, 3)), 'shape': np.array([0, 3], np.int32)}, {})],
    'Pad': [({'input': array((0, 3)), 'paddings': np.array([[1, 1], [0, 0]], np.int32)}, {})],
    'PadV2': [({'input': array((0, 3)), 'paddings': np.array([[1, 1], [0, 0]], np.int32), 'constant_values': np.array(7., np.float32)}, {})],
    'Slice': [({'input': array((2, 3)), 'begin': np.array([0, 0], np.int32), 'size': np.array([0, 3], np.int32)}, {})],
    'ReverseV2': [({'tensor': array((0, 3)), 'axis': np.array([0], np.int32)}, {})],
    'DiagPart': [({'input': array((0, 0))}, {})],
    'MatrixDiagPartV3': [({'input': array((0, 3, 3)), 'k': np.array(0, np.int32), 'padding_value': np.array(0., np.float32)}, {})],
    'MatrixBandPart': [({'input': array((0, 3, 3)), 'num_lower': np.array(-1, np.int64), 'num_upper': np.array(0, np.int64)}, {})],
    'InvertPermutation': [({'x': np.array([], np.int32)}, {})],
    'Where': [({'condition': array((0, 3), np.bool_)}, {})],
    'GatherV2': [({'params': array((2, 3)), 'indices': np.array([], np.int32), 'axis': np.array(0, np.int32)}, {})],
    'GatherNd': [({'params': array((2, 3)), 'indices': np.zeros((0, 1), np.int32)}, {})],
    'ScatterNd': [({'indices': np.zeros((0, 1), np.int32), 'updates': array((0, 3)), 'shape': np.array([2, 3], np.int32)}, {})],
    'UnsortedSegmentSum': [({'data': array((0, 3)), 'segment_ids': np.array([], np.int32), 'num_segments': np.array(2, np.int32)}, {})],
    'ArgMax': [({'input': array((0, 3)), 'dimension': np.array(1, np.int32)}, {})],
    'Softmax': [({'logits': array((0, 3))}, {}), ({'logits': array((2, 0))}, {})],
    'BiasAdd': [({'value': array((0, 3)), 'bias': array((3,))}, {})],
    'BiasAddGrad': [({'out_backprop': array((0, 3))}, {})],
    'SparseSoftmaxCrossEntropyWithLogits': [({'features': array((0, 3)), 'labels': np.array([], np.int64)}, {})],
    'TopKV2': [({'input': array((0, 3)), 'k': np.array(2, np.int32)}, {}), ({'input': array((2, 3)), 'k': np.array(0, np.int32)}, {})],
    'InTopKV2': [({'predictions': array((0, 3)), 'targets': np.array([], np.int32), 'k': np.array(1, np.int32)}, {})],
    'Range': [({'start': np.array(0, np.int32), 'limit': np.array(0, np.int32), 'delta': np.array(1, np.int32)}, {})],
    'AsString': [({'input': array((0, 3))}, {})],
    'StringToHashBucketFast': [({'input': np.empty((0, 3), dtype=object)}, {'num_buckets': 10})],
    'StaticRegexFullMatch': [({'input': np.empty((0,), dtype=object)}, {'pattern': '.*'})],
}
for name in ['MatMul', 'BatchMatMulV2']:
  CASES[name] = [({'a' if name == 'MatMul' else 'x': array(a),
                  'b' if name == 'MatMul' else 'y': array(b)}, {})
                 for a, b in [((0, 3), (3, 2)), ((2, 0), (0, 3)), ((2, 3), (3, 0))]]
for name, keys in [('ReluGrad', ('gradients', 'features')), ('LeakyReluGrad', ('gradients', 'features')),
                   ('TanhGrad', ('y', 'dy')), ('SigmoidGrad', ('y', 'dy')),
                   ('SqrtGrad', ('y', 'dy')), ('RsqrtGrad', ('y', 'dy'))]:
  CASES[name] = [({k: array((0, 3)) for k in keys}, {})]
for name, cases in CASES.items():
  add_cases(name, [(str(i), inputs, attrs, False) for i, (inputs, attrs) in enumerate(cases)])

MORE_CASES = {
    'AddN': [({'inputs': [array((0, 3)), array((0, 3))]}, {})],
    'IdentityN': [({'input': [array((0, 3)), array((2, 0))]}, {})],
    'ShapeN': [({'input': [array((0, 3)), array((2, 0))]}, {})],
    'Pack': [({'values': [array((0, 3)), array((0, 3))]}, {'axis': 0})],
    'Unpack': [({'value': array((2, 0, 3))}, {'num': 2, 'axis': 0})],
    'ConcatV2': [({'values': [array((0, 3)), array((2, 3))], 'axis': np.array(0, np.int32)}, {}),
                 ({'values': [array((0, 3)), array((0, 3))], 'axis': np.array(1, np.int32)}, {})],
    'Split': [({'value': array((0, 4)), 'axis': np.array(1, np.int32)}, {'num_split': 2})],
    'SplitV': [({'value': array((0, 4)), 'size_splits': np.array([0, 0], np.int32), 'axis': np.array(0, np.int32)}, {'num_split': 2})],
    'Unique': [({'x': array((0,), np.int32)}, {})],
    'Select': [({'condition': array((0,), np.bool_), 'x': array((0, 3)), 'y': array((0, 3))}, {})],
    'SelectV2': [({'condition': array((0, 1), np.bool_), 't': array((1, 3)), 'e': array(())}, {})],
    'Einsum': [({'inputs': [array((2, 0)), array((0, 3))]}, {'equation': 'ij,jk->ik'})],
    'StridedSlice': [({'input': array((0, 3)), 'begin': np.array([0, 0], np.int32), 'end': np.array([0, 3], np.int32), 'strides': np.array([1, 1], np.int32)}, {})],
    'StridedSliceGrad': [({'shape': np.array([2, 3], np.int32), 'begin': np.array([0, 0], np.int32), 'end': np.array([0, 3], np.int32), 'strides': np.array([1, 1], np.int32), 'dy': array((0, 3))}, {})],
    'MatrixSetDiagV3': [({'input': array((0, 3, 3)), 'diagonal': array((0, 3)), 'k': np.array(0, np.int32)}, {})],
    'SparseTensorDenseMatMul': [({'a_indices': np.empty((0, 2), np.int64), 'a_values': array((0,)), 'a_shape': np.array([2, 3], np.int64), 'b': array((3, 4))}, {})],
    'SparseSlice': [({'indices': np.empty((0, 2), np.int64), 'values': array((0,)), 'shape': np.array([2, 3], np.int64), 'start': np.array([0, 0], np.int64), 'size': np.array([2, 3], np.int64)}, {})],
    'BroadcastGradientArgs': [({'s0': np.array([0, 3], np.int32), 's1': np.array([1, 3], np.int32)}, {})],
    'ConcatOffset': [({'concat_dim': np.array(0, np.int32), 'shape': [np.array([0, 3], np.int32), np.array([2, 3], np.int32)]}, {})],
    'StringJoin': [({'inputs': [np.empty((0,), object), np.empty((0,), object)]}, {})],
}
for name in ['RandomUniform', 'RandomStandardNormal', 'TruncatedNormal']:
  MORE_CASES[name] = [({'shape': np.array([2, 0, 3], np.int32)}, {'dtype': tf.float32})]
MORE_CASES['RandomUniformInt'] = [({'shape': np.array([0, 3], np.int32), 'minval': np.array(0, np.int32), 'maxval': np.array(10, np.int32)}, {})]
for name in ['Conv2D', 'MaxPool', 'ExtractImagePatches']:
  inputs = {'input' if name != 'ExtractImagePatches' else 'images': array((0, 4, 4, 2))}
  attrs = {'strides': [1, 1, 1, 1], 'padding': 'VALID'}
  if name == 'Conv2D':
    inputs['filter'] = array((2, 2, 2, 3))
  else:
    attrs['ksizes'] = [1, 2, 2, 1]
    if name == 'MaxPool':
      attrs['ksize'] = attrs.pop('ksizes')
    else:
      attrs['rates'] = [1, 1, 1, 1]
  MORE_CASES[name] = [(inputs, attrs)]

MORE_CASES['Conv2DBackpropInput'] = [({'input_sizes': np.array([0, 4, 4, 2], np.int32), 'filter': array((2, 2, 2, 3)), 'out_backprop': array((0, 3, 3, 3))}, {'strides': [1, 1, 1, 1], 'padding': 'VALID'})]
MORE_CASES['Conv2DBackpropFilter'] = [({'input': array((0, 4, 4, 2)), 'filter_sizes': np.array([2, 2, 2, 3], np.int32), 'out_backprop': array((0, 3, 3, 3))}, {'strides': [1, 1, 1, 1], 'padding': 'VALID'})]
MORE_CASES['MaxPoolGrad'] = [({'orig_input': array((0, 4, 4, 2)), 'orig_output': array((0, 3, 3, 2)), 'grad': array((0, 3, 3, 2))}, {'ksize': [1, 2, 2, 1], 'strides': [1, 1, 1, 1], 'padding': 'VALID'})]
MORE_CASES['FusedBatchNormV3'] = [({'x': array((0, 2, 2, 3)), 'scale': array((3,)), 'offset': array((3,)), 'mean': array((0,)), 'variance': array((0,))}, {'is_training': True})]

for name in ['FusedBatchNorm', 'FusedBatchNormV2']:
  MORE_CASES[name] = MORE_CASES['FusedBatchNormV3']
for name in ['FusedBatchNormGrad', 'FusedBatchNormGradV2', 'FusedBatchNormGradV3']:
  inputs = {'y_backprop': array((0, 2, 2, 3)), 'x': array((0, 2, 2, 3)),
            'scale': array((3,)), 'reserve_space_1': array((3,)), 'reserve_space_2': array((3,))}
  if name.endswith('V3'):
    inputs['reserve_space_3'] = array(())
  MORE_CASES[name] = [(inputs, {'is_training': True})]
for name, cases in MORE_CASES.items():
  add_cases(name, [(str(i), inputs, attrs, False) for i, (inputs, attrs) in enumerate(cases)])


def _invalid_bias(self):
  self.check_case(tf.raw_ops.BiasAdd,
                  {'value': array((0, 3)), 'bias': array((0,))}, invalid=True)
EmptyDimensionsOpTest.testInvalidEmptyBiasAdd = _invalid_bias


# Gradient kernels must reject a nonempty upstream gradient for empty data.
def _invalid_gradient(self):
  for name in ['TanhGrad', 'SigmoidGrad', 'RsqrtGrad', 'SqrtGrad']:
    for y, dy in [((0, 3), (1,)), ((1,), (0, 3))]:
      with self.subTest(op=name, y=y, dy=dy):
        self.check_case(getattr(tf.raw_ops, name),
                        {'y': array(y), 'dy': array(dy)}, invalid=True)
EmptyDimensionsOpTest.testInvalidEmptyGradient = _invalid_gradient

if __name__ == '__main__':
  tf.test.main()
