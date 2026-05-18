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

"""Gradient registrations for MUSA extension ops."""

from tensorflow.python.framework import ops as tf_ops

from . import raw_ops


@tf_ops.RegisterGradient("MusaLayerNorm")
def _musa_layer_norm_grad(op, grad):
    epsilon = op.get_attr("epsilon")
    dx, dgamma, dbeta = raw_ops.musa_layer_norm_grad(
        dy=grad,
        x=op.inputs[0],
        gamma=op.inputs[1],
        beta=op.inputs[2],
        epsilon=epsilon,
    )
    return dx, dgamma, dbeta
