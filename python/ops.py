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

"""Convenience wrappers for selected MUSA extension ops."""

from . import raw_ops


def clip(x, lo, hi, name=None):
    return raw_ops.musa_clip(x=x, lo=lo, hi=hi, name=name)


def layer_norm(x, gamma, beta, epsilon=0.00001, name=None):
    return raw_ops.musa_layer_norm(
        x=x,
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
        name=name,
    )


def shifted_affine_map(data_left, mask, sliced_var_right, name=None):
    return raw_ops.musa_shifted_affine_map(
        data_left=data_left,
        mask=mask,
        sliced_var_right=sliced_var_right,
        name=name,
    )


def interact(input, name=None):
    return raw_ops.musa_interact(input=input, name=name)


def dropout(x, rate=0.5, seed=0, offset=0, name=None):
    return raw_ops.musa_dropout(
        x=x,
        rate=rate,
        seed=seed,
        offset=offset,
        name=name,
    )


def dropout_grad(grad, mask, rate=0.5, name=None):
    return raw_ops.musa_dropout_grad(
        grad=grad,
        mask=mask,
        rate=rate,
        name=name,
    )


def resource_sparse_apply_adam(
    var,
    m,
    v,
    beta1_power,
    beta2_power,
    lr,
    beta1,
    beta2,
    epsilon,
    grad,
    indices,
    use_locking=False,
    name=None,
):
    return raw_ops.musa_resource_sparse_apply_adam(
        var=var,
        m=m,
        v=v,
        beta1_power=beta1_power,
        beta2_power=beta2_power,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        grad=grad,
        indices=indices,
        use_locking=use_locking,
        name=name,
    )


def gelu(x, approximate=False, name=None):
    return raw_ops.musa_gelu(
        x=x,
        approximate=approximate,
        name=name,
    )


def reshape_mat_mul(x, w, transpose_b=False, name=None):
    return raw_ops.musa_reshape_mat_mul(
        x=x,
        w=w,
        transpose_b=transpose_b,
        name=name,
    )


def matmul_bias_add(a, b, bias, transpose_a=False, transpose_b=False, name=None):
    return raw_ops.musa_mat_mul_bias_add(
        a=a,
        b=b,
        bias=bias,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        name=name,
    )


def resource_apply_nadam(
    var,
    m,
    v,
    beta1_power,
    beta2_power,
    lr,
    beta1,
    beta2,
    epsilon,
    grad,
    use_locking=False,
    name=None,
):
    return raw_ops.ResourceApplyNadam(
        var=var,
        m=m,
        v=v,
        beta1_power=beta1_power,
        beta2_power=beta2_power,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        grad=grad,
        use_locking=use_locking,
        name=name,
    )


__all__ = [
    "clip",
    "dropout",
    "gelu",
    "interact",
    "matmul_bias_add",
    "reshape_mat_mul",
    "resource_apply_nadam",
    "resource_sparse_apply_adam",
    "shifted_affine_map",
]
