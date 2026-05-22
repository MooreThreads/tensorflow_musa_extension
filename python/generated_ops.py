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

"""Generated public wrappers for MUSA extension ops."""

from . import raw_ops


def batch_mat_mul_v2(x, y, adj_x=False, adj_y=False, name=None):
    return raw_ops.musa_batch_mat_mul_v2(
        x=x,
        y=y,
        adj_x=adj_x,
        adj_y=adj_y,
        name=name,
    )


def bias_add_relu_mat_mul(input, bias, other, relu_input_slot, transpose_a=False, transpose_b=False, name=None):
    return raw_ops.musa_bias_add_relu_mat_mul(
        input=input,
        bias=bias,
        other=other,
        relu_input_slot=relu_input_slot,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        name=name,
    )


def clip(x, lo, hi, name=None):
    return raw_ops.musa_clip(
        x=x,
        lo=lo,
        hi=hi,
        name=name,
    )


def concat_mat_mul(inputs, axis, other, concat_input_idx, transpose_a=False, transpose_b=False, name=None):
    return raw_ops.musa_concat_mat_mul(
        inputs=inputs,
        axis=axis,
        other=other,
        concat_input_idx=concat_input_idx,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        name=name,
    )


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


def gelu(x, approximate=False, name=None):
    return raw_ops.musa_gelu(x=x, approximate=approximate, name=name)


def interact(input, name=None):
    return raw_ops.musa_interact(input=input, name=name)


def layer_norm(x, gamma, beta, epsilon=0.00001, name=None):
    return raw_ops.musa_layer_norm(
        x=x,
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
        name=name,
    )


def linear_activation(a, b, bias, activation='relu', alpha=0.0, transpose_a=False, transpose_b=False, name=None):
    return raw_ops.musa_linear_activation(
        a=a,
        b=b,
        bias=bias,
        activation=activation,
        alpha=alpha,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        name=name,
    )


def mat_mul(a, b, transpose_a=False, transpose_b=False, name=None):
    return raw_ops.musa_mat_mul(
        a=a,
        b=b,
        transpose_a=transpose_a,
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


def maximum(x, y, name=None):
    return raw_ops.musa_maximum(x=x, y=y, name=name)


def mean(input, reduction_indices, keep_dims=False, name=None):
    return raw_ops.musa_mean(
        input=input,
        reduction_indices=reduction_indices,
        keep_dims=keep_dims,
        name=name,
    )


def normalize(x, gamma, beta, epsilon=1e-11, max_std=float('inf'), name=None):
    return raw_ops.musa_normalize(
        x=x,
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
        max_std=max_std,
        name=name,
    )


def pln_cascade(norm_out, adpos, add_input, bias_input, use_table=False, table_index=0, select_on_true=True, name=None):
    return raw_ops.musa_pln_cascade(
        norm_out=norm_out,
        adpos=adpos,
        add_input=add_input,
        bias_input=bias_input,
        use_table=use_table,
        table_index=table_index,
        select_on_true=select_on_true,
        name=name,
    )


def pln_cascade_block(norm_out, add_input, bias_input, gates, table_indices, select_on_true, name=None):
    return raw_ops.musa_pln_cascade_block(
        norm_out=norm_out,
        add_input=add_input,
        bias_input=bias_input,
        gates=gates,
        table_indices=table_indices,
        select_on_true=select_on_true,
        name=name,
    )


def prelu(x, alpha, name=None):
    return raw_ops.musa_p_relu(x=x, alpha=alpha, name=name)


def reshape_mat_mul(x, w, transpose_b=False, name=None):
    return raw_ops.musa_reshape_mat_mul(
        x=x,
        w=w,
        transpose_b=transpose_b,
        name=name,
    )


def resource_apply_adam_mixed(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking=False, use_nesterov=False, name=None):
    return raw_ops.musa_resource_apply_adam_mixed(
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
        use_nesterov=use_nesterov,
        name=name,
    )


def resource_sparse_apply_adam(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, indices, use_locking=False, name=None):
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


def shifted_affine_map(data_left, mask, sliced_var_right, name=None):
    return raw_ops.musa_shifted_affine_map(
        data_left=data_left,
        mask=mask,
        sliced_var_right=sliced_var_right,
        name=name,
    )


def tensor_dot(a, b, axes_a, axes_b, name=None):
    return raw_ops.musa_tensor_dot(
        a=a,
        b=b,
        axes_a=axes_a,
        axes_b=axes_b,
        name=name,
    )


def tensor_dot_bias(a, b, bias, axes_a, axes_b, name=None):
    return raw_ops.musa_tensor_dot_bias(
        a=a,
        b=b,
        bias=bias,
        axes_a=axes_a,
        axes_b=axes_b,
        name=name,
    )


def token_mixer(x, num_T, num_H, d_k, name=None):
    return raw_ops.musa_token_mixer(
        x=x,
        num_T=num_T,
        num_H=num_H,
        d_k=d_k,
        name=name,
    )


def resource_apply_nadam(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking=False, name=None):
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
    "batch_mat_mul_v2",
    "bias_add_relu_mat_mul",
    "clip",
    "concat_mat_mul",
    "dropout",
    "dropout_grad",
    "gelu",
    "interact",
    "layer_norm",
    "linear_activation",
    "mat_mul",
    "matmul_bias_add",
    "maximum",
    "mean",
    "normalize",
    "pln_cascade",
    "pln_cascade_block",
    "prelu",
    "reshape_mat_mul",
    "resource_apply_adam_mixed",
    "resource_apply_nadam",
    "resource_sparse_apply_adam",
    "shifted_affine_map",
    "tensor_dot",
    "tensor_dot_bias",
    "token_mixer",
]
