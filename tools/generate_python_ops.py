#!/usr/bin/env python3
"""Generate Python wrappers for custom MUSA TensorFlow ops."""

from __future__ import annotations

import ast
import pprint
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS_ROOT = ROOT / "musa_ext" / "kernels"
GENERATED_OPS = ROOT / "python" / "generated_ops.py"
OP_MANIFEST = ROOT / "python" / "op_manifest.py"

OP_DEFINITIONS = [
    {
        "op": "MusaBatchMatMulV2",
        "raw": "musa_batch_mat_mul_v2",
        "api": "batch_mat_mul_v2",
        "source": "musa_ext/kernels/math/musa_matmul_op.cc",
        "args": ["x", "y", "adj_x=False", "adj_y=False", "name=None"],
        "call": ["x=x", "y=y", "adj_x=adj_x", "adj_y=adj_y", "name=name"],
    },
    {
        "op": "MusaBiasAddReluMatMul",
        "raw": "musa_bias_add_relu_mat_mul",
        "api": "bias_add_relu_mat_mul",
        "source": "musa_ext/kernels/fusion/musa_rgprojection_fusion_op.cc",
        "args": ["input", "bias", "other", "relu_input_slot", "transpose_a=False", "transpose_b=False", "name=None"],
        "call": ["input=input", "bias=bias", "other=other", "relu_input_slot=relu_input_slot", "transpose_a=transpose_a", "transpose_b=transpose_b", "name=name"],
    },
    {
        "op": "MusaClip",
        "raw": "musa_clip",
        "api": "clip",
        "source": "musa_ext/kernels/fusion/musa_clip_op.cc",
        "args": ["x", "lo", "hi", "name=None"],
        "call": ["x=x", "lo=lo", "hi=hi", "name=name"],
    },
    {
        "op": "MusaConcatMatMul",
        "raw": "musa_concat_mat_mul",
        "api": "concat_mat_mul",
        "source": "musa_ext/kernels/fusion/musa_concat_matmul_op.cc",
        "args": ["inputs", "axis", "other", "concat_input_idx", "transpose_a=False", "transpose_b=False", "name=None"],
        "call": ["inputs=inputs", "axis=axis", "other=other", "concat_input_idx=concat_input_idx", "transpose_a=transpose_a", "transpose_b=transpose_b", "name=name"],
    },
    {
        "op": "MusaDropout",
        "raw": "musa_dropout",
        "api": "dropout",
        "source": "musa_ext/kernels/nn/musa_dropout_op.cc",
        "args": ["x", "rate=0.5", "seed=0", "offset=0", "name=None"],
        "call": ["x=x", "rate=rate", "seed=seed", "offset=offset", "name=name"],
    },
    {
        "op": "MusaDropoutGrad",
        "raw": "musa_dropout_grad",
        "api": "dropout_grad",
        "source": "musa_ext/kernels/nn/musa_dropout_op.cc",
        "args": ["grad", "mask", "rate=0.5", "name=None"],
        "call": ["grad=grad", "mask=mask", "rate=rate", "name=name"],
    },
    {
        "op": "MusaGelu",
        "raw": "musa_gelu",
        "api": "gelu",
        "source": "musa_ext/kernels/fusion/musa_gelu_op.cc",
        "args": ["x", "approximate=False", "name=None"],
        "call": ["x=x", "approximate=approximate", "name=name"],
    },
    {
        "op": "MusaInteract",
        "raw": "musa_interact",
        "api": "interact",
        "source": "musa_ext/kernels/array/musa_tensorinteraction_op.cc",
        "args": ["input", "name=None"],
        "call": ["input=input", "name=name"],
    },
    {
        "op": "MusaLayerNorm",
        "raw": "musa_layer_norm",
        "api": "layer_norm",
        "source": "musa_ext/kernels/fusion/musa_layernorm_op.cc",
        "args": ["x", "gamma", "beta", "epsilon=0.00001", "name=None"],
        "call": ["x=x", "gamma=gamma", "beta=beta", "epsilon=epsilon", "name=name"],
    },
    {
        "op": "MusaLinearActivation",
        "raw": "musa_linear_activation",
        "api": "linear_activation",
        "source": "musa_ext/kernels/fusion/musa_linear_relu_op.cc",
        "args": ["a", "b", "bias", "activation='relu'", "alpha=0.0", "transpose_a=False", "transpose_b=False", "name=None"],
        "call": ["a=a", "b=b", "bias=bias", "activation=activation", "alpha=alpha", "transpose_a=transpose_a", "transpose_b=transpose_b", "name=name"],
    },
    {
        "op": "MusaMatMul",
        "raw": "musa_mat_mul",
        "api": "mat_mul",
        "source": "musa_ext/kernels/math/musa_matmul_op.cc",
        "args": ["a", "b", "transpose_a=False", "transpose_b=False", "name=None"],
        "call": ["a=a", "b=b", "transpose_a=transpose_a", "transpose_b=transpose_b", "name=name"],
    },
    {
        "op": "MusaMatMulBiasAdd",
        "raw": "musa_mat_mul_bias_add",
        "api": "matmul_bias_add",
        "source": "musa_ext/kernels/fusion/musa_matmul_bias_op.cc",
        "args": ["a", "b", "bias", "transpose_a=False", "transpose_b=False", "name=None"],
        "call": ["a=a", "b=b", "bias=bias", "transpose_a=transpose_a", "transpose_b=transpose_b", "name=name"],
    },
    {
        "op": "MusaMaximum",
        "raw": "musa_maximum",
        "api": "maximum",
        "source": "musa_ext/kernels/math/musa_maximum_op.cc",
        "args": ["x", "y", "name=None"],
        "call": ["x=x", "y=y", "name=name"],
    },
    {
        "op": "MusaMean",
        "raw": "musa_mean",
        "api": "mean",
        "source": "musa_ext/kernels/math/musa_mean_op.cc",
        "args": ["input", "reduction_indices", "keep_dims=False", "name=None"],
        "call": ["input=input", "reduction_indices=reduction_indices", "keep_dims=keep_dims", "name=name"],
    },
    {
        "op": "MusaNormalize",
        "raw": "musa_normalize",
        "api": "normalize",
        "source": "musa_ext/kernels/fusion/musa_normalize_fusion_op.cc",
        "args": ["x", "gamma", "beta", "epsilon=1e-11", "max_std=float('inf')", "name=None"],
        "call": ["x=x", "gamma=gamma", "beta=beta", "epsilon=epsilon", "max_std=max_std", "name=name"],
    },
    {
        "op": "MusaPlnCascade",
        "raw": "musa_pln_cascade",
        "api": "pln_cascade",
        "source": "musa_ext/kernels/fusion/musa_pln_cascade_op.cc",
        "args": ["norm_out", "adpos", "add_input", "bias_input", "use_table=False", "table_index=0", "select_on_true=True", "name=None"],
        "call": ["norm_out=norm_out", "adpos=adpos", "add_input=add_input", "bias_input=bias_input", "use_table=use_table", "table_index=table_index", "select_on_true=select_on_true", "name=name"],
    },
    {
        "op": "MusaPlnCascadeBlock",
        "raw": "musa_pln_cascade_block",
        "api": "pln_cascade_block",
        "source": "musa_ext/kernels/fusion/musa_pln_cascade_block_op.cc",
        "args": ["norm_out", "add_input", "bias_input", "gates", "table_indices", "select_on_true", "name=None"],
        "call": ["norm_out=norm_out", "add_input=add_input", "bias_input=bias_input", "gates=gates", "table_indices=table_indices", "select_on_true=select_on_true", "name=name"],
    },
    {
        "op": "MusaPRelu",
        "raw": "musa_p_relu",
        "api": "prelu",
        "source": "musa_ext/kernels/fusion/musa_prelu_fusion_op.cc",
        "args": ["x", "alpha", "name=None"],
        "call": ["x=x", "alpha=alpha", "name=name"],
    },
    {
        "op": "MusaReshapeMatMul",
        "raw": "musa_reshape_mat_mul",
        "api": "reshape_mat_mul",
        "source": "musa_ext/kernels/fusion/musa_reshape_matmul_op.cc",
        "args": ["x", "w", "transpose_b=False", "name=None"],
        "call": ["x=x", "w=w", "transpose_b=transpose_b", "name=name"],
    },
    {
        "op": "MusaResourceApplyAdamMixed",
        "raw": "musa_resource_apply_adam_mixed",
        "api": "resource_apply_adam_mixed",
        "source": "musa_ext/kernels/training/musa_applyadam_mixed_op.cc",
        "args": ["var", "m", "v", "beta1_power", "beta2_power", "lr", "beta1", "beta2", "epsilon", "grad", "use_locking=False", "use_nesterov=False", "name=None"],
        "call": ["var=var", "m=m", "v=v", "beta1_power=beta1_power", "beta2_power=beta2_power", "lr=lr", "beta1=beta1", "beta2=beta2", "epsilon=epsilon", "grad=grad", "use_locking=use_locking", "use_nesterov=use_nesterov", "name=name"],
    },
    {
        "op": "MusaResourceSparseApplyAdam",
        "raw": "musa_resource_sparse_apply_adam",
        "api": "resource_sparse_apply_adam",
        "source": "musa_ext/kernels/training/musa_apply_sparse_adam_op.cc",
        "args": ["var", "m", "v", "beta1_power", "beta2_power", "lr", "beta1", "beta2", "epsilon", "grad", "indices", "use_locking=False", "name=None"],
        "call": ["var=var", "m=m", "v=v", "beta1_power=beta1_power", "beta2_power=beta2_power", "lr=lr", "beta1=beta1", "beta2=beta2", "epsilon=epsilon", "grad=grad", "indices=indices", "use_locking=use_locking", "name=name"],
    },
    {
        "op": "MusaShiftedAffineMap",
        "raw": "musa_shifted_affine_map",
        "api": "shifted_affine_map",
        "source": "musa_ext/kernels/fusion/musa_shifted_affine_map_op.cc",
        "args": ["data_left", "mask", "sliced_var_right", "name=None"],
        "call": ["data_left=data_left", "mask=mask", "sliced_var_right=sliced_var_right", "name=name"],
    },
    {
        "op": "MusaTensorDot",
        "raw": "musa_tensor_dot",
        "api": "tensor_dot",
        "source": "musa_ext/kernels/fusion/musa_tensordot_op.cc",
        "args": ["a", "b", "axes_a", "axes_b", "name=None"],
        "call": ["a=a", "b=b", "axes_a=axes_a", "axes_b=axes_b", "name=name"],
    },
    {
        "op": "MusaTensorDotBias",
        "raw": "musa_tensor_dot_bias",
        "api": "tensor_dot_bias",
        "source": "musa_ext/kernels/fusion/musa_tensordot_bias_op.cc",
        "args": ["a", "b", "bias", "axes_a", "axes_b", "name=None"],
        "call": ["a=a", "b=b", "bias=bias", "axes_a=axes_a", "axes_b=axes_b", "name=name"],
    },
    {
        "op": "MusaTokenMixer",
        "raw": "musa_token_mixer",
        "api": "token_mixer",
        "source": "musa_ext/kernels/fusion/musa_tokenmixer_op.cc",
        "args": ["x", "num_T", "num_H", "d_k", "name=None"],
        "call": ["x=x", "num_T=num_T", "num_H=num_H", "d_k=d_k", "name=name"],
    },
    {
        "op": "ResourceApplyNadam",
        "raw": "ResourceApplyNadam",
        "api": "resource_apply_nadam",
        "source": "musa_ext/kernels/training/musa_resource_apply_nadam_op.cc",
        "args": ["var", "m", "v", "beta1_power", "beta2_power", "lr", "beta1", "beta2", "epsilon", "grad", "use_locking=False", "name=None"],
        "call": ["var=var", "m=m", "v=v", "beta1_power=beta1_power", "beta2_power=beta2_power", "lr=lr", "beta1=beta1", "beta2=beta2", "epsilon=epsilon", "grad=grad", "use_locking=use_locking", "name=name"],
    },
]

HEADER = """# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.\n#\n# Licensed under the Apache License, Version 2.0 (the \"License\");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#     http://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an \"AS IS\" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License.\n# ==============================================================================\n\n"""


def _registered_ops_from_source() -> set[str]:
    pattern = re.compile(r'REGISTER_OP\("([^"]+)"\)')
    registered = set()
    for path in KERNELS_ROOT.rglob("*.cc"):
        registered.update(pattern.findall(path.read_text(encoding="utf-8")))
    return registered


def _validate_definitions() -> None:
    source_ops = _registered_ops_from_source()
    defined_ops = {entry["op"] for entry in OP_DEFINITIONS}
    missing = sorted(source_ops - defined_ops)
    stale = sorted(defined_ops - source_ops)
    if missing or stale:
        messages = []
        if missing:
            messages.append(f"missing from OP_DEFINITIONS: {missing}")
        if stale:
            messages.append(f"not found in REGISTER_OP sources: {stale}")
        raise SystemExit("Custom op manifest is out of sync: " + "; ".join(messages))

    for entry in OP_DEFINITIONS:
        args_src = ", ".join(entry["args"])
        ast.parse(f"def {entry['api']}({args_src}):\n    pass\n")


def _format_call(raw_name: str, call_args: list[str]) -> str:
    if len(call_args) <= 3:
        return f"    return raw_ops.{raw_name}({', '.join(call_args)})\n"
    lines = [f"    return raw_ops.{raw_name}(\n"]
    lines.extend(f"        {arg},\n" for arg in call_args)
    lines.append("    )\n")
    return "".join(lines)


def _generate_ops() -> str:
    lines = [HEADER, '"""Generated public wrappers for MUSA extension ops."""\n\n', "from . import raw_ops\n\n\n"]
    for entry in OP_DEFINITIONS:
        lines.append(f"def {entry['api']}({', '.join(entry['args'])}):\n")
        lines.append(_format_call(entry["raw"], entry["call"]))
        lines.append("\n\n")

    lines.append("__all__ = [\n")
    for api_name in sorted(entry["api"] for entry in OP_DEFINITIONS):
        lines.append(f'    "{api_name}",\n')
    lines.append("]\n")
    return "".join(lines)


def _generate_manifest() -> str:
    serializable = [
        {
            "op": entry["op"],
            "raw": entry["raw"],
            "api": entry["api"],
            "source": entry["source"],
        }
        for entry in OP_DEFINITIONS
    ]
    lines = [HEADER, '"""Manifest of custom MUSA TensorFlow ops exposed to Python."""\n\n']
    lines.append("CUSTOM_OPS = ")
    lines.append(pprint.pformat(serializable, width=88, sort_dicts=False))
    lines.append("\n\n")
    lines.append("CUSTOM_OP_NAMES = tuple(entry[\"op\"] for entry in CUSTOM_OPS)\n")
    lines.append("RAW_OP_NAMES = tuple(entry[\"raw\"] for entry in CUSTOM_OPS)\n")
    lines.append("PUBLIC_API_NAMES = tuple(entry[\"api\"] for entry in CUSTOM_OPS)\n")
    return "".join(lines)


def main() -> None:
    _validate_definitions()
    GENERATED_OPS.write_text(_generate_ops(), encoding="utf-8")
    OP_MANIFEST.write_text(_generate_manifest(), encoding="utf-8")


if __name__ == "__main__":
    main()
