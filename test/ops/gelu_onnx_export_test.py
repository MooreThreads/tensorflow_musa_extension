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

"""Export ONNX graphs for the fused MusaGelu operator."""

import os
import tempfile
import unittest

try:
    import onnx
    from onnx import TensorProto, checker, helper
except ImportError as exc:
    onnx = None
    _ONNX_IMPORT_ERROR = exc
else:
    _ONNX_IMPORT_ERROR = None


CUSTOM_DOMAIN = "com.mthreads.musa"


def build_musa_gelu_onnx_model(approximate):
    """Build a minimal ONNX model containing one MusaGelu node."""
    if onnx is None:
        raise RuntimeError(
            "Failed to import `onnx`. Install it in the current environment with "
            "`python -m pip install onnx`. Original error: "
            f"{_ONNX_IMPORT_ERROR}"
        )

    suffix = "approx" if approximate else "exact"
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", 8]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch", 8]
    )

    gelu_node = helper.make_node(
        "MusaGelu",
        inputs=["input"],
        outputs=["output"],
        name=f"musa_gelu_{suffix}",
        domain=CUSTOM_DOMAIN,
        approximate=1 if approximate else 0,
    )

    graph = helper.make_graph(
        nodes=[gelu_node],
        name=f"musa_gelu_{suffix}_graph",
        inputs=[input_info],
        outputs=[output_info],
        initializer=[],
    )

    model = helper.make_model(
        graph,
        producer_name="tensorflow_musa_extension",
        producer_version="debug",
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid(CUSTOM_DOMAIN, 1),
        ],
    )
    model.ir_version = onnx.IR_VERSION
    checker.check_model(model)
    return model


def export_one(approximate):
    """Export one ONNX file and print a concise summary."""
    model = build_musa_gelu_onnx_model(approximate)
    suffix = "approx" if approximate else "exact"

    with tempfile.NamedTemporaryFile(
        prefix=f"musa_gelu_{suffix}_",
        suffix=".onnx",
        delete=False,
    ) as handle:
        output_path = handle.name

    onnx.save(model, output_path)

    print("\n" + "=" * 70)
    print(f"MusaGelu {suffix} ONNX")
    print("=" * 70)
    print(f"ONNX saved to: {output_path}")
    print("Nodes:")
    for node in model.graph.node:
        attrs = []
        for attr in node.attribute:
            value = attr.i if attr.type == onnx.AttributeProto.INT else "?"
            attrs.append(f"{attr.name}={value}")
        print(
            f"  {node.name:20s} op={node.op_type:12s} "
            f"domain={node.domain:18s} inputs={list(node.input)} "
            f"outputs={list(node.output)} attrs={attrs}"
        )

    return output_path


class GeluOnnxExportTest(unittest.TestCase):
    """Generate ONNX files for fused MusaGelu graphs."""

    def test_export_musa_gelu_exact_onnx(self):
        path = export_one(approximate=False)
        self.assertTrue(os.path.exists(path))

    def test_export_musa_gelu_approximate_onnx(self):
        path = export_one(approximate=True)
        self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
