#!/usr/bin/env python3
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

"""Export TensorFlow GraphDef files for GELU graphs.

This script writes `.pb` and `.pbtxt` files that can be opened with Netron.
By default it exports:
1. raw exact GELU graph
2. raw approximate GELU graph

Optionally, it can also export direct fused `MusaGelu` graphs if the MUSA
plugin library is present.
"""

import argparse
import os
import tempfile

import tensorflow as tf
from google.protobuf import text_format


tf.compat.v1.disable_eager_execution()


def _default_output_dir():
    return tempfile.mkdtemp(prefix="musa_gelu_graph_")


def _write_graph(graph_def, output_dir, graph_name):
    pb_path = os.path.join(output_dir, f"{graph_name}.pb")
    pbtxt_path = os.path.join(output_dir, f"{graph_name}.pbtxt")

    with tf.io.gfile.GFile(pb_path, "wb") as handle:
        handle.write(graph_def.SerializeToString())

    with tf.io.gfile.GFile(pbtxt_path, "w") as handle:
        handle.write(text_format.MessageToString(graph_def))

    print("\n" + "=" * 72)
    print(f"{graph_name}")
    print("=" * 72)
    print(f"pb:    {pb_path}")
    print(f"pbtxt: {pbtxt_path}")
    print("nodes:")
    for node in graph_def.node:
        print(f"  {node.name:24s} op={node.op:12s} inputs={list(node.input)}")


def build_exact_gelu_graph():
    """Build the exact erf-based GELU graph shape used by the fusion matcher."""
    graph = tf.Graph()
    with graph.as_default():
        x = tf.compat.v1.placeholder(
            tf.float32, shape=[None, 344], name="input"
        )
        sqrt2 = tf.constant(1.41421356237, dtype=tf.float32, name="sqrt2")
        div = tf.math.divide(x, sqrt2, name="gelu_div")
        erf = tf.math.erf(div, name="gelu_erf")
        one = tf.constant(1.0, dtype=tf.float32, name="one")
        add = tf.math.add(erf, one, name="gelu_add")
        half = tf.constant(0.5, dtype=tf.float32, name="half")
        half_mul = tf.math.multiply(x, half, name="gelu_half_mul")
        tf.math.multiply(add, half_mul, name="gelu_output")
    return graph.as_graph_def()


def build_approximate_gelu_graph():
    """Build the tanh-based approximate GELU graph used by the fusion matcher."""
    graph = tf.Graph()
    with graph.as_default():
        x = tf.compat.v1.placeholder(
            tf.float32, shape=[None, 344], name="input"
        )
        pow3 = tf.math.pow(
            x, tf.constant(3.0, dtype=tf.float32, name="pow3_exponent"), name="pow3"
        )
        coeff = tf.constant(0.044715, dtype=tf.float32, name="approx_coeff")
        cubic_mul = tf.math.multiply(pow3, coeff, name="cubic_mul")
        inner_add = tf.math.add(x, cubic_mul, name="inner_add")
        scale = tf.constant(0.7978845608, dtype=tf.float32, name="approx_scale")
        tanh_scale_mul = tf.math.multiply(inner_add, scale, name="tanh_scale_mul")
        tanh = tf.math.tanh(tanh_scale_mul, name="gelu_tanh")
        one = tf.constant(1.0, dtype=tf.float32, name="one")
        factor = tf.math.add(one, tanh, name="gelu_factor")
        half = tf.constant(0.5, dtype=tf.float32, name="half")
        half_mul = tf.math.multiply(x, half, name="gelu_half_mul")
        tf.math.multiply(half_mul, factor, name="gelu_output")
    return graph.as_graph_def()


def _find_plugin_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(current_dir, "..", "..", "build", "libmusa_plugin.so"),
        os.path.join(os.getcwd(), "build", "libmusa_plugin.so"),
        os.path.join(os.getcwd(), "..", "build", "libmusa_plugin.so"),
    ]

    for path in candidate_paths:
        normalized = os.path.normpath(path)
        if os.path.exists(normalized):
            return normalized
    return None


def build_fused_musa_gelu_graph(approximate):
    """Build a direct fused MusaGelu graph if the plugin is available."""
    plugin_path = _find_plugin_path()
    if not plugin_path:
        raise FileNotFoundError(
            "libmusa_plugin.so not found. Build the plugin first if you want "
            "to export a direct fused MusaGelu graph."
        )

    musa_ops = tf.load_op_library(plugin_path)
    graph = tf.Graph()
    with graph.as_default():
        x = tf.compat.v1.placeholder(
            tf.float32, shape=[None, 344], name="input"
        )
        musa_ops.musa_gelu(
            x=x, approximate=approximate, name="musa_gelu_output"
        )
    return graph.as_graph_def()


def main():
    parser = argparse.ArgumentParser(
        description="Export TensorFlow `.pb/.pbtxt` graphs for GELU."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for generated graph files. Defaults to a temp dir.",
    )
    parser.add_argument(
        "--include_fused",
        action="store_true",
        help="Also export direct fused MusaGelu graphs if the plugin exists.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    _write_graph(build_exact_gelu_graph(), output_dir, "gelu_exact_raw")
    _write_graph(build_approximate_gelu_graph(), output_dir, "gelu_approx_raw")

    if args.include_fused:
        _write_graph(
            build_fused_musa_gelu_graph(approximate=False),
            output_dir,
            "gelu_exact_fused",
        )
        _write_graph(
            build_fused_musa_gelu_graph(approximate=True),
            output_dir,
            "gelu_approx_fused",
        )

    print("\nUse Netron to open the generated `.pb` or `.pbtxt` files.")


if __name__ == "__main__":
    main()
