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

"""Convert TensorFlow GraphDef `.pbtxt` files to binary `.pb` files."""

import argparse
import os

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2


def ensure_pb_extension(path):
    if path.endswith(".pb"):
        return path
    return path + ".pb"


def convert_one(pbtxt_path, output_path=None):
    graph_def = graph_pb2.GraphDef()
    with open(pbtxt_path, "r", encoding="utf-8") as handle:
        text_format.Parse(handle.read(), graph_def)

    if output_path is None:
        if pbtxt_path.endswith(".pbtxt"):
            output_path = pbtxt_path[:-6] + ".pb"
        else:
            output_path = pbtxt_path + ".pb"
    else:
        output_path = ensure_pb_extension(output_path)

    with open(output_path, "wb") as handle:
        handle.write(graph_def.SerializeToString())

    print(
        f"Converted: {pbtxt_path} -> {output_path} "
        f"(nodes={len(graph_def.node)})"
    )
    return output_path


def collect_pbtxt_files(input_path, recursive):
    if os.path.isfile(input_path):
        return [input_path]

    pbtxt_files = []
    if recursive:
        for root, _, files in os.walk(input_path):
            for name in files:
                if name.endswith(".pbtxt"):
                    pbtxt_files.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(input_path)):
            full_path = os.path.join(input_path, name)
            if os.path.isfile(full_path) and name.endswith(".pbtxt"):
                pbtxt_files.append(full_path)

    return sorted(pbtxt_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert TensorFlow GraphDef `.pbtxt` files to `.pb`."
    )
    parser.add_argument(
        "input_path",
        help="A `.pbtxt` file or a directory containing `.pbtxt` files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the same directory as the input file.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for `.pbtxt` files when input_path is a directory.",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(os.path.expanduser(args.input_path))
    output_dir = (
        os.path.abspath(os.path.expanduser(args.output_dir))
        if args.output_dir
        else None
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pbtxt_files = collect_pbtxt_files(input_path, args.recursive)
    if not pbtxt_files:
        raise FileNotFoundError(f"No `.pbtxt` files found under: {input_path}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(pbtxt_files)} pbtxt file(s).")
    for pbtxt_path in pbtxt_files:
        if output_dir:
            base_name = os.path.basename(pbtxt_path)
            output_path = os.path.join(output_dir, base_name[:-6] + ".pb")
        else:
            output_path = None
        convert_one(pbtxt_path, output_path)


if __name__ == "__main__":
    main()
