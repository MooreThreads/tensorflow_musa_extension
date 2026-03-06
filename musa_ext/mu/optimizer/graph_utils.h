/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_UTILS_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_UTILS_H_

#include <string>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
namespace musa {

// Check if MUSA_DUMP_GRAPHDEF environment variable is set
bool IsGraphDefDumpingEnabled();

// Dump GraphDef to a pbtxt file with a descriptive name
// The filename will be: {prefix}_{stage_description}.pbtxt
// Example: musa_optimizer_before_layout.pbtxt
Status DumpGraphDef(const GraphDef& graph_def, const std::string& prefix,
                    const std::string& stage_description);

// Helper class to dump graph at various optimization stages
class GraphDefDumper {
 public:
  explicit GraphDefDumper(const std::string& optimizer_name);
  ~GraphDefDumper();

  // Dump graph at a specific optimization stage
  // stage: description of the current stage (e.g., "before_layout", "after_fusion")
  void DumpAtStage(const GraphDef& graph, const std::string& stage);

  // Dump graph before a specific optimization pass
  void DumpBeforePass(const GraphDef& graph, const std::string& pass_name);

  // Dump graph after a specific optimization pass
  void DumpAfterPass(const GraphDef& graph, const std::string& pass_name);

  // Dump initial graph (before any optimization)
  void DumpInitial(const GraphDef& graph);

  // Dump final graph (after all optimizations)
  void DumpFinal(const GraphDef& graph);

 private:
  std::string optimizer_name_;
  int dump_id_;
  static int global_dump_counter_;
};

}  // namespace musa
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_UTILS_H_
