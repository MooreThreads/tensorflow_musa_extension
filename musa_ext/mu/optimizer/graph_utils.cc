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

#include "mu/optimizer/graph_utils.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tsl/platform/errors.h"

namespace tensorflow {
namespace grappler {
namespace musa {

namespace {

// Get dump directory from environment or use default
std::string GetDumpDirectory() {
  const char* env_dir = std::getenv("MUSA_DUMP_GRAPHDEF_DIR");
  if (env_dir != nullptr && strlen(env_dir) > 0) {
    return std::string(env_dir);
  }
  // Default to current directory
  return ".";
}

}  // namespace

bool IsGraphDefDumpingEnabled() {
  const char* env_val = std::getenv("MUSA_DUMP_GRAPHDEF");
  return env_val != nullptr &&
         (std::string(env_val) == "1" || std::string(env_val) == "true" ||
          std::string(env_val) == "TRUE" || std::string(env_val) == "yes");
}

Status DumpGraphDef(const GraphDef& graph_def, const std::string& prefix,
                    const std::string& stage_description) {
  if (!IsGraphDefDumpingEnabled()) {
    return ::tsl::OkStatus();
  }

  std::string dump_dir = GetDumpDirectory();

  // Create directory if it doesn't exist
  tensorflow::Env* env = tensorflow::Env::Default();
  if (!env->FileExists(dump_dir).ok()) {
    TF_RETURN_IF_ERROR(env->CreateDir(dump_dir));
  }

  // Construct filename: {dump_dir}/{prefix}_{stage_description}.pbtxt
  std::stringstream filename;
  filename << dump_dir << "/" << prefix;
  if (!stage_description.empty()) {
    filename << "_" << stage_description;
  }
  filename << ".pbtxt";

  std::string filepath = filename.str();

  // Serialize GraphDef to text format
  std::string graph_txt;
  if (!protobuf::TextFormat::PrintToString(graph_def, &graph_txt)) {
    return tsl::errors::Internal("Failed to serialize GraphDef to text format");
  }

  // Write to file
  std::ofstream file(filepath, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    return tsl::errors::Internal("Failed to open file for writing: " + filepath);
  }

  file << graph_txt;
  file.close();

  LOG(INFO) << "MusaGraphOptimizer: Dumped GraphDef to " << filepath
            << " (nodes: " << graph_def.node_size() << ")";

  return ::tsl::OkStatus();
}

// Initialize static member
int GraphDefDumper::global_dump_counter_ = 0;

GraphDefDumper::GraphDefDumper(const std::string& optimizer_name)
    : optimizer_name_(optimizer_name), dump_id_(++global_dump_counter_) {}

GraphDefDumper::~GraphDefDumper() {}

void GraphDefDumper::DumpAtStage(const GraphDef& graph,
                                  const std::string& stage) {
  if (!IsGraphDefDumpingEnabled()) return;

  // Create prefix: {optimizer_name}_{dump_id}
  std::stringstream prefix;
  prefix << optimizer_name_ << "_" << std::setfill('0') << std::setw(4)
         << dump_id_;

  Status status = DumpGraphDef(graph, prefix.str(), stage);
  if (!status.ok()) {
    LOG(WARNING) << "MusaGraphOptimizer: Failed to dump graph at stage ["
                 << stage << "]: " << status;
  }
}

void GraphDefDumper::DumpBeforePass(const GraphDef& graph,
                                     const std::string& pass_name) {
  DumpAtStage(graph, "before_" + pass_name);
}

void GraphDefDumper::DumpAfterPass(const GraphDef& graph,
                                    const std::string& pass_name) {
  DumpAtStage(graph, "after_" + pass_name);
}

void GraphDefDumper::DumpInitial(const GraphDef& graph) {
  DumpAtStage(graph, "initial");
}

void GraphDefDumper::DumpFinal(const GraphDef& graph) {
  DumpAtStage(graph, "final");
}

}  // namespace musa
}  // namespace grappler
}  // namespace tensorflow
