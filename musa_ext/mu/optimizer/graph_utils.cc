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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa {

namespace {

constexpr const char* kDumpDirEnv = "MUSA_DUMP_GRAPHDEF_DIR";
constexpr const char* kDumpFormatEnv = "MUSA_DUMP_GRAPHDEF_FORMAT";

enum class DumpFormatMode {
  kAuto,
  kPbtxt,
  kPb,
  kBoth,
};

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  return value;
}

std::string GetDumpDirectory() {
  const char* env_dir = std::getenv(kDumpDirEnv);
  if (env_dir != nullptr && std::strlen(env_dir) > 0) {
    return std::string(env_dir);
  }
  return ".";
}

DumpFormatMode GetDumpFormatMode() {
  const char* env_val = std::getenv(kDumpFormatEnv);
  if (env_val == nullptr || std::strlen(env_val) == 0) {
    return DumpFormatMode::kAuto;
  }

  const std::string mode = ToLower(std::string(env_val));
  if (mode == "pbtxt" || mode == "text") {
    return DumpFormatMode::kPbtxt;
  }
  if (mode == "pb" || mode == "binary") {
    return DumpFormatMode::kPb;
  }
  if (mode == "both") {
    return DumpFormatMode::kBoth;
  }
  return DumpFormatMode::kAuto;
}

std::string BuildDumpBasePath(const std::string& dump_dir,
                              const std::string& prefix,
                              const std::string& stage_description) {
  std::stringstream filename;
  filename << dump_dir << "/" << prefix;
  if (!stage_description.empty()) {
    filename << "_" << stage_description;
  }
  return filename.str();
}

Status EnsureDumpDirectoryExists(const std::string& dump_dir) {
  tensorflow::Env* env = tensorflow::Env::Default();
  if (!env->FileExists(dump_dir).ok()) {
    TF_RETURN_IF_ERROR(env->CreateDir(dump_dir));
  }
  return Status::OK();
}

Status WriteStringToFile(const std::string& path, const std::string& contents,
                         std::ios::openmode mode) {
  std::ofstream file(path, mode);
  if (!file.is_open()) {
    return Status(tensorflow::error::INTERNAL,
                  "Failed to open file for writing: " + path);
  }
  file << contents;
  file.close();
  return Status::OK();
}

Status DumpGraphDefAsPbtxt(const GraphDef& graph_def,
                           const std::string& base_path,
                           std::string* dumped_path) {
  std::string graph_txt;
  if (!google::protobuf::TextFormat::PrintToString(graph_def, &graph_txt)) {
    return Status(tensorflow::error::INTERNAL,
                  "Failed to serialize GraphDef to text format");
  }

  const std::string path = base_path + ".pbtxt";
  TF_RETURN_IF_ERROR(
      WriteStringToFile(path, graph_txt, std::ios::out | std::ios::trunc));
  *dumped_path = path;
  return Status::OK();
}

Status DumpGraphDefAsPb(const GraphDef& graph_def, const std::string& base_path,
                        std::string* dumped_path) {
  std::string graph_bin;
  if (!graph_def.SerializeToString(&graph_bin)) {
    return Status(tensorflow::error::INTERNAL,
                  "Failed to serialize GraphDef to binary format");
  }

  const std::string path = base_path + ".pb";
  TF_RETURN_IF_ERROR(WriteStringToFile(
      path, graph_bin, std::ios::out | std::ios::trunc | std::ios::binary));
  *dumped_path = path;
  return Status::OK();
}

std::string JoinStrings(const std::vector<std::string>& values,
                        const std::string& separator) {
  std::ostringstream os;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      os << separator;
    }
    os << values[i];
  }
  return os.str();
}

std::string DumpModeToString(DumpFormatMode mode) {
  switch (mode) {
    case DumpFormatMode::kAuto:
      return "auto";
    case DumpFormatMode::kPbtxt:
      return "pbtxt";
    case DumpFormatMode::kPb:
      return "pb";
    case DumpFormatMode::kBoth:
      return "both";
  }
  return "auto";
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
    return Status::OK();
  }

  const std::string dump_dir = GetDumpDirectory();
  TF_RETURN_IF_ERROR(EnsureDumpDirectoryExists(dump_dir));

  const std::string base_path =
      BuildDumpBasePath(dump_dir, prefix, stage_description);
  const DumpFormatMode mode = GetDumpFormatMode();

  std::vector<std::string> dumped_paths;
  std::vector<std::string> error_messages;

  auto try_dump_pbtxt = [&]() {
    std::string path;
    Status status = DumpGraphDefAsPbtxt(graph_def, base_path, &path);
    if (status.ok()) {
      dumped_paths.push_back(path);
    } else {
      error_messages.push_back("pbtxt: " + status.ToString());
    }
    return status;
  };

  auto try_dump_pb = [&]() {
    std::string path;
    Status status = DumpGraphDefAsPb(graph_def, base_path, &path);
    if (status.ok()) {
      dumped_paths.push_back(path);
    } else {
      error_messages.push_back("pb: " + status.ToString());
    }
    return status;
  };

  switch (mode) {
    case DumpFormatMode::kPbtxt:
      TF_RETURN_IF_ERROR(try_dump_pbtxt());
      break;
    case DumpFormatMode::kPb:
      TF_RETURN_IF_ERROR(try_dump_pb());
      break;
    case DumpFormatMode::kBoth:
      try_dump_pbtxt();
      try_dump_pb();
      break;
    case DumpFormatMode::kAuto:
      if (!try_dump_pbtxt().ok()) {
        LOG(WARNING) << "MusaGraphOptimizer: pbtxt dump failed for " << base_path
                     << ", falling back to binary pb";
        try_dump_pb();
      }
      break;
  }

  if (dumped_paths.empty()) {
    return Status(tensorflow::error::INTERNAL,
                  "Failed to dump GraphDef in mode '" +
                      DumpModeToString(mode) + "': " +
                      JoinStrings(error_messages, "; "));
  }

  if (!error_messages.empty()) {
    LOG(WARNING) << "MusaGraphOptimizer: Partial dump issues for " << base_path
                 << ": " << JoinStrings(error_messages, "; ");
  }

  LOG(INFO) << "MusaGraphOptimizer: Dumped GraphDef to "
            << JoinStrings(dumped_paths, ", ") << " (nodes: "
            << graph_def.node_size() << ", mode=" << DumpModeToString(mode)
            << ")";

  return Status::OK();
}

int GraphDefDumper::global_dump_counter_ = 0;

GraphDefDumper::GraphDefDumper(const std::string& optimizer_name)
    : optimizer_name_(optimizer_name), dump_id_(++global_dump_counter_) {}

GraphDefDumper::~GraphDefDumper() {}

void GraphDefDumper::DumpAtStage(const GraphDef& graph,
                                 const std::string& stage) {
  if (!IsGraphDefDumpingEnabled()) return;

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