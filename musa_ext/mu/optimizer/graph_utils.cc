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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa {

namespace {

constexpr const char* kSlimDumpEnv = "MUSA_DUMP_GRAPHDEF_SLIM";
constexpr size_t kSlimConstTensorContentBytesThreshold = 4096;
constexpr size_t kSlimConstPayloadItemThreshold = 256;
constexpr const char* kSlimDropAttrNames[] = {
    "_class",
    "_output_shapes",
    "_XlaScope",
    "_XlaInternalScope",
    "_XlaCompile",
    "_XlaSeparateCompiledGradients",
    "_XlaAlreadyClustered",
    "_kernel",
};

struct SlimGraphStats {
  int nodes_processed = 0;
  int attrs_removed = 0;
  int devices_cleared = 0;
  int debug_fields_cleared = 0;
  int consts_truncated = 0;
};

bool EnvFlagEnabled(const char* env_name) {
  const char* env_val = std::getenv(env_name);
  return env_val != nullptr &&
         (std::string(env_val) == "1" || std::string(env_val) == "true" ||
          std::string(env_val) == "TRUE" || std::string(env_val) == "yes");
}

// Get dump directory from environment or use default
std::string GetDumpDirectory() {
  const char* env_dir = std::getenv("MUSA_DUMP_GRAPHDEF_DIR");
  if (env_dir != nullptr && std::strlen(env_dir) > 0) {
    return std::string(env_dir);
  }
  // Default to current directory
  return ".";
}

bool IsSlimGraphDefDumpEnabled() { return EnvFlagEnabled(kSlimDumpEnv); }

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

Status WriteGraphDefPbtxt(const GraphDef& graph_def,
                          const std::string& filepath) {
  std::ofstream file(filepath, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    return Status(tensorflow::error::INTERNAL,
                  "Failed to open file for writing: " + filepath);
  }

  {
    // Use default Printer settings so the dump is complete: unknown fields
    // (e.g. NodeDef.experimental_debug_info, field 7) are preserved as raw
    // field numbers in the pbtxt.  The conversion script already passes
    // allow_unknown_field=True to text_format.Parse(), so those numeric lines
    // are silently accepted on the Python side without a ParseError.
    protobuf::io::OstreamOutputStream output_stream(&file);
    protobuf::TextFormat::Printer printer;
    if (!printer.Print(graph_def, &output_stream)) {
      return Status(tensorflow::error::INTERNAL,
                    "Failed to serialize GraphDef to text format");
    }
  }

  file.flush();
  if (!file.good()) {
    return Status(tensorflow::error::INTERNAL,
                  "Failed to flush GraphDef text to file: " + filepath);
  }
  file.close();
  if (file.fail()) {
    return Status(tensorflow::error::INTERNAL,
                  "Failed to close GraphDef text file: " + filepath);
  }
  return Status::OK();
}

Status WriteGraphDefBinary(const GraphDef& graph_def,
                           const std::string& filepath) {
  std::ofstream file(filepath,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.is_open()) {
    return Status(tensorflow::error::INTERNAL,
                  "Failed to open file for writing: " + filepath);
  }

  if (!graph_def.SerializeToOstream(&file)) {
    return Status(tensorflow::error::INTERNAL,
                  "Failed to serialize GraphDef to binary format");
  }

  file.close();
  return Status::OK();
}

void ClearKnownFieldIfPresent(protobuf::Message* message,
                              const char* field_name,
                              int* cleared_count) {
  if (message == nullptr) {
    return;
  }
  const protobuf::Descriptor* descriptor = message->GetDescriptor();
  const protobuf::Reflection* reflection = message->GetReflection();
  const protobuf::FieldDescriptor* field = descriptor->FindFieldByName(field_name);
  if (field == nullptr || field->is_repeated() ||
      !reflection->HasField(*message, field)) {
    return;
  }
  reflection->ClearField(message, field);
  if (cleared_count != nullptr) {
    ++(*cleared_count);
  }
}

size_t ApproximateTensorPayloadItems(const TensorProto& tensor) {
  size_t items = 0;
  items += tensor.half_val_size();
  items += tensor.float_val_size();
  items += tensor.double_val_size();
  items += tensor.int_val_size();
  items += tensor.string_val_size();
  items += tensor.scomplex_val_size();
  items += tensor.int64_val_size();
  items += tensor.bool_val_size();
  items += tensor.dcomplex_val_size();
  items += tensor.resource_handle_val_size();
  items += tensor.variant_val_size();
  items += tensor.uint32_val_size();
  items += tensor.uint64_val_size();
  if (!tensor.tensor_content().empty()) {
    items += 1;
  }
  return items;
}

size_t ApproximateTensorPayloadBytes(const TensorProto& tensor) {
  size_t bytes = tensor.tensor_content().size();
  bytes += tensor.half_val_size() * sizeof(int32);
  bytes += tensor.float_val_size() * sizeof(float);
  bytes += tensor.double_val_size() * sizeof(double);
  bytes += tensor.int_val_size() * sizeof(int32);
  for (int i = 0; i < tensor.string_val_size(); ++i) {
    bytes += tensor.string_val(i).size();
  }
  bytes += tensor.scomplex_val_size() * sizeof(float);
  bytes += tensor.int64_val_size() * sizeof(int64_t);
  bytes += tensor.bool_val_size() * sizeof(bool);
  bytes += tensor.dcomplex_val_size() * sizeof(double);
  for (int i = 0; i < tensor.resource_handle_val_size(); ++i) {
    bytes += tensor.resource_handle_val(i).ByteSizeLong();
  }
  for (int i = 0; i < tensor.variant_val_size(); ++i) {
    bytes += tensor.variant_val(i).ByteSizeLong();
  }
  bytes += tensor.uint32_val_size() * sizeof(uint32);
  bytes += tensor.uint64_val_size() * sizeof(uint64);
  return bytes;
}

void ClearTensorPayload(TensorProto* tensor) {
  if (tensor == nullptr) {
    return;
  }
  tensor->clear_tensor_content();
  tensor->clear_half_val();
  tensor->clear_float_val();
  tensor->clear_double_val();
  tensor->clear_int_val();
  tensor->clear_string_val();
  tensor->clear_scomplex_val();
  tensor->clear_int64_val();
  tensor->clear_bool_val();
  tensor->clear_dcomplex_val();
  tensor->clear_resource_handle_val();
  tensor->clear_variant_val();
  tensor->clear_uint32_val();
  tensor->clear_uint64_val();
}

void SetDefaultTensorPayload(TensorProto* tensor) {
  if (tensor == nullptr) {
    return;
  }

  bool has_elements = true;
  for (int i = 0; i < tensor->tensor_shape().dim_size(); ++i) {
    if (tensor->tensor_shape().dim(i).size() == 0) {
      has_elements = false;
      break;
    }
  }
  if (!has_elements) {
    return;
  }

  switch (tensor->dtype()) {
    case DT_HALF:
    case DT_BFLOAT16:
      tensor->add_half_val(0);
      break;
    case DT_FLOAT:
      tensor->add_float_val(0.0f);
      break;
    case DT_DOUBLE:
      tensor->add_double_val(0.0);
      break;
    case DT_INT8:
    case DT_INT16:
    case DT_INT32:
    case DT_UINT8:
    case DT_UINT16:
    case DT_QINT8:
    case DT_QUINT8:
    case DT_QINT16:
    case DT_QUINT16:
    case DT_QINT32:
      tensor->add_int_val(0);
      break;
    case DT_INT64:
      tensor->add_int64_val(0);
      break;
    case DT_UINT32:
      tensor->add_uint32_val(0);
      break;
    case DT_UINT64:
      tensor->add_uint64_val(0);
      break;
    case DT_BOOL:
      tensor->add_bool_val(false);
      break;
    case DT_COMPLEX64:
      tensor->add_scomplex_val(0.0f);
      tensor->add_scomplex_val(0.0f);
      break;
    case DT_COMPLEX128:
      tensor->add_dcomplex_val(0.0);
      tensor->add_dcomplex_val(0.0);
      break;
    case DT_STRING:
      tensor->add_string_val("");
      break;
    default:
      break;
  }
}

bool MaybeTruncateTensorProto(TensorProto* tensor) {
  if (tensor == nullptr) {
    return false;
  }
  if (ApproximateTensorPayloadBytes(*tensor) <=
          kSlimConstTensorContentBytesThreshold &&
      ApproximateTensorPayloadItems(*tensor) <= kSlimConstPayloadItemThreshold) {
    return false;
  }

  ClearTensorPayload(tensor);
  SetDefaultTensorPayload(tensor);
  tensor->DiscardUnknownFields();
  return true;
}

void SlimNodeDef(NodeDef* node, SlimGraphStats* stats) {
  if (node == nullptr || stats == nullptr) {
    return;
  }

  ++stats->nodes_processed;

  if (!node->device().empty()) {
    node->clear_device();
    ++stats->devices_cleared;
  }

  for (const char* attr_name : kSlimDropAttrNames) {
    if (node->mutable_attr()->erase(attr_name) > 0) {
      ++stats->attrs_removed;
    }
  }

  ClearKnownFieldIfPresent(node, "experimental_type", &stats->debug_fields_cleared);
  ClearKnownFieldIfPresent(node, "experimental_debug_info",
                           &stats->debug_fields_cleared);

  auto attr_it = node->mutable_attr()->find("value");
  if (node->op() == "Const" && attr_it != node->mutable_attr()->end()) {
    if (MaybeTruncateTensorProto(attr_it->second.mutable_tensor())) {
      ++stats->consts_truncated;
    }
  }

  node->DiscardUnknownFields();
}

GraphDef CreateSlimGraphDef(const GraphDef& graph_def, SlimGraphStats* stats) {
  GraphDef slim_graph = graph_def;

  for (int i = 0; i < slim_graph.node_size(); ++i) {
    SlimNodeDef(slim_graph.mutable_node(i), stats);
  }

  FunctionDefLibrary* library = slim_graph.mutable_library();
  for (int i = 0; i < library->function_size(); ++i) {
    FunctionDef* function = library->mutable_function(i);
    for (int j = 0; j < function->node_def_size(); ++j) {
      SlimNodeDef(function->mutable_node_def(j), stats);
    }
    function->DiscardUnknownFields();
  }

  slim_graph.DiscardUnknownFields();
  return slim_graph;
}

}  // namespace

bool IsGraphDefDumpingEnabled() { return EnvFlagEnabled("MUSA_DUMP_GRAPHDEF"); }

Status DumpGraphDef(const GraphDef& graph_def, const std::string& prefix,
                    const std::string& stage_description) {
  if (!IsGraphDefDumpingEnabled()) {
    return Status::OK();
  }

  std::string dump_dir = GetDumpDirectory();

  // Create directory if it doesn't exist
  tensorflow::Env* env = tensorflow::Env::Default();
  if (!env->FileExists(dump_dir).ok()) {
    TF_RETURN_IF_ERROR(env->CreateDir(dump_dir));
  }

  const std::string base_path =
      BuildDumpBasePath(dump_dir, prefix, stage_description);

  // -----------------------------------------------------------------------
  // Primary format: binary protobuf (.pb)
  //
  // Binary proto carries unknown fields (e.g. NodeDef.experimental_debug_info,
  // field 7) byte-for-byte without requiring the receiver to know the schema.
  // This is the ONLY format that guarantees zero information loss when the
  // model was serialised by a newer TF version than the one this extension was
  // compiled against.  pbtxt cannot represent unknown fields by their named
  // descriptors, so those fields appear as bare numbers ("7 { ... }") which
  // Python's text_format.Parse() refuses to load.
  // -----------------------------------------------------------------------
  const std::string pb_path = base_path + ".pb";
  TF_RETURN_IF_ERROR(WriteGraphDefBinary(graph_def, pb_path));
  LOG(INFO) << "MusaGraphOptimizer: Dumped GraphDef (binary) to " << pb_path
            << " (nodes: " << graph_def.node_size() << ")";

  bool SlimGraphStatus = IsSlimGraphDefDumpEnabled();

  if (SlimGraphStatus) {
    SlimGraphStats slim_stats;
    GraphDef slim_graph = CreateSlimGraphDef(graph_def, &slim_stats);
    const std::string slim_pb_path = base_path + ".slim.pb";
    Status slim_status = WriteGraphDefBinary(slim_graph, slim_pb_path);
    if (slim_status.ok()) {
      LOG(INFO) << "MusaGraphOptimizer: Dumped GraphDef (slim binary) to "
                << slim_pb_path << " (nodes: " << graph_def.node_size()
                << ", nodes_processed: " << slim_stats.nodes_processed
                << ", consts_truncated: " << slim_stats.consts_truncated
                << ", attrs_removed: " << slim_stats.attrs_removed
                << ", devices_cleared: " << slim_stats.devices_cleared
                << ", debug_fields_cleared: "
                << slim_stats.debug_fields_cleared << ")";
    } else {
      LOG(WARNING) << "MusaGraphOptimizer: slim pb dump failed (non-fatal): "
                   << slim_status;
    }
  }

  // -----------------------------------------------------------------------
  // Optional text format (.pbtxt) for human inspection.
  //
  // Enable with: MUSA_DUMP_GRAPHDEF_TEXT=1
  // Note: unknown fields will appear as raw field numbers in the text output.
  // The file can be inspected but may not round-trip cleanly through
  // text_format.Parse() without allow_unknown_field=True.
  // -----------------------------------------------------------------------
  const char* dump_text_env = std::getenv("MUSA_DUMP_GRAPHDEF_TEXT");
  const bool dump_text = dump_text_env != nullptr &&
                         (std::string(dump_text_env) == "1" ||
                          std::string(dump_text_env) == "true");
  if (dump_text) {
    const std::string pbtxt_path = base_path + ".pbtxt";
    Status text_status = WriteGraphDefPbtxt(graph_def, pbtxt_path);
    if (text_status.ok()) {
      LOG(INFO) << "MusaGraphOptimizer: Dumped GraphDef (text) to "
                << pbtxt_path;
    } else {
      LOG(WARNING) << "MusaGraphOptimizer: pbtxt dump failed (non-fatal): "
                   << text_status;
    }
  }

  return Status::OK();
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
