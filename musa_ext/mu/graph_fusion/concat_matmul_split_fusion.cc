#include "mu/graph_fusion/concat_matmul_split_fusion.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

struct SliceSpec {
  const NodeDef* node = nullptr;
  std::vector<int64_t> begin;
  std::vector<int64_t> size;
  std::vector<std::string> removable_const_nodes;
};

struct SlicePlan {
  bool valid = false;
  std::string producer_name;
  int slice_axis = -1;
  std::vector<const NodeDef*> ordered_slice_nodes;
  std::vector<int64_t> slice_sizes;
  std::vector<std::string> removable_nodes;
  std::vector<AttrValue> output_shapes;
};

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

bool HasOriginalSuffix(const std::string& node_name) {
  static const std::string kOriginalSuffix = "_original";
  return node_name.size() >= kOriginalSuffix.size() &&
         node_name.compare(node_name.size() - kOriginalSuffix.size(),
                           kOriginalSuffix.size(), kOriginalSuffix) == 0;
}

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  const std::string node_name = FusionGraphUtils::GetProducerNodeName(input);
  if (node_name.empty()) {
    return nullptr;
  }
  return FusionGraphUtils::GetNodeByName(graph, node_name);
}

bool GetConstIntValues(const NodeDef& node, std::vector<int64_t>* values) {
  if (!IsOp(node, "Const")) {
    return false;
  }

  auto it = node.attr().find("value");
  if (it == node.attr().end()) {
    return false;
  }

  const auto& tensor_proto = it->second.tensor();
  if (tensor_proto.dtype() == DT_INT32) {
    if (tensor_proto.int_val_size() > 0) {
      for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
        values->push_back(static_cast<int64_t>(tensor_proto.int_val(i)));
      }
    } else if (!tensor_proto.tensor_content().empty()) {
      const int32* data =
          reinterpret_cast<const int32*>(tensor_proto.tensor_content().data());
      const int num =
          tensor_proto.tensor_content().size() / sizeof(int32);
      for (int i = 0; i < num; ++i) {
        values->push_back(static_cast<int64_t>(data[i]));
      }
    }
  } else if (tensor_proto.dtype() == DT_INT64) {
    if (tensor_proto.int64_val_size() > 0) {
      for (int i = 0; i < tensor_proto.int64_val_size(); ++i) {
        values->push_back(tensor_proto.int64_val(i));
      }
    } else if (!tensor_proto.tensor_content().empty()) {
      const int64* data =
          reinterpret_cast<const int64*>(tensor_proto.tensor_content().data());
      const int num =
          tensor_proto.tensor_content().size() / sizeof(int64);
      for (int i = 0; i < num; ++i) {
        values->push_back(data[i]);
      }
    }
  } else {
    return false;
  }

  return !values->empty();
}

bool ParseSliceNode(const GraphDef& graph, const NodeDef& node, SliceSpec* spec) {
  spec->node = &node;
  spec->begin.clear();
  spec->size.clear();
  spec->removable_const_nodes.clear();

  if (IsOp(node, "Slice")) {
    if (node.input_size() != 3) {
      return false;
    }
    const NodeDef* begin_node = FindProducer(graph, node.input(1));
    const NodeDef* size_node = FindProducer(graph, node.input(2));
    if (!begin_node || !size_node) {
      return false;
    }
    if (!GetConstIntValues(*begin_node, &spec->begin) ||
        !GetConstIntValues(*size_node, &spec->size)) {
      return false;
    }
    if (spec->begin.size() != spec->size.size()) {
      return false;
    }
    for (size_t i = 0; i < spec->size.size(); ++i) {
      if (spec->begin[i] < 0 || spec->size[i] <= 0) {
        return false;
      }
    }
    spec->removable_const_nodes.push_back(begin_node->name());
    spec->removable_const_nodes.push_back(size_node->name());
    return true;
  }

  if (IsOp(node, "StridedSlice")) {
    if (node.input_size() != 4) {
      return false;
    }
    const auto begin_mask_it = node.attr().find("begin_mask");
    const auto end_mask_it = node.attr().find("end_mask");
    const auto ellipsis_mask_it = node.attr().find("ellipsis_mask");
    const auto new_axis_mask_it = node.attr().find("new_axis_mask");
    const auto shrink_axis_mask_it = node.attr().find("shrink_axis_mask");
    if (begin_mask_it == node.attr().end() || end_mask_it == node.attr().end() ||
        ellipsis_mask_it == node.attr().end() ||
        new_axis_mask_it == node.attr().end() ||
        shrink_axis_mask_it == node.attr().end()) {
      return false;
    }
    if (begin_mask_it->second.i() != 0 || end_mask_it->second.i() != 0 ||
        ellipsis_mask_it->second.i() != 0 || new_axis_mask_it->second.i() != 0 ||
        shrink_axis_mask_it->second.i() != 0) {
      return false;
    }

    const NodeDef* begin_node = FindProducer(graph, node.input(1));
    const NodeDef* end_node = FindProducer(graph, node.input(2));
    const NodeDef* strides_node = FindProducer(graph, node.input(3));
    if (!begin_node || !end_node || !strides_node) {
      return false;
    }
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    if (!GetConstIntValues(*begin_node, &spec->begin) ||
        !GetConstIntValues(*end_node, &end) ||
        !GetConstIntValues(*strides_node, &strides)) {
      return false;
    }
    if (spec->begin.size() != end.size() || spec->begin.size() != strides.size()) {
      return false;
    }

    spec->size.resize(spec->begin.size(), 0);
    for (size_t i = 0; i < spec->begin.size(); ++i) {
      if (spec->begin[i] < 0 || end[i] <= spec->begin[i] || strides[i] != 1) {
        return false;
      }
      spec->size[i] = end[i] - spec->begin[i];
    }
    spec->removable_const_nodes.push_back(begin_node->name());
    spec->removable_const_nodes.push_back(end_node->name());
    spec->removable_const_nodes.push_back(strides_node->name());
    return true;
  }

  return false;
}

std::vector<const NodeDef*> FindConsumers(const GraphDef& graph,
                                          const std::string& producer_name) {
  std::vector<const NodeDef*> consumers;
  std::unordered_set<std::string> seen_names;
  for (const auto& node : graph.node()) {
    for (int i = 0; i < node.input_size(); ++i) {
      if (FusionGraphUtils::GetProducerNodeName(node.input(i)) != producer_name) {
        continue;
      }
      if (seen_names.insert(node.name()).second) {
        consumers.push_back(&node);
      }
    }
  }
  return consumers;
}

SlicePlan BuildSlicePlan(const GraphDef& graph, const NodeDef& producer) {
  SlicePlan plan;
  if (!IsOp(producer, "MusaConcatMatMul") || HasOriginalSuffix(producer.name())) {
    return plan;
  }

  const auto consumers = FindConsumers(graph, producer.name());
  if (consumers.size() < 2) {
    return plan;
  }

  std::vector<SliceSpec> specs;
  specs.reserve(consumers.size());
  for (const NodeDef* consumer : consumers) {
    if (HasOriginalSuffix(consumer->name())) {
      return plan;
    }
    SliceSpec spec;
    if (!ParseSliceNode(graph, *consumer, &spec)) {
      return plan;
    }
    specs.push_back(std::move(spec));
  }

  const int rank = static_cast<int>(specs[0].size.size());
  for (const auto& spec : specs) {
    if (static_cast<int>(spec.size.size()) != rank) {
      return plan;
    }
  }

  int matched_axis = -1;
  for (int axis = 0; axis < rank; ++axis) {
    bool valid_axis = true;
    std::vector<int64_t> common_dims(rank, -1);
    std::vector<int> order(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
      order[i] = static_cast<int>(i);
      for (int dim = 0; dim < rank; ++dim) {
        if (dim == axis) {
          continue;
        }
        if (specs[i].begin[dim] != 0 || specs[i].size[dim] <= 0) {
          valid_axis = false;
          break;
        }
        if (common_dims[dim] < 0) {
          common_dims[dim] = specs[i].size[dim];
        } else if (common_dims[dim] != specs[i].size[dim]) {
          valid_axis = false;
          break;
        }
      }
      if (!valid_axis) {
        break;
      }
    }
    if (!valid_axis) {
      continue;
    }

    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      if (specs[lhs].begin[axis] != specs[rhs].begin[axis]) {
        return specs[lhs].begin[axis] < specs[rhs].begin[axis];
      }
      return specs[lhs].node->name() < specs[rhs].node->name();
    });

    int64_t expected_begin = 0;
    bool contiguous = true;
    for (int idx : order) {
      if (specs[idx].begin[axis] != expected_begin ||
          specs[idx].size[axis] <= 0) {
        contiguous = false;
        break;
      }
      expected_begin += specs[idx].size[axis];
    }
    if (!contiguous) {
      continue;
    }

    if (matched_axis >= 0) {
      return plan;
    }
    matched_axis = axis;
  }

  if (matched_axis < 0) {
    return plan;
  }

  std::sort(specs.begin(), specs.end(), [&](const SliceSpec& lhs,
                                            const SliceSpec& rhs) {
    if (lhs.begin[matched_axis] != rhs.begin[matched_axis]) {
      return lhs.begin[matched_axis] < rhs.begin[matched_axis];
    }
    return lhs.node->name() < rhs.node->name();
  });

  plan.valid = true;
  plan.producer_name = producer.name();
  plan.slice_axis = matched_axis;
  plan.removable_nodes.push_back(producer.name() + "_original");
  bool all_output_shapes_available = true;
  for (const auto& spec : specs) {
    plan.ordered_slice_nodes.push_back(spec.node);
    plan.slice_sizes.push_back(spec.size[matched_axis]);
    plan.removable_nodes.push_back(spec.node->name());
    for (const auto& const_name : spec.removable_const_nodes) {
      plan.removable_nodes.push_back(const_name);
    }
    const auto output_shapes_it = spec.node->attr().find("_output_shapes");
    if (output_shapes_it != spec.node->attr().end() &&
        output_shapes_it->second.list().shape_size() > 0) {
      AttrValue output_shape_attr;
      output_shape_attr.mutable_list()->add_shape()->CopyFrom(
          output_shapes_it->second.list().shape(0));
      plan.output_shapes.push_back(output_shape_attr);
    } else {
      all_output_shapes_available = false;
    }
  }
  if (!all_output_shapes_available) {
    plan.output_shapes.clear();
  }

  return plan;
}

void SetIntListAttr(NodeDef* node, const std::string& attr_name,
                    const std::vector<int64_t>& values) {
  auto* list = (*node->mutable_attr())[attr_name].mutable_list();
  list->clear_i();
  for (int64_t value : values) {
    list->add_i(value);
  }
}

void ProtectInputProducer(std::unordered_set<std::string>* protected_names,
                          const std::string& input_name) {
  if (!input_name.empty()) {
    protected_names->insert(FusionGraphUtils::GetProducerNodeName(input_name));
  }
}

}  // namespace

bool ConcatMatMulSplitFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult ConcatMatMulSplitFusion::Match(
    const GraphDef& graph, int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& node = graph.node(start_node_idx);
  const SlicePlan plan = BuildSlicePlan(graph, node);
  if (!plan.valid) {
    return result;
  }

  result.matched = true;
  result.matched_nodes.push_back(&node);
  result.captured_attrs["producer_name"] = node.name();
  return result;
}

Status ConcatMatMulSplitFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid ConcatMatMulSplit match result");
  }
  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  const auto producer_name_it =
      match_result.captured_attrs.find("producer_name");
  if (producer_name_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing producer_name in ConcatMatMulSplit match result");
  }

  const NodeDef* producer =
      FusionGraphUtils::GetNodeByName(*graph, producer_name_it->second);
  if (!producer) {
    return Status(error::INVALID_ARGUMENT,
                  "Producer node disappeared before apply");
  }

  const SlicePlan plan = BuildSlicePlan(*graph, *producer);
  if (!plan.valid) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to rebuild ConcatMatMulSplit plan during apply");
  }

  const std::string original_name = producer->name();
  const std::string original_producer_name = original_name + "_original";
  const int producer_idx = FusionGraphUtils::FindNodeIndex(*graph, original_name);
  if (producer_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to locate producer node for ConcatMatMulSplit");
  }

  std::vector<std::string> producer_inputs(producer->input().begin(),
                                           producer->input().end());
  const std::string device = producer->device();
  const auto producer_attrs = producer->attr();

  NodeDef* producer_mutable = graph->mutable_node(producer_idx);
  producer_mutable->set_name(original_producer_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name);
  fused_node->set_op("MusaConcatMatMulSplit");
  fused_node->set_device(device);
  for (const auto& input : producer_inputs) {
    fused_node->add_input(input);
  }

  auto copy_cached_attr = [&](const std::string& name) {
    const auto it = producer_attrs.find(name);
    if (it != producer_attrs.end()) {
      (*fused_node->mutable_attr())[name] = it->second;
    }
  };
  copy_cached_attr("T");
  copy_cached_attr("transpose_a");
  copy_cached_attr("transpose_b");
  copy_cached_attr("num_concat");
  copy_cached_attr("concat_input_idx");
  copy_cached_attr("fused_ops");
  copy_cached_attr("num_args");
  copy_cached_attr("activation_alpha");
  (*fused_node->mutable_attr())["num_outputs"].set_i(
      static_cast<int64_t>(plan.ordered_slice_nodes.size()));
  (*fused_node->mutable_attr())["slice_axis"].set_i(plan.slice_axis);
  SetIntListAttr(fused_node, "slice_sizes", plan.slice_sizes);

  if (plan.output_shapes.size() == plan.ordered_slice_nodes.size()) {
    auto* shape_list = (*fused_node->mutable_attr())["_output_shapes"].mutable_list();
    shape_list->clear_shape();
    for (const auto& shape_attr : plan.output_shapes) {
      shape_list->add_shape()->CopyFrom(shape_attr.list().shape(0));
    }
  }

  std::vector<std::string> removable_names;
  removable_names.reserve(plan.removable_nodes.size());
  removable_names.push_back(original_producer_name);

  for (size_t i = 0; i < plan.ordered_slice_nodes.size(); ++i) {
    const NodeDef* slice_node = plan.ordered_slice_nodes[i];
    const int slice_idx = FusionGraphUtils::FindNodeIndex(*graph, slice_node->name());
    if (slice_idx < 0) {
      return Status(error::INVALID_ARGUMENT,
                    "Failed to locate slice node for ConcatMatMulSplit: " +
                        slice_node->name());
    }

    const std::string original_slice_name = slice_node->name();
    const std::string renamed_slice_name = original_slice_name + "_original";
    const auto slice_attrs = graph->node(slice_idx).attr();
    const std::string slice_device = graph->node(slice_idx).device();

    NodeDef* slice_mutable = graph->mutable_node(slice_idx);
    slice_mutable->set_name(renamed_slice_name);
    removable_names.push_back(renamed_slice_name);

    NodeDef* identity_node = graph->add_node();
    identity_node->set_name(original_slice_name);
    identity_node->set_op("Identity");
    identity_node->set_device(slice_device);
    identity_node->add_input(original_name + ":" + std::to_string(i));

    const auto t_it = slice_attrs.find("T");
    if (t_it != slice_attrs.end()) {
      (*identity_node->mutable_attr())["T"] = t_it->second;
    } else {
      (*identity_node->mutable_attr())["T"] = producer_attrs.at("T");
    }

    const auto output_shapes_it = slice_attrs.find("_output_shapes");
    if (output_shapes_it != slice_attrs.end()) {
      (*identity_node->mutable_attr())["_output_shapes"] = output_shapes_it->second;
    }
  }

  for (const auto& removable_name : plan.removable_nodes) {
    if (removable_name == original_name + "_original") {
      continue;
    }
    if (std::find(removable_names.begin(), removable_names.end(), removable_name) ==
        removable_names.end()) {
      removable_names.push_back(removable_name);
    }
  }

  std::unordered_set<std::string> protected_names = {original_name};
  for (const auto& input : producer_inputs) {
    ProtectInputProducer(&protected_names, input);
  }

  FusionGraphUtils::RemoveNodesIfUnused(graph, removable_names,
                                        protected_names);
  return Status::OK();
}

REGISTER_FUSION_PATTERN(ConcatMatMulSplitFusion);
REGISTER_FUSION_KERNEL(ConcatMatMulSplitFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
