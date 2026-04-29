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

#include "kernels/fusion/fusion_pattern_manager.h"

#include <algorithm>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

// =============================================================================
// FusionKernelRegistry Implementation
// =============================================================================

FusionKernelRegistry& FusionKernelRegistry::GetInstance() {
  static FusionKernelRegistry instance;
  return instance;
}

void FusionKernelRegistry::RegisterKernel(
    const std::string& pattern_name, std::function<bool()> is_available_func) {
  if (pattern_name.empty()) {
    LOG(WARNING) << "Cannot register kernel with empty pattern name";
    return;
  }
  kernel_availability_[pattern_name] = is_available_func;
  VLOG(2) << "Registered fusion kernel: " << pattern_name;
}

bool FusionKernelRegistry::IsKernelAvailable(
    const std::string& pattern_name) const {
  // First check if it's in the implemented set
  if (implemented_kernels_.count(pattern_name) > 0) {
    return true;
  }

  // Otherwise check if there's a custom availability checker
  auto it = kernel_availability_.find(pattern_name);
  if (it != kernel_availability_.end() && it->second) {
    return it->second();
  }

  return false;
}

std::vector<std::string> FusionKernelRegistry::GetAvailableKernels() const {
  std::vector<std::string> available;

  // Add all implemented kernels
  for (const auto& kernel : implemented_kernels_) {
    available.push_back(kernel);
  }

  // Check dynamic availability
  for (const auto& kv : kernel_availability_) {
    if (kv.second && kv.second() && implemented_kernels_.count(kv.first) == 0) {
      available.push_back(kv.first);
    }
  }

  return available;
}

void FusionKernelRegistry::MarkKernelAsImplemented(
    const std::string& pattern_name) {
  implemented_kernels_.insert(pattern_name);
  VLOG(1) << "Marked kernel as implemented: " << pattern_name;
}

// =============================================================================
// FusionPatternManager Implementation
// =============================================================================

FusionPatternManager& FusionPatternManager::GetInstance() {
  static FusionPatternManager instance;
  return instance;
}

void FusionPatternManager::RegisterPattern(
    std::unique_ptr<FusionPattern> pattern) {
  if (!pattern) {
    LOG(WARNING) << "Cannot register null fusion pattern";
    return;
  }

  const std::string& name = pattern->GetName();
  if (name.empty()) {
    LOG(WARNING) << "Cannot register fusion pattern with empty name";
    return;
  }

  // Check if pattern with same name already exists
  for (const auto& existing : patterns_) {
    if (existing->GetName() == name) {
      LOG(WARNING) << "Fusion pattern '" << name
                   << "' already registered, skipping";
      return;
    }
  }

  // Check kernel availability
  bool kernel_available = pattern->IsKernelAvailable();
  if (!kernel_available) {
    VLOG(1) << "Fusion pattern '" << name
            << "' registered but kernel is not available, will use fallback";
  } else {
    VLOG(1) << "Fusion pattern '" << name
            << "' registered with available kernel";
  }

  patterns_.push_back(std::move(pattern));
  needs_sort_ = true;
}

std::vector<const FusionPattern*> FusionPatternManager::GetSortedPatterns()
    const {
  SortPatternsIfNeeded();

  std::vector<const FusionPattern*> result;
  for (const auto& pattern : patterns_) {
    if (pattern->IsEnabled()) {
      result.push_back(pattern.get());
    }
  }
  return result;
}

bool FusionPatternManager::HasAvailableKernel(
    const std::string& pattern_name) const {
  return FusionKernelRegistry::GetInstance().IsKernelAvailable(pattern_name);
}

void FusionPatternManager::SetPatternEnabled(const std::string& pattern_name,
                                             bool enabled) {
  for (auto& pattern : patterns_) {
    if (pattern->GetName() == pattern_name) {
      pattern->SetEnabled(enabled);
      VLOG(1) << "Fusion pattern '" << pattern_name << "' "
              << (enabled ? "enabled" : "disabled");
      return;
    }
  }
  LOG(WARNING) << "Fusion pattern '" << pattern_name << "' not found";
}

const FusionPattern* FusionPatternManager::GetPattern(
    const std::string& pattern_name) const {
  for (const auto& pattern : patterns_) {
    if (pattern->GetName() == pattern_name) {
      return pattern.get();
    }
  }
  return nullptr;
}

std::vector<std::string> FusionPatternManager::GetRegisteredPatternNames()
    const {
  std::vector<std::string> names;
  for (const auto& pattern : patterns_) {
    names.push_back(pattern->GetName());
  }
  return names;
}

void FusionPatternManager::ClearPatterns() {
  patterns_.clear();
  needs_sort_ = false;
}

void FusionPatternManager::SortPatternsIfNeeded() const {
  if (!needs_sort_) return;

  // Sort by priority (descending), then by pattern length (descending)
  std::sort(patterns_.begin(), patterns_.end(),
            [](const std::unique_ptr<FusionPattern>& a,
               const std::unique_ptr<FusionPattern>& b) {
              int prio_a = a->GetPriority();
              int prio_b = b->GetPriority();
              if (prio_a != prio_b) {
                return prio_a > prio_b;  // Higher priority first
              }
              // For same priority, use alphabetical order for determinism
              return a->GetName() < b->GetName();
            });

  needs_sort_ = false;

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Sorted fusion patterns by priority:";
    for (const auto& pattern : patterns_) {
      VLOG(2) << "  " << pattern->GetName()
              << " (priority=" << pattern->GetPriority()
              << ", enabled=" << pattern->IsEnabled()
              << ", kernel=" << pattern->IsKernelAvailable() << ")";
    }
  }
}

// =============================================================================
// FusionGraphUtils Implementation
// =============================================================================

int FusionGraphUtils::FindNodeIndex(const GraphDef& graph,
                                    const std::string& node_name) {
  for (int i = 0; i < graph.node_size(); ++i) {
    if (graph.node(i).name() == node_name) {
      return i;
    }
  }
  return -1;
}

const NodeDef* FusionGraphUtils::GetNodeByName(const GraphDef& graph,
                                               const std::string& node_name) {
  int idx = FindNodeIndex(graph, node_name);
  if (idx >= 0) {
    return &graph.node(idx);
  }
  return nullptr;
}

bool FusionGraphUtils::HasInput(const NodeDef& node,
                                const std::string& input_name) {
  for (int i = 0; i < node.input_size(); ++i) {
    if (node.input(i) == input_name) {
      return true;
    }
  }
  return false;
}

std::string FusionGraphUtils::GetProducerNodeName(const std::string& input) {
  if (input.empty()) return "";

  // Handle control dependencies (^prefix)
  if (input[0] == '^') {
    return input.substr(1);
  }

  // Handle output ports (:suffix)
  size_t colon_pos = input.find(':');
  if (colon_pos != std::string::npos) {
    return input.substr(0, colon_pos);
  }

  return input;
}

void FusionGraphUtils::RemoveNode(GraphDef* graph, int node_idx) {
  if (node_idx < 0 || node_idx >= graph->node_size()) return;

  // Swap with last and remove
  int last_idx = graph->node_size() - 1;
  if (node_idx != last_idx) {
    graph->mutable_node()->SwapElements(node_idx, last_idx);
  }
  graph->mutable_node()->RemoveLast();
}

bool FusionGraphUtils::HasAnyConsumer(const GraphDef& graph,
                                      const std::string& node_name) {
  for (const auto& node : graph.node()) {
    if (node.name() == node_name) {
      continue;
    }

    for (int i = 0; i < node.input_size(); ++i) {
      if (GetProducerNodeName(node.input(i)) == node_name) {
        return true;
      }
    }
  }

  return false;
}

int FusionGraphUtils::RemoveNodesIfUnused(
    GraphDef* graph, const std::vector<std::string>& node_names,
    const std::unordered_set<std::string>& protected_node_names) {
  std::vector<std::string> ordered_candidates;
  std::unordered_set<std::string> removable_node_names;

  for (const auto& node_name : node_names) {
    if (node_name.empty() ||
        protected_node_names.find(node_name) != protected_node_names.end()) {
      continue;
    }
    if (removable_node_names.insert(node_name).second) {
      ordered_candidates.push_back(node_name);
    }
  }

  int removed_count = 0;
  bool changed = true;
  while (changed) {
    changed = false;

    for (const auto& node_name : ordered_candidates) {
      if (removable_node_names.find(node_name) == removable_node_names.end()) {
        continue;
      }

      const int node_idx = FindNodeIndex(*graph, node_name);
      if (node_idx < 0) {
        removable_node_names.erase(node_name);
        continue;
      }

      if (HasAnyConsumer(*graph, node_name)) {
        continue;
      }

      RemoveNode(graph, node_idx);
      removable_node_names.erase(node_name);
      removed_count++;
      changed = true;
    }
  }

  return removed_count;
}

void FusionGraphUtils::RedirectInputs(GraphDef* graph,
                                      const std::string& old_node_name,
                                      const std::string& new_node_name) {
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    if (node->name() == new_node_name) continue;

    for (int j = 0; j < node->input_size(); ++j) {
      std::string input = node->input(j);
      std::string producer = GetProducerNodeName(input);

      if (producer == old_node_name) {
        // Preserve output port and control dependency prefix
        if (input[0] == '^') {
          node->set_input(j, "^" + new_node_name);
        } else if (input.find(':') != std::string::npos) {
          size_t colon_pos = input.find(':');
          std::string port = input.substr(colon_pos);
          node->set_input(j, new_node_name + port);
        } else {
          node->set_input(j, new_node_name);
        }
      }
    }
  }
}

bool FusionGraphUtils::IsMusaNode(const NodeDef& node) {
  return node.device().find("MUSA") != std::string::npos ||
         node.device().find("/device:MUSA") != std::string::npos;
}

bool FusionGraphUtils::IsOpType(const NodeDef& node,
                                const std::string& op_type) {
  return node.op() == op_type;
}

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
