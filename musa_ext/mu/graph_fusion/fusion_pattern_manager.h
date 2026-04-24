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

#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_FUSION_PATTERN_MANAGER_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_FUSION_PATTERN_MANAGER_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

// Match result for a fusion pattern
struct FusionMatchResult {
  bool matched = false;
  std::vector<const NodeDef*> matched_nodes;
  std::map<std::string, const NodeDef*> captured_nodes;
  std::map<std::string, std::string> captured_attrs;
  
  // Helper method to check if match is valid
  bool IsValid() const { return matched && !matched_nodes.empty(); }
  
  // Reset the match result
  void Reset() {
    matched = false;
    matched_nodes.clear();
    captured_nodes.clear();
    captured_attrs.clear();
  }
};

// Fusion pattern interface
class FusionPattern {
 public:
  virtual ~FusionPattern() = default;
  
  // Match the pattern starting from a node in the graph
  virtual FusionMatchResult Match(const GraphDef& graph, int start_node_idx) const = 0;
  
  // Apply the fusion transformation to the graph
  virtual Status Apply(GraphDef* graph, const FusionMatchResult& match_result) const = 0;
  
  // Get the priority of this pattern (higher = applied first)
  virtual int GetPriority() const = 0;
  
  // Check if the corresponding kernel is available
  virtual bool IsKernelAvailable() const = 0;
  
  // Get the name of this fusion pattern
  virtual std::string GetName() const = 0;
  
  // Get fallback reason if kernel is not available
  virtual std::string GetFallbackReason() const { return ""; }
  
  // Enable/disable this pattern
  void SetEnabled(bool enabled) { enabled_ = enabled; }
  bool IsEnabled() const { return enabled_; }

 protected:
  // Check whether this fusion pattern is disabled by environment variable.
  bool IsDisabledByEnv() const;

  bool enabled_ = true;
};

// Kernel registry for tracking fusion kernel availability
class FusionKernelRegistry {
 public:
  static FusionKernelRegistry& GetInstance();
  
  // Register a kernel availability checker
  void RegisterKernel(const std::string& pattern_name, 
                      std::function<bool()> is_available_func);
  
  // Check if kernel is available
  bool IsKernelAvailable(const std::string& pattern_name) const;
  
  // Get all available kernels
  std::vector<std::string> GetAvailableKernels() const;
  
  // Mark a kernel as implemented (for built-in kernels)
  void MarkKernelAsImplemented(const std::string& pattern_name);

 private:
  FusionKernelRegistry() = default;
  
  std::unordered_map<std::string, std::function<bool()>> kernel_availability_;
  std::unordered_set<std::string> implemented_kernels_;
};

// Manager for all fusion patterns
class FusionPatternManager {
 public:
  static FusionPatternManager& GetInstance();
  
  // Register a new fusion pattern
  void RegisterPattern(std::unique_ptr<FusionPattern> pattern);
  
  // Get all patterns sorted by priority
  std::vector<const FusionPattern*> GetSortedPatterns() const;
  
  // Check if a pattern has available kernel
  bool HasAvailableKernel(const std::string& pattern_name) const;
  
  // Enable/disable a pattern by name
  void SetPatternEnabled(const std::string& pattern_name, bool enabled);
  
  // Get pattern by name
  const FusionPattern* GetPattern(const std::string& pattern_name) const;
  
  // Get all registered pattern names
  std::vector<std::string> GetRegisteredPatternNames() const;
  
  // Clear all patterns (mainly for testing)
  void ClearPatterns();

 private:
  FusionPatternManager() = default;
  
  mutable std::vector<std::unique_ptr<FusionPattern>> patterns_;
  mutable bool needs_sort_ = false;
  
  void SortPatternsIfNeeded() const;
};

// Helper macros for pattern registration
#define REGISTER_FUSION_PATTERN(PatternClass) \
  static struct PatternClass##Registrar { \
    PatternClass##Registrar() { \
      ::tensorflow::grappler::musa_fusion::FusionPatternManager::GetInstance() \
          .RegisterPattern(std::make_unique<PatternClass>()); \
    } \
  } g_##PatternClass##_registrar;

#define REGISTER_FUSION_KERNEL(PatternName, IsAvailableFunc) \
  static struct PatternName##KernelRegistrar { \
    PatternName##KernelRegistrar() { \
      ::tensorflow::grappler::musa_fusion::FusionKernelRegistry::GetInstance() \
          .RegisterKernel(#PatternName, IsAvailableFunc); \
    } \
  } g_##PatternName##_kernel_registrar;

// Graph utilities for fusion patterns
class FusionGraphUtils {
 public:
  // Find node index by name
  static int FindNodeIndex(const GraphDef& graph, const std::string& node_name);
  
  // Get node by name
  static const NodeDef* GetNodeByName(const GraphDef& graph, const std::string& node_name);
  
  // Check if node has specific input
  static bool HasInput(const NodeDef& node, const std::string& input_name);
  
  // Get the producer node name from input (handles control dependencies and ports)
  static std::string GetProducerNodeName(const std::string& input);
  
  // Remove a node from the graph
  static void RemoveNode(GraphDef* graph, int node_idx);

  // Check whether a node still has any consumers in the graph.
  static bool HasAnyConsumer(const GraphDef& graph, const std::string& node_name);

  // Remove nodes that are no longer referenced outside the removable set.
  // Protected nodes are excluded from deletion even if listed in node_names.
  static int RemoveNodesIfUnused(
      GraphDef* graph, const std::vector<std::string>& node_names,
      const std::unordered_set<std::string>& protected_node_names = {});
  
  // Redirect all inputs referencing old_node to new_node
  static void RedirectInputs(GraphDef* graph, const std::string& old_node_name,
                             const std::string& new_node_name);
  
  // Check if node is on MUSA device
  static bool IsMusaNode(const NodeDef& node);
  
  // Check if node op matches
  static bool IsOpType(const NodeDef& node, const std::string& op_type);
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_FUSION_PATTERN_MANAGER_H_
