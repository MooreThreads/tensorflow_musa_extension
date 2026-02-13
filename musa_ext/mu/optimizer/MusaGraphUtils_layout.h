#ifndef TENSORFLOW_GRAPPLER_OPTIMIZERS_MUSA_GRAPH_UTILS_LAYOUT_H_
#define TENSORFLOW_GRAPPLER_OPTIMIZERS_MUSA_GRAPH_UTILS_LAYOUT_H_

#include <set>
#include <string>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

class MusaGraphUtils {
 public:
  static NodeDef* CreateConstNode(GraphDef* graph, const string& name,
                                  const std::vector<int32>& values,
                                  const string& device);

  static NodeDef* InsertTranspose(GraphDef* graph, const string& base_name,
                                  const string& input_name,
                                  const std::vector<int32>& perm,
                                  DataType dtype, const string& device);

  static void RedirectEdges(GraphDef* graph, const string& old_node_name,
                            const string& new_node_name);

  static void RewriteLayoutAttributes(NodeDef* node);

  static bool IsMusaNCHWSupported(const NodeDef& node);

  static bool kLayoutSensitiveOps(const NodeDef& node);

  static bool kLayoutAgnosticOps(const NodeDef& node);

  static void CleanupUnusedNodes(GraphDef* graph);

  static NodeDef* InsertCast(GraphDef* graph, const string& name,
                             const string& input_name, DataType src_dtype,
                             DataType dst_dtype, const string& device);
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OPTIMIZERS_MUSA_GRAPH_UTILS_LAYOUT_H_
