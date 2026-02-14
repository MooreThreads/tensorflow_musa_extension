#include "MusaGraphUtils_layout.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

class MusaLayoutOptimizer : public CustomGraphOptimizer {
 public:
  MusaLayoutOptimizer() {}
  ~MusaLayoutOptimizer() override {}

  std::string name() const override { return "musa_layout_optimizer"; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer *config) override {
    return Status::OK();
  }
  bool UsesFunctionLibrary() const override { return false; }
  void Feedback(Cluster *cluster, const GrapplerItem &item,
                const GraphDef &optimized_graph, double result) override {}
  Status Optimize(Cluster *cluster, const GrapplerItem &item,
                  GraphDef *optimized_graph) override {
    *optimized_graph = item.graph;

    bool changed = true;
    int iteration = 0;
    while (changed && iteration < 5) {
      changed = false;
      iteration++;

      for (int i = 0; i < optimized_graph->node_size(); ++i) {
        NodeDef *node = optimized_graph->mutable_node(i);

        if (!MusaGraphUtils::IsMusaNCHWSupported(*node)) {
          continue;
        }

        auto *attr = node->mutable_attr();
        bool is_already_nchw = (attr->count("data_format") &&
                                (*attr)["data_format"].s() == "NCHW");
        if (is_already_nchw) continue;

        bool has_nchw_upstream = false;
        if (node->input_size() > 0) {
          if (node->input(0).find("/post_transpose_nhwc") !=
              std::string::npos) {
            has_nchw_upstream = true;
          }
        }

        bool should_transform = false;
        if (MusaGraphUtils::kLayoutSensitiveOps(*node)) {
          should_transform = true;
        } else if (MusaGraphUtils::kLayoutAgnosticOps(*node) &&
                   has_nchw_upstream) {
          should_transform = true;
        }

        if (should_transform) {
          std::string op_name = node->name();
          DataType dtype = (*attr)["T"].type();
          std::string device = node->device();

          if (has_nchw_upstream) {
            std::string real_src = node->input(0).substr(
                0, node->input(0).find("/post_transpose_nhwc"));
            node->set_input(0, real_src);
          } else {
            std::string pre_name = op_name + "/pre_transpose_nchw";
            MusaGraphUtils::InsertTranspose(optimized_graph, pre_name,
                                            node->input(0), {0, 3, 1, 2}, dtype,
                                            device);
            node->set_input(0, pre_name);
          }

          (*attr)["data_format"].set_s("NCHW");
          if (MusaGraphUtils::kLayoutSensitiveOps(*node)) {
            MusaGraphUtils::RewriteLayoutAttributes(node);
          }

          std::string post_name = op_name + "/post_transpose_nhwc";
          MusaGraphUtils::InsertTranspose(optimized_graph, post_name, op_name,
                                          {0, 2, 3, 1}, dtype, device);
          MusaGraphUtils::RedirectEdges(optimized_graph, op_name, post_name);

          changed = true;
        }
      }
    }

    MusaGraphUtils::CleanupUnusedNodes(optimized_graph);

    return Status::OK();
  }
};

// REGISTER_GRAPH_OPTIMIZER_AS(MusaLayoutOptimizer, "MusaLayoutOptimizer");

}  // namespace grappler
}  // namespace tensorflow
