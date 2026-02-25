
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {
namespace musa {

using Labels = absl::InlinedVector<int, 8UL>;
using OperandLabels = absl::InlinedVector<Labels, 2UL>;
using LabelCounts = absl::InlinedVector<int, 8UL>;
using OperandLabelCounts = absl::InlinedVector<LabelCounts, 2UL>;

// Dummy axis label used to denote an ellipsis in an input or output subscript.
constexpr int kEllipsisLabel = -1;

enum EinsumDimensionType {
  // Batch dimensions are those present in two inputs as well as the output.
  // They are part of the batch dimensions during Tensor contraction. Such
  // dimensions may be broadcasting dimensions (those mapping to ellipsis)
  // or explicit batch dimensions corresponding to named axis labels.
  kBroadcasting = 0,
  kBatch = 1,
  // Free dimensions are present in exactly one of the inputs, and also the
  // output. These are non-contracted axes in the Tensor contraction.
  kFree = 2,
  // Contract dimensions are present in two inputs, but not the output. These
  // dimensions are contracted in Tensor contraction.
  kContract = 3,
  // Reduce dimensions are present in exactly one input; and not in the output
  // and are summed over prior to Tensor contraction.
  kReduce = 4,
};

// Returns the EinsumDimensionType given whether the corresponding label is
// present in exactly one input subscript (is_unique) and whether it is absent
// from the output subscripts (is_removed). Does not handle broadcasting
// dimensions.
EinsumDimensionType GetDimensionType(bool is_removed, bool is_unique) {
  if (!is_removed && !is_unique)
    return kBatch;
  else if (!is_removed && is_unique)
    return kFree;
  else if (is_removed && !is_unique)
    return kContract;
  else  // is_removed && is_unique
    return kReduce;
}

Status ValidateEinsumEquation(
    const std::string& equation,
    absl::InlinedVector<std::string, 2UL>* input_subscripts,
    std::string* output_subscript) {
  absl::InlinedVector<std::string, 2UL> inputs_and_output_subscripts =
      absl::StrSplit(equation, "->");
  if (inputs_and_output_subscripts.size() != 2) {
    return errors::InvalidArgument(
        "Expecting exactly one '->' in einsum equation: ", equation);
  }
  *output_subscript = std::move(inputs_and_output_subscripts[1]);
  *input_subscripts =
      absl::StrSplit(std::move(inputs_and_output_subscripts[0]), ',');
  if (input_subscripts->size() != 1 && input_subscripts->size() != 2) {
    return errors::InvalidArgument(
        "Expecting 1 or 2 input subscripts in equation '", equation,
        "' but got: ", input_subscripts->size());
  }
  return Status::OK();
}

// Maps the character labels to consecutive integers.
void MapToLabels(const std::string& subscript, Labels* labels,
                 absl::flat_hash_map<char, int>* label_mapping) {
  for (int i = 0; i < subscript.size(); ++i) {
    const char label_char = subscript[i];
    if (label_char == '.') {
      labels->push_back(kEllipsisLabel);
      i += 2;  // Skip next 2 characters as well.
      continue;
    }
    if (!label_mapping->contains(label_char)) {
      const int next_label = label_mapping->size();
      (*label_mapping)[label_char] = next_label;
    }
    const int mapped_label = (*label_mapping)[label_char];
    labels->push_back(mapped_label);
  }
}

Status ParseEinsumEquation(const std::string& equation,
                           OperandLabels* input_labels, Labels* output_labels,
                           std::vector<EinsumDimensionType>* label_types,
                           OperandLabelCounts* input_label_counts,
                           LabelCounts* output_label_counts,
                           absl::InlinedVector<bool, 2UL>* input_has_ellipsis,
                           bool* output_has_ellipsis) {
  absl::InlinedVector<std::string, 2UL> input_str;
  std::string output_str;
  TF_RETURN_IF_ERROR(ValidateEinsumEquation(equation, &input_str, &output_str));

  // Temporary map from single character labels to (consecutive) integer labels.
  absl::flat_hash_map<char, int> label_mapping;
  int num_inputs = input_str.size();
  input_labels->resize(num_inputs);

  // Map from single characters to integer labels.
  for (int i = 0; i < num_inputs; ++i) {
    MapToLabels(input_str[i], &input_labels->at(i), &label_mapping);
  }
  MapToLabels(output_str, output_labels, &label_mapping);

  // Compute counts for input and output labels.
  int num_labels = label_mapping.size();
  input_label_counts->resize(num_inputs);
  input_has_ellipsis->resize(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_label_counts->at(i).resize(num_labels);
    input_has_ellipsis->at(i) = false;
    for (const int label : input_labels->at(i)) {
      if (label != kEllipsisLabel)
        input_label_counts->at(i)[label] += 1;
      else
        input_has_ellipsis->at(i) = true;
    }
  }
  output_label_counts->resize(num_labels);
  *output_has_ellipsis = false;
  for (const int label : *output_labels) {
    if (label != kEllipsisLabel)
      output_label_counts->at(label) += 1;
    else
      *output_has_ellipsis = true;
  }

  // Map each label to a unique EinsumDimensionType.
  label_types->resize(num_labels);
  for (int label = 0; label < num_labels; ++label) {
    if (label == kEllipsisLabel) continue;
    bool removed = (*output_label_counts)[label] == 0;
    bool unique = num_inputs == 1 || (*input_label_counts)[0][label] == 0 ||
                  (*input_label_counts)[1][label] == 0;
    (*label_types)[label] = GetDimensionType(removed, unique);
  }
  return Status::OK();
}

}  // namespace musa
}  // namespace tensorflow