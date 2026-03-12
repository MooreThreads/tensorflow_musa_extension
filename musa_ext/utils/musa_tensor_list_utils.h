#ifndef TENSORFLOW_MUSA_EXTENSION_MUSA_EXT_UTILS_MUSA_TENSOR_LIST_UTILS_H_
#define TENSORFLOW_MUSA_EXTENSION_MUSA_EXT_UTILS_MUSA_TENSOR_LIST_UTILS_H_

#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/tensor_list.h"

namespace tensorflow {

// These functions are needed to make TensorList work as a Variant type
// in our custom MUSA kernels. Since TensorFlow doesn't export these
// symbols for TensorList, we provide our own implementations.

template <>
void EncodeVariant<TensorList>(const TensorList& value, std::string* buf);

template <>
bool DecodeVariant<TensorList>(std::string* buf, TensorList* value);

template <>
void EncodeVariant<TensorList>(const TensorList& value, VariantTensorData* data);

template <>
bool DecodeVariant<TensorList>(VariantTensorData* data, TensorList* value);

template <>
std::string TypeNameVariant<TensorList>(const TensorList& value);

template <>
std::string DebugStringVariant<TensorList>(const TensorList& value);

}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_MUSA_EXT_UTILS_MUSA_TENSOR_LIST_UTILS_H_
