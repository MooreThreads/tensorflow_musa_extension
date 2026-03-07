#include "musa_tensor_list_utils.h"

namespace tensorflow {

// Define the static member kTypeName which is not exported by TF
const char TensorList::kTypeName[] = "tensorflow::TensorList";

void TensorList::Encode(VariantTensorData* data) const {
  data->set_type_name(TypeName());
  data->set_metadata("");
  for (const Tensor& t : tensors()) {
    *data->add_tensors() = t;
  }
}

bool TensorList::Decode(const VariantTensorData& data) {
  if (data.type_name() != TypeName()) {
    return false;
  }
  // The tensors_ object is managed by the class constructor/destructor
  tensors_->values_ = data.tensors();
  return true;
}

TensorList::~TensorList() { 
  if (tensors_) { 
    tensors_->Unref(); 
  } 
}

// Variant registration functions for TensorList

template <>
void EncodeVariant<TensorList>(const TensorList& value, std::string* buf) {
  VariantTensorData data;
  value.Encode(&data);
  data.set_type_name(value.TypeName());
  data.SerializeToString(buf);
}

template <>
bool DecodeVariant<TensorList>(std::string* buf, TensorList* value) {
  VariantTensorData data;
  if (!data.ParseFromString(*buf)) return false;
  return value->Decode(data);
}

template <>
void EncodeVariant<TensorList>(const TensorList& value, VariantTensorData* data) {
  value.Encode(data);
  data->set_type_name(value.TypeName());
}

template <>
bool DecodeVariant<TensorList>(VariantTensorData* data, TensorList* value) {
  return value->Decode(*data);
}

template <>
std::string TypeNameVariant<TensorList>(const TensorList& value) {
  return value.TypeName();
}

template <>
std::string DebugStringVariant<TensorList>(const TensorList& value) {
  return "TensorList";
}

}  // namespace tensorflow
