namespace tensorflow {
namespace musa {

void DoTranspose(OpKernelContext* ctx, const Tensor& input,
                 const std::vector<int64_t>& permutation, Tensor* output);

}  // namespace musa
}  // namespace tensorflow