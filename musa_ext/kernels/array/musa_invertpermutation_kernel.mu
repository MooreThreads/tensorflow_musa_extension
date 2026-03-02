namespace tensorflow {
namespace musa {

template <typename T>
__global__ void InvertPermutationKernel(
    const T* __restrict__ perm,
    T* __restrict__ inv_perm,
    int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        inv_perm[static_cast<size_t>(perm[idx])] = static_cast<T>(idx);
    }
}

template <typename T>
void MusaInvertPermutationKernelLauncher(
    const void* perm, void* inv_perm, int64_t n, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    InvertPermutationKernel<T><<<grid_size, block_size, 0, stream>>>(
        static_cast<const T*>(perm),
        static_cast<T*>(inv_perm),
        n);
}

template void MusaInvertPermutationKernelLauncher<int>(const void*, void*, int64_t, musaStream_t);

template void MusaInvertPermutationKernelLauncher<long long>(const void*, void*, int64_t, musaStream_t);

} // namespace musa
} // namespace tensorflow
