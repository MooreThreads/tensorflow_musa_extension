#include <musa_runtime.h>

namespace tensorflow {
namespace musa {

__device__ inline float atomicMul(float* address, float val) {
    int* address_as_ull = (int*)address;
    int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __float_as_int(val * __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ inline int atomicMul(int* address, int val) {
    int old = *address, assumed;
    do {
        assumed = old;
        old = atomicCAS(address, assumed, assumed * val);
    } while (assumed != old);
    return old;
}

struct TensorInfo {
    int dims[8];
    int strides[8];
    int rank;
};

template <typename T>
__global__ void FillOneKernel(T* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = static_cast<T>(1);
    }
}

template <typename T>
__global__ void ProdKernel(const T* input, T* output, int num_elements,
                           TensorInfo in_info, 
                           TensorInfo eff_out_info) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    T val = input[idx];
    int temp_idx = idx;
    int out_idx = 0;

    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d >= in_info.rank) break;
        int coord = (temp_idx / in_info.strides[d]) % in_info.dims[d];
        out_idx += coord * eff_out_info.strides[d];
    }

    atomicMul(output + out_idx, val);
}

template <typename T>
void LaunchFillOne(T* data, int n, musaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    FillOneKernel<T><<<grid_size, block_size, 0, stream>>>(data, n);
}

template <typename T>
void LaunchProd(const T* input, T* output, int input_size,
                const int* in_dims, const int* in_strides,
                const int* eff_out_strides, int rank,
                musaStream_t stream) {
    
    TensorInfo in_info;
    TensorInfo eff_out_info;
    
    in_info.rank = rank;
    eff_out_info.rank = rank;

    for(int i=0; i<rank; ++i) {
        in_info.dims[i] = in_dims[i];
        in_info.strides[i] = in_strides[i];
        eff_out_info.strides[i] = eff_out_strides[i];
    }

    int block_size = 256;
    int grid_size = (input_size + block_size - 1) / block_size;

    ProdKernel<T><<<grid_size, block_size, 0, stream>>>(
        input, output, input_size, in_info, eff_out_info);
}

template void LaunchFillOne<float>(float*, int, musaStream_t);
template void LaunchFillOne<int>(int*, int, musaStream_t);

template void LaunchProd<float>(const float*, float*, int, const int*, const int*, const int*, int, musaStream_t);
template void LaunchProd<int>(const int*, int*, int, const int*, const int*, const int*, int, musaStream_t);

} // namespace musa
} // namespace tensorflow
