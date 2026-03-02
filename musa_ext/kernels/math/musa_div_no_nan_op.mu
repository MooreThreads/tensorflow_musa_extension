#include <musa_runtime.h>

namespace tensorflow {
namespace musa {

struct DivNoNanStrides {
  int s0, s1, s2, s3;
};

struct DivNoNanDims {
  int d0, d1, d2, d3;
};

template <typename T>
__global__ void DivNoNanKernel(const T* x, DivNoNanStrides x_st,
                               const T* y, DivNoNanStrides y_st,
                               T* out, DivNoNanDims dims,
                               int n_total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_total) {
    int temp = i;
    int w = temp % dims.d3; temp /= dims.d3;
    int h = temp % dims.d2; temp /= dims.d2;
    int c = temp % dims.d1; temp /= dims.d1;
    int n = temp;

    int idx_x = n * x_st.s0 + c * x_st.s1 + h * x_st.s2 + w * x_st.s3;
    int idx_y = n * y_st.s0 + c * y_st.s1 + h * y_st.s2 + w * y_st.s3;
    
    T val_y = y[idx_y];

    if (val_y == T(0)) {
      out[i] = T(0);
    } else {
      out[i] = x[idx_x] / val_y;
    }
  }
}

template <typename T>
void LaunchDivNoNan(const T* in0, const T* in1, T* out,
                    DivNoNanStrides s_in0, DivNoNanStrides s_in1, DivNoNanDims dims,
                    int total_elements, musaStream_t stream) {
  if (total_elements == 0) return;
  int block_size = 256;
  int grid_size = (total_elements + block_size - 1) / block_size;
  DivNoNanKernel<T><<<grid_size, block_size, 0, stream>>>(
      in0, s_in0, in1, s_in1, out, dims, total_elements);
}

template void LaunchDivNoNan<float>(const float*, const float*, float*, DivNoNanStrides, DivNoNanStrides, DivNoNanDims, int, musaStream_t);
template void LaunchDivNoNan<double>(const double*, const double*, double*, DivNoNanStrides, DivNoNanStrides, DivNoNanDims, int, musaStream_t);

} // namespace musa
} // namespace tensorflow