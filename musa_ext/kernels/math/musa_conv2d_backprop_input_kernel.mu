#include <cstdint>

#include <musa_runtime.h>

__global__ void Conv2DBackpropInputFloatKernel(
    const float* dy, const float* filter, float* dx, int64_t total_elements,
    int64_t in_h, int64_t in_w, int64_t in_c, int64_t out_h, int64_t out_w,
    int64_t out_c, int64_t filter_h, int64_t filter_w, int stride_h,
    int stride_w, int dilation_h, int dilation_w, int pad_top, int pad_left) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);

  for (int64_t idx = thread_id; idx < total_elements; idx += stride) {
    int64_t tmp = idx;
    const int64_t ic = tmp % in_c;
    tmp /= in_c;
    const int64_t iw = tmp % in_w;
    tmp /= in_w;
    const int64_t ih = tmp % in_h;
    const int64_t n = tmp / in_h;

    float sum = 0.0f;
    for (int64_t kh = 0; kh < filter_h; ++kh) {
      const int64_t oh_raw = ih + pad_top - kh * dilation_h;
      if (oh_raw < 0 || oh_raw % stride_h != 0) {
        continue;
      }
      const int64_t oh = oh_raw / stride_h;
      if (oh < 0 || oh >= out_h) {
        continue;
      }

      for (int64_t kw = 0; kw < filter_w; ++kw) {
        const int64_t ow_raw = iw + pad_left - kw * dilation_w;
        if (ow_raw < 0 || ow_raw % stride_w != 0) {
          continue;
        }
        const int64_t ow = ow_raw / stride_w;
        if (ow < 0 || ow >= out_w) {
          continue;
        }

        const int64_t dy_base = ((n * out_h + oh) * out_w + ow) * out_c;
        const int64_t filter_base = (kh * filter_w + kw) * in_c * out_c +
                                    ic * out_c;
        for (int64_t oc = 0; oc < out_c; ++oc) {
          sum += dy[dy_base + oc] * filter[filter_base + oc];
        }
      }
    }
    dx[idx] = sum;
  }
}

extern "C" void LaunchConv2DBackpropInputFloat(
    void* stream, const float* dy, const float* filter, float* dx,
    int64_t total_elements, int64_t batch, int64_t in_h, int64_t in_w,
    int64_t in_c, int64_t out_h, int64_t out_w, int64_t out_c,
    int64_t filter_h, int64_t filter_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int pad_top, int pad_left) {
  (void)batch;
  const int block_size = 256;
  int grid_size = static_cast<int>((total_elements + block_size - 1) /
                                   block_size);
  if (grid_size > 1024) {
    grid_size = 1024;
  }
  Conv2DBackpropInputFloatKernel<<<grid_size, block_size, 0,
                                   static_cast<musaStream_t>(stream)>>>(
      dy, filter, dx, total_elements, in_h, in_w, in_c, out_h, out_w, out_c,
      filter_h, filter_w, stride_h, stride_w, dilation_h, dilation_w, pad_top,
      pad_left);
}
