#include <cstdint>

#include <musa_runtime.h>

__global__ void Conv2DBackpropFilterFloatKernel(
    const float* input, const float* dy, float* filter_backprop,
    int64_t total_elements, int64_t batch, int64_t in_h, int64_t in_w,
    int64_t in_c, int64_t out_h, int64_t out_w, int64_t out_c,
    int64_t filter_h, int64_t filter_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int pad_top, int pad_left) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);

  for (int64_t idx = thread_id; idx < total_elements; idx += stride) {
    int64_t tmp = idx;
    const int64_t oc = tmp % out_c;
    tmp /= out_c;
    const int64_t ic = tmp % in_c;
    tmp /= in_c;
    const int64_t kw = tmp % filter_w;
    const int64_t kh = tmp / filter_w;

    float sum = 0.0f;
    for (int64_t n = 0; n < batch; ++n) {
      for (int64_t oh = 0; oh < out_h; ++oh) {
        const int64_t ih = oh * stride_h + kh * dilation_h - pad_top;
        if (ih < 0 || ih >= in_h) {
          continue;
        }

        for (int64_t ow = 0; ow < out_w; ++ow) {
          const int64_t iw = ow * stride_w + kw * dilation_w - pad_left;
          if (iw < 0 || iw >= in_w) {
            continue;
          }

          const int64_t input_index = ((n * in_h + ih) * in_w + iw) * in_c + ic;
          const int64_t dy_index = ((n * out_h + oh) * out_w + ow) * out_c + oc;
          sum += input[input_index] * dy[dy_index];
        }
      }
    }
    filter_backprop[idx] = sum;
  }
}

extern "C" void LaunchConv2DBackpropFilterFloat(
    void* stream, const float* input, const float* dy, float* filter_backprop,
    int64_t total_elements, int64_t batch, int64_t in_h, int64_t in_w,
    int64_t in_c, int64_t out_h, int64_t out_w, int64_t out_c,
    int64_t filter_h, int64_t filter_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int pad_top, int pad_left) {
  const int block_size = 256;
  int grid_size = static_cast<int>((total_elements + block_size - 1) /
                                   block_size);
  if (grid_size > 1024) {
    grid_size = 1024;
  }
  Conv2DBackpropFilterFloatKernel<<<grid_size, block_size, 0,
                                    static_cast<musaStream_t>(stream)>>>(
      input, dy, filter_backprop, total_elements, batch, in_h, in_w, in_c,
      out_h, out_w, out_c, filter_h, filter_w, stride_h, stride_w, dilation_h,
      dilation_w, pad_top, pad_left);
}
