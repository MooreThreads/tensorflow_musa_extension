#include <cstdint>

#include <musa_runtime.h>

__global__ void MaxPoolGradFloatKernel(
    const float* input, const float* grad_output, float* grad_input,
    int64_t total_elements, int64_t in_h, int64_t in_w, int64_t channels,
    int64_t out_h, int64_t out_w, int window_h, int window_w, int stride_h,
    int stride_w, int pad_top, int pad_left) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);

  for (int64_t idx = thread_id; idx < total_elements; idx += grid_stride) {
    int64_t tmp = idx;
    const int64_t c = tmp % channels;
    tmp /= channels;
    const int64_t iw = tmp % in_w;
    tmp /= in_w;
    const int64_t ih = tmp % in_h;
    const int64_t n = tmp / in_h;

    const float input_value = input[idx];
    float sum = 0.0f;

    for (int64_t oh = 0; oh < out_h; ++oh) {
      const int64_t h_start = oh * stride_h - pad_top;
      const int64_t h_end = h_start + window_h;
      if (ih < h_start || ih >= h_end) {
        continue;
      }

      for (int64_t ow = 0; ow < out_w; ++ow) {
        const int64_t w_start = ow * stride_w - pad_left;
        const int64_t w_end = w_start + window_w;
        if (iw < w_start || iw >= w_end) {
          continue;
        }

        float max_value = -3.4028234663852886e38f;
        for (int kh = 0; kh < window_h; ++kh) {
          const int64_t cur_h = h_start + kh;
          if (cur_h < 0 || cur_h >= in_h) {
            continue;
          }
          for (int kw = 0; kw < window_w; ++kw) {
            const int64_t cur_w = w_start + kw;
            if (cur_w < 0 || cur_w >= in_w) {
              continue;
            }
            const int64_t input_index =
                ((n * in_h + cur_h) * in_w + cur_w) * channels + c;
            const float value = input[input_index];
            if (value > max_value) {
              max_value = value;
            }
          }
        }

        if (input_value == max_value) {
          const int64_t grad_index =
              ((n * out_h + oh) * out_w + ow) * channels + c;
          sum += grad_output[grad_index];
        }
      }
    }

    grad_input[idx] = sum;
  }
}

extern "C" void LaunchMaxPoolGradFloat(
    void* stream, const float* input, const float* grad_output,
    float* grad_input, int64_t total_elements, int64_t batch, int64_t in_h,
    int64_t in_w, int64_t channels, int64_t out_h, int64_t out_w,
    int window_h, int window_w, int stride_h, int stride_w, int pad_top,
    int pad_left) {
  (void)batch;
  const int block_size = 256;
  int grid_size = static_cast<int>((total_elements + block_size - 1) /
                                   block_size);
  if (grid_size > 1024) {
    grid_size = 1024;
  }
  MaxPoolGradFloatKernel<<<grid_size, block_size, 0,
                           static_cast<musaStream_t>(stream)>>>(
      input, grad_output, grad_input, total_elements, in_h, in_w, channels,
      out_h, out_w, window_h, window_w, stride_h, stride_w, pad_top, pad_left);
}
