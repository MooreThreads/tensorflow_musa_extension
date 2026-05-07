/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
 * TensorDot 辅助 kernel
 */

 #include <musa_runtime.h>

 extern "C" {

 // 用于小规模 tensordot 的直接计算 kernel（避免 transpose 开销）
 __global__ void TensorDotDirectKernel_float(
     const float* __restrict__ a,
     const float* __restrict__ b,
     float* __restrict__ output,
     int64_t a_batch_size,
     int64_t a_contract_size,
     int64_t b_batch_size,
     int64_t total_elements) {

   int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx < total_elements) {
     int64_t out_row = idx / b_batch_size;
     int64_t out_col = idx % b_batch_size;

     float sum = 0.0f;
     for (int64_t k = 0; k < a_contract_size; ++k) {
       sum += a[out_row * a_contract_size + k] * b[k * b_batch_size + out_col];
     }

     output[idx] = sum;
   }
 }

 void LaunchTensorDotDirectFloat(
     const float* a,
     const float* b,
     float* output,
     int64_t a_batch_size,
     int64_t a_contract_size,
     int64_t b_batch_size,
     musaStream_t stream) {

   int64_t total = a_batch_size * b_batch_size;
   int threads = 256;
   int blocks = (total + threads - 1) / threads;

   TensorDotDirectKernel_float<<<blocks, threads, 0, stream>>>(
       a, b, output, a_batch_size, a_contract_size, b_batch_size, total);
 }

 }  // extern "C"
