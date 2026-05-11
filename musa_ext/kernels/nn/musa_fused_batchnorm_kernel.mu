// MUSA FusedBatchNorm auxiliary kernels
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// BesselCorrectionKernel
//
// Applies Bessel correction in-place: data[i] *= factor
// Used to convert muDNN's population variance (1/N) to sample variance
// (1/(N-1)) for the batch_var output of FusedBatchNormV3.
// ---------------------------------------------------------------------------
__global__ void BesselCorrectionKernel(float* data, float factor, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] *= factor;
  }
}

extern "C" {

// Launch Bessel correction kernel on the given stream.
// data    : device pointer to the variance buffer to scale in-place
// factor  : N / (N - 1) where N = batch_size * height * width
// count   : number of elements (= number of channels C)
// stream  : MUSA stream for async execution
void LaunchBesselCorrection(float* data, float factor, int count,
                            musaStream_t stream) {
  if (count <= 0 || factor == 1.0f) return;
  int block_size = 256;
  int grid_size = (count + block_size - 1) / block_size;
  BesselCorrectionKernel<<<grid_size, block_size, 0, stream>>>(
      data, factor, count);
}

}  // extern "C"
