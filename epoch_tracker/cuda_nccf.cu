/*
Copyright 2015 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
// GPU-accelerated NCCF computation for REAPER epoch tracking.

#ifdef REAPER_USE_CUDA

#include "epoch_tracker/cuda_nccf.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Each block handles one peak. Threads within the block are distributed
// across lags. The reference energy (zero-lag autocorrelation) is computed
// once by thread 0 and broadcast via shared memory.
__global__ void nccf_kernel(
    const float* __restrict__ mixture,
    const int* __restrict__ peak_starts,
    float* __restrict__ nccf_out,
    int n_peaks,
    int n_lags,
    int first_lag,
    int window_size,
    int signal_length) {
  int peak_idx = blockIdx.x;
  if (peak_idx >= n_peaks) return;

  int start = peak_starts[peak_idx];

  extern __shared__ float s_energy[];

  if (threadIdx.x == 0) {
    float energy = 0.0f;
    int end = start + window_size;
    if (end <= signal_length) {
      for (int i = start; i < end; ++i) {
        float val = mixture[i];
        energy += val * val;
      }
    }
    s_energy[0] = energy;
  }
  __syncthreads();

  float energy = s_energy[0];
  float* out_row = nccf_out + peak_idx * n_lags;

  if (energy == 0.0f) {
    for (int lag_off = threadIdx.x; lag_off < n_lags;
         lag_off += blockDim.x) {
      out_row[lag_off] = 0.0f;
    }
    return;
  }

  for (int lag_off = threadIdx.x; lag_off < n_lags;
       lag_off += blockDim.x) {
    int lag = first_lag + lag_off;
    int ref_end = start + lag + window_size;
    if (ref_end > signal_length) {
      out_row[lag_off] = 0.0f;
      continue;
    }

    float lag_energy = 0.0f;
    float cross = 0.0f;
    for (int i = 0; i < window_size; ++i) {
      float a = mixture[start + i];
      float b = mixture[start + lag + i];
      cross += a * b;
      lag_energy += b * b;
    }

    if (lag_energy <= 0.0f) lag_energy = 1.0f;
    out_row[lag_off] = cross / sqrtf(energy * lag_energy);
  }
}

GpuNccfComputer::GpuNccfComputer()
    : d_mixture_(nullptr), d_peak_starts_(nullptr), d_nccf_out_(nullptr),
      max_signal_length_(0), max_peaks_(0), max_lags_(0),
      gpu_available_(false), initialized_(false) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  gpu_available_ = (err == cudaSuccess && device_count > 0);
}

GpuNccfComputer::~GpuNccfComputer() {
  if (d_mixture_) cudaFree(d_mixture_);
  if (d_peak_starts_) cudaFree(d_peak_starts_);
  if (d_nccf_out_) cudaFree(d_nccf_out_);
}

bool GpuNccfComputer::Initialize(int32_t max_signal_length,
                                  int32_t max_peaks,
                                  int32_t max_lags) {
  if (!gpu_available_) return false;

  max_signal_length_ = max_signal_length;
  max_peaks_ = max_peaks;
  max_lags_ = max_lags;

  cudaError_t err;
  err = cudaMalloc(&d_mixture_, max_signal_length * sizeof(float));
  if (err != cudaSuccess) return false;
  err = cudaMalloc(&d_peak_starts_, max_peaks * sizeof(int32_t));
  if (err != cudaSuccess) return false;
  err = cudaMalloc(&d_nccf_out_, max_peaks * max_lags * sizeof(float));
  if (err != cudaSuccess) return false;

  initialized_ = true;
  return true;
}

bool GpuNccfComputer::ComputeBatchNccf(
    const float* mixture, int32_t signal_length,
    const int32_t* peak_starts, int32_t n_peaks,
    int32_t first_lag, int32_t n_lags,
    int32_t window_size, float* nccf_out) {
  if (!initialized_ || n_peaks <= 0) return false;
  if (signal_length > max_signal_length_ || n_peaks > max_peaks_ ||
      n_lags > max_lags_)
    return false;

  cudaMemcpy(d_mixture_, mixture, signal_length * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_peak_starts_, peak_starts, n_peaks * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  int threads_per_block = 256;
  if (n_lags < threads_per_block) threads_per_block = n_lags;
  int shared_mem = sizeof(float);

  nccf_kernel<<<n_peaks, threads_per_block, shared_mem>>>(
      d_mixture_, d_peak_starts_, d_nccf_out_,
      n_peaks, n_lags, first_lag, window_size, signal_length);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    return false;
  }

  cudaDeviceSynchronize();
  cudaMemcpy(nccf_out, d_nccf_out_, n_peaks * n_lags * sizeof(float),
             cudaMemcpyDeviceToHost);
  return true;
}

#endif  // REAPER_USE_CUDA
