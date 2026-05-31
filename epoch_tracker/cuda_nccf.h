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

#ifndef _CUDA_NCCF_H_
#define _CUDA_NCCF_H_

#ifdef REAPER_USE_CUDA

#include <stdint.h>
#include <vector>

struct GpuNccfConfig {
  int32_t n_peaks;
  int32_t n_lags;
  int32_t first_lag;
  int32_t window_size;
  int32_t signal_length;
};

class GpuNccfComputer {
 public:
  GpuNccfComputer();
  ~GpuNccfComputer();

  bool Initialize(int32_t max_signal_length, int32_t max_peaks,
                  int32_t max_lags);

  // Compute NCCF for all peaks in one GPU batch.
  // peak_starts[i] is the start index into mixture for peak i.
  // Output: nccf_out is flattened [n_peaks x n_lags].
  bool ComputeBatchNccf(const float* mixture, int32_t signal_length,
                        const int32_t* peak_starts, int32_t n_peaks,
                        int32_t first_lag, int32_t n_lags,
                        int32_t window_size, float* nccf_out);

  bool IsAvailable() const { return gpu_available_; }

 private:
  float* d_mixture_;
  int32_t* d_peak_starts_;
  float* d_nccf_out_;
  int32_t max_signal_length_;
  int32_t max_peaks_;
  int32_t max_lags_;
  bool gpu_available_;
  bool initialized_;
};

#endif  // REAPER_USE_CUDA
#endif  // _CUDA_NCCF_H_
