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
// Author: dtalkin@google.com (David Talkin)

// Implementation of the EpochTracker class.  This does all of the
// processing necessary to estimate the F0, voicing state and epochs
// (glottal-closure instants) in human speech signals.  See
// epoch_tracker.h for details.

#include "epoch_tracker/epoch_tracker.h"

#include <string>
#include <vector>
#include <cmath>

#include "epoch_tracker/fd_filter.h"
#include "epoch_tracker/lpc_analyzer.h"
#include "epoch_tracker/fft.h"

#if defined(REAPER_USE_AVX2)
#include <immintrin.h>
#elif defined(REAPER_USE_SSE41)
#include <smmintrin.h>
#include <xmmintrin.h>
#elif defined(REAPER_USE_NEON)
#include <arm_neon.h>
#endif

const int kMinSampleRate = 6000;

EpochTracker::EpochTracker(void) : sample_rate_(-1.0), num_threads_(0) {
  SetParameters();
#ifdef REAPER_USE_CUDA
  gpu_nccf_.reset(new GpuNccfComputer());
#endif
}

EpochTracker::~EpochTracker(void) {
  CleanUp();
}

static inline int32_t RoundUp(float val) {
  return static_cast<int32_t>(val + 0.5);
}

void EpochTracker::CleanUp(void) {
  for (size_t i = 0; i < resid_peaks_.size(); ++i) {
    for (size_t j = 0; j < resid_peaks_[i].future.size(); ++j) {
      delete resid_peaks_[i].future[j];
    }
  }
  resid_peaks_.clear();
  output_.clear();
  best_corr_.clear();
}

void EpochTracker::SetParameters(void) {
  external_frame_interval_ = kExternalFrameInterval;
  do_highpass_ = kDoHighpass;
  do_hilbert_transform_ = kDoHilbertTransform;
  max_f0_search_ = kMaxF0Search;
  min_f0_search_ = kMinF0Search;
  unvoiced_pulse_interval_ = kUnvoicedPulseInterval;
  debug_name_ = kDebugName;

  internal_frame_interval_ = kInternalFrameInterval;
  corner_frequency_ = 80.0;
  filter_duration_ = 0.05;
  frame_duration_ = 0.02;
  lpc_frame_interval_ = 0.01;
  preemphasis_ = 0.98;
  noise_floor_ = 70.0;
  peak_delay_ = 0.0004;
  skew_delay_ = 0.00015;
  peak_val_wt_ = 0.1;
  peak_prominence_wt_ = 0.3;
  peak_skew_wt_ = 0.1;
  peak_quality_floor_ = 0.01;
  time_span_ = 0.020;
  level_change_den_ = 30.0;
  min_rms_db_ = 20.0;
  ref_dur_ = 0.02;
  min_freq_for_rms_ = 100.0;
  max_freq_for_rms_ = 1000.0;
  rms_window_dur_ = 0.025;
  correlation_dur_ = 0.0075;
  correlation_thresh_ = 0.2;

  reward_ = -1.5;
  period_deviation_wt_ = 1.0;
  peak_quality_wt_ = 1.3;
  unvoiced_cost_ = kUnvoicedCost;
  nccf_uv_peak_wt_ = 0.9;
  period_wt_ = 0.0002;
  level_wt_ = 0.8;
  freq_trans_wt_ = 1.8;
  voice_transition_factor_ = 1.4;

  endpoint_padding_ = 0.01;
}

bool EpochTracker::Init(const int16_t* input, int32_t n_input, float sample_rate,
                        float min_f0_search, float max_f0_search,
                        bool do_highpass, bool do_hilbert_transform) {
  if (input && (sample_rate > 6000.0) && (n_input > (sample_rate * 0.05)) &&
      (min_f0_search < max_f0_search) && (min_f0_search > 0.0)) {
    CleanUp();
    min_f0_search_ = min_f0_search;
    max_f0_search_ = max_f0_search;
    sample_rate_ = sample_rate;
    int16_t* input_p = const_cast<int16_t *>(input);
    if (do_highpass) {
      input_p = HighpassFilter(input_p, n_input, sample_rate,
                               corner_frequency_, filter_duration_);
    }
    signal_.resize(n_input);
    if (do_hilbert_transform) {
      HilbertTransform(input_p, n_input, &(signal_.front()));
    } else {
      for (int32_t i = 0; i < n_input; ++i) {
        signal_[i] = input_p[i];
      }
    }
    if (input_p != input) {
      delete [] input_p;
    }
    return true;
  }
  return false;
}

void EpochTracker::HilbertTransform(int16_t* input, int32_t n_input,
                                    float* output) {
  FFT ft = FFT(FFT::fft_pow2_from_window_size(n_input));
  int32_t n_fft = ft.get_fftSize();
  float* re = new float[n_fft];
  float* im = new float[n_fft];
  for (int i = 0; i < n_input; ++i) {
    re[i] = input[i];
    im[i] = 0.0;
  }
  for (int i = n_input; i < n_fft; ++i) {
    re[i] = 0.0;
    im[i] = 0.0;
  }
  ft.fft(re, im);
  for (int i = 1; i < n_fft/2; ++i) {
    float tmp = im[i];
    im[i] = -re[i];
    re[i] = tmp;
  }
  re[0] = im[0] = 0.0;
  for (int i = n_fft/2 + 1; i < n_fft; ++i) {
    float tmp = im[i];
    im[i] = re[i];
    re[i] = -tmp;
  }
  ft.ifft(re, im);
  float inv_n = 1.0f / n_fft;
  for (int i = 0; i < n_input; ++i) {
    output[i] = re[i] * inv_n;
  }
  delete [] re;
  delete [] im;
}


int16_t* EpochTracker::HighpassFilter(int16_t* input, int32_t n_input,
                                      float sample_rate, float corner_freq,
				      float fir_duration) {
  FdFilter filter(sample_rate, corner_freq, true, fir_duration, false);
  int16_t* filtered_data = new int16_t[n_input];
  int32_t max_buffer_size = filter.GetMaxInputSize();
  int32_t to_process = n_input;
  bool start = true;
  bool end = false;
  int32_t input_index = 0;
  int32_t output_index = 0;
  while (to_process > 0) {
    int32_t to_send = to_process;
    if (to_send > max_buffer_size) {
      to_send = max_buffer_size;
    } else {
      end = true;
    }
    int32_t samples_returned = filter.FilterArray(input + input_index, to_send,
                                                  start, end,
                                                  filtered_data + output_index,
                                                  n_input - output_index);
    input_index += to_send;
    to_process -= to_send;
    output_index += samples_returned;
    start = false;
  }
  return filtered_data;
}


static float LpcDcGain(float* lpc, int32_t order) {
  float sum = 0.0;
  for (int32_t i = 0; i <= order; ++i) {
    sum += lpc[i];
  }
  if (sum > 0.0) {
    return sum;
  } else {
    return 1.0;
  }
}


static void MakeDeltas(float* now, float* next, int32_t size, int32_t n_steps,
                       float* deltas) {
  float inv_steps = 1.0f / n_steps;
  for (int32_t i = 0; i < size; ++i) {
    deltas[i] = (next[i] - now[i]) * inv_steps;
  }
}


bool EpochTracker::GetLpcResidual(const std::vector<float>& input, float sample_rate,
                                  std::vector<float>* output) {
  int32_t n_input = input.size();
  if (!((n_input > 0) && (sample_rate > 0.0) && output)) {
    return false;
  }
  output->resize(n_input);
  int32_t frame_step = RoundUp(sample_rate * lpc_frame_interval_);
  int32_t frame_size = RoundUp(sample_rate * frame_duration_);
  int32_t n_frames = 1 + ((n_input - frame_size) / frame_step);
  int32_t n_analyzed = ((n_frames - 1) * frame_step) + frame_size;
  if (n_analyzed <= n_input) {
    n_frames--;
    if (n_frames <= 0) {
      return false;
    }
  }
  LpcAnalyzer lp;
  int32_t order = lp.GetLpcOrder(sample_rate);
  float* lpc = new float[order + 1];
  float* old_lpc = new float[order + 1];
  float* delta_lpc = new float[order + 1];
  float norm_error = 0.0;
  float preemp_rms = 0.0;

#define  RELEASE_MEMORY() {                     \
    delete [] lpc;                              \
    delete [] old_lpc;                          \
    delete [] delta_lpc;                        \
  }

  if (!lp.ComputeLpc(order, noise_floor_, frame_size, &(input.front()),
                     old_lpc, NULL, NULL, &norm_error, &preemp_rms,
                     preemphasis_)) {
    RELEASE_MEMORY();
    return false;
  }
  for (int32_t i = 0; i <= order; ++i) {
    delta_lpc[i] = 0.0;
    (*output)[i] = 0.0;
  }
  float old_gain = LpcDcGain(old_lpc, order);
  float new_gain = 1.0;
  int32_t n_to_filter = (frame_size / 2) - order;
  int32_t input_p = 0;
  int32_t output_p = order;
  int32_t proc_p = 0;

  const float* input_data = &(input.front());
  float* output_data = &(output->front());

  for ( ; n_frames > 0; --n_frames, input_p += frame_step,
            n_to_filter = frame_step) {
    if (!lp.ComputeLpc(order, noise_floor_, frame_size,
                       input_data + input_p, lpc, NULL, NULL,
                       &norm_error, &preemp_rms, preemphasis_)) {
      RELEASE_MEMORY();
      return false;
    }
    new_gain = LpcDcGain(lpc, order);
    float delta_gain = (new_gain - old_gain) / n_to_filter;
    MakeDeltas(old_lpc, lpc, order+1, n_to_filter, delta_lpc);
    for (int32_t sample = 0; sample < n_to_filter; ++sample, ++proc_p,
             ++output_p) {
      float sum = 0.0;
      int32_t mem = proc_p;
      for (int32_t k = order; k > 0; --k, ++mem) {
        sum += (old_lpc[k] * input_data[mem]);
        old_lpc[k] += delta_lpc[k];
      }
      sum += input_data[mem];
      output_data[output_p] = sum / old_gain;
      old_gain += delta_gain;
    }
  }
  RELEASE_MEMORY();
  return true;
}

void EpochTracker::GetResidualPulses(void) {
  int32_t peak_ind = RoundUp(peak_delay_ * sample_rate_);
  int32_t skew_ind = RoundUp(skew_delay_ * sample_rate_);
  float min_peak = -1.0;
  int32_t limit = norm_residual_.size() - peak_ind;
  resid_peaks_.resize(0);
  peaks_debug_.resize(residual_.size());
  for (size_t i = 0; i < peaks_debug_.size(); ++i) {
    peaks_debug_[i] = 0.0;
  }
  for (int32_t i = peak_ind; i < limit; ++i) {
    float val = norm_residual_[i];
    if (val > min_peak) {
      continue;
    }
    if ((norm_residual_[i-1] > val) && (val <= norm_residual_[i+1])) {
      float vm_peak = norm_residual_[i - peak_ind];
      float vp_peak = norm_residual_[i + peak_ind];
      if ((vm_peak < val) || (vp_peak < val)) {
        continue;
      }
      float vm_skew = norm_residual_[i - skew_ind];
      float vp_skew = norm_residual_[i + skew_ind];
      float sharp = (0.5 * (vp_peak + vm_peak)) - val;
      float skew = -(vm_skew - vp_skew);
      ResidPeak p;
      p.resid_index = i;
      float time = static_cast<float>(i) / sample_rate_;
      p.frame_index = RoundUp(time / internal_frame_interval_);
      if (p.frame_index >= n_feature_frames_) {
        p.frame_index = n_feature_frames_ - 1;
      }
      p.peak_quality = (-val * peak_val_wt_) + (skew * peak_skew_wt_) +
          (sharp * peak_prominence_wt_);
      if (p.peak_quality < peak_quality_floor_) {
        p.peak_quality = peak_quality_floor_;
      }
      resid_peaks_.push_back(p);
      peaks_debug_[i] = p.peak_quality;
    }
  }
}


void EpochTracker::GetVoiceTransitionFeatures(void) {
  int32_t frame_offset = RoundUp(0.5 * time_span_ / internal_frame_interval_);
  if (frame_offset <= 0) {
    frame_offset = 1;
  }
  voice_onset_prob_.resize(n_feature_frames_);
  voice_offset_prob_.resize(n_feature_frames_);
  int32_t limit = n_feature_frames_ - frame_offset;

  // Parallelize the independent frame computations
#ifdef REAPER_USE_OPENMP
  #pragma omp parallel for schedule(static) if(limit - frame_offset > 500) \
      num_threads(num_threads_ > 0 ? num_threads_ : omp_get_max_threads())
#endif
  for (int32_t frame = frame_offset; frame < limit; ++frame) {
    float delta_rms = (bandpassed_rms_[frame + frame_offset] -
                       bandpassed_rms_[frame - frame_offset]) / level_change_den_;
    if (delta_rms > 1.0) {
      delta_rms = 1.0;
    } else {
      if (delta_rms < -1.0) {
        delta_rms = -1.0;
      }
    }
    float prob_onset = delta_rms;
    float prob_offset = -prob_onset;
    if (prob_onset > 1.0) {
      prob_onset = 1.0;
    } else {
      if (prob_onset < 0.0) {
        prob_onset = 0.0;
      }
    }
    if (prob_offset > 1.0) {
      prob_offset = 1.0;
    } else {
      if (prob_offset < 0.0) {
        prob_offset = 0.0;
      }
    }
    voice_onset_prob_[frame] = prob_onset;
    voice_offset_prob_[frame] = prob_offset;
  }
  for (int32_t frame = 0; frame < frame_offset; ++frame) {
    int32_t bframe = n_feature_frames_ - 1 - frame;
    voice_onset_prob_[frame] = voice_offset_prob_[frame] = 0.0;
    voice_onset_prob_[bframe] = voice_offset_prob_[bframe] = 0.0;
  }
}


void EpochTracker::GetRmsVoicingModulator(void) {
  float min_val = bandpassed_rms_[0];
  float max_val = min_val;

  prob_voiced_.resize(bandpassed_rms_.size());
  for (size_t i = 1; i < bandpassed_rms_.size(); ++i) {
    float val = bandpassed_rms_[i];
    if (val < min_val) {
      min_val = val;
    } else {
      if (val > max_val) {
        max_val = val;
      }
    }
  }
  if (min_val < min_rms_db_) {
    min_val = min_rms_db_;
  }
  float range = max_val - min_val;
  if (range < 1.0) {
    range = 1.0;
  }
  float inv_range = 1.0f / range;
  for (size_t i = 0; i < bandpassed_rms_.size(); ++i) {
    float v = (bandpassed_rms_[i] - min_val) * inv_range;
    prob_voiced_[i] = (v < 0.0f) ? 0.0f : v;
  }
}


int32_t EpochTracker::FindNccfPeaks(const std::vector<float>& input, float thresh,
                                    std::vector<int16_t>* output) {
  int32_t limit = input.size() - 1;
  uint32_t n_peaks = 0;
  float max_val = 0.0;
  int16_t max_index = 1;
  int16_t max_out_index = 0;
  output->resize(0);
  for (int16_t i = 1; i < limit; ++i) {
    float val = input[i];
    if ((val > thresh) && (val > input[i-1]) && (val >= input[i+1])) {
      if (val > max_val) {
        max_val = val;
        max_out_index = n_peaks;
        max_index = i;
      }
      n_peaks++;
      output->push_back(i);
    }
  }
  if ((n_peaks > 1) && (max_out_index > 0)) {
    int16_t hold = (*output)[0];
    (*output)[0] = (*output)[max_out_index];
    (*output)[max_out_index] = hold;
  } else {
    if (n_peaks <= 0) {
      n_peaks = 1;
      output->push_back(max_index);
    }
  }
  return n_peaks;
}


// SIMD-optimized dot product helper
static inline float DotProduct(const float* a, const float* b, int32_t n) {
  float sum = 0.0f;
#if defined(REAPER_USE_AVX2) && defined(REAPER_USE_FMA)
  __m256 v_sum = _mm256_setzero_ps();
  int32_t i = 0;
  for (; i + 7 < n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    v_sum = _mm256_fmadd_ps(va, vb, v_sum);
  }
  // Horizontal reduction
  __m128 hi = _mm256_extractf128_ps(v_sum, 1);
  __m128 lo = _mm256_castps256_ps128(v_sum);
  __m128 s4 = _mm_add_ps(lo, hi);
  s4 = _mm_hadd_ps(s4, s4);
  s4 = _mm_hadd_ps(s4, s4);
  sum = _mm_cvtss_f32(s4);
  for (; i < n; ++i) sum += a[i] * b[i];
#elif defined(REAPER_USE_AVX2)
  __m256 v_sum = _mm256_setzero_ps();
  int32_t i = 0;
  for (; i + 7 < n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(va, vb));
  }
  __m128 hi = _mm256_extractf128_ps(v_sum, 1);
  __m128 lo = _mm256_castps256_ps128(v_sum);
  __m128 s4 = _mm_add_ps(lo, hi);
  s4 = _mm_hadd_ps(s4, s4);
  s4 = _mm_hadd_ps(s4, s4);
  sum = _mm_cvtss_f32(s4);
  for (; i < n; ++i) sum += a[i] * b[i];
#elif defined(REAPER_USE_SSE41)
  __m128 v_sum = _mm_setzero_ps();
  int32_t i = 0;
  for (; i + 3 < n; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    v_sum = _mm_add_ps(v_sum, _mm_mul_ps(va, vb));
  }
  v_sum = _mm_hadd_ps(v_sum, v_sum);
  v_sum = _mm_hadd_ps(v_sum, v_sum);
  sum = _mm_cvtss_f32(v_sum);
  for (; i < n; ++i) sum += a[i] * b[i];
#elif defined(REAPER_USE_NEON)
  float32x4_t v_sum = vdupq_n_f32(0.0f);
  int32_t i = 0;
  for (; i + 3 < n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    v_sum = vmlaq_f32(v_sum, va, vb);
  }
  float32x2_t v2 = vadd_f32(vget_low_f32(v_sum), vget_high_f32(v_sum));
  sum = vget_lane_f32(vpadd_f32(v2, v2), 0);
  for (; i < n; ++i) sum += a[i] * b[i];
#else
  for (int32_t i = 0; i < n; ++i) sum += a[i] * b[i];
#endif
  return sum;
}

// SIMD-optimized sum-of-squares
static inline float SumOfSquares(const float* a, int32_t n) {
  return DotProduct(a, a, n);
}


void EpochTracker::CrossCorrelation(const std::vector<float>& data, int32_t start,
                                    int32_t first_lag, int32_t n_lags,
                                    int32_t size, std::vector<float>* corr) {
  const float* input = (&(data.front())) + start;
  corr->resize(n_lags);

  float energy = SumOfSquares(input, size);

  if (energy == 0.0) {
    for (int32_t i = 0; i < n_lags; ++i) {
      (*corr)[i] = 0.0;
    }
    return;
  }

  int32_t limit = first_lag + size;
  double lag_energy = 0.0;
  for (int32_t i = first_lag; i < limit; ++i) {
    lag_energy += input[i] * input[i];
  }

  int32_t last_lag = first_lag + n_lags;
  int32_t oind = 0;
  for (int32_t lag = first_lag; lag < last_lag; ++lag, ++oind) {
    float sum = DotProduct(input, input + lag, size);
    double le = (lag_energy <= 0.0) ? 1.0 : lag_energy;
    (*corr)[oind] = sum / sqrt(energy * le);
    int32_t lag_end = lag + size;
    lag_energy -= input[lag] * input[lag];
    lag_energy += input[lag_end] * input[lag_end];
  }
}


#ifdef REAPER_USE_CUDA
bool EpochTracker::GetPulseCorrelationsGPU(float window_dur, float peak_thresh,
                                           const std::vector<float>& mixture) {
  if (!gpu_nccf_ || !gpu_nccf_->IsAvailable()) return false;

  first_nccf_lag_ = RoundUp(sample_rate_ / max_f0_search_);
  int32_t max_lag = RoundUp(sample_rate_ / min_f0_search_);
  n_nccf_lags_ = max_lag - first_nccf_lag_;
  int32_t window_size = RoundUp(window_dur * sample_rate_);
  int32_t half_wind = window_size / 2;
  int32_t frame_size = window_size + max_lag;

  const float kMinCorrelationStep = 0.001;
  int32_t min_step = RoundUp(sample_rate_ * kMinCorrelationStep);

  // Phase 1: determine which peaks need fresh computation vs. copy
  size_t n_peaks = resid_peaks_.size();
  std::vector<bool> needs_compute(n_peaks, false);
  std::vector<int32_t> copy_source(n_peaks, -1);
  int32_t old_start = -(2.0 * min_step);
  int32_t last_computed = -1;

  for (size_t peak = 0; peak < n_peaks; ++peak) {
    int32_t start = resid_peaks_[peak].resid_index - half_wind;
    if (start < 0) start = 0;
    size_t end = start + frame_size;
    if ((end >= mixture.size()) || ((start - old_start) < min_step)) {
      copy_source[peak] = (last_computed >= 0) ? last_computed : 0;
    } else {
      needs_compute[peak] = true;
      last_computed = peak;
      old_start = start;
    }
  }

  // Phase 2: collect peaks that need GPU computation
  std::vector<int32_t> compute_indices;
  std::vector<int32_t> peak_starts;
  for (size_t peak = 0; peak < n_peaks; ++peak) {
    if (needs_compute[peak]) {
      compute_indices.push_back(peak);
      int32_t start = resid_peaks_[peak].resid_index - half_wind;
      if (start < 0) start = 0;
      peak_starts.push_back(start);
    }
  }

  int32_t n_compute = compute_indices.size();
  if (n_compute == 0) return false;

  // Initialize GPU buffers
  if (!gpu_nccf_->Initialize(mixture.size(), n_compute, n_nccf_lags_)) {
    return false;
  }

  // Launch GPU kernel
  std::vector<float> nccf_flat(n_compute * n_nccf_lags_);
  if (!gpu_nccf_->ComputeBatchNccf(
          &mixture.front(), mixture.size(),
          &peak_starts.front(), n_compute,
          first_nccf_lag_, n_nccf_lags_,
          window_size, &nccf_flat.front())) {
    return false;
  }

  // Phase 3: distribute results back to resid_peaks_
  for (int32_t ci = 0; ci < n_compute; ++ci) {
    int32_t peak = compute_indices[ci];
    resid_peaks_[peak].nccf.assign(
        nccf_flat.begin() + ci * n_nccf_lags_,
        nccf_flat.begin() + (ci + 1) * n_nccf_lags_);
    FindNccfPeaks(resid_peaks_[peak].nccf, peak_thresh,
                  &(resid_peaks_[peak].nccf_periods));
    for (size_t i = 0; i < resid_peaks_[peak].nccf_periods.size(); ++i) {
      resid_peaks_[peak].nccf_periods[i] += first_nccf_lag_;
    }
  }

  // Phase 4: fill in copies
  for (size_t peak = 0; peak < n_peaks; ++peak) {
    if (!needs_compute[peak] && copy_source[peak] >= 0) {
      resid_peaks_[peak].nccf = resid_peaks_[copy_source[peak]].nccf;
      resid_peaks_[peak].nccf_periods = resid_peaks_[copy_source[peak]].nccf_periods;
    }
  }

  return true;
}
#endif  // REAPER_USE_CUDA


void EpochTracker::GetPulseCorrelations(float window_dur, float peak_thresh) {
  first_nccf_lag_ = RoundUp(sample_rate_ / max_f0_search_);
  int32_t max_lag = RoundUp(sample_rate_ / min_f0_search_);
  n_nccf_lags_ = max_lag - first_nccf_lag_;
  int32_t window_size = RoundUp(window_dur * sample_rate_);
  int32_t half_wind = window_size / 2;
  int32_t frame_size = window_size + max_lag;

  std::vector<float> mixture;
  mixture.resize(residual_.size());
  const float kMinCorrelationStep = 0.001;
  const float kResidFract = 0.7;
  const float kPcmFract = 1.0 - kResidFract;

  // Vectorized mixture computation
  size_t mix_size = residual_.size();
#ifdef REAPER_USE_OPENMP
  #pragma omp parallel for schedule(static) if(mix_size > 10000) \
      num_threads(num_threads_ > 0 ? num_threads_ : omp_get_max_threads())
#endif
  for (size_t i = 0; i < mix_size; ++i) {
    mixture[i] = (kResidFract * residual_[i]) + (kPcmFract * signal_[i]);
  }

#ifdef REAPER_USE_CUDA
  if (gpu_nccf_ && gpu_nccf_->IsAvailable()) {
    if (GetPulseCorrelationsGPU(window_dur, peak_thresh, mixture)) {
      return;
    }
    // Fall through to CPU path if GPU fails
  }
#endif

  int32_t min_step = RoundUp(sample_rate_ * kMinCorrelationStep);
  size_t n_peaks = resid_peaks_.size();

  // Phase 1: determine which peaks need fresh computation
  std::vector<bool> needs_compute(n_peaks, false);
  std::vector<int32_t> copy_source(n_peaks, -1);
  std::vector<int32_t> starts(n_peaks);
  int32_t old_start = -(2.0 * min_step);
  int32_t last_computed = -1;

  for (size_t peak = 0; peak < n_peaks; ++peak) {
    int32_t start = resid_peaks_[peak].resid_index - half_wind;
    if (start < 0) start = 0;
    starts[peak] = start;
    size_t end = start + frame_size;
    if ((end >= mixture.size()) || ((start - old_start) < min_step)) {
      copy_source[peak] = (last_computed >= 0) ? last_computed : 0;
    } else {
      needs_compute[peak] = true;
      last_computed = peak;
      old_start = start;
    }
  }

  // Phase 2: parallel NCCF computation for independent peaks
#ifdef REAPER_USE_OPENMP
  int n_threads = num_threads_ > 0 ? num_threads_ : omp_get_max_threads();
  #pragma omp parallel for schedule(dynamic, 8) num_threads(n_threads)
#endif
  for (size_t peak = 0; peak < n_peaks; ++peak) {
    if (!needs_compute[peak]) continue;
    CrossCorrelation(mixture, starts[peak], first_nccf_lag_, n_nccf_lags_,
                     window_size, &(resid_peaks_[peak].nccf));
    FindNccfPeaks(resid_peaks_[peak].nccf, peak_thresh,
                  &(resid_peaks_[peak].nccf_periods));
    for (size_t i = 0; i < resid_peaks_[peak].nccf_periods.size(); ++i) {
      resid_peaks_[peak].nccf_periods[i] += first_nccf_lag_;
    }
  }

  // Phase 3: fill in copies (sequential - dependencies on computed peaks)
  for (size_t peak = 0; peak < n_peaks; ++peak) {
    if (!needs_compute[peak] && copy_source[peak] >= 0) {
      resid_peaks_[peak].nccf = resid_peaks_[copy_source[peak]].nccf;
      resid_peaks_[peak].nccf_periods = resid_peaks_[copy_source[peak]].nccf_periods;
    }
  }
}


void EpochTracker::Window(const std::vector<float>& input, int32_t offset, size_t size,
                          float* output) {
  if (size != window_.size()) {
    window_.resize(size);
    float arg = 2.0 * M_PI / size;
    for (size_t i = 0; i < size; ++i) {
      window_[i] = 0.5 - (0.5 * cos((i + 0.5) * arg));
    }
  }
  const float* data = (&(input.front())) + offset;
  for (size_t i = 0; i < size; ++i) {
    output[i] = data[i] * window_[i];
  }
}


bool EpochTracker::GetBandpassedRmsSignal(const std::vector<float>& input,
                                          float sample_rate,
                                          float low_limit, float high_limit,
                                          float frame_interval,
                                          float frame_dur,
                                          std::vector<float>* output_rms) {
  size_t frame_step = RoundUp(sample_rate * frame_interval);
  size_t frame_size = RoundUp(sample_rate * frame_dur);
  size_t n_frames = 1 + ((input.size() - frame_size) / frame_step);
  if (n_frames < 2) {
    fprintf(stderr, "input too small (%d) in GetBandpassedRmsSignal\n",
            static_cast<int>(input.size()));
    output_rms->resize(0);
    return false;
  }
  output_rms->resize(n_frames);

  int32_t pow2 = FFT::fft_pow2_from_window_size(frame_size);
  int32_t fft_size = 1 << pow2;
  int32_t first_bin = RoundUp(fft_size * low_limit / sample_rate);
  int32_t last_bin = RoundUp(fft_size * high_limit / sample_rate);

  size_t first_frame = frame_size / (2 * frame_step);
  if ((first_frame * 2 * frame_step) < frame_size) {
    first_frame++;
  }

  // Pre-compute window (thread-safe since all threads use the same size)
  std::vector<float> hann_window(frame_size);
  float arg = 2.0 * M_PI / frame_size;
  for (size_t i = 0; i < frame_size; ++i) {
    hann_window[i] = 0.5 - (0.5 * cos((i + 0.5) * arg));
  }

  float first_rms = 0.0f;
  bool first_rms_computed = false;

#ifdef REAPER_USE_OPENMP
  int n_threads = num_threads_ > 0 ? num_threads_ : omp_get_max_threads();
  #pragma omp parallel num_threads(n_threads) if(n_frames - first_frame > 20)
  {
    // Thread-local FFT and buffers
    FFT local_fft(pow2);
    int32_t local_fft_size = local_fft.get_fftSize();
    std::vector<float> re(local_fft_size, 0.0f);
    std::vector<float> im(local_fft_size, 0.0f);

    #pragma omp for schedule(static)
    for (size_t frame = first_frame; frame < n_frames; ++frame) {
      size_t offset = (frame - first_frame) * frame_step;
      const float* data = &(input.front()) + offset;
      for (size_t i = 0; i < frame_size; ++i) {
        re[i] = data[i] * hann_window[i];
        im[i] = 0.0f;
      }
      for (int32_t i = frame_size; i < local_fft_size; ++i) {
        re[i] = im[i] = 0.0f;
      }
      local_fft.fft(&re[0], &im[0]);
      float rms = 20.0f *
          log10(1.0f + local_fft.get_band_rms(&re[0], &im[0], first_bin, last_bin));
      (*output_rms)[frame] = rms;
    }
  }  // end omp parallel

  // Fill in early frames
  if (first_frame < n_frames) {
    first_rms = (*output_rms)[first_frame];
    for (size_t bframe = 0; bframe < first_frame; ++bframe) {
      (*output_rms)[bframe] = first_rms;
    }
  }
#else
  // Sequential path
  FFT ft(pow2);
  int32_t fft_size_actual = ft.get_fftSize();
  float* re = new float[fft_size_actual];
  float* im = new float[fft_size_actual];
  for (size_t frame = first_frame; frame < n_frames; ++frame) {
    size_t offset = (frame - first_frame) * frame_step;
    const float* data = &(input.front()) + offset;
    for (size_t i = 0; i < frame_size; ++i) {
      re[i] = data[i] * hann_window[i];
      im[i] = 0.0f;
    }
    for (int32_t i = frame_size; i < fft_size_actual; ++i) {
      re[i] = im[i] = 0.0f;
    }
    ft.fft(re, im);
    float rms = 20.0f *
        log10(1.0f + ft.get_band_rms(re, im, first_bin, last_bin));
    (*output_rms)[frame] = rms;
    if (frame == first_frame) {
      for (size_t bframe = 0; bframe < first_frame; ++bframe) {
        (*output_rms)[bframe] = rms;
      }
    }
  }
  delete [] re;
  delete [] im;
#endif
  return true;
}


void EpochTracker::GetSymmetryStats(const std::vector<float>& data, float* pos_rms,
                                    float* neg_rms, float* mean) {
  int32_t n_input = data.size();
  double p_sum = 0.0;
  double n_sum = 0.0;
  double sum = 0.0;
  int32_t n_p = 0;
  int32_t n_n = 0;

  // Compute mean
  for (int32_t i = 0; i < n_input; ++i) {
    sum += data[i];
  }
  *mean = sum / n_input;

  // Compute pos/neg RMS
  float m = *mean;
#ifdef REAPER_USE_OPENMP
  #pragma omp parallel for reduction(+:p_sum,n_sum,n_p,n_n) schedule(static) \
      if(n_input > 50000) \
      num_threads(num_threads_ > 0 ? num_threads_ : omp_get_max_threads())
#endif
  for (int32_t i = 0; i < n_input; ++i) {
    double val = data[i] - m;
    if (val > 0.0) {
      p_sum += (val * val);
      n_p++;
    } else if (val < 0.0) {
      n_sum += (val * val);
      n_n++;
    }
  }
  *pos_rms = sqrt(p_sum / n_p);
  *neg_rms = sqrt(n_sum / n_n);
}


void EpochTracker::NormalizeAmplitude(const std::vector<float>& input,
                                      float sample_rate,
                                      std::vector<float>* output) {
  int32_t n_input = input.size();
  int32_t ref_size = RoundUp(sample_rate * ref_dur_);

  output->resize(n_input);

  // Compute Hann window
  std::vector<float> norm_window(ref_size);
  float w_arg = 2.0 * M_PI / ref_size;
  for (int32_t i = 0; i < ref_size; ++i) {
    norm_window[i] = 0.5 - (0.5 * cos((i + 0.5) * w_arg));
  }

  int32_t ref_by_2 = ref_size / 2;
  int32_t frame_step = RoundUp(sample_rate * internal_frame_interval_);
  int32_t limit = n_input - ref_size;
  int32_t frame_limit = ref_by_2;
  int32_t data_p = 0;
  int32_t frame_p = 0;
  double old_inv_rms = 0.0;

  const float* in_data = &(input.front());
  float* out_data = &(output->front());

  while (frame_p < limit) {
    double ref_energy = 1.0;
    for (int32_t i = 0; i < ref_size; ++i) {
      double val = norm_window[i] * in_data[i + frame_p];
      ref_energy += (val * val);
    }
    double inv_rms = sqrt(static_cast<double>(ref_size) / ref_energy);
    double delta_inv_rms = 0.0;
    if (frame_p > 0) {
      delta_inv_rms = (inv_rms - old_inv_rms) / frame_step;
    } else {
      old_inv_rms = inv_rms;
    }
    for (int i = 0; i < frame_limit; ++i, ++data_p) {
      out_data[data_p] = in_data[data_p] * old_inv_rms;
      old_inv_rms += delta_inv_rms;
    }
    frame_limit = frame_step;
    frame_p += frame_step;
  }
  for ( ; data_p < n_input; ++data_p) {
    out_data[data_p] = in_data[data_p] * old_inv_rms;
  }
}

bool EpochTracker::ComputePolarity(int *polarity) {
  if (sample_rate_ <= 0.0) {
    fprintf(stderr, "EpochTracker not initialized in ComputeFeatures\n");
    return false;
  }
  if (!GetBandpassedRmsSignal(signal_, sample_rate_, min_freq_for_rms_,
                              max_freq_for_rms_, internal_frame_interval_,
                              rms_window_dur_, &bandpassed_rms_)) {
    fprintf(stderr, "Failure in GetBandpassedRmsSignal\n");
    return false;
  }
  if (!GetLpcResidual(signal_, sample_rate_, &residual_)) {
    fprintf(stderr, "Failure in GetLpcResidual\n");
    return false;
  }
  float mean = 0.0;
  GetSymmetryStats(residual_, &positive_rms_, &negative_rms_, &mean);
  *polarity = -1;
  if (positive_rms_ > negative_rms_) {
    *polarity = 1;
  }
  return true;
}

bool EpochTracker::ComputeFeatures(void) {
  if (sample_rate_ <= 0.0) {
    fprintf(stderr, "EpochTracker not initialized in ComputeFeatures\n");
    return false;
  }
  if (!GetBandpassedRmsSignal(signal_, sample_rate_, min_freq_for_rms_,
                              max_freq_for_rms_, internal_frame_interval_,
                              rms_window_dur_, &bandpassed_rms_)) {
    fprintf(stderr, "Failure in GetBandpassedRmsSignal\n");
    return false;
  }
  if (!GetLpcResidual(signal_, sample_rate_, &residual_)) {
    fprintf(stderr, "Failure in GetLpcResidual\n");
    return false;
  }
  n_feature_frames_ = bandpassed_rms_.size();
  float mean = 0.0;
  GetSymmetryStats(residual_, &positive_rms_, &negative_rms_, &mean);
  fprintf(stdout, "Residual symmetry: P:%f  N:%f  MEAN:%f\n",
	  positive_rms_, negative_rms_, mean);
  if (positive_rms_ > negative_rms_) {
    fprintf(stdout, "Inverting signal\n");
    size_t resid_size = residual_.size();
#ifdef REAPER_USE_OPENMP
    #pragma omp parallel for schedule(static) if(resid_size > 50000) \
        num_threads(num_threads_ > 0 ? num_threads_ : omp_get_max_threads())
#endif
    for (size_t i = 0; i < resid_size; ++i) {
      residual_[i] = -residual_[i];
      signal_[i] = -signal_[i];
    }
  }
  NormalizeAmplitude(residual_, sample_rate_, &norm_residual_);
  GetResidualPulses();
  GetPulseCorrelations(correlation_dur_, correlation_thresh_);
  GetVoiceTransitionFeatures();
  GetRmsVoicingModulator();
  return true;
}


bool EpochTracker::TrackEpochs(void) {
  CreatePeriodLattice();
  DoDynamicProgramming();
  return BacktrackAndSaveOutput();
}


void EpochTracker::CreatePeriodLattice(void) {
  int32_t low_period = RoundUp(sample_rate_ / max_f0_search_);
  int32_t high_period = RoundUp(sample_rate_ / min_f0_search_);
  int32_t total_cands = 0;

  for (size_t peak = 0; peak < resid_peaks_.size(); ++peak) {
    size_t frame_index = resid_peaks_[peak].frame_index;
    size_t resid_index = resid_peaks_[peak].resid_index;
    int32_t min_period = resid_index + low_period;
    int32_t max_period = resid_index + high_period;
    float lowest_cost = 1.0e30;
    float time = resid_index / sample_rate_;
    int32_t best_nccf_period = resid_peaks_[peak].nccf_periods[0];
    float best_cc_val =
        resid_peaks_[peak].nccf[best_nccf_period - first_nccf_lag_];
    best_corr_.push_back(time);
    best_corr_.push_back(best_cc_val);
    EpochCand* uv_cand = new EpochCand;
    uv_cand->voiced = false;
    uv_cand->start_peak = peak;
    uv_cand->cost_sum = 0.0;
    uv_cand->local_cost = 0.0;
    uv_cand->best_prev_cand = -1;
    int32_t next_cands_created = 0;
    for (size_t npeak = peak + 1; npeak < resid_peaks_.size(); ++npeak) {
      int32_t iperiod = resid_peaks_[npeak].resid_index - resid_index;
      if (resid_peaks_[npeak].resid_index >= min_period) {
        float fperiod = iperiod;
        int32_t cc_peak = 0;
        float min_period_diff = fabs(log(fperiod / best_nccf_period));
        for (size_t cc_peak_ind = 1;
             cc_peak_ind < resid_peaks_[peak].nccf_periods.size();
             ++cc_peak_ind) {
          int32_t nccf_period =  resid_peaks_[peak].nccf_periods[cc_peak_ind];
          float test_diff = fabs(log(fperiod / nccf_period));
          if (test_diff < min_period_diff) {
            min_period_diff = test_diff;
            cc_peak = cc_peak_ind;
          }
        }
        EpochCand* v_cand = new EpochCand;
        v_cand->voiced = true;
        v_cand->period = iperiod;
        int32_t cc_index = iperiod - first_nccf_lag_;
        float cc_value = 0.0;
        if ((cc_index >= 0) && (cc_index < n_nccf_lags_)) {
          cc_value = resid_peaks_[peak].nccf[cc_index];
        } else {
          int32_t peak_cc_index = resid_peaks_[peak].nccf_periods[cc_peak] -
              first_nccf_lag_;
          cc_value =  resid_peaks_[peak].nccf[peak_cc_index];
        }
        float per_dev_cost = period_deviation_wt_ * min_period_diff;
        float level_cost = level_wt_ * (1.0 - prob_voiced_[frame_index]);
        float period_cost = fperiod * period_wt_;
        float peak_qual_cost = peak_quality_wt_ /
            (resid_peaks_[npeak].peak_quality + resid_peaks_[peak].peak_quality);
        float local_cost =  (1.0 - cc_value) + per_dev_cost + peak_qual_cost +
            level_cost + period_cost + reward_;
        v_cand->local_cost = local_cost;
        if (local_cost < lowest_cost) {
          lowest_cost = local_cost;
          uv_cand->period = iperiod;
          level_cost = level_wt_ * prob_voiced_[frame_index];
          uv_cand->local_cost = (nccf_uv_peak_wt_ * cc_value) +
              level_cost + unvoiced_cost_ + reward_;
          uv_cand->end_peak = npeak;
          uv_cand->closest_nccf_period =
              resid_peaks_[peak].nccf_periods[cc_peak];
        }
        v_cand->start_peak = peak;
        v_cand->end_peak = npeak;
        v_cand->closest_nccf_period = resid_peaks_[peak].nccf_periods[cc_peak];
        v_cand->cost_sum = 0.0;
        v_cand->best_prev_cand = -1;
        resid_peaks_[peak].future.push_back(v_cand);
        resid_peaks_[npeak].past.push_back(v_cand);
        total_cands++;
        next_cands_created++;
        if (resid_peaks_[npeak].resid_index >= max_period) {
          break;
        }
      }
    }
    if (next_cands_created) {
      resid_peaks_[peak].future.push_back(uv_cand);
      resid_peaks_[uv_cand->end_peak].past.push_back(uv_cand);
      total_cands++;
    } else {
      delete uv_cand;
    }

    if (resid_peaks_[peak].past.size() == 0) {
      for (size_t pp = 0; pp < resid_peaks_[peak].future.size(); ++pp) {
        resid_peaks_[peak].future[pp]->cost_sum =
            resid_peaks_[peak].future[pp]->local_cost;
        resid_peaks_[peak].future[pp]->best_prev_cand = -1;
      }
    } else {
      int32_t uv_hyps_found = 0;
      float lowest_cost =  resid_peaks_[peak].past[0]->local_cost;
      size_t lowest_index = 0;
      for (size_t pcand = 0; pcand < resid_peaks_[peak].past.size(); ++pcand) {
        if (!resid_peaks_[peak].past[pcand]->voiced) {
          uv_hyps_found++;
        } else {
          if (resid_peaks_[peak].past[pcand]->local_cost < lowest_cost) {
            lowest_index = pcand;
            lowest_cost = resid_peaks_[peak].past[pcand]->local_cost;
          }
        }
      }
      if (!uv_hyps_found) {
        size_t start_peak = resid_peaks_[peak].past[lowest_index]->start_peak;
        EpochCand* uv_cand = new EpochCand;
        uv_cand->voiced = false;
        uv_cand->start_peak = start_peak;
        uv_cand->end_peak = peak;
        uv_cand->period =  resid_peaks_[peak].past[lowest_index]->period;
        uv_cand->closest_nccf_period =
            resid_peaks_[peak].past[lowest_index]->closest_nccf_period;
        uv_cand->cost_sum = 0.0;
        uv_cand->local_cost = 0.0;
        uv_cand->best_prev_cand = -1;
        float llevel_cost = level_wt_ *
            prob_voiced_[resid_peaks_[start_peak].frame_index];
        int32_t lcc_index = uv_cand->period - first_nccf_lag_;
        float lcc_value = 0.0;
        if ((lcc_index >= 0) && (lcc_index < n_nccf_lags_)) {
          lcc_value = resid_peaks_[start_peak].nccf[lcc_index];
        } else {
          int32_t peak_cc_index = uv_cand->closest_nccf_period - first_nccf_lag_;
          lcc_value =  resid_peaks_[start_peak].nccf[peak_cc_index];
        }
        uv_cand->local_cost = (nccf_uv_peak_wt_ * lcc_value) + llevel_cost +
            unvoiced_cost_ + reward_;
        resid_peaks_[start_peak].future.push_back(uv_cand);
        resid_peaks_[peak].past.push_back(uv_cand);
        total_cands++;
      }
    }
  }
}


void EpochTracker::DoDynamicProgramming(void) {
  for (size_t peak = 0; peak < resid_peaks_.size(); ++peak) {
    if (resid_peaks_[peak].past.size() == 0) {
      continue;
    }
    size_t n_future = resid_peaks_[peak].future.size();
    size_t n_past = resid_peaks_[peak].past.size();
    int32_t frame_index = resid_peaks_[peak].frame_index;

    // Pre-fetch voicing probabilities for this peak
    float onset_prob = voice_onset_prob_[frame_index];
    float offset_prob = voice_offset_prob_[frame_index];

    for (size_t fhyp = 0; fhyp < n_future; ++fhyp) {
      float min_cost = 1.0e30;
      size_t min_index = 0;
      float forward_period = resid_peaks_[peak].future[fhyp]->period;
      bool fhyp_voiced = resid_peaks_[peak].future[fhyp]->voiced;

      for (size_t phyp = 0; phyp < n_past; ++phyp) {
        float sum_cost;
        bool phyp_voiced = resid_peaks_[peak].past[phyp]->voiced;
        float past_cost_sum = resid_peaks_[peak].past[phyp]->cost_sum;

        if (fhyp_voiced && phyp_voiced) {
          float f_trans_cost = freq_trans_wt_ *
              fabs(log(forward_period / resid_peaks_[peak].past[phyp]->period));
          sum_cost = f_trans_cost + past_cost_sum;
        } else if (fhyp_voiced && !phyp_voiced) {
          float v_transition_cost = voice_transition_factor_ * (1.0 - onset_prob);
          sum_cost = past_cost_sum + v_transition_cost;
        } else if (!fhyp_voiced && phyp_voiced) {
          float v_transition_cost = voice_transition_factor_ * (1.0 - offset_prob);
          sum_cost = past_cost_sum + v_transition_cost;
        } else {
          sum_cost = past_cost_sum;
        }

        if (sum_cost < min_cost) {
          min_cost = sum_cost;
          min_index = phyp;
        }
      }
      resid_peaks_[peak].future[fhyp]->cost_sum =
          resid_peaks_[peak].future[fhyp]->local_cost + min_cost;
      resid_peaks_[peak].future[fhyp]->best_prev_cand = min_index;
    }
  }
}


bool EpochTracker::BacktrackAndSaveOutput(void) {
  if (resid_peaks_.size() == 0) {
    fprintf(stderr, "Can't backtrack with no residual peaks\n");
    return false;
  }
  float min_cost = 1.0e30;
  int32_t min_index = 0;
  size_t end = 0;
  for (size_t peak = resid_peaks_.size() - 1; peak > 0; --peak) {
    if ((resid_peaks_[peak].past.size() > 1)) {
      for (size_t ind = 0; ind < resid_peaks_[peak].past.size(); ++ind) {
        if (resid_peaks_[peak].past[ind]->cost_sum < min_cost) {
          min_cost = resid_peaks_[peak].past[ind]->cost_sum;
          min_index = ind;
        }
      }
      end = peak;
      break;
    }
  }
  if (end == 0) {
    fprintf(stderr, "No terminal peak found in DynamicProgramming\n");
    return false;
  }
  output_.clear();
  while (1) {
    int32_t start_peak = resid_peaks_[end].past[min_index]->start_peak;
    TrackerResults tr;
    tr.resid_index = resid_peaks_[start_peak].resid_index;
    if (resid_peaks_[end].past[min_index]->voiced) {
      float nccf_period =
          resid_peaks_[end].past[min_index]->closest_nccf_period;
      tr.f0 = sample_rate_ / nccf_period;
      tr.voiced = true;
    } else {
      tr.f0 = 0.0;
      tr.voiced = false;
    }
    int32_t cc_index = resid_peaks_[end].past[min_index]->period -
        first_nccf_lag_;
    if ((cc_index >= 0) && (cc_index < n_nccf_lags_)) {
      tr.nccf_value = resid_peaks_[start_peak].nccf[cc_index];
    } else {
      int32_t peak_cc_index =
          resid_peaks_[end].past[min_index]->closest_nccf_period -
          first_nccf_lag_;
      tr.nccf_value =  resid_peaks_[start_peak].nccf[peak_cc_index];
    }
    output_.push_back(tr);
    size_t new_end =  resid_peaks_[end].past[min_index]->start_peak;
    min_index = resid_peaks_[end].past[min_index]->best_prev_cand;
    if (min_index < 0) {
      break;
    }
    end = new_end;
  }
  return true;
}


void EpochTracker::GetFilledEpochs(float unvoiced_pm_interval,
                                   std::vector<float>* times,
                                   std::vector<int16_t>* voicing) {
  times->clear();
  voicing->clear();
  float final_time = norm_residual_.size() / sample_rate_;
  int32_t limit = output_.size() - 1;
  int32_t i = limit;
  while (i >= 0) {
    int32_t i_old = i;
    float time = output_[i].resid_index / sample_rate_;
    if (output_[i].voiced || ((i < limit) && (output_[i+1].voiced))) {
      times->push_back(time);
      voicing->push_back(1);
      i--;
    }
    if (i == limit) {
      time = 0.0;
    }
    if ((i > 0) && (!output_[i].voiced) && (time < final_time)) {
      for ( ; i > 0; --i) {
        if (output_[i].voiced) {
          break;
        }
      }
      float next_time = final_time;
      int32_t fill_ind = 1;
      if (i > 0) {
        next_time = (output_[i].resid_index / sample_rate_) -
            (1.0 / max_f0_search_);
      }
      float now = time + (fill_ind * unvoiced_pm_interval);
      while (now < next_time) {
        times->push_back(now);
        voicing->push_back(0);
        fill_ind++;
        now = time + (fill_ind * unvoiced_pm_interval);
      }
    }
    if (i == i_old) {
      i--;
    }
  }
}


bool EpochTracker::ResampleAndReturnResults(float resample_interval,
                                            std::vector<float>* f0,
                                            std::vector<float>* correlations) {
  if ((sample_rate_ <= 0.0) || (output_.size() == 0)) {
    fprintf(stderr,
            "Un-initialized EpochTracker or no output_ in ResampleAndReturnF0\n");
    return false;
  }
  if (resample_interval <= 0.0) {
    fprintf(stderr, "resample_interval <= 0.0 in ResampleAndReturnF0\n");
    return false;
  }
  float last_time = (output_[0].resid_index / sample_rate_) + endpoint_padding_;
  int32_t n_frames = RoundUp(last_time / resample_interval);
  f0->resize(0);
  correlations->resize(0);
  f0->insert(f0->begin(), n_frames, 0.0);
  correlations->insert(correlations->begin(), n_frames, 0.0);
  int32_t limit = output_.size() - 1;
  int32_t prev_frame = 0;
  float prev_f0 = output_[limit].f0;
  float prev_corr = output_[limit].nccf_value;
  for (int32_t i = limit; i >= 0; --i) {
    int32_t frame = RoundUp(output_[i].resid_index /
                            (sample_rate_ * resample_interval));
    (*f0)[frame] = output_[i].f0;
    (*correlations)[frame] = output_[i].nccf_value;
    if ((frame - prev_frame) > 1) {
      for (int32_t fr = prev_frame + 1; fr < frame; ++fr) {
        (*f0)[fr] = prev_f0;
        (*correlations)[fr] = prev_corr;
      }
    }
    prev_frame = frame;
    prev_corr = output_[i].nccf_value;
    prev_f0 = output_[i].f0;
  }
  for (int32_t frame = prev_frame; frame < n_frames; ++frame) {
    (*f0)[frame] = prev_f0;
    (*correlations)[frame] = prev_corr;
  }
  return true;
}


bool EpochTracker::WriteDebugData(const std::vector<float>& data,
                                  const std::string& extension) {
  if (debug_name_.empty()) {
    return true;
  }
  std::string filename = debug_name_ + "." + extension;
  if (data.size() == 0) {
    fprintf(stdout, "Data size==0 for %s in WriteDebugData\n",
               filename.c_str());
    return false;
  }
  FILE* out = fopen(filename.c_str(), "w");
  if (!out) {
    fprintf(stderr, "Can't open %s for debug output\n", filename.c_str());
    return false;
  }
  size_t  written = fwrite(&(data.front()), sizeof(data.front()),
                           data.size(), out);
  fclose(out);
  if (written != data.size()) {
    fprintf(stderr, "Problems writing debug data (%d %d)\n",
            static_cast<int>(written), static_cast<int>(data.size()));
    return false;
  }
  return true;
}

bool EpochTracker::WriteDiagnostics(const std::string& file_base) {
  if (!file_base.empty()) {
    set_debug_name(file_base);
  }
  WriteDebugData(signal_, "pcm");
  WriteDebugData(residual_, "resid");
  WriteDebugData(norm_residual_, "nresid");
  WriteDebugData(bandpassed_rms_, "bprms");
  WriteDebugData(voice_onset_prob_, "onsetp");
  WriteDebugData(voice_offset_prob_, "offsetp");
  WriteDebugData(peaks_debug_, "pvals");
  WriteDebugData(prob_voiced_, "pvoiced");
  WriteDebugData(best_corr_, "bestcorr");
  if ((!debug_name_.empty()) && (output_.size() > 2)) {
    std::string pm_name = debug_name_ + ".pmlab";
    FILE* pmfile = fopen(pm_name.c_str(), "w");
    fprintf(pmfile, "#\n");
    std::vector<float> f0;
    int32_t limit = output_.size() - 1;
    for (int32_t i = limit; i >= 0; --i) {
      float time = output_[i].resid_index / sample_rate_;
      if (output_[i].voiced || ((i < limit) && (output_[i+1].voiced))) {
        fprintf(pmfile, "%f blue \n", time);
      } else {
        fprintf(pmfile, "%f red \n", time);
      }
      f0.push_back(time);
      f0.push_back(output_[i].f0);
      f0.push_back(output_[i].nccf_value);
    }
    fclose(pmfile);
    WriteDebugData(f0, "f0ap");
  }
  return true;
}
