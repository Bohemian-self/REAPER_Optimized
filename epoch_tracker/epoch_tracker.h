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

// EpochTracker estimates the location of glottal closure instants
// (GCI), also known as "epochs" from digitized acoustic speech
// signals.  It simultaneously estimates the local fundamental
// frequency (F0) and voicing state of the speech on a per-epoch
// basis.  Various output methods are available for retrieving the
// results.
//
// The processing stages are:
//  * Optionally highpass the signal at 80 Hz to remove rumble, etc.
//  * Compute the LPC residual, obtaining an approximation of the
//    differentiated glottal flow.
//  * Normalize the amplitude of the residual by a local RMS measure.
//  * Pick the prominent peaks in the glottal flow, and grade them by
//    peakiness, skew and relative amplitude.
//  * Compute correlates of voicing to serve as pseudo-probabilities
//    of voicing, voicing onset and voicing offset.
//  * For every peak selected from the residual, compute a normalized
//    cross-correlation function (NCCF) of the LPC residual with a
//    relatively short reference window centered on the peak.
//  * For each peak in the residual, hypothesize all following peaks
//    within a specified F0 seaqrch range, that might be the end of a
//    period starting on that peak.
//    * Grade each of these hypothesized periods on local measures of
//      "voicedness" using the NCCF and the pseudo-probability of voicing
//      feature.
//    * Generate an unvoiced hypothesis for each period and grade it
//      for "voicelessness".
//    * Do a dynamic programming iteration to grade the goodness of
//      continuity between all hypotheses that start on a peak and
//      those that end on the same peak.  For voiced-voiced
//      connections add a cost for F0 transitions.  For
//      unvoiced-voiced and voiced-unvoiced transitions add a cost
//      that is modulated by the voicing onset or offset inverse
//      pseudo-probability.  Unvoiced-unvoiced transitions incur no cost.
//  * Backtrack through the lowest-cost path developed during the
//    dynamic-programming stage to determine the best peak collection
//    in the residual.  At each voiced peak, find the peak in the NCCF
//    (computed above) that corresponds to the duration closest to the
//    inter-peak interval, and use that as the inverse F0 for the
//    peak.
//
// A typical calling sequence might look like:
/* ==============================================================
   EpochTracker et;
   et.Init();  // Prepare the instance for, possibly, multiple calls.
   et.set_num_threads(4);  // Use 4 threads for parallel computation.
   Track* f0;  // for returning the F0 track
   Track* pm;  // for returning the epoch track
   if (!et.ComputeEpochs(my_input_waveform, &pm, &f0)) {
     exit(-2);  // problems in the epoch computations
   }
   DoSomethingWithTracks(f0, pm);
   delete f0;
   delete pm;
   ============================================================== */
//
// NOTE: Any client of this code inherits the Google command-line flags
// defined in epoch_tracker.cc.  These flags are processed in the Init()
// method, and override both default and params-sourced settings.
//
// As currently written, this is a batch process.  Very little has
// been done to conserve either memory or CPU.  The aim was simply to
// make the best possible tracker.  As will be seen in the
// implementation, there are many parameters that can be adjusted to
// influence the processing.  It is very unlikely that the best
// parameter setting is currently expressed in the code!  However, the
// performance, as written, appears to be quite good on a variety of voices.

#ifndef _EPOCH_TRACKER_H_
#define _EPOCH_TRACKER_H_

#include <memory>
#include <stdint.h>
#include <vector>
#include <string>

#ifdef REAPER_USE_OPENMP
#include <omp.h>
#endif

#ifdef REAPER_USE_CUDA
#include "epoch_tracker/cuda_nccf.h"
#endif

static const float kExternalFrameInterval = 0.005;
static const float kInternalFrameInterval = 0.002;
static const float kMinF0Search = 40.0;
static const float kMaxF0Search = 500.0;
static const float kUnvoicedPulseInterval = 0.01;
static const float kUnvoicedCost = 0.9;
static const bool kDoHighpass = true;
static const bool kDoHilbertTransform = false;
static const char kDebugName[] = "";


class EpochTracker {
 public:
  EpochTracker(void);

  virtual ~EpochTracker(void);

  void SetParameters(void);

  // NOTE: The following methods are exposed primarily for algorithm
  // development purposes, where EpochTracker is used in a developer's test
  // harness.  These need not/should not be called directly in normal use.

  // DEPRECATED Init method - retained for legacy code.
  bool Init(const int16_t* input, int32_t n_input, float sample_rate,
            float min_f0_search, float max_f0_search,
            bool do_highpass, bool do_hilbert_transform);

  void set_debug_name(const std::string& debug_name) {
    debug_name_ = debug_name;
  }

  std::string debug_name(void) { return debug_name_; }

  void HilbertTransform(int16_t* input, int32_t n_input, float* output);

  int16_t* HighpassFilter(int16_t* input, int32_t n_input,
                          float sample_rate, float corner_freq,
                          float fir_duration);

  bool GetLpcResidual(const std::vector<float>& input, float sample_rate,
                      std::vector<float>* output);

  void CrossCorrelation(const std::vector<float>& data, int32_t start,
                        int32_t first_lag, int32_t n_lags,
                        int32_t size, std::vector<float>* corr);

  bool GetBandpassedRmsSignal(const std::vector<float>& input, float sample_rate,
                              float low_limit, float high_limit, float frame_interval,
                              float frame_dur,  std::vector<float>* output_rms);

  void GetSymmetryStats(const std::vector<float>& data, float* pos_rms,
                        float* neg_rms, float* mean);

  void NormalizeAmplitude(const std::vector<float>& input, float sample_rate,
                          std::vector<float>* output);

  void Window(const std::vector<float>& input, int32_t offset, size_t size,
              float* output);

  bool ComputePolarity(int *polarity);

  bool ComputeFeatures(void);

  bool WriteDebugData(const std::vector<float>& data,
                      const std::string& extension);

  bool WriteDiagnostics(const std::string& file_base);

  bool TrackEpochs(void);

  void CreatePeriodLattice(void);

  void DoDynamicProgramming(void);

  bool BacktrackAndSaveOutput(void);

  bool ResampleAndReturnResults(float resample_interval,
                                std::vector<float>* f0,
                                std::vector<float>* correlations);

  void GetFilledEpochs(float unvoiced_pm_interval, std::vector<float>* times,
                       std::vector<int16_t>* voicing);

  // Setters.
  void set_do_hilbert_transform(bool v) { do_hilbert_transform_ = v; }
  void set_do_highpass(bool v) { do_highpass_ = v; }
  void set_external_frame_interval(float v) { external_frame_interval_ = v; }
  void set_unvoiced_pulse_interval(float v) { unvoiced_pulse_interval_ = v; }
  void set_min_f0_search(float v) { min_f0_search_ = v; }
  void set_max_f0_search(float v) { max_f0_search_ = v; }
  void set_unvoiced_cost(float v) { unvoiced_cost_ = v; }

  // Set thread count for OpenMP parallel regions. 0 = use all available.
  void set_num_threads(int n) { num_threads_ = n; }
  int num_threads(void) const { return num_threads_; }

 private:
  void GetResidualPulses(void);

  void GetVoiceTransitionFeatures(void);

  void GetRmsVoicingModulator(void);

  void CleanUp(void);

  int32_t FindNccfPeaks(const std::vector<float>& input, float thresh,
                        std::vector<int16_t>* output);

  void GetPulseCorrelations(float window_dur, float peak_thresh);

#ifdef REAPER_USE_CUDA
  bool GetPulseCorrelationsGPU(float window_dur, float peak_thresh,
                               const std::vector<float>& mixture);
#endif

 private:
  struct EpochCand {
    int32_t period;
    float local_cost;
    float cost_sum;
    int32_t start_peak;
    int32_t end_peak;
    int32_t best_prev_cand;
    int32_t closest_nccf_period;
    bool voiced;
  };

  typedef std::vector<EpochCand*> CandList;

  struct ResidPeak {
    int32_t resid_index;
    int32_t frame_index;
    float peak_quality;
    std::vector<float> nccf;
    std::vector<int16_t> nccf_periods;
    CandList future;
    CandList past;
  };

  struct TrackerResults {
    bool voiced;
    float f0;
    int32_t resid_index;
    float nccf_value;
  };
  typedef std::vector<TrackerResults> TrackerOutput;

 protected:
  std::vector<ResidPeak> resid_peaks_;
  TrackerOutput output_;
  std::vector<float> signal_;
  std::vector<float> residual_;
  std::vector<float> norm_residual_;
  std::vector<float> peaks_debug_;
  std::vector<float> bandpassed_rms_;
  std::vector<float> voice_onset_prob_;
  std::vector<float> voice_offset_prob_;
  std::vector<float> prob_voiced_;
  std::vector<float> best_corr_;
  std::vector<float> window_;
  float sample_rate_;

  float positive_rms_;
  float negative_rms_;
  int32_t n_feature_frames_;
  int32_t first_nccf_lag_;
  int32_t n_nccf_lags_;
  std::string debug_name_;

  // Threading control
  int num_threads_;

#ifdef REAPER_USE_CUDA
  std::unique_ptr<GpuNccfComputer> gpu_nccf_;
#endif

  // Control parameters available to clients of EpochTracker.
  float external_frame_interval_;
  float unvoiced_pulse_interval_;
  float min_f0_search_;
  float max_f0_search_;
  bool do_highpass_;
  bool do_hilbert_transform_;

  // Internal feature-computation Parameters:
  float internal_frame_interval_;
  float corner_frequency_;
  float filter_duration_;
  float frame_duration_;
  float lpc_frame_interval_;
  float preemphasis_;
  float noise_floor_;
  float peak_delay_;
  float skew_delay_;
  float peak_val_wt_;
  float peak_prominence_wt_;
  float peak_skew_wt_;
  float peak_quality_floor_;
  float time_span_;
  float level_change_den_;
  float min_rms_db_;
  float ref_dur_;
  float min_freq_for_rms_;
  float max_freq_for_rms_;
  float rms_window_dur_;
  float correlation_dur_;
  float correlation_thresh_;
  float reward_;
  float period_deviation_wt_;
  float peak_quality_wt_;
  float unvoiced_cost_;
  float nccf_uv_peak_wt_;
  float period_wt_;
  float level_wt_;
  float freq_trans_wt_;
  float voice_transition_factor_;
  float endpoint_padding_;
};


#endif  // _EPOCH_TRACKER_H_
