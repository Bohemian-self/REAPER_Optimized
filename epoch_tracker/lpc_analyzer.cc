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
//  Implementation of the LpcAnalyzer class.
//
// Note that this is derived from legacy code written by David Talkin
// before the flood.  Hence the archaic style, etc.

#include "epoch_tracker/lpc_analyzer.h"
#include <stdlib.h>
#include <math.h>

#if defined(REAPER_USE_AVX2)
#include <immintrin.h>
#elif defined(REAPER_USE_SSE41)
#include <smmintrin.h>
#include <xmmintrin.h>
#elif defined(REAPER_USE_NEON)
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI (3.14159265359)
#endif

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void LpcAnalyzer::HannWindow(const float* din, float* dout, int n,
                             float preemp) {
  int i;
  const float *p;

  if (window_.size() != static_cast<size_t>(n)) {
    double arg, half = 0.5;
    window_.resize(n);
    for (i = 0, arg = M_PI * 2.0 / n; i < n; ++i)
      window_[i] = (half - half * cos((half + i) * arg));
  }
  if (preemp != 0.0) {
    for (i = 0, p = din + 1; i < n; ++i)
      *dout++ = window_[i] * (*p++ - (preemp * *din++));
  } else {
    for (i = 0; i < n; ++i)
      *dout++ = window_[i] * *din++;
  }
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void LpcAnalyzer::GetWindow(int n) {
  if (energywind_.size() != static_cast<size_t>(n)) {
    double arg = M_PI * 2.0 / n, half = 0.5;
    energywind_.resize(n);
    for (int i = 0; i < n; ++i)
      energywind_[i] = (half - half * cos((half + i) * arg));
  }
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void LpcAnalyzer::Autoc(int windowsize, float* s, int p, float* r, float* e) {
  int i, j;
  float sum;

  // Compute zero-lag energy with SIMD
  float sum0 = 0.0f;
#if defined(REAPER_USE_AVX2)
  {
    __m256 v_sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < windowsize; i += 8) {
      __m256 vs = _mm256_loadu_ps(s + i);
      v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(vs, vs));
    }
    __m128 hi = _mm256_extractf128_ps(v_sum, 1);
    __m128 lo = _mm256_castps256_ps128(v_sum);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    sum0 = _mm_cvtss_f32(s4);
    for (; i < windowsize; ++i) {
      sum0 += s[i] * s[i];
    }
  }
#elif defined(REAPER_USE_SSE41)
  {
    __m128 v_sum = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < windowsize; i += 4) {
      __m128 vs = _mm_loadu_ps(s + i);
      v_sum = _mm_add_ps(v_sum, _mm_mul_ps(vs, vs));
    }
    v_sum = _mm_hadd_ps(v_sum, v_sum);
    v_sum = _mm_hadd_ps(v_sum, v_sum);
    sum0 = _mm_cvtss_f32(v_sum);
    for (; i < windowsize; ++i) {
      sum0 += s[i] * s[i];
    }
  }
#elif defined(REAPER_USE_NEON)
  {
    float32x4_t v_sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < windowsize; i += 4) {
      float32x4_t vs = vld1q_f32(s + i);
      v_sum = vmlaq_f32(v_sum, vs, vs);
    }
    float32x2_t v2 = vadd_f32(vget_low_f32(v_sum), vget_high_f32(v_sum));
    sum0 = vget_lane_f32(vpadd_f32(v2, v2), 0);
    for (; i < windowsize; ++i) {
      sum0 += s[i] * s[i];
    }
  }
#else
  {
    float *q;
    for (i = windowsize, q = s; i--;) {
      sum = *q++;
      sum0 += sum*sum;
    }
  }
#endif

  *r = 1.;
  if (sum0 == 0.0) {
    *e = 1.;
    for (i = 1; i <= p; i++) {
      r[i] = 0.;
    }
    return;
  }
  *e = sqrt(sum0 / windowsize);
  float inv_sum0 = 1.0f / sum0;

  // Compute autocorrelation lags with SIMD
  for (i = 1; i <= p; i++) {
    int count = windowsize - i;
    float lag_sum = 0.0f;
#if defined(REAPER_USE_AVX2)
    {
      __m256 v_sum = _mm256_setzero_ps();
      int k = 0;
      for (; k + 7 < count; k += 8) {
        __m256 va = _mm256_loadu_ps(s + k);
        __m256 vb = _mm256_loadu_ps(s + k + i);
        v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(va, vb));
      }
      __m128 hi = _mm256_extractf128_ps(v_sum, 1);
      __m128 lo = _mm256_castps256_ps128(v_sum);
      __m128 s4 = _mm_add_ps(lo, hi);
      s4 = _mm_hadd_ps(s4, s4);
      s4 = _mm_hadd_ps(s4, s4);
      lag_sum = _mm_cvtss_f32(s4);
      for (; k < count; ++k)
        lag_sum += s[k] * s[k + i];
    }
#elif defined(REAPER_USE_SSE41)
    {
      __m128 v_sum = _mm_setzero_ps();
      int k = 0;
      for (; k + 3 < count; k += 4) {
        __m128 va = _mm_loadu_ps(s + k);
        __m128 vb = _mm_loadu_ps(s + k + i);
        v_sum = _mm_add_ps(v_sum, _mm_mul_ps(va, vb));
      }
      v_sum = _mm_hadd_ps(v_sum, v_sum);
      v_sum = _mm_hadd_ps(v_sum, v_sum);
      lag_sum = _mm_cvtss_f32(v_sum);
      for (; k < count; ++k)
        lag_sum += s[k] * s[k + i];
    }
#elif defined(REAPER_USE_NEON)
    {
      float32x4_t v_sum = vdupq_n_f32(0.0f);
      int k = 0;
      for (; k + 3 < count; k += 4) {
        float32x4_t va = vld1q_f32(s + k);
        float32x4_t vb = vld1q_f32(s + k + i);
        v_sum = vmlaq_f32(v_sum, va, vb);
      }
      float32x2_t v2 = vadd_f32(vget_low_f32(v_sum), vget_high_f32(v_sum));
      lag_sum = vget_lane_f32(vpadd_f32(v2, v2), 0);
      for (; k < count; ++k)
        lag_sum += s[k] * s[k + i];
    }
#else
    {
      float *q, *t;
      for (sum = 0.0f, j = count, q = s, t = s + i; j--;)
        lag_sum += (*q++) * (*t++);
    }
#endif
    *(++r) = lag_sum * inv_sum0;
  }
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void LpcAnalyzer::Durbin(float* r, float* k, float* a, int p, float* ex) {
  float  bb[BIGSORD];
  int i, j;
  float e, s, *b = bb;

  e = *r;
  *k = -r[1] / e;
  *a = *k;
  e *= (1.0 - (*k) * (*k));
  for (i = 1; i < p; i++) {
    s = 0;
    for (j = 0; j < i; j++) {
      s -= a[j] * r[i - j];
    }
    k[i] = (s - r[i + 1]) / e;
    a[i] = k[i];
    for (j = 0; j <= i; j++) {
      b[j] = a[j];
    }
    for (j = 0; j < i; j++) {
      a[j] += k[i] * b[i - j - 1];
    }
    e *= (1.0 - (k[i] * k[i]));
  }
  *ex = e;
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void LpcAnalyzer::PcToAutocorPc(float* a, float* b, float* c, int p) {
  float  s, *ap, *a0;
  int  i, j;

  for (s = 1., ap = a, i = p; i--; ap++)
    s += *ap * *ap;

  *c = s;
  for (i = 1; i <= p; i++) {
    s = a[i - 1];
    for (a0 = a, ap = a + i, j = p - i; j--;)
      s += (*a0++ * *ap++);
    *b++ = 2.0 * s;
  }
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
float LpcAnalyzer::ItakuraDistance(int p, float*  b, float* c, float* r,
                                    float gain) {
  float s;

  for (s = *c; p--;)
    s += *r++ * *b++;

  return s / gain;
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
float LpcAnalyzer::WindowedRms(float* data, int size) {
  float sum, f;
  int i;

  GetWindow(size);
  sum = 0.0f;
  for (i = 0; i < size; i++) {
    f = energywind_[i] * (*data++);
    sum += f * f;
  }
  return sqrt(sum / size);
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
int LpcAnalyzer::ComputeLpc(int lpc_ord, float noise_floor, int wsize,
                            const float* data, float* lpca, float* ar,
                            float* lpck, float* normerr, float* rms,
                            float preemp) {
  float rho[BIGSORD+1], k[BIGSORD], a[BIGSORD+1], *r, *kp, *ap, en, er;

  if ((wsize <= 0) || (!data) || (lpc_ord > BIGSORD))
    return false;

  float *dwind = new float[wsize];

  HannWindow(data, dwind, wsize, preemp);
  if (!(r = ar)) r = rho;
  if (!(kp = lpck)) kp = k;
  if (!(ap = lpca)) ap = a;
  Autoc(wsize, dwind, lpc_ord, r, &en);
  if (noise_floor > 1.0) {
    int i;
    float ffact;
    ffact = 1.0 / (1.0 + exp((-noise_floor / 20.0) * log(10.0)));
    for (i = 1; i <= lpc_ord; i++)
      rho[i] = ffact * r[i];
    *rho = *r;
    r = rho;
    if (ar) {
      for (i = 0; i <= lpc_ord; i++)
        ar[i] = r[i];
    }
  }
  Durbin(r, kp, ap + 1, lpc_ord, &er);
  float wfact = .612372;
  ap[0] = 1.0;
  if (rms)
    *rms = en / wfact;
  if (normerr)
    *normerr = er;
  delete [] dwind;
  return true;
}
