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
// Author: David Talkin (dtalkin@google.com)

#include "epoch_tracker/fft.h"

#if defined(REAPER_USE_AVX2)
#include <immintrin.h>
#elif defined(REAPER_USE_SSE41)
#include <smmintrin.h>
#include <xmmintrin.h>
#elif defined(REAPER_USE_NEON)
#include <arm_neon.h>
#endif

FFT::FFT(int power) {
  makefttable(power);
}

FFT::~FFT() {
  delete [] fsine;
  delete [] fcosine;
}

/*-----------------------------------------------------------------------*/
bool FFT::flog_mag(float *x, float *y, float *z, int n) {
  if (x && y && z && n) {
    for (int i = n - 1; i >= 0; --i) {
      float t1 = x[i];
      float t2 = y[i];
      float ssq = (t1 * t1) + (t2 * t2);
      z[i] = (ssq > 0.0f) ? 10.0f * log10(ssq) : -200.0f;
    }
    return true;
  }
  return false;
}

/*-----------------------------------------------------------------------*/
float FFT::get_band_rms(float *x, float *y, int first_bin, int last_bin) {
  int n = last_bin - first_bin + 1;
  if (n <= 0) return 0.0f;
  double sum = 0.0;
#if defined(REAPER_USE_AVX2)
  __m256 v_sum = _mm256_setzero_ps();
  int i = first_bin;
  for (; i + 7 <= last_bin; i += 8) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = _mm256_loadu_ps(y + i);
    __m256 vx2 = _mm256_mul_ps(vx, vx);
    __m256 vy2 = _mm256_mul_ps(vy, vy);
    v_sum = _mm256_add_ps(v_sum, _mm256_add_ps(vx2, vy2));
  }
  __m128 hi = _mm256_extractf128_ps(v_sum, 1);
  __m128 lo = _mm256_castps256_ps128(v_sum);
  __m128 s4 = _mm_add_ps(lo, hi);
  s4 = _mm_hadd_ps(s4, s4);
  s4 = _mm_hadd_ps(s4, s4);
  sum = _mm_cvtss_f32(s4);
  for (; i <= last_bin; ++i)
    sum += (x[i] * x[i]) + (y[i] * y[i]);
#elif defined(REAPER_USE_SSE41)
  __m128 v_sum = _mm_setzero_ps();
  int i = first_bin;
  for (; i + 3 <= last_bin; i += 4) {
    __m128 vx = _mm_loadu_ps(x + i);
    __m128 vy = _mm_loadu_ps(y + i);
    __m128 vx2 = _mm_mul_ps(vx, vx);
    __m128 vy2 = _mm_mul_ps(vy, vy);
    v_sum = _mm_add_ps(v_sum, _mm_add_ps(vx2, vy2));
  }
  v_sum = _mm_hadd_ps(v_sum, v_sum);
  v_sum = _mm_hadd_ps(v_sum, v_sum);
  sum = _mm_cvtss_f32(v_sum);
  for (; i <= last_bin; ++i)
    sum += (x[i] * x[i]) + (y[i] * y[i]);
#elif defined(REAPER_USE_NEON)
  float32x4_t v_sum = vdupq_n_f32(0.0f);
  int i = first_bin;
  for (; i + 3 <= last_bin; i += 4) {
    float32x4_t vx = vld1q_f32(x + i);
    float32x4_t vy = vld1q_f32(y + i);
    v_sum = vmlaq_f32(v_sum, vx, vx);
    v_sum = vmlaq_f32(v_sum, vy, vy);
  }
  float32x2_t v2 = vadd_f32(vget_low_f32(v_sum), vget_high_f32(v_sum));
  sum = vget_lane_f32(vpadd_f32(v2, v2), 0);
  for (; i <= last_bin; ++i)
    sum += (x[i] * x[i]) + (y[i] * y[i]);
#else
  for (int i = first_bin; i <= last_bin; ++i)
    sum += (x[i] * x[i]) + (y[i] * y[i]);
#endif
  return sqrt(sum / n);
}

/*-----------------------------------------------------------------------*/
int FFT::makefttable(int pow2) {
  int lmx, lm;
  float *c, *s;
  double scl, arg;

  fftSize = 1 << pow2;
  fft_ftablesize = lmx = fftSize/2;
  fsine = new float[lmx];
  fcosine = new float[lmx];
  scl = (M_PI * 2.0) / fftSize;
  for (s = fsine, c = fcosine, lm = 0; lm < lmx; ++lm) {
    arg = scl * lm;
    *s++ = sin(arg);
    *c++ = cos(arg);
  }
  kbase = (fft_ftablesize * 2) / fftSize;
  power2 = pow2;
  return(fft_ftablesize);
}

/*-----------------------------------------------------------------------*/
void FFT::fft(float *x, float *y) {
  float c, s,  t1, t2;
  int j1, j2, li, lix, i;
  int lmx, lo, lixnp, lm, j, nv2, k = kbase, im, jm, l = power2;

  for (lmx = fftSize, lo = 0; lo < l; lo++, k *= 2) {
    lix = lmx;
    lmx /= 2;
    lixnp = fftSize - lix;
    for (i = 0, lm = 0; lm < lmx; lm++, i += k) {
      c = fcosine[i];
      s = fsine[i];
      for (li = lixnp + lm, j1 = lm, j2 = lm + lmx; j1 <= li;
            j1 += lix, j2 += lix) {
        t1 = x[j1] - x[j2];
        t2 = y[j1] - y[j2];
        x[j1] += x[j2];
        y[j1] += y[j2];
        x[j2] = (c * t1) + (s * t2);
        y[j2] = (c * t2) - (s * t1);
      }
    }
  }

  j = 1;
  nv2 = fftSize / 2;
  for (i = 1; i < fftSize; i++) {
    if (j < i) {
      jm = j - 1;
      im = i - 1;
      t1 = x[jm];
      t2 = y[jm];
      x[jm] = x[im];
      y[jm] = y[im];
      x[im] = t1;
      y[im] = t2;
    }
    k = nv2;
    while (j > k) {
      j -= k;
      k /= 2;
    }
    j += k;
  }
}

/*-----------------------------------------------------------------------*/
void FFT::ifft(float *x, float *y) {
  float c, s,  t1, t2;
  int j1, j2, li, lix, i;
  int lmx, lo, lixnp, lm, j, nv2, k = kbase, im, jm, l = power2;

  for (lmx = fftSize, lo = 0; lo < l; lo++, k *= 2) {
    lix = lmx;
    lmx /= 2;
    lixnp = fftSize - lix;
    for (i = 0, lm = 0; lm < lmx; lm++, i += k) {
      c = fcosine[i];
      s = -fsine[i];
      for (li = lixnp + lm, j1 = lm, j2 = lm + lmx; j1 <= li;
            j1 += lix, j2 += lix) {
        t1 = x[j1] - x[j2];
        t2 = y[j1] - y[j2];
        x[j1] += x[j2];
        y[j1] += y[j2];
        x[j2] = (c * t1) + (s * t2);
        y[j2] = (c * t2) - (s * t1);
      }
    }
  }

  j = 1;
  nv2 = fftSize / 2;
  for (i = 1; i < fftSize; i++) {
    if (j < i) {
      jm = j-1;
      im = i-1;
      t1 = x[jm];
      t2 = y[jm];
      x[jm] = x[im];
      y[jm] = y[im];
      x[im] = t1;
      y[im] = t2;
    }
    k = nv2;
    while (j > k) {
      j -= k;
      k /= 2;
    }
    j += k;
  }
}
