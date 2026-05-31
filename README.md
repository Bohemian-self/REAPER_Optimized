# REAPER_Optimized: 高性能多架构优化版本

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## 项目简介

REAPER (Robust Epoch And Pitch EstimatoR) 是一个高品质的语音处理系统。`reaper` 程序使用 `EpochTracker` 类，用于同时估计浊语音的"韵律单元"或声门闭合瞬间（GCI）、浊音状态（浊音或清音）和基频（F0 或"音高"），并定义局部瞬时 F0 为连续 GCI 时间间隔的倒数。

原始代码由 David Talkin 在 Google 开发。本项目在保留原始算法精度的基础上，对 REAPER 进行了深度优化，显著提升了计算性能。

## 主要优化亮点

### 1. 构建系统优化
- 编译器优化：`-O3 -march=native -ffast-math` 标志用于激进优化
- 指令集检测：自动检测和利用 AVX2/FMA、SSE4.1、ARM NEON 指令集
- 并行框架：自动检测 OpenMP，实现 CPU 多线程并行
- GPU 支持：可选 CUDA 加速 (`-DUSE_CUDA=ON`)
- 条件编译：根据运行时环境智能选择最优代码路径

### 2. 核心算法 SIMD 加速
- **点积/平方和**：`DotProduct` 和 `SumOfSquares` 函数支持 AVX2+FMA、SSE4.1、ARM NEON 指令集加速
- **自相关计算**：`LpcAnalyzer` 中的零延迟能量计算和自相关滞点计算均使用 SIMD 向量化
- **频带 RMS**：`get_band_rms` 函数使用 SIMD 向量化加速频谱能量计算
- **特征计算**：向量化信号反转，消除不必要的内存拷贝

### 3. CPU 多线程并行 (OpenMP)
- **GetPulseCorrelations**：三阶段并行（分类 → 并行计算 → 顺序合并），使用 `schedule(dynamic, 8)`
- **GetBandpassedRmsSignal**：每帧 FFT+RMS 独立并行，线程局部 FFT 实例避免竞争
- **GetVoiceTransitionFeatures**：独立帧计算完全并行
- **GetSymmetryStats**：使用并行归约计算 RMS

### 4. GPU 加速 (CUDA)
- **cuda_nccf.h / cuda_nccf.cu**：NCCF 计算的 CUDA 实现
- **并行策略**：每个 block 处理一个峰值，线程间分配滞点计算任务
- **共享内存优化**：峰值能量在 block 内共享，避免重复计算
- **四阶段流水线**：分类 → 数据传输与计算 → 结果分发 → 内存回传
- **无缝降级**：GPU 不可用时自动回退到 CPU 实现

### 5. 运行时控制
- 新增 `-j <N>` 命令行选项，用于指定线程数量
- 线程数通过 `EpochTracker::set_num_threads()` 传递
- 运行时输出当前使用的线程数

## 性能提升

相比原始 REAPER，优化版本在典型场景下的性能提升：

- **批处理场景**（100 条 1‑2 秒语音）：**待测试**
- **长语音处理**（25 秒 48kHz 768kbit/s）：**约 12~25倍加速**
- **GPU 场景**（批处理 + CUDA）：可获得额外加速（视硬件配置而定）
- **内存开销**：保持可控，j=32时 25秒音频 GPU 内存占用约 215 MB

*注：实际提升幅度受硬件配置、音频长度及批处理规模影响。*

## 构建与安装

### 环境要求
- CMake 3.10+
- C++14 兼容编译器（GCC 7+ / Clang 6+ / MSVC 2019+）
- 可选：CUDA Toolkit 10.0+（用于 GPU 加速）
- 可选：OpenMP（编译器通常内置，或需安装 `libomp-dev`）

### 构建步骤


# 克隆仓库
git clone https://github.com/Bohemian-self/REAPER_Optimized.git
cd REAPER_Optimized

# 创建构建目录
mkdir build && cd build

# CPU 模式（自动检测 SIMD + OpenMP）
cmake .. -DCMAKE_BUILD_TYPE=Release

# GPU 模式（需要 CUDA 环境）
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON

# 编译
make -j$(nproc)

# 编译后的可执行文件位于 build/reaper

# 使用
./reaper -i input.wav -f out.f0 -p out.pm -j 8 # Use 8 threads

# REAPER_Optimized

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## English

REAPER (Robust Epoch And Pitch EstimatoR) is a high-quality speech processing system. This optimized fork adds **GPU acceleration (CUDA)**, **SIMD vectorization** (AVX2/FMA, SSE4.1, ARM NEON), and **OpenMP multi‑threading** to the original Google REAPER. All algorithmic behavior is preserved, while compute‑intensive parts run much faster.

### Key Optimizations

- **Build system**  
  `-O3 -march=native -ffast-math`, auto‑detection of OpenMP, CUDA, SSE4.1, AVX2/FMA, NEON. Use `-DUSE_CUDA=ON` to enable GPU.
- **SIMD core functions**  
  `DotProduct` / `SumOfSquares` (AVX2+FMA, SSE4.1, NEON), used in cross‑correlation – the hottest path.  
  Vectorized band‑RMS and autocorrelation.
- **OpenMP parallelism**  
  `GetPulseCorrelations` (3‑phase, `schedule(dynamic,8)`),  
  `GetBandpassedRmsSignal` (thread‑local FFTs),  
  `GetVoiceTransitionFeatures`, `GetSymmetryStats` (parallel reduction).
- **CUDA GPU acceleration** (`cuda_nccf.h` / `.cu`)  
  One block per peak, shared memory for reference energy, 4‑phase pipeline. Automatic fallback to CPU.
- **Runtime thread control**  
  New `-j <N>` flag to set the number of threads.

### Performance

- Batch processing (100 short utterances): **Wait for test **  
- Single long file (25s 48kHz 768kbit/s): **~12-25× speedup**  
- GPU + batch: additional gain depending on hardware  
- GPU memory ~215 MB for 25s audio (j=32)

### Build & Use

git clone https://github.com/Bohemian-self/REAPER_Optimized.git
cd REAPER_Optimized
mkdir build && cd build

# CPU-only (auto SIMD + OpenMP)
cmake .. -DCMAKE_BUILD_TYPE=Release

# With GPU (CUDA required)
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON

# Make it
make -j$(nproc)

# Usage
./reaper -i input.wav -f out.f0 -p out.pm -j 8 # Use 8 threads
