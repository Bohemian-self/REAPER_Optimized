cmake_minimum_required(VERSION 3.10)
project(Reaper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Enable fast math for non-critical paths (safe for audio DSP at this precision)
CHECK_CXX_COMPILER_FLAG("-ffast-math" COMPILER_SUPPORTS_FAST_MATH)
if(COMPILER_SUPPORTS_FAST_MATH)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
endif()

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  add_definitions(-DREAPER_USE_OPENMP)
  message(STATUS "OpenMP found: multi-threaded execution enabled")
else()
  message(STATUS "OpenMP not found: falling back to single-threaded execution")
endif()

# Optional CUDA
option(USE_CUDA "Enable CUDA GPU acceleration for NCCF computation" OFF)
if(USE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
      add_definitions(-DREAPER_USE_CUDA)
      set(CMAKE_CUDA_STANDARD 11)
      message(STATUS "CUDA found: GPU-accelerated NCCF enabled")
    else()
      message(WARNING "CUDA compiler found but toolkit not available, building without GPU support")
      set(USE_CUDA OFF)
    endif()
  else()
    message(WARNING "CUDA compiler not found, building without GPU support")
    set(USE_CUDA OFF)
  endif()
endif()

# SIMD detection
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
CHECK_CXX_COMPILER_FLAG("-mfma" COMPILER_SUPPORTS_FMA)
CHECK_CXX_COMPILER_FLAG("-msse4.1" COMPILER_SUPPORTS_SSE41)

if(COMPILER_SUPPORTS_AVX2)
  add_definitions(-DREAPER_USE_AVX2)
  if(COMPILER_SUPPORTS_FMA)
    add_definitions(-DREAPER_USE_FMA)
  endif()
  message(STATUS "AVX2 SIMD enabled")
elseif(COMPILER_SUPPORTS_SSE41)
  add_definitions(-DREAPER_USE_SSE41)
  message(STATUS "SSE4.1 SIMD enabled")
endif()

# ARM NEON (auto-enabled on aarch64, flag needed on armv7)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  add_definitions(-DREAPER_USE_NEON)
  message(STATUS "ARM NEON SIMD enabled (aarch64)")
else()
  CHECK_CXX_COMPILER_FLAG("-mfpu=neon" COMPILER_SUPPORTS_NEON)
  if(COMPILER_SUPPORTS_NEON)
    add_definitions(-DREAPER_USE_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    message(STATUS "ARM NEON SIMD enabled")
  endif()
endif()

include_directories(".")

add_library(core STATIC
  core/file_resource.h core/file_resource.cc
  core/float_matrix.h core/float_matrix-inl.h core/float_matrix.cc
  core/track.h core/track.cc)

add_library(wave STATIC
  wave/codec_riff.h wave/codec_riff.cc
  wave/codec_api.h wave/codec_api-inl.h
  wave/wave.h wave/wave.cc
  wave/wave_io.h wave/wave_io-inl.h wave/wave_io.cc)

set(EPOCH_TRACKER_SOURCES
  epoch_tracker/fft.h epoch_tracker/fft.cc
  epoch_tracker/fd_filter.h epoch_tracker/fd_filter.cc
  epoch_tracker/lpc_analyzer.cc epoch_tracker/lpc_analyzer.h
  epoch_tracker/epoch_tracker.cc epoch_tracker/epoch_tracker.h)

if(USE_CUDA)
  list(APPEND EPOCH_TRACKER_SOURCES
    epoch_tracker/cuda_nccf.cu epoch_tracker/cuda_nccf.h)
endif()

add_library(epoch_tracker STATIC ${EPOCH_TRACKER_SOURCES})

if(USE_CUDA)
  target_link_libraries(epoch_tracker CUDA::cudart)
  set_target_properties(epoch_tracker PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(core OpenMP::OpenMP_CXX)
  target_link_libraries(epoch_tracker OpenMP::OpenMP_CXX)
endif()

add_executable(reaper epoch_tracker_main.cc)
target_link_libraries(wave core)
target_link_libraries(epoch_tracker wave)
target_link_libraries(reaper epoch_tracker)

if(OpenMP_CXX_FOUND)
  target_link_libraries(reaper OpenMP::OpenMP_CXX)
endif()
