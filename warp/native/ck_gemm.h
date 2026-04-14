// SPDX-FileCopyrightText: Copyright (c) 2025 Filippo Luca Ferretti. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Block-level GEMM using AMD MFMA (Matrix Fused Multiply-Add) intrinsics.
// Inspired by AMD Composable Kernel (CK) blockwise GEMM approach.
//
// MFMA thread mapping for 16x16 tile (mfma_f32_16x16x4f32):
//   64 threads (one wavefront), grouped as 4 blocks of 16 lanes.
//   Thread T: lane = T % 16, blk = T / 16
//   Input A:  A[lane][blk] (one element per thread per K step)
//   Input B:  B[blk][lane] (one element per thread per K step)
//   Output C: C[lane][blk*4 + 0..3] (four elements per thread)

#pragma once

#if WP_ENABLE_HIP && WP_ENABLE_CK && defined(__HIPCC__)

#include <type_traits>

#ifndef CK_WAVE_SIZE
#define CK_WAVE_SIZE 64
#endif

#define CK_MFMA_TILE 16
#define CK_MFMA_K_F32 4
#define CK_MFMA_K_F64 4

typedef float mfma_float4 __attribute__((ext_vector_type(4)));
typedef double mfma_double4 __attribute__((ext_vector_type(4)));

namespace ck_gemm {

template <int M, int N, int K,
          int StrA0, int StrA1,
          int StrB0, int StrB1,
          int StrC0, int StrC1>
inline __device__ void mfma_tile_f32(
    int lane, int blk, int base_m, int base_n,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* alpha, const float* beta,
    float* __restrict__ C)
{
    mfma_float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < K; k += CK_MFMA_K_F32)
    {
        int a_row = base_m + lane;
        int a_col = k + blk;
        float a_val = (a_row < M && a_col < K) ? A[a_row * StrA0 + a_col * StrA1] : 0.0f;

        int b_row = k + blk;
        int b_col = base_n + lane;
        float b_val = (b_row < K && b_col < N) ? B[b_row * StrB0 + b_col * StrB1] : 0.0f;

        acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_val, acc, 0, 0, 0);
    }

    int out_row = base_m + lane;
    if (out_row < M)
    {
        float a = *alpha;
        float b = *beta;
        #pragma unroll
        for (int c = 0; c < 4; c++)
        {
            int out_col = base_n + blk * 4 + c;
            if (out_col < N)
            {
                int idx = out_row * StrC0 + out_col * StrC1;
                float val = a * acc[c];
                if (b != 0.0f)
                    val += b * C[idx];
                C[idx] = val;
            }
        }
    }
}

template <int M, int N, int K,
          int StrA0, int StrA1,
          int StrB0, int StrB1,
          int StrC0, int StrC1>
inline __device__ void mfma_tile_f64(
    int lane, int blk, int base_m, int base_n,
    const double* __restrict__ A,
    const double* __restrict__ B,
    const double* alpha, const double* beta,
    double* __restrict__ C)
{
    mfma_double4 acc = {0.0, 0.0, 0.0, 0.0};

    for (int k = 0; k < K; k += CK_MFMA_K_F64)
    {
        int a_row = base_m + lane;
        int a_col = k + blk;
        double a_val = (a_row < M && a_col < K) ? A[a_row * StrA0 + a_col * StrA1] : 0.0;

        int b_row = k + blk;
        int b_col = base_n + lane;
        double b_val = (b_row < K && b_col < N) ? B[b_row * StrB0 + b_col * StrB1] : 0.0;

        acc = __builtin_amdgcn_mfma_f64_16x16x4f64(a_val, b_val, acc, 0, 0, 0);
    }

    int out_row = base_m + lane;
    if (out_row < M)
    {
        double a = *alpha;
        double b = *beta;
        #pragma unroll
        for (int c = 0; c < 4; c++)
        {
            int out_col = base_n + blk * 4 + c;
            if (out_col < N)
            {
                int idx = out_row * StrC0 + out_col * StrC1;
                double val = a * acc[c];
                if (b != 0.0)
                    val += b * C[idx];
                C[idx] = val;
            }
        }
    }
}

template <int M, int N, int K,
          int StrA0, int StrA1,
          int StrB0, int StrB1,
          int StrC0, int StrC1>
inline __device__ void mfma_tile_f16(
    int lane, int blk, int base_m, int base_n,
    const wp::float16* __restrict__ A,
    const wp::float16* __restrict__ B,
    const wp::float16* alpha, const wp::float16* beta,
    wp::float16* __restrict__ C)
{
    mfma_float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < K; k += CK_MFMA_K_F32)
    {
        int a_row = base_m + lane;
        int a_col = k + blk;
        float a_val = (a_row < M && a_col < K)
            ? float(A[a_row * StrA0 + a_col * StrA1]) : 0.0f;

        int b_row = k + blk;
        int b_col = base_n + lane;
        float b_val = (b_row < K && b_col < N)
            ? float(B[b_row * StrB0 + b_col * StrB1]) : 0.0f;

        acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_val, acc, 0, 0, 0);
    }

    int out_row = base_m + lane;
    if (out_row < M)
    {
        float a = float(*alpha);
        float b = float(*beta);
        #pragma unroll
        for (int c = 0; c < 4; c++)
        {
            int out_col = base_n + blk * 4 + c;
            if (out_col < N)
            {
                int idx = out_row * StrC0 + out_col * StrC1;
                float val = a * acc[c];
                if (b != 0.0f)
                    val += b * float(C[idx]);
                C[idx] = wp::float16(val);
            }
        }
    }
}

// Block-level GEMM: distributes 16x16 output tiles across wavefronts.
// All threads in the block must call this function.
// C[M x N] = alpha * A[M x K] * B[K x N] + beta * C[M x N]
template <typename T, int M, int N, int K,
          int StrA0, int StrA1,
          int StrB0, int StrB1,
          int StrC0, int StrC1>
__device__ void block_gemm(T* alpha, T* A, T* B, T* beta, T* C)
{
    const int tid = threadIdx.x;
    const int wave_id = tid / CK_WAVE_SIZE;
    const int lane_in_wave = tid % CK_WAVE_SIZE;
    const int num_waves = blockDim.x / CK_WAVE_SIZE;

    const int lane = lane_in_wave % CK_MFMA_TILE;
    const int blk = lane_in_wave / CK_MFMA_TILE;

    constexpr int tiles_m = (M + CK_MFMA_TILE - 1) / CK_MFMA_TILE;
    constexpr int tiles_n = (N + CK_MFMA_TILE - 1) / CK_MFMA_TILE;
    constexpr int total_tiles = tiles_m * tiles_n;

    for (int tile_idx = wave_id; tile_idx < total_tiles; tile_idx += num_waves)
    {
        const int tile_i = tile_idx / tiles_n;
        const int tile_j = tile_idx % tiles_n;
        const int base_m = tile_i * CK_MFMA_TILE;
        const int base_n = tile_j * CK_MFMA_TILE;

        if constexpr (std::is_same<T, float>::value)
            mfma_tile_f32<M, N, K, StrA0, StrA1, StrB0, StrB1, StrC0, StrC1>(
                lane, blk, base_m, base_n,
                reinterpret_cast<const float*>(A),
                reinterpret_cast<const float*>(B),
                reinterpret_cast<const float*>(alpha),
                reinterpret_cast<const float*>(beta),
                reinterpret_cast<float*>(C));
        else if constexpr (std::is_same<T, double>::value)
            mfma_tile_f64<M, N, K, StrA0, StrA1, StrB0, StrB1, StrC0, StrC1>(
                lane, blk, base_m, base_n,
                reinterpret_cast<const double*>(A),
                reinterpret_cast<const double*>(B),
                reinterpret_cast<const double*>(alpha),
                reinterpret_cast<const double*>(beta),
                reinterpret_cast<double*>(C));
        else if constexpr (std::is_same<T, wp::float16>::value)
            mfma_tile_f16<M, N, K, StrA0, StrA1, StrB0, StrB1, StrC0, StrC1>(
                lane, blk, base_m, base_n,
                reinterpret_cast<const wp::float16*>(A),
                reinterpret_cast<const wp::float16*>(B),
                reinterpret_cast<const wp::float16*>(alpha),
                reinterpret_cast<const wp::float16*>(beta),
                reinterpret_cast<wp::float16*>(C));
    }
}

} // namespace ck_gemm

#endif // WP_ENABLE_HIP && WP_ENABLE_CK && __HIPCC__
