/*
dct_4x4_avx2.cpp

This file is part of DCTFilter

Copyright (c) 2016, OKA Motofumi <chikuzen.mo at gmail dot com>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
*/


#ifndef __AVX2__
#error /arch:avx2 is not set.
#endif

#include <cstdint>
#include <immintrin.h>

#include "dct.h"


constexpr float r2 = 1.3065630198f; // cos(2 / 16.0 * PI) * SQRT2
constexpr float r6 = 0.5411961079f; // cos(6 / 16.0 * PI) * SQRT2


template <typename T>
static __forceinline __m256 load_and_cvt_to_float_x8_avx2(
    const T* srcp, const __m256& factor) noexcept
{
    if (sizeof(T) == 4) {
        return _mm256_load_ps(reinterpret_cast<const float*>(srcp));
    }

    __m256 ret;
    if (sizeof(T) == 2) {
        __m128i s = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));
        ret = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s));
    } else {
        __m128i s = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp));
        ret = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(s));
    }
    return _mm256_mul_ps(ret, factor);
}


template <typename T>
static __forceinline __m256 load_and_cvt_to_float_x4x2_avx2(
    const T* srcp, const __m256i& vindex, const __m256& factor) noexcept
{
    __m256i s = _mm256_i32gather_epi32(
        reinterpret_cast<const int*>(srcp), vindex, sizeof(T));

    if (sizeof(T) == 4) {
        return _mm256_castsi256_ps(s);
    }

    constexpr int shift = (4 - sizeof(T)) * 8;
    s = _mm256_srli_epi32(_mm256_slli_epi32(s, shift), shift);
    return _mm256_mul_ps(_mm256_cvtepi32_ps(s), factor);
}


static __forceinline void transpose_4x4x2_avx(
    __m256& a4e4, __m256& b4f4, __m256& c4g4, __m256& d4h4) noexcept
{
    // a4e4 : a0 a1 a2 a3 | e0 e1 e2 e3
    // b4f4 : b0 b1 b2 b3 | f0 f1 f2 f3
    // c4g4 : c0 c1 c2 c3 | g0 g1 g2 g3
    // d4h4 : d0 d1 d2 d3 | h0 h1 h2 h3

    __m256 ac01eg01 = _mm256_unpacklo_ps(a4e4, c4g4); // a0 c0 a1 c1 | e0 g0 e1 g1
    __m256 ac23eg23 = _mm256_unpackhi_ps(a4e4, c4g4); // a2 c2 a3 c3 | e2 g2 e3 g3
    __m256 bd01fh01 = _mm256_unpacklo_ps(b4f4, d4h4); // b0 d0 b1 d1 | f0 h0 f1 h1
    __m256 bd23fh23 = _mm256_unpackhi_ps(b4f4, d4h4); // b2 d2 b3 d3 | f2 h2 f3 h3

    a4e4 = _mm256_unpacklo_ps(ac01eg01, bd01fh01); // a0 b0 c0 d0 | e0 f0 g0 h0
    b4f4 = _mm256_unpackhi_ps(ac01eg01, bd01fh01); // a1 b1 c1 d1 | e1 f1 g1 h1
    c4g4 = _mm256_unpacklo_ps(ac23eg23, bd23fh23); // a2 b2 c2 d2 | e2 f2 g2 h2
    d4h4 = _mm256_unpackhi_ps(ac23eg23, bd23fh23); // a3 b3 c3 d3 | e3 f3 g3 h3
}


static __forceinline void fdct_4x4x2_llm_fma3(
    __m256& s0, __m256& s1, __m256& s2, __m256& s3, const __m256& xr2,
    const __m256& xr6) noexcept
{
    __m256 p03 = _mm256_add_ps(s0, s3);
    __m256 p12 = _mm256_add_ps(s1, s2);
    __m256 m03 = _mm256_sub_ps(s0, s3);
    __m256 m12 = _mm256_sub_ps(s1, s2);

    s0 = _mm256_add_ps(p03, p12);
    s1 = _mm256_fmadd_ps(m03, xr2, _mm256_mul_ps(m12, xr6));
    s2 = _mm256_sub_ps(p03, p12);
    s3 = _mm256_fmsub_ps(m03, xr6, _mm256_mul_ps(m12, xr2));
}


static __forceinline void idct_4x4x2_llm_fma3(
    __m256& s0, __m256& s1, __m256& s2, __m256& s3, const __m256& xr2,
    const __m256& xr6) noexcept
{
    __m256 p02 = _mm256_add_ps(s0, s2);
    __m256 m02 = _mm256_sub_ps(s0, s2);
    __m256 p13 = _mm256_fmadd_ps(s1, xr2, _mm256_mul_ps(s3, xr6));
    __m256 m13 = _mm256_fmsub_ps(s1, xr6, _mm256_mul_ps(s3, xr2));

    s0 = _mm256_add_ps(p02, p13);
    s1 = _mm256_add_ps(m02, m13);
    s2 = _mm256_sub_ps(m02, m13);
    s3 = _mm256_sub_ps(p02, p13);
}


static __forceinline void proc_dct_avx2(
    __m256& s0, __m256& s1, __m256& s2, __m256& s3, const __m256& xr2,
    const __m256& xr6, const float* f) noexcept
{
    transpose_4x4x2_avx(s0, s1, s2, s3);
    fdct_4x4x2_llm_fma3(s0, s1, s2, s3, xr2, xr6);
    transpose_4x4x2_avx(s0, s1, s2, s3);
    fdct_4x4x2_llm_fma3(s0, s1, s2, s3, xr2, xr6);

    s0 = _mm256_mul_ps(s0, _mm256_load_ps(f +  0));
    s1 = _mm256_mul_ps(s1, _mm256_load_ps(f +  8));
    s2 = _mm256_mul_ps(s2, _mm256_load_ps(f + 16));
    s3 = _mm256_mul_ps(s3, _mm256_load_ps(f + 24));

    transpose_4x4x2_avx(s0, s1, s2, s3);
    idct_4x4x2_llm_fma3(s0, s1, s2, s3, xr2, xr6);
    transpose_4x4x2_avx(s0, s1, s2, s3);
    idct_4x4x2_llm_fma3(s0, s1, s2, s3, xr2, xr6);
}


template <typename T>
static __forceinline void store_x8_to_dst_avx2(
    const __m256& src, T* dstp, const __m256& factor) noexcept
{
    __m256 s0 = _mm256_mul_ps(src, factor);

    if (sizeof(T) == 4) {
        _mm256_store_ps(reinterpret_cast<float*>(dstp), s0);
        return;
    }

    __m256i d0 = _mm256_cvtps_epi32(s0);
    d0 = _mm256_packus_epi32(d0, d0);
    d0 = _mm256_permute4x64_epi64(d0, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i d1 = _mm256_extracti128_si256(d0, 0);
    if (sizeof(T) == 2) {
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp), d1);
    } else {
        d1 = _mm_packus_epi16(d1, d1);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp), d1);
    }
}


template <typename T>
static void store_x4x2_to_dst_avx2(
    const __m256& src, T* dstp0, T* dstp1, const __m256& factor)
{
    __m256 s0 = _mm256_mul_ps(src, factor);

    if (sizeof(T) == 4) {
        _mm_store_ps(reinterpret_cast<float*>(dstp0), _mm256_extractf128_ps(s0, 0));
        _mm_store_ps(reinterpret_cast<float*>(dstp1), _mm256_extractf128_ps(s0, 1));
        return;
    }

    __m256i s1 = _mm256_cvtps_epi32(s0);
    __m128i d0 = _mm256_extracti128_si256(s1, 0);
    __m128i d1 = _mm256_extracti128_si256(s1, 1);
    d0 = _mm_packus_epi32(d0, d0);
    d1 = _mm_packus_epi32(d1, d1);
    if (sizeof(T) == 2) {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp0), d0);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp1), d1);
    } else {
        *reinterpret_cast<int*>(dstp0) = _mm_cvtsi128_si32(_mm_packus_epi16(d0, d0));
        *reinterpret_cast<int*>(dstp1) = _mm_cvtsi128_si32(_mm_packus_epi16(d1, d1));
    }
}


template <typename T>
static void fdct_idct_4x4x2_avx2(
    const T* srcp, T* dstp, const float* f, int spitch, int dpitch,
    const float* load, const float* store) noexcept
{
    const __m256 factor_load = _mm256_load_ps(load);

    __m256 s0 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 0, factor_load);
    __m256 s1 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 1, factor_load);
    __m256 s2 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 2, factor_load);
    __m256 s3 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 3, factor_load);

    const __m256 xr2 = _mm256_set1_ps(r2);
    const __m256 xr6 = _mm256_set1_ps(r6);
    proc_dct_avx2(s0, s1, s2, s3, xr2, xr6, f);

    const __m256 factor_store = _mm256_load_ps(store);

    store_x8_to_dst_avx2(s0, dstp + 0 * dpitch, factor_store);
    store_x8_to_dst_avx2(s1, dstp + 1 * dpitch, factor_store);
    store_x8_to_dst_avx2(s2, dstp + 2 * dpitch, factor_store);
    store_x8_to_dst_avx2(s3, dstp + 3 * dpitch, factor_store);
}


template <typename T>
static void fdct_idct_4x4x2v_avx2(
    const T* srcp, T* dstp, const float* f, int spitch, int dpitch,
    const float* load, const float* store) noexcept
{
    int sp4 = spitch * 4;
    __m256i vindex = _mm256_setr_epi32(0, 1, 2, 3, sp4, sp4 + 1, sp4 + 2, sp4 + 3);

    const __m256 factor_load = _mm256_load_ps(load);

    __m256 s0 = load_and_cvt_to_float_x4x2_avx2(srcp + spitch * 0, vindex, factor_load);
    __m256 s1 = load_and_cvt_to_float_x4x2_avx2(srcp + spitch * 1, vindex, factor_load);
    __m256 s2 = load_and_cvt_to_float_x4x2_avx2(srcp + spitch * 2, vindex, factor_load);
    __m256 s3 = load_and_cvt_to_float_x4x2_avx2(srcp + spitch * 3, vindex, factor_load);

    const __m256 xr2 = _mm256_set1_ps(r2);
    const __m256 xr6 = _mm256_set1_ps(r6);
    proc_dct_avx2(s0, s1, s2, s3, xr2, xr6, f);

    const __m256 factor_store = _mm256_load_ps(store);

    store_x4x2_to_dst_avx2(s0, dstp + 0 * dpitch, dstp + 4 * dpitch, factor_store);
    store_x4x2_to_dst_avx2(s1, dstp + 1 * dpitch, dstp + 5 * dpitch, factor_store);
    store_x4x2_to_dst_avx2(s2, dstp + 2 * dpitch, dstp + 6 * dpitch, factor_store);
    store_x4x2_to_dst_avx2(s3, dstp + 3 * dpitch, dstp + 7 * dpitch, factor_store);
}


template <typename T>
static inline void fdct_idct_avx2(
    const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
    int width, int height, float*, const float* factors,
    const float* load, const float* store, int) noexcept
{
    const T* s = reinterpret_cast<const T*>(srcp);
    T* d = reinterpret_cast<T*>(dstp);
    src_pitch /= sizeof(T);
    dst_pitch /= sizeof(T);

    if (width == 4 && height != 4) {
        for (int y = 0; y < height; y += 8) {
            fdct_idct_4x4x2v_avx2(s, d, factors, src_pitch, dst_pitch, load, store);
            s += src_pitch * 8;
            d += dst_pitch * 8;
        }
        return;
    }

    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += 8) {
            fdct_idct_4x4x2_avx2(s + x, d + x, factors, src_pitch, dst_pitch, load, store);
        }
        s += src_pitch * 4;
        d += dst_pitch * 4;
    }
}


void fdct_idct_4x4_avx2_8(
        const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
        int width, int height, float*, const float* factors,
        const float* load, const float* store, int) noexcept
{
    fdct_idct_avx2<uint8_t>(srcp, dstp, src_pitch, dst_pitch, width, height,
                            nullptr, factors, load, store, 0);
}


void fdct_idct_4x4_avx2_16(
        const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
        int width, int height, float*, const float* factors,
        const float* load, const float* store, int) noexcept
{
    fdct_idct_avx2<uint16_t>(srcp, dstp, src_pitch, dst_pitch, width, height,
                             nullptr, factors, load, store, 0);
}


void fdct_idct_4x4_avx2_32(
        const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
        int width, int height, float*, const float* factors,
        const float* load, const float* store, int) noexcept
{
    fdct_idct_avx2<float>(srcp, dstp, src_pitch, dst_pitch, width, height,
                          nullptr, factors, load, store, 0);
}

