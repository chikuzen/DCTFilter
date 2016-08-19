/*
dct_8x8_avx2.cpp

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


constexpr float r1 = 1.3870398998f; // cos(1 / 16.0 * PI) * SQRT2
constexpr float r2 = 1.3065630198f; // cos(2 / 16.0 * PI) * SQRT2
constexpr float r3 = 1.1758755445f; // cos(3 / 16.0 * PI) * SQRT2
constexpr float r5 = 0.7856949568f; // cos(5 / 16.0 * PI) * SQRT2
constexpr float r6 = 0.5411961079f; // cos(6 / 16.0 * PI) * SQRT2
constexpr float r7 = 0.2758993804f; // cos(7 / 16.0 * PI) * SQRT2
constexpr float isqrt2 = 0.7071067812f; // 1.0f / SQRT2


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


static __forceinline void transpose_8x8_avx(
    __m256& a, __m256& b, __m256& c, __m256& d, __m256& e, __m256& f,
    __m256& g, __m256& h) noexcept
{
    __m256 ac0145 = _mm256_unpacklo_ps(a, c); // a0 c0 a1 c1 a4 c4 a5 c5
    __m256 ac2367 = _mm256_unpackhi_ps(a, c); // a2 c2 a3 c3 a6 c6 a7 c7
    __m256 bd0145 = _mm256_unpacklo_ps(b, d); // b0 d0 b1 d1 b4 d4 b5 d5
    __m256 bd2367 = _mm256_unpackhi_ps(b, d); // b2 d2 b3 d3 b6 d6 b7 d7
    __m256 eg0145 = _mm256_unpacklo_ps(e, g); // e0 g0 e1 g1 e4 g4 e5 g5
    __m256 eg2367 = _mm256_unpackhi_ps(e, g); // e2 g2 e3 g3 e6 g6 e7 g7
    __m256 fh0145 = _mm256_unpacklo_ps(f, h); // f0 h0 f1 h1 f4 h4 f5 h5
    __m256 fh2367 = _mm256_unpackhi_ps(f, h); // f2 h2 f3 h3 f6 h6 f7 h7

    __m256 abcd04 = _mm256_unpacklo_ps(ac0145, bd0145); // a0 b0 c0 d0 a4 b4 c4 d4
    __m256 abcd15 = _mm256_unpackhi_ps(ac0145, bd0145); // a1 b1 c1 d1 a5 b5 c5 d5
    __m256 abcd26 = _mm256_unpacklo_ps(ac2367, bd2367); // a2 b2 c2 d2 a6 b6 c6 d6
    __m256 abcd37 = _mm256_unpackhi_ps(ac2367, bd2367); // a3 b3 c3 d3 a7 b7 c7 d7
    __m256 efgh04 = _mm256_unpacklo_ps(eg0145, fh0145); // e0 f0 g0 h0 e4 f4 g4 h4
    __m256 efgh15 = _mm256_unpackhi_ps(eg0145, fh0145); // e1 f1 g1 h1 e5 f5 g5 h5
    __m256 efgh26 = _mm256_unpacklo_ps(eg2367, fh2367); // e2 f2 g2 h2 e6 f6 g6 h6
    __m256 efgh37 = _mm256_unpackhi_ps(eg2367, fh2367); // e3 f3 g3 h3 e7 f7 g7 h7

    a = _mm256_permute2f128_ps(abcd04, efgh04, (2 << 4) | 0); //a0 b0 c0 d0 e0 f0 g0 h0
    e = _mm256_permute2f128_ps(abcd04, efgh04, (3 << 4) | 1); //a4 b4 c4 d4 e4 f4 g4 h4
    b = _mm256_permute2f128_ps(abcd15, efgh15, (2 << 4) | 0); //a1 b1 c1 d1 e1 f1 g1 h1
    f = _mm256_permute2f128_ps(abcd15, efgh15, (3 << 4) | 1); //a5 b5 c5 d5 e5 f5 g5 h5
    c = _mm256_permute2f128_ps(abcd26, efgh26, (2 << 4) | 0); //a2 b2 c2 d2 e2 f2 g2 h2
    g = _mm256_permute2f128_ps(abcd26, efgh26, (3 << 4) | 1); //a6 b6 c6 d6 e6 f6 g6 h6
    d = _mm256_permute2f128_ps(abcd37, efgh37, (2 << 4) | 0); //a3 b3 c3 d3 e3 f3 g3 h3
    h = _mm256_permute2f128_ps(abcd37, efgh37, (3 << 4) | 1); //a7 b7 c7 d7 e7 f7 g7 h7
}


static __forceinline void fdct_8x8_llm_fma3(
    __m256& s0, __m256& s1, __m256& s2, __m256& s3, __m256& s4, __m256& s5,
    __m256& s6, __m256& s7) noexcept
{
    const __m256 xr1 = _mm256_set1_ps(r1);
    const __m256 xr2 = _mm256_set1_ps(r2);
    const __m256 xr3 = _mm256_set1_ps(r3);
    const __m256 xr5 = _mm256_set1_ps(r5);
    const __m256 xr6 = _mm256_set1_ps(r6);
    const __m256 xr7 = _mm256_set1_ps(r7);
    const __m256 xisqrt2 = _mm256_set1_ps(isqrt2);

    __m256 t0 = _mm256_add_ps(s0, s7);
    __m256 t7 = _mm256_sub_ps(s0, s7);
    __m256 t1 = _mm256_add_ps(s1, s6);
    __m256 t6 = _mm256_sub_ps(s1, s6);
    __m256 t2 = _mm256_add_ps(s2, s5);
    __m256 t5 = _mm256_sub_ps(s2, s5);
    __m256 t3 = _mm256_add_ps(s3, s4);
    __m256 t4 = _mm256_sub_ps(s3, s4);

    __m256 c0 = _mm256_add_ps(t0, t3);
    __m256 c3 = _mm256_sub_ps(t0, t3);
    __m256 c1 = _mm256_add_ps(t1, t2);
    __m256 c2 = _mm256_sub_ps(t1, t2);

    s0 = _mm256_add_ps(c0, c1);
    s4 = _mm256_sub_ps(c0, c1);
    s2 = _mm256_fmadd_ps(c2, xr6, _mm256_mul_ps(c3, xr2));
    s6 = _mm256_fmsub_ps(c3, xr6, _mm256_mul_ps(c2, xr2));

    c3 = _mm256_fmadd_ps(t4, xr3, _mm256_mul_ps(t7, xr5));
    c0 = _mm256_fmsub_ps(t7, xr3, _mm256_mul_ps(t4, xr5));
    c2 = _mm256_fmadd_ps(t5, xr1, _mm256_mul_ps(t6, xr7));
    c1 = _mm256_fmsub_ps(t6, xr1, _mm256_mul_ps(t5, xr7));

    s3 = _mm256_sub_ps(c0, c2);
    s5 = _mm256_sub_ps(c3, c1);

    c0 = _mm256_mul_ps(_mm256_add_ps(c0, c2), xisqrt2);
    c3 = _mm256_mul_ps(_mm256_add_ps(c1, c3), xisqrt2);

    s1 = _mm256_add_ps(c0, c3);
    s7 = _mm256_sub_ps(c0, c3);

}


static __forceinline void idct_8x8_llm_fma3(
    __m256& s0, __m256& s1, __m256& s2, __m256& s3, __m256& s4, __m256& s5,
    __m256& s6, __m256& s7) noexcept
{
    __m256 z0 = _mm256_add_ps(s1, s7);
    __m256 z1 = _mm256_add_ps(s3, s5);
    __m256 z4 = _mm256_mul_ps(_mm256_add_ps(z0, z1), _mm256_set1_ps(r3));
    __m256 z2 = _mm256_fmadd_ps(_mm256_set1_ps(-r3 - r5), _mm256_add_ps(s3, s7), z4);
    __m256 z3 = _mm256_fmadd_ps(_mm256_set1_ps(-r3 + r5), _mm256_add_ps(s1, s5), z4);
    z0 = _mm256_mul_ps(z0, _mm256_set1_ps(-r3 + r7));
    z1 = _mm256_mul_ps(z1, _mm256_set1_ps(-r3 - r1));

    __m256 b3 = _mm256_fmadd_ps(s7, _mm256_set1_ps(-r1 + r3 + r5 - r7), _mm256_add_ps(z0, z2));
    __m256 b2 = _mm256_fmadd_ps(s5, _mm256_set1_ps(r1 + r3 - r5 + r7), _mm256_add_ps(z1, z3));
    __m256 b1 = _mm256_fmadd_ps(s3, _mm256_set1_ps(r1 + r3 + r5 - r7), _mm256_add_ps(z1, z2));
    __m256 b0 = _mm256_fmadd_ps(s1, _mm256_set1_ps(r1 + r3 - r5 - r7), _mm256_add_ps(z0, z3));

    z0 = _mm256_add_ps(s0, s4);
    z1 = _mm256_sub_ps(s0, s4);
    z4 = _mm256_mul_ps(_mm256_add_ps(s2, s6), _mm256_set1_ps(r6));

    z2 = _mm256_sub_ps(z4, _mm256_mul_ps(s6, _mm256_set1_ps(r2 + r6)));
    z3 = _mm256_fmadd_ps(s2, _mm256_set1_ps(r2 - r6), z4);

    __m256 a0 = _mm256_add_ps(z0, z3);
    __m256 a3 = _mm256_sub_ps(z0, z3);
    __m256 a1 = _mm256_add_ps(z1, z2);
    __m256 a2 = _mm256_sub_ps(z1, z2);

    s0 = _mm256_add_ps(a0, b0);
    s7 = _mm256_sub_ps(a0, b0);
    s1 = _mm256_add_ps(a1, b1);
    s6 = _mm256_sub_ps(a1, b1);
    s2 = _mm256_add_ps(a2, b2);
    s5 = _mm256_sub_ps(a2, b2);
    s3 = _mm256_add_ps(a3, b3);
    s4 = _mm256_sub_ps(a3, b3);
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
static void fdct_idct_8x8_avx2(
    const T* srcp, T* dstp, const float* f, int spitch, int dpitch,
    const float* load, const float* store) noexcept
{
    const __m256 factor_load = _mm256_load_ps(load);

    __m256 s0 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 0, factor_load);
    __m256 s1 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 1, factor_load);
    __m256 s2 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 2, factor_load);
    __m256 s3 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 3, factor_load);
    __m256 s4 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 4, factor_load);
    __m256 s5 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 5, factor_load);
    __m256 s6 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 6, factor_load);
    __m256 s7 = load_and_cvt_to_float_x8_avx2(srcp + spitch * 7, factor_load);

    transpose_8x8_avx(s0, s1, s2, s3, s4, s5, s6, s7);
    fdct_8x8_llm_fma3(s0, s1, s2, s3, s4, s5, s6, s7);
    transpose_8x8_avx(s0, s1, s2, s3, s4, s5, s6, s7);
    fdct_8x8_llm_fma3(s0, s1, s2, s3, s4, s5, s6, s7);

    s0 = _mm256_mul_ps(s0, _mm256_load_ps(f +  0));
    s1 = _mm256_mul_ps(s1, _mm256_load_ps(f +  8));
    s2 = _mm256_mul_ps(s2, _mm256_load_ps(f + 16));
    s3 = _mm256_mul_ps(s3, _mm256_load_ps(f + 24));
    s4 = _mm256_mul_ps(s4, _mm256_load_ps(f + 32));
    s5 = _mm256_mul_ps(s5, _mm256_load_ps(f + 40));
    s6 = _mm256_mul_ps(s6, _mm256_load_ps(f + 48));
    s7 = _mm256_mul_ps(s7, _mm256_load_ps(f + 56));

    transpose_8x8_avx(s0, s1, s2, s3, s4, s5, s6, s7);
    idct_8x8_llm_fma3(s0, s1, s2, s3, s4, s5, s6, s7);
    transpose_8x8_avx(s0, s1, s2, s3, s4, s5, s6, s7);
    idct_8x8_llm_fma3(s0, s1, s2, s3, s4, s5, s6, s7);

    const __m256 factor_store = _mm256_load_ps(store);

    store_x8_to_dst_avx2(s0, dstp + 0 * dpitch, factor_store);
    store_x8_to_dst_avx2(s1, dstp + 1 * dpitch, factor_store);
    store_x8_to_dst_avx2(s2, dstp + 2 * dpitch, factor_store);
    store_x8_to_dst_avx2(s3, dstp + 3 * dpitch, factor_store);
    store_x8_to_dst_avx2(s4, dstp + 4 * dpitch, factor_store);
    store_x8_to_dst_avx2(s5, dstp + 5 * dpitch, factor_store);
    store_x8_to_dst_avx2(s6, dstp + 6 * dpitch, factor_store);
    store_x8_to_dst_avx2(s7, dstp + 7 * dpitch, factor_store);
}


template <typename T>
static inline void fdct_idct_avx2(
        const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
        int width, int height, float*, const float* factors, const float* load,
        const float* store, int) noexcept
{
    const T* s = reinterpret_cast<const T*>(srcp);
    T* d = reinterpret_cast<T*>(dstp);
    src_pitch /= sizeof(T);
    dst_pitch /= sizeof(T);

    for (int y = 0; y < height; y += 8) {
        for (int x = 0; x < width; x += 8) {
            fdct_idct_8x8_avx2<T>(s + x, d + x, factors, src_pitch, dst_pitch, load, store);
        }
        s += src_pitch * 8;
        d += dst_pitch * 8;
    }
}


void fdct_idct_8x8_avx2_8(
        const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
        int width, int height, float*, const float* factors, const float* load,
        const float* store, int) noexcept
{
    fdct_idct_avx2<uint8_t>(srcp, dstp, src_pitch, dst_pitch, width, height,
                            nullptr, factors, load, store, 0);
}


void fdct_idct_8x8_avx2_16(
        const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
        int width, int height, float*, const float* factors, const float* load,
        const float* store, int) noexcept
{
    fdct_idct_avx2<uint16_t>(srcp, dstp, src_pitch, dst_pitch, width, height,
                             nullptr, factors, load, store, 0);
}


void fdct_idct_8x8_avx2_32(
        const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
        int width, int height, float*, const float* factors, const float* load,
        const float* store, int) noexcept
{
    fdct_idct_avx2<float>(srcp, dstp, src_pitch, dst_pitch, width, height,
                          nullptr, factors, load, store, 0);
}
