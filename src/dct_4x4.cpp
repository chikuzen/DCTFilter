/*
dct_4x4.cpp

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


#include <cstdint>
#include <smmintrin.h>
#include "dct.h"


constexpr float r2 = 1.3065630198f; // cos(2 / 16.0 * PI) * SQRT2
constexpr float r6 = 0.5411961079f; // cos(6 / 16.0 * PI) * SQRT2


static void fdct_4x4_llm_cpp(const float* s, float* d) noexcept
{
    for(int i = 0; i < 4; ++i) {
        float p03 = s[0] + s[3];
        float m03 = s[0] - s[3];
        float p12 = s[1] + s[2];
        float m12 = s[1] - s[2];

        d[0] = p03 + p12;
        d[1] = r2 * m03 + r6 * m12;
        d[2] = p03 - p12;
        d[3] = r6 * m03 - r2 * m12;

        s += 4;
        d += 4;
    }
}


static void idct_4x4_llm_cpp(const float* s, float* d) noexcept
{
    for (int i = 0; i < 4; ++i) {
        float t10 = s[0] + s[2];
        float t12 = s[0] - s[2];
        float t0 = r2 * s[1] + r6 * s[3];
        float t2 = r6 * s[1] - r2 * s[3];

        d[0] = t10 + t0;
        d[1] = t12 + t2;
        d[2] = t12 - t2;
        d[3] = t10 - t0;

        s += 4;
        d += 4;
    }
}


template <typename T>
static void fdct_idct_cpp(
        const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
        int width, int height, float* buff0, const float* factors,
        const float* load, const float* store, int) noexcept
{
    const T* s = reinterpret_cast<const T*>(srcp);
    T* d = reinterpret_cast<T*>(dstp);
    float* buff1 = buff0 + 16;
    src_pitch /= sizeof(T);
    dst_pitch /= sizeof(T);

    const float f_load = load[0];
    const float f_store = store[0];

    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += 4) {

            src_to_float_XxX_cpp<T, 4>(s + x, buff0, src_pitch, load[0]);

            transpose_XxX_cpp<4>(buff0, buff1);
            fdct_4x4_llm_cpp(buff1, buff0);
            transpose_XxX_cpp<4>(buff0, buff1);
            fdct_4x4_llm_cpp(buff1, buff0);

            for (int i = 0; i < 16; ++i) {
                buff0[i] *= factors[i];
            }

            transpose_XxX_cpp<4>(buff0, buff1);
            idct_4x4_llm_cpp(buff1, buff0);
            transpose_XxX_cpp<4>(buff0, buff1);
            idct_4x4_llm_cpp(buff1, buff0);

            float_to_dst_XxX_cpp<T, 4>(buff0, d + x, dst_pitch, store[0]);
        }
        s += src_pitch * 4;
        d += dst_pitch * 4;
    }
}


template <typename T>
static __forceinline __m128 load_and_cvt_to_float_x4_sse2(
    const T* srcp, const __m128& factor) noexcept
{
    if (sizeof(T) == 4) {
        return _mm_load_ps(reinterpret_cast<const float*>(srcp));
    }

    const __m128i zero = _mm_setzero_si128();
    __m128i s;
    if (sizeof(T) == 2) {
        s = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp));
    } else {
        s = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(srcp));
        s = _mm_unpacklo_epi8(s, zero);
    }
    __m128 ret = _mm_cvtepi32_ps(_mm_unpacklo_epi16(s, zero));
    return _mm_mul_ps(ret, factor);
}


static __forceinline void fdct_4x4_llm_sse(
        __m128& s0, __m128& s1, __m128& s2, __m128& s3, const __m128& xr2,
        const __m128& xr6) noexcept
{
    __m128 p03 = _mm_add_ps(s0, s3);
    __m128 p12 = _mm_add_ps(s1, s2);
    __m128 m03 = _mm_sub_ps(s0, s3);
    __m128 m12 = _mm_sub_ps(s1, s2);

    s0 = _mm_add_ps(p03, p12);
    s1 = _mm_add_ps(_mm_mul_ps(m03, xr2), _mm_mul_ps(m12, xr6));
    s2 = _mm_sub_ps(p03, p12);
    s3 = _mm_sub_ps(_mm_mul_ps(m03, xr6), _mm_mul_ps(m12, xr2));
}


static __forceinline void idct_4x4_llm_sse(
        __m128& s0, __m128& s1, __m128& s2, __m128& s3, const __m128& xr2,
        const __m128& xr6) noexcept
{
    __m128 p02 = _mm_add_ps(s0, s2);
    __m128 m02 = _mm_sub_ps(s0, s2);
    __m128 p13 = _mm_add_ps(_mm_mul_ps(s1, xr2), _mm_mul_ps(s3, xr6));
    __m128 m13 = _mm_sub_ps(_mm_mul_ps(s1, xr6), _mm_mul_ps(s3, xr2));

    s0 = _mm_add_ps(p02, p13);
    s1 = _mm_add_ps(m02, m13);
    s2 = _mm_sub_ps(m02, m13);
    s3 = _mm_sub_ps(p02, p13);
}


static __forceinline __m128i packus_epi32(
        const __m128i& x, const int bits) noexcept
{
    if (bits < 16) {
        return _mm_packs_epi32(x, x);
    } else {
        __m128i t0 = _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 0, 3, 1));
        t0 = _mm_shufflehi_epi16(t0, _MM_SHUFFLE(3, 1, 2, 0));
        return _mm_srli_si128(t0, 4);
    }
}


template <typename T>
static __forceinline void store_x4_to_dst_sse2(
        const __m128& src, T* dstp, const __m128& factor, const int bits)
        noexcept
{
    __m128 s0 = _mm_mul_ps(src, factor);

    if (sizeof(T) == 4) {
        _mm_store_ps(reinterpret_cast<float*>(dstp), s0);
        return;
    }

    __m128i d0 = packus_epi32(_mm_cvtps_epi32(s0), bits);
    if (sizeof(T) == 2) {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp), d0);
    } else {
        d0 = _mm_packus_epi16(d0, d0);
        *reinterpret_cast<int*>(dstp) = _mm_cvtsi128_si32(d0);
    }

}


template <typename T>
static void fdct_idct_4x4_sse2(
        const T* srcp, T* dstp, const float* f, int spitch, int dpitch,
        const float* load, const float* store, int bits) noexcept
{
    const __m128 factor_load = _mm_load_ps(load);

    __m128 s0 = load_and_cvt_to_float_x4_sse2(srcp + spitch * 0, factor_load);
    __m128 s1 = load_and_cvt_to_float_x4_sse2(srcp + spitch * 1, factor_load);
    __m128 s2 = load_and_cvt_to_float_x4_sse2(srcp + spitch * 2, factor_load);
    __m128 s3 = load_and_cvt_to_float_x4_sse2(srcp + spitch * 3, factor_load);

    const __m128 xr2 = _mm_set1_ps(r2);
    const __m128 xr6 = _mm_set1_ps(r6);

    _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
    fdct_4x4_llm_sse(s0, s1, s2, s3, xr2, xr6);
    _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
    fdct_4x4_llm_sse(s0, s1, s2, s3, xr2, xr6);

    s0 = _mm_mul_ps(s0, _mm_load_ps(f +  0));
    s1 = _mm_mul_ps(s1, _mm_load_ps(f +  4));
    s2 = _mm_mul_ps(s2, _mm_load_ps(f +  8));
    s3 = _mm_mul_ps(s3, _mm_load_ps(f + 12));

    _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
    idct_4x4_llm_sse(s0, s1, s2, s3, xr2, xr6);
    _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
    idct_4x4_llm_sse(s0, s1, s2, s3, xr2, xr6);

    const __m128 factor_store = _mm_load_ps(store);

    store_x4_to_dst_sse2(s0, dstp + 0 * dpitch, factor_store, bits);
    store_x4_to_dst_sse2(s1, dstp + 1 * dpitch, factor_store, bits);
    store_x4_to_dst_sse2(s2, dstp + 2 * dpitch, factor_store, bits);
    store_x4_to_dst_sse2(s3, dstp + 3 * dpitch, factor_store, bits);
}


template <typename T>
static void fdct_idct_sse2(
        const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
        int width, int height, float*, const float* factors,
        const float* load, const float* store, int bits) noexcept
{
    const T* s = reinterpret_cast<const T*>(srcp);
    T* d = reinterpret_cast<T*>(dstp);
    src_pitch /= sizeof(T);
    dst_pitch /= sizeof(T);

    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += 4) {
            fdct_idct_4x4_sse2(s + x, d + x, factors, src_pitch, dst_pitch, load, store, bits);
        }
        s += src_pitch * 4;
        d += dst_pitch * 4;
    }
}


extern void fdct_idct_4x4_avx2_8(
    const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
    int width, int height, float*, const float* factors,
    const float* load, const float* store, int) noexcept;

extern void fdct_idct_4x4_avx2_16(
    const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
    int width, int height, float*, const float* factors,
    const float* load, const float* store, int) noexcept;

extern void fdct_idct_4x4_avx2_32(
    const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
    int width, int height, float*, const float* factors,
    const float* load, const float* store, int) noexcept;


fdct_idct_func_t get_main_proc_4x4(int component_size, int opt) noexcept
{
    if (opt > 2) {
        switch (component_size) {
        case 1: return fdct_idct_4x4_avx2_8;
        case 2: return fdct_idct_4x4_avx2_16;
        default: return fdct_idct_4x4_avx2_32;
        }
    }
    if (opt > 0) {
        switch (component_size) {
        case 1: return fdct_idct_sse2<uint8_t>;
        case 2: return fdct_idct_sse2<uint16_t>;
        default: return fdct_idct_sse2<float>;
        }
    }

    switch (component_size) {
    case 1: return fdct_idct_cpp<uint8_t>;
    case 2: return fdct_idct_cpp<uint16_t>;
    default: return fdct_idct_cpp<float>;
    }
}
