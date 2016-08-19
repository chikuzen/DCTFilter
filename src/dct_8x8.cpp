/*
dct_8x8.cpp

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


constexpr float r1 = 1.3870398998f; // cos(1 / 16.0 * PI) * SQRT2
constexpr float r2 = 1.3065630198f; // cos(2 / 16.0 * PI) * SQRT2
constexpr float r3 = 1.1758755445f; // cos(3 / 16.0 * PI) * SQRT2
constexpr float r5 = 0.7856949568f; // cos(5 / 16.0 * PI) * SQRT2
constexpr float r6 = 0.5411961079f; // cos(6 / 16.0 * PI) * SQRT2
constexpr float r7 = 0.2758993804f; // cos(7 / 16.0 * PI) * SQRT2
constexpr float isqrt2 = 0.7071067812f; // 1.0f / SQRT2


static void fdct_8x8_llm_cpp(const float* s, float* d) noexcept
{
    for (int i = 0; i < 8; ++i) {
        float t0 = s[0] + s[7];
        float t7 = s[0] - s[7];
        float t1 = s[1] + s[6];
        float t6 = s[1] - s[6];
        float t2 = s[2] + s[5];
        float t5 = s[2] - s[5];
        float t3 = s[3] + s[4];
        float t4 = s[3] - s[4];

        float c0 = t0 + t3;
        float c3 = t0 - t3;
        float c1 = t1 + t2;
        float c2 = t1 - t2;

        d[0] = c0 + c1;
        d[4] = c0 - c1;
        d[2] = c2 * r6 + c3 * r2;
        d[6] = c3 * r6 - c2 * r2;

        c3 = t4 * r3 + t7 * r5;
        c0 = t7 * r3 - t4 * r5;
        c2 = t5 * r1 + t6 * r7;
        c1 = t6 * r1 - t5 * r7;

        d[5] = c3 - c1;
        d[3] = c0 - c2;
        c0 = (c0 + c2) * isqrt2;
        c3 = (c3 + c1) * isqrt2;
        d[1] = c0 + c3;
        d[7] = c0 - c3;

        s += 8;
        d += 8;
    }
}


static void idct_8x8_llm_cpp(const float* s, float* d) noexcept
{
    for (int i = 0; i < 8; ++i) {
        float z0 = s[1] + s[7];
        float z1 = s[3] + s[5];
        float z4 = (z0 + z1) * r3;
        float z2 = (s[3] + s[7]) * (-r3 - r5) + z4;
        float z3 = (s[1] + s[5]) * (-r3 + r5) + z4;

        z0 *= (-r3 + r7);
        z1 *= (-r3 - r1);

        float b3 = s[7] * (-r1 + r3 + r5 - r7) + z0 + z2;
        float b2 = s[5] * (r1 + r3 - r5 + r7) + z1 + z3;
        float b1 = s[3] * (r1 + r3 + r5 - r7) + z1 + z2;
        float b0 = s[1] * (r1 + r3 - r5 - r7) + z0 + z3;

        z4 = (s[2] + s[6]) * r6;
        z0 = s[0] + s[4];
        z1 = s[0] - s[4];
        z2 = z4 - s[6] * (r2 + r6);
        z3 = z4 + s[2] * (r2 - r6);

        float a0 = z0 + z3;
        float a3 = z0 - z3;
        float a1 = z1 + z2;
        float a2 = z1 - z2;

        d[0] = a0 + b0;
        d[7] = a0 - b0;
        d[1] = a1 + b1;
        d[6] = a1 - b1;
        d[2] = a2 + b2;
        d[5] = a2 - b2;
        d[3] = a3 + b3;
        d[4] = a3 - b3;

        s += 8;
        d += 8;
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
    float* buff1 = buff0 + 64;
    src_pitch /= sizeof(T);
    dst_pitch /= sizeof(T);

    for (int y = 0; y < height; y += 8) {
        for (int x = 0; x < width; x += 8) {

            src_to_float_XxX_cpp<T, 8>(s + x, buff0, src_pitch, load[0]);

            transpose_XxX_cpp<8>(buff0, buff1);
            fdct_8x8_llm_cpp(buff1, buff0);
            transpose_XxX_cpp<8>(buff0, buff1);
            fdct_8x8_llm_cpp(buff1, buff0);

            for (int i = 0; i < 64; ++i) {
                buff0[i] *= factors[i];
            }

            transpose_XxX_cpp<8>(buff0, buff1);
            idct_8x8_llm_cpp(buff1, buff0);
            transpose_XxX_cpp<8>(buff0, buff1);
            idct_8x8_llm_cpp(buff1, buff0);

            float_to_dst_XxX_cpp<T, 8>(buff0, d + x, dst_pitch, store[0]);
        }
        s += src_pitch * 8;
        d += dst_pitch * 8;
    }
}

/******************* SIMD version ***************************/

template <bool HAS_SSE41>
static __forceinline __m128i packus_epi32(
        const __m128i& x, const __m128i& y, int bits) noexcept
{
    if (HAS_SSE41) {
        return _mm_packus_epi32(x, y);
    }

    if (bits < 16) {
        return _mm_packs_epi32(x, y);
    } else {
        __m128i t0 = _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 0, 3, 1));
        __m128i t1 = _mm_shufflelo_epi16(y, _MM_SHUFFLE(2, 0, 3, 1));
        t0 = _mm_shufflehi_epi16(t0, _MM_SHUFFLE(3, 1, 2, 0));
        t1 = _mm_shufflehi_epi16(t1, _MM_SHUFFLE(3, 1, 2, 0));
        t0 = _mm_srli_si128(t0, 4);
        t1 = _mm_srli_si128(t1, 4);
        return _mm_unpacklo_epi64(t0, t1);
    }
}


template <typename T>
static __forceinline void load_x8_to_float_sse2(
    const T* srcp, __m128& s0, __m128& s1, const __m128& factor) noexcept
{
    if (sizeof(T) == 4) {
        s0 = _mm_load_ps(reinterpret_cast<const float*>(srcp));
        s1 = _mm_load_ps(reinterpret_cast<const float*>(srcp + 4));
        return;
    }

    const __m128i zero = _mm_setzero_si128();
    __m128i s;
    if (sizeof(T) == 2) {
        s = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));
    } else {
        s = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp)), zero);
    }
    s0 = _mm_mul_ps(factor, _mm_cvtepi32_ps(_mm_unpacklo_epi16(s, zero)));
    s1 = _mm_mul_ps(factor, _mm_cvtepi32_ps(_mm_unpackhi_epi16(s, zero)));
}


template <typename T>
static void src_to_float_8x8_sse2(
        const T* srcp, float* dstp, int spitch, const float* f) noexcept
{
    const __m128 factor = _mm_load_ps(f);

    for (int y = 0; y < 8; ++y) {
        __m128 s0, s1;
        load_x8_to_float_sse2(srcp, s0, s1, factor);
        _mm_store_ps(dstp, s0);
        _mm_store_ps(dstp + 4, s1);
        dstp += 8;
        srcp += spitch;
    }
}


template <typename T, bool HAS_SSE41>
static void float_to_dst_8x8_sse2(
        const float* srcp, T* dstp, int dpitch, const float* f, int bits) noexcept
{
    const __m128 factor = _mm_load_ps(f);

    for (int y = 0; y < 8; ++y) {
        __m128 s0 = _mm_mul_ps(_mm_load_ps(srcp), factor);
        __m128 s1 = _mm_mul_ps(_mm_load_ps(srcp + 4), factor);
        if (sizeof(T) == 4) {
            _mm_store_ps(reinterpret_cast<float*>(dstp), s0);
            _mm_store_ps(reinterpret_cast<float*>(dstp) + 4, s1);
        } else {
            __m128i d0 = packus_epi32<HAS_SSE41>(
                _mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1), bits);
            if (sizeof(T) == 2) {
                _mm_store_si128(reinterpret_cast<__m128i*>(dstp), d0);
            } else {
                d0 = _mm_packus_epi16(d0, d0);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp), d0);
            }
        }
        srcp += 8;
        dstp += dpitch;
    }
}


static void fdct_8x8_llm_with_transpose_sse(const float* s, float* d) noexcept
{
    const __m128 xr1 = _mm_set1_ps(r1);
    const __m128 xr2 = _mm_set1_ps(r2);
    const __m128 xr3 = _mm_set1_ps(r3);
    const __m128 xr5 = _mm_set1_ps(r5);
    const __m128 xr6 = _mm_set1_ps(r6);
    const __m128 xr7 = _mm_set1_ps(r7);
    const __m128 xisqrt2 = _mm_set1_ps(isqrt2);

    for (int i = 0; i < 2; ++i) {
        __m128 s0 = _mm_load_ps(s +  0);
        __m128 s1 = _mm_load_ps(s +  8);
        __m128 s2 = _mm_load_ps(s + 16);
        __m128 s3 = _mm_load_ps(s + 24);
        _MM_TRANSPOSE4_PS(s0, s1, s2, s3);

        __m128 s4 = _mm_load_ps(s +  4);
        __m128 s5 = _mm_load_ps(s + 12);
        __m128 s6 = _mm_load_ps(s + 20);
        __m128 s7 = _mm_load_ps(s + 28);
        _MM_TRANSPOSE4_PS(s4, s5, s6, s7);

        __m128 t0 = _mm_add_ps(s0, s7);
        __m128 t7 = _mm_sub_ps(s0, s7);
        __m128 t1 = _mm_add_ps(s1, s6);
        __m128 t6 = _mm_sub_ps(s1, s6);
        __m128 t2 = _mm_add_ps(s2, s5);
        __m128 t5 = _mm_sub_ps(s2, s5);
        __m128 t3 = _mm_add_ps(s3, s4);
        __m128 t4 = _mm_sub_ps(s3, s4);

        __m128 c0 = _mm_add_ps(t0, t3);
        __m128 c3 = _mm_sub_ps(t0, t3);
        __m128 c1 = _mm_add_ps(t1, t2);
        __m128 c2 = _mm_sub_ps(t1, t2);

        _mm_store_ps(d +  0, _mm_add_ps(c0, c1));
        _mm_store_ps(d + 32, _mm_sub_ps(c0, c1));
        _mm_store_ps(d + 16, _mm_add_ps(_mm_mul_ps(c2, xr6), _mm_mul_ps(c3, xr2)));
        _mm_store_ps(d + 48, _mm_sub_ps(_mm_mul_ps(c3, xr6), _mm_mul_ps(c2, xr2)));

        c3 = _mm_add_ps(_mm_mul_ps(t4, xr3), _mm_mul_ps(t7, xr5));
        c0 = _mm_sub_ps(_mm_mul_ps(t7, xr3), _mm_mul_ps(t4, xr5));
        c2 = _mm_add_ps(_mm_mul_ps(t5, xr1), _mm_mul_ps(t6, xr7));
        c1 = _mm_sub_ps(_mm_mul_ps(t6, xr1), _mm_mul_ps(t5, xr7));

        _mm_store_ps(d + 24, _mm_sub_ps(c0, c2));
        _mm_store_ps(d + 40, _mm_sub_ps(c3, c1));

        c0 = _mm_mul_ps(_mm_add_ps(c0, c2), xisqrt2);
        c3 = _mm_mul_ps(_mm_add_ps(c1, c3), xisqrt2);

        _mm_store_ps(d +  8, _mm_add_ps(c0, c3));
        _mm_store_ps(d + 56, _mm_sub_ps(c0, c3));

        s += 32;
        d += 4;
    }
}


static void idct_8x8_llm_with_transpose_sse(const float* s, float* d) noexcept
{
    for (int i = 0; i < 2; ++i) {
        __m128 s0 = _mm_load_ps(s +  0);
        __m128 s1 = _mm_load_ps(s +  8);
        __m128 s2 = _mm_load_ps(s + 16);
        __m128 s3 = _mm_load_ps(s + 24);
        _MM_TRANSPOSE4_PS(s0, s1, s2, s3);

        __m128 s4 = _mm_load_ps(s +  4);
        __m128 s5 = _mm_load_ps(s + 12);
        __m128 s6 = _mm_load_ps(s + 20);
        __m128 s7 = _mm_load_ps(s + 28);
        _MM_TRANSPOSE4_PS(s4, s5, s6, s7);

        __m128 z0 = _mm_add_ps(s1, s7);
        __m128 z1 = _mm_add_ps(s3, s5);

        __m128 z4 = _mm_mul_ps(_mm_add_ps(z0, z1), _mm_set1_ps(r3));

        __m128 z2 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-r3 - r5), _mm_add_ps(s3, s7)), z4);
        __m128 z3 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-r3 + r5), _mm_add_ps(s1, s5)), z4);
        z0 = _mm_mul_ps(z0, _mm_set1_ps(-r3 + r7));
        z1 = _mm_mul_ps(z1, _mm_set1_ps(-r3 - r1));

        __m128 b3 = _mm_add_ps(_mm_mul_ps(s7, _mm_set1_ps(-r1 + r3 + r5 - r7)), _mm_add_ps(z0, z2));
        __m128 b2 = _mm_add_ps(_mm_mul_ps(s5, _mm_set1_ps(r1 + r3 - r5 + r7)), _mm_add_ps(z1, z3));
        __m128 b1 = _mm_add_ps(_mm_mul_ps(s3, _mm_set1_ps(r1 + r3 + r5 - r7)), _mm_add_ps(z1, z2));
        __m128 b0 = _mm_add_ps(_mm_mul_ps(s1, _mm_set1_ps(r1 + r3 - r5 - r7)), _mm_add_ps(z0, z3));

        z0 = _mm_add_ps(s0, s4);
        z1 = _mm_sub_ps(s0, s4);
        z4 = _mm_mul_ps(_mm_add_ps(s2, s6), _mm_set1_ps(r6));

        z2 = _mm_sub_ps(z4, _mm_mul_ps(s6, _mm_set1_ps(r2 + r6)));
        z3 = _mm_add_ps(_mm_mul_ps(s2, _mm_set1_ps(r2 - r6)), z4);
        
        __m128 a0 = _mm_add_ps(z0, z3);
        __m128 a3 = _mm_sub_ps(z0, z3);
        __m128 a1 = _mm_add_ps(z1, z2);
        __m128 a2 = _mm_sub_ps(z1, z2);

        _mm_store_ps(d +  0, _mm_add_ps(a0, b0));
        _mm_store_ps(d + 56, _mm_sub_ps(a0, b0));
        _mm_store_ps(d +  8, _mm_add_ps(a1, b1));
        _mm_store_ps(d + 48, _mm_sub_ps(a1, b1));
        _mm_store_ps(d + 16, _mm_add_ps(a2, b2));
        _mm_store_ps(d + 40, _mm_sub_ps(a2, b2));
        _mm_store_ps(d + 24, _mm_add_ps(a3, b3));
        _mm_store_ps(d + 32, _mm_sub_ps(a3, b3));

        s += 32;
        d += 4;
    }
}


template <typename T, bool HAS_SSE41>
static void fdct_idct_sse2(
        const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
        int width, int height, float* buff0, const float* factors,
        const float* load, const float* store, int bits) noexcept
{
    const T* s = reinterpret_cast<const T*>(srcp);
    T* d = reinterpret_cast<T*>(dstp);
    float* buff1 = buff0 + 64;
    src_pitch /= sizeof(T);
    dst_pitch /= sizeof(T);

    for (int y = 0; y < height; y += 8) {
        for (int x = 0; x < width; x += 8) {

            src_to_float_8x8_sse2(s + x, buff0, src_pitch, load);

            fdct_8x8_llm_with_transpose_sse(buff0, buff1);
            fdct_8x8_llm_with_transpose_sse(buff1, buff0);

            for (int i = 0; i < 64; i += 4) {
                __m128 t0 = _mm_load_ps(buff0 + i);
                __m128 t1 = _mm_load_ps(factors + i);
                _mm_store_ps(buff0 + i, _mm_mul_ps(t0, t1));
            }

            idct_8x8_llm_with_transpose_sse(buff0, buff1);
            idct_8x8_llm_with_transpose_sse(buff1, buff0);

            float_to_dst_8x8_sse2<T, HAS_SSE41>(buff0, d + x, dst_pitch, store, bits);
        }
        s += src_pitch * 8;
        d += dst_pitch * 8;
    }
}


extern void fdct_idct_8x8_avx2_8(
    const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
    int width, int height, float*, const float* factors,
    const float* load, const float* store, int) noexcept;

extern void fdct_idct_8x8_avx2_16(
    const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
    int width, int height, float*, const float* factors,
    const float* load, const float* store, int) noexcept;

extern void fdct_idct_8x8_avx2_32(
    const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
    int width, int height, float*, const float* factors,
    const float* load, const float* store, int) noexcept;


fdct_idct_func_t get_main_proc_8x8(int component_size, int opt) noexcept
{
    if (opt > 2) {
        switch (component_size) {
        case 1: return fdct_idct_8x8_avx2_8;
        case 2: return fdct_idct_8x8_avx2_16;
        default: return fdct_idct_8x8_avx2_32;
        }
    }
    if (opt > 1) {
        switch (component_size) {
        case 1: return fdct_idct_sse2<uint8_t, true>;
        case 2: return fdct_idct_sse2<uint16_t, true>;
        default: return fdct_idct_sse2<float, false>;
        }
    }
    if (opt > 0) {
        switch (component_size) {
        case 1: return fdct_idct_sse2<uint8_t, false>;
        case 2: return fdct_idct_sse2<uint16_t, false>;
        default: return fdct_idct_sse2<float, false>;
        }
    }
    switch (component_size) {
    case 1: return fdct_idct_cpp<uint8_t>;
    case 2: return fdct_idct_cpp<uint16_t>;
    default: return fdct_idct_cpp<float>;
    }
}

