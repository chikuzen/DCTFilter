/*
dct.cpp

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

//#define __AVX2__
#include <cstdint>
#if defined(__AVX2__)
#include <immintrin.h>
#else
#include <smmintrin.h>
#endif


template <typename T>
static void src_to_float_8x8_cpp(const T* srcp, float* dstp, int spitch,
        int bits) noexcept
{
    const float factor = 1.0f / ((1 << bits) - 1);

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            dstp[x] = sizeof(T) == 4 ? srcp[x] : factor * srcp[x];
        }
        dstp += 8;
        srcp += spitch;
    }
}


template <typename T>
static void float_to_dst_8x8_cpp(const float* srcp, T* dstp, int dpitch,
        int bits) noexcept
{
    const float factor = sizeof(T) == 4 ? 0.1250f : ((1 << bits) - 1) * 0.1250f;

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            dstp[x] = static_cast<T>(factor * srcp[x]);
        }
        srcp += 8;
        dstp += dpitch;
    }
}


static void transpose_8x8_cpp(const float* s, float* d) noexcept
{
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            d[8 * x] = s[x];
        }
        s += 8;
        ++d;
    }
}


constexpr float r1 = 1.3870398998f; // cos(1 / 16.0 * PI) * SQRT2
constexpr float r2 = 1.3065630198f; // cos(2 / 16.0 * PI) * SQRT2
constexpr float r3 = 1.1758755445f; // cos(3 / 16.0 * PI) * SQRT2
constexpr float r5 = 0.7856949568f; // cos(5 / 16.0 * PI) * SQRT2
constexpr float r6 = 0.5411961079f; // cos(6 / 16.0 * PI) * SQRT2
constexpr float r7 = 0.2758993804f; // cos(7 / 16.0 * PI) * SQRT2
constexpr float isqrt2 = 0.7071067812f; // 1.0f / SQRT2


static void dct_8x8_llm_cpp(const float* s, float* d) noexcept
{
    float t0, t1, t2, t3, t4, t5, t6, t7, c0, c1, c2, c3;

    for (int i = 0; i < 8; ++i) {
        t0 = s[0] + s[7];
        t7 = s[0] - s[7];
        t1 = s[1] + s[6];
        t6 = s[1] - s[6];
        t2 = s[2] + s[5];
        t5 = s[2] - s[5];
        t3 = s[3] + s[4];
        t4 = s[3] - s[4];

        c0 = t0 + t3;
        c3 = t0 - t3;
        c1 = t1 + t2;
        c2 = t1 - t2;

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
    float a0, a1, a2, a3, b0, b1, b2, b3, z0, z1, z2, z3, z4;

    for (int i = 0; i < 8; ++i) {
        z0 = s[1] + s[7];
        z1 = s[3] + s[5];

        z4 = (z0 + z1) * r3;

        z2 = (s[3] + s[7]) * (-r3 - r5) + z4;
        z3 = (s[1] + s[5]) * (-r3 + r5) + z4;
        z0 *= (-r3 + r7);
        z1 *= (-r3 - r1);

        b3 = s[7] * (-r1 + r3 + r5 - r7) + z0 + z2;
        b2 = s[5] * (r1 + r3 - r5 + r7) + z1 + z3;
        b1 = s[3] * (r1 + r3 + r5 - r7) + z1 + z2;
        b0 = s[1] * (r1 + r3 - r5 - r7) + z0 + z3;

        z4 = (s[2] + s[6]) * r6;
        z0 = s[0] + s[4];
        z1 = s[0] - s[4];
        z2 = z4 - s[6] * (r2 + r6);
        z3 = z4 + s[2] * (r2 - r6);
        a0 = z0 + z3;
        a3 = z0 - z3;
        a1 = z1 + z2;
        a2 = z1 - z2;

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
static void dct_idct_cpp(const uint8_t* srcp, uint8_t* dstp, int src_pitch,
        int dst_pitch, int rowsize, int height, float* buff0,
        const float* factors, int bits) noexcept
{
    const T* s = reinterpret_cast<const T*>(srcp);
    T* d = reinterpret_cast<T*>(dstp);
    float* buff1 = buff0 + 64;
    rowsize /= sizeof(T);
    src_pitch /= sizeof(T);
    dst_pitch /= sizeof(T);

    for (int y = 0; y < height; y += 8) {
        for (int x = 0; x < rowsize; x += 8) {

            src_to_float_8x8_cpp(s + x, buff0, src_pitch, bits);

            transpose_8x8_cpp(buff0, buff1);
            dct_8x8_llm_cpp(buff1, buff0);
            transpose_8x8_cpp(buff0, buff1);
            dct_8x8_llm_cpp(buff1, buff0);

            for (int i = 0; i < 64; ++i) {
                buff0[i] *= factors[i];
            }

            transpose_8x8_cpp(buff0, buff1);
            idct_8x8_llm_cpp(buff1, buff0);
            transpose_8x8_cpp(buff0, buff1);
            idct_8x8_llm_cpp(buff1, buff0);

            float_to_dst_8x8_cpp(buff0, d + x, dst_pitch, bits);
        }
        s += src_pitch * 8;
        d += dst_pitch * 8;
    }
}

/******************* SIMD version ***************************/

template <typename T>
static __forceinline void
load_x8_to_float_sse2(const T* srcp, __m128& s0, __m128& s1)
{
    if (sizeof(T) == 4) {
        s0 = _mm_loadu_ps(reinterpret_cast<const float*>(srcp));
        s1 = _mm_loadu_ps(reinterpret_cast<const float*>(srcp + 4));
        return;
    }

    const __m128i zero = _mm_setzero_si128();

    if (sizeof(T) == 2) {
        __m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp));
        s0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(s, zero));
        s1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(s, zero));
        return;
    }
    __m128i s = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp));
    s = _mm_unpacklo_epi8(s, zero);
    s0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(s, zero));
    s1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(s, zero));
}


template <typename T>
static void
src_to_float_8x8_sse2(const T* srcp, float* dstp, int spitch, int bits) noexcept
{
    const __m128 factor = _mm_set1_ps(1.0f / ((1 << bits) - 1));

    for (int y = 0; y < 8; ++y) {
        __m128 s0, s1;
        load_x8_to_float_sse2<T>(srcp, s0, s1);
        if (sizeof(T) != 4) {
            s0 = _mm_mul_ps(s0, factor);
            s1 = _mm_mul_ps(s1, factor);
        }
        _mm_store_ps(dstp, s0);
        _mm_store_ps(dstp + 4, s1);
        dstp += 8;
        srcp += spitch;
    }
}


static __forceinline __m128i packus_epi32(const __m128i& x, const __m128i& y)
{
    __m128i t0 = _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 0, 3, 1));
    t0 = _mm_shufflehi_epi16(t0, _MM_SHUFFLE(3, 1, 2, 0));
    t0 = _mm_srli_si128(t0, 4);
    __m128i t1 = _mm_shufflelo_epi16(y, _MM_SHUFFLE(2, 0, 3, 1));
    t1 = _mm_shufflehi_epi16(t1, _MM_SHUFFLE(3, 1, 2, 0));
    t1 = _mm_srli_si128(t1, 4);
    return _mm_unpacklo_epi64(t0, t1);
}


template <typename T>
static void
float_to_dst_8x8_sse2(const float* srcp, T* dstp, int dpitch, int bits) noexcept
{
    constexpr float setval = sizeof(T) == 4 ? 0.1250f
        : ((1LLU << (sizeof(T) * 8)) - 1) * 0.1250f;

    const __m128 factor = _mm_set1_ps(setval);
    const __m128i mask = _mm_setr_epi32(0, -1, 0, -1);

    for (int y = 0; y < 8; ++y) {
        __m128 s0 = _mm_mul_ps(_mm_load_ps(srcp), factor);
        __m128 s1 = _mm_mul_ps(_mm_load_ps(srcp + 4), factor);
        if (sizeof(T) == 4) {
            _mm_store_ps(reinterpret_cast<float*>(dstp), s0);
            _mm_store_ps(reinterpret_cast<float*>(dstp) + 4, s1);
        } else {
            __m128i d0 = _mm_cvtps_epi32(s0);
            __m128i d1 = _mm_cvtps_epi32(s1);
            d0 = packus_epi32(d0, d1);
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


static void dct_8x8_llm_sse(const float* s, float* d) noexcept
{
    static const __m128 xr1 = _mm_set1_ps(r1);
    static const __m128 xr2 = _mm_set1_ps(r2);
    static const __m128 xr3 = _mm_set1_ps(r3);
    static const __m128 xr5 = _mm_set1_ps(r5);
    static const __m128 xr6 = _mm_set1_ps(r6);
    static const __m128 xr7 = _mm_set1_ps(r7);
    static const __m128 xisqrt2 = _mm_set1_ps(isqrt2);

    for (int i = 0; i < 2; ++i) {
        __m128 s0 = _mm_load_ps(s);
        __m128 s1 = _mm_load_ps(s + 8);
        __m128 s2 = _mm_load_ps(s + 16);
        __m128 s3 = _mm_load_ps(s + 24);
        _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
        __m128 s4 = _mm_load_ps(s + 4);
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

        _mm_store_ps(d + 8, _mm_add_ps(c0, c3));
        _mm_store_ps(d + 56, _mm_sub_ps(c0, c3));

        s += 32;
        d += 4;
    }
}


static void idct_8x8_llm_sse(const float* s, float* d) noexcept
{
    for (int i = 0; i < 2; ++i) {
        __m128 s0 = _mm_load_ps(s);
        __m128 s1 = _mm_load_ps(s + 8);
        __m128 s2 = _mm_load_ps(s + 16);
        __m128 s3 = _mm_load_ps(s + 24);
        _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
        __m128 s4 = _mm_load_ps(s + 4);
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


#if defined(__AVX2__)

template <typename T>
static __forceinline __m256
load_and_cvt_to_float_x8_avx2(const T* srcp)
{
    if (sizeof(T) == 4) {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(srcp));
    } else if (sizeof(T) == 2) {
        __m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp));
        return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s));
    } else {
        __m128i s = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp));
        return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(s));
    }
}


static __forceinline void
transpose_8x8_avx(__m256& a, __m256& b, __m256& c, __m256& d, __m256& e,
        __m256& f, __m256& g, __m256& h) noexcept
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


static __forceinline void
dct_8x8_llm_fma3(__m256& s0, __m256& s1, __m256& s2, __m256& s3, __m256& s4,
        __m256& s5, __m256& s6, __m256& s7) noexcept
{
    static const __m256 xr1 = _mm256_set1_ps(r1);
    static const __m256 xr2 = _mm256_set1_ps(r2);
    static const __m256 xr3 = _mm256_set1_ps(r3);
    static const __m256 xr5 = _mm256_set1_ps(r5);
    static const __m256 xr6 = _mm256_set1_ps(r6);
    static const __m256 xr7 = _mm256_set1_ps(r7);
    static const __m256 xisqrt2 = _mm256_set1_ps(isqrt2);

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


static __forceinline void
idct_8x8_llm_fma3(__m256& s0, __m256& s1, __m256& s2, __m256& s3, __m256& s4,
        __m256& s5, __m256& s6, __m256& s7) noexcept
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
static __forceinline void
store_x8_to_dst_avx2(const __m256& src, T* dstp, int bits) noexcept
{
    constexpr float setval = sizeof(T) == 4 ? 0.1250f
        : ((1LLU << (sizeof(T) * 8)) - 1) * 0.1250f;

    static const __m256 factor = _mm256_set1_ps(setval);

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
static void
dct_idct_8x8_avx2(const T* srcp, T* dstp, const float* f, int spitch, int dpitch, int bits) noexcept
{
    static const __m256 factor = _mm256_set1_ps(1.0f / ((1 << bits) - 1));

    __m256 s0 = load_and_cvt_to_float_x8_avx2<T>(srcp + spitch * 0);
    __m256 s1 = load_and_cvt_to_float_x8_avx2<T>(srcp + spitch * 1);
    __m256 s2 = load_and_cvt_to_float_x8_avx2<T>(srcp + spitch * 2);
    __m256 s3 = load_and_cvt_to_float_x8_avx2<T>(srcp + spitch * 3);
    __m256 s4 = load_and_cvt_to_float_x8_avx2<T>(srcp + spitch * 4);
    __m256 s5 = load_and_cvt_to_float_x8_avx2<T>(srcp + spitch * 5);
    __m256 s6 = load_and_cvt_to_float_x8_avx2<T>(srcp + spitch * 6);
    __m256 s7 = load_and_cvt_to_float_x8_avx2<T>(srcp + spitch * 7);

    if (sizeof(T) != 4) {
        s0 = _mm256_mul_ps(s0, factor);
        s1 = _mm256_mul_ps(s1, factor);
        s2 = _mm256_mul_ps(s2, factor);
        s3 = _mm256_mul_ps(s3, factor);
        s4 = _mm256_mul_ps(s4, factor);
        s5 = _mm256_mul_ps(s5, factor);
        s6 = _mm256_mul_ps(s6, factor);
        s7 = _mm256_mul_ps(s7, factor);
    }

    transpose_8x8_avx(s0, s1, s2, s3, s4, s5, s6, s7);
    dct_8x8_llm_fma3(s0, s1, s2, s3, s4, s5, s6, s7);
    transpose_8x8_avx(s0, s1, s2, s3, s4, s5, s6, s7);
    dct_8x8_llm_fma3(s0, s1, s2, s3, s4, s5, s6, s7);

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

    store_x8_to_dst_avx2<T>(s0, dstp + 0 * dpitch, bits);
    store_x8_to_dst_avx2<T>(s1, dstp + 1 * dpitch, bits);
    store_x8_to_dst_avx2<T>(s2, dstp + 2 * dpitch, bits);
    store_x8_to_dst_avx2<T>(s3, dstp + 3 * dpitch, bits);
    store_x8_to_dst_avx2<T>(s4, dstp + 4 * dpitch, bits);
    store_x8_to_dst_avx2<T>(s5, dstp + 5 * dpitch, bits);
    store_x8_to_dst_avx2<T>(s6, dstp + 6 * dpitch, bits);
    store_x8_to_dst_avx2<T>(s7, dstp + 7 * dpitch, bits);
}

#endif


template <typename T, bool USE_AVX2>
static void dct_idct_simd(const uint8_t* srcp, uint8_t* dstp, int src_pitch,
    int dst_pitch, int rowsize, int height, float* buff0,
    const float* factors, int bits) noexcept
{
    const T* s = reinterpret_cast<const T*>(srcp);
    T* d = reinterpret_cast<T*>(dstp);
    float* buff1 = buff0 + 64;
    rowsize /= sizeof(T);
    src_pitch /= sizeof(T);
    dst_pitch /= sizeof(T);

    for (int y = 0; y < height; y += 8) {
        for (int x = 0; x < rowsize; x += 8) {
#if defined(__AVX2__)
            if (USE_AVX2) {
                dct_idct_8x8_avx2(s + x, d + x, factors, src_pitch, dst_pitch, bits);
                continue;
            }
#endif
            src_to_float_8x8_sse2(s + x, buff0, src_pitch, bits);

            dct_8x8_llm_sse(buff0, buff1);
            dct_8x8_llm_sse(buff1, buff0);

            for (int i = 0; i < 64; i += 4) {
                __m128 t0 = _mm_load_ps(buff0 + i);
                __m128 t1 = _mm_load_ps(factors + i);
                _mm_store_ps(buff0 + i, _mm_mul_ps(t0, t1));
            }

            idct_8x8_llm_sse(buff0, buff1);
            idct_8x8_llm_sse(buff1, buff0);

            float_to_dst_8x8_sse2(buff0, d + x, dst_pitch, bits);
        }
        s += src_pitch * 8;
        d += dst_pitch * 8;
    }
}


typedef void(*dct_idct_func_t)(
    const uint8_t*, uint8_t*, int, int, int, int, float*, const float*, int);

dct_idct_func_t get_main_proc(int component_size, int opt)
{
#if defined(__AVX2__)
    if (opt > 1) {
        if (component_size == 1) {
            return dct_idct_simd<uint8_t, true>;
        } else if (component_size == 2) {
            return dct_idct_simd<uint16_t, true>;
        } else {
            return dct_idct_simd<float, true>;
        }
    }
#endif
    if (opt > 0) {
        if (component_size == 1) {
            return dct_idct_simd<uint8_t, false>;
        } else if (component_size == 2) {
            return dct_idct_simd<uint16_t, false>;
        } else {
            return dct_idct_simd<float, false>;
        }
    }
    if (component_size == 1) {
        return dct_idct_cpp<uint8_t>;
    }
    if (component_size == 2) {
        return dct_idct_cpp<uint16_t>;
    }
    return dct_idct_cpp<float>;
}