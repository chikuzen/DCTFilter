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


#include <cstdint>


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
static void dct_idct_llm_cpp(const uint8_t* srcp, uint8_t* dstp, int src_pitch,
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


typedef void(*dct_idct_func_t)(
    const uint8_t*, uint8_t*, int, int, int, int, float*, const float*, int);

dct_idct_func_t get_main_proc(int component_size)
{
    if (component_size == 1) {
        return dct_idct_llm_cpp<uint8_t>;
    }
    if (component_size == 2) {
        return dct_idct_llm_cpp<uint16_t>;
    }
    return dct_idct_llm_cpp<float>;
}