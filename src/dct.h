/*
dct.h

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


#ifndef DCTFILTER_DCT_H
#define DCTFILTER_DCT_H

#include <cstdint>


typedef void(*fdct_idct_func_t)(
    const uint8_t* srcp, uint8_t* dstp, int src_pitch, int dst_pitch,
    int width, int height, float* buff, const float* factors,
    const float* load, const float* store, int bits);


fdct_idct_func_t get_main_proc_4x4(int component_size, int opt) noexcept;

fdct_idct_func_t get_main_proc_8x8(int component_size, int opt) noexcept;


template <typename T, int SIZE>
static void src_to_float_XxX_cpp(
        const T* srcp, float* dstp, const int spitch, const float factor) noexcept
{
    for (int y = 0; y < SIZE; ++y) {
        for (int x = 0; x < SIZE; ++x) {
            dstp[x] = sizeof(T) == 4 ? srcp[x] : factor * srcp[x];
        }
        dstp += SIZE;
        srcp += spitch;
    }
}


template <typename T, int SIZE>
static void float_to_dst_XxX_cpp(
        const float* srcp, T* dstp, const int dpitch, const float factor) noexcept
{
    for (int y = 0; y < SIZE; ++y) {
        for (int x = 0; x < SIZE; ++x) {
            dstp[x] = static_cast<T>(factor * srcp[x]);
        }
        srcp += SIZE;
        dstp += dpitch;
    }
}


template <int SIZE>
static void transpose_XxX_cpp(const float* s, float* d) noexcept
{
    for (int y = 0; y < SIZE; ++y) {
        for (int x = 0; x < SIZE; ++x) {
            d[SIZE * x] = s[x];
        }
        s += SIZE;
        ++d;
    }
}

#endif

