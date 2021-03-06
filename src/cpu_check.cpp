/*
cpu_check.cpp

This file is a part of DCTFilter

Copyright (C) 2016 OKA Motofumi

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
#include <intrin.h>


enum {
    CPU_NO_X86_SIMD          = 0x000000,
    CPU_SSE2_SUPPORT         = 0x000001,
    CPU_SSE3_SUPPORT         = 0x000002,
    CPU_SSSE3_SUPPORT        = 0x000004,
    CPU_SSE4_1_SUPPORT       = 0x000008,
    CPU_SSE4_2_SUPPORT       = 0x000010,
    CPU_SSE4_A_SUPPORT       = 0x000020,
    CPU_FMA4_SUPPORT         = 0x000040,
    CPU_FMA3_SUPPORT         = 0x000080,
    CPU_AVX_SUPPORT          = 0x000100,
    CPU_AVX2_SUPPORT         = 0x000200,
    CPU_AVX512F_SUPPORT      = 0x000400,
    CPU_AVX512DQ_SUPPORT     = 0x000800,
    CPU_AVX512IFMA52_SUPPORT = 0x001000,
    CPU_AVX512PF_SUPPORT     = 0x002000,
    CPU_AVX512ER_SUPPORT     = 0x004000,
    CPU_AVX512CD_SUPPORT     = 0x008000,
    CPU_AVX512BW_SUPPORT     = 0x010000,
    CPU_AVX512VL_SUPPORT     = 0x020000,
    CPU_AVX512VBMI_SUPPORT   = 0x040000,
};




static __forceinline bool is_bit_set(int bitfield, int bit)
{
    return (bitfield & (1 << bit)) != 0;
}

static uint32_t get_simd_support_info(void)
{
    uint32_t ret = 0;
    int regs[4] = {0};

    __cpuid(regs, 0x00000001);
    if (is_bit_set(regs[3], 26)) {
        ret |= CPU_SSE2_SUPPORT;
    }
    if (is_bit_set(regs[2], 0)) {
        ret |= CPU_SSE3_SUPPORT;
    }
    if (is_bit_set(regs[2], 9)) {
        ret |= CPU_SSSE3_SUPPORT;
    }
    if (is_bit_set(regs[2], 19)) {
        ret |= CPU_SSE4_1_SUPPORT;
    }
    if (is_bit_set(regs[2], 26)) {
        ret |= CPU_SSE4_2_SUPPORT;
    }
    if (is_bit_set(regs[2], 27)) {
        if (is_bit_set(regs[2], 28)) {
            ret |= CPU_AVX_SUPPORT;
        }
        if (is_bit_set(regs[2], 12)) {
            ret |= CPU_FMA3_SUPPORT;
        }
    }

    regs[3] = 0;
    __cpuid(regs, 0x80000001);
    if (is_bit_set(regs[3], 6)) {
        ret |= CPU_SSE4_A_SUPPORT;
    }
    if (is_bit_set(regs[3], 16)) {
        ret |= CPU_FMA4_SUPPORT;
    }

    __cpuid(regs, 0x00000000);
    if (regs[0] < 7) {
        return ret;
    }

    __cpuidex(regs, 0x00000007, 0);
    if (is_bit_set(regs[1], 5)) {
        ret |= CPU_AVX2_SUPPORT;
    }
    if (!is_bit_set(regs[1], 16)) {
        return ret;
    }

    ret |= CPU_AVX512F_SUPPORT;
    if (is_bit_set(regs[1], 17)) {
        ret |= CPU_AVX512DQ_SUPPORT;
    }
    if (is_bit_set(regs[1], 21)) {
        ret |= CPU_AVX512IFMA52_SUPPORT;
    }
    if (is_bit_set(regs[1], 26)) {
        ret |= CPU_AVX512PF_SUPPORT;
    }
    if (is_bit_set(regs[1], 27)) {
        ret |= CPU_AVX512ER_SUPPORT;
    }
    if (is_bit_set(regs[1], 28)) {
        ret |= CPU_AVX512CD_SUPPORT;
    }
    if (is_bit_set(regs[1], 30)) {
        ret |= CPU_AVX512BW_SUPPORT;
    }
    if (is_bit_set(regs[1], 31)) {
        ret |= CPU_AVX512VL_SUPPORT;
    }
    if (is_bit_set(regs[2], 1)) {
        ret |= CPU_AVX512VBMI_SUPPORT;
    }

    return ret;
}

bool has_sse2()
{
    return (get_simd_support_info() & CPU_SSE2_SUPPORT) != 0;
}

bool has_sse41()
{
    return (get_simd_support_info() & CPU_SSE4_1_SUPPORT) != 0;
}

bool has_avx2()
{
    uint32_t flags = CPU_AVX2_SUPPORT | CPU_FMA3_SUPPORT;
    return (get_simd_support_info() & flags) == flags;
}
