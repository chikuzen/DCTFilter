/*
DCTFilter.cpp

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
#include <stdexcept>
#include <algorithm>
#include <malloc.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NOGDI
#define VC_EXTRALEAN
#include <windows.h>
#include <avisynth.h>

#include "dct.h"



#define DCT_FILTER_VERSION "0.4.0"


typedef IScriptEnvironment ise_t;

extern bool has_sse2();
extern bool has_sse41();
extern bool has_avx2();


static int check_opt(int opt)
{
    if (opt == 0 || !has_sse2()) {
        return 0;
    } else if (opt == 1 || !has_sse41()) {
        return 1;
#if !defined(__AVX2__)
    }
    return 2;
#else
    } else if (opt == 2 || !has_avx2()){
        return 2;
    }
    return 3;
#endif
}


static void set_factors(
        float* factors, const int size, const int dcount, const double* f,
        const float constant, const int opt) noexcept
{
    int step = (size == 4 && opt == 3) ? 8 : size;
    if (dcount == 0) {
        if (f == nullptr) {
            return;
        }
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                factors[x] = static_cast<float>(f[y] * f[x] * constant);
            }
            factors += step;
        }
    } else {
        const int dc_max = size * 2 - 2;
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                factors[x] = (x + y) > (dc_max - dcount) ? 0 : constant;
            }
            factors += step;
        }
    }
    if (size == 4 && opt == 3) {
        for (int y = 0; y < 4; ++y) {
            factors -= 8;
            memcpy(factors + 4, factors, 4 * sizeof(float));
        }
    }
}


enum {
    MODE_8X8,
    MODE_4X4,
};

class DCTFilter : public GenericVideoFilter {
    int numPlanes;
    int planes[3];
    int chroma;
    bool isPlus;
    float* factorsLoad;
    float* factorsStore8x8;
    float* factorsStore4x4;
    float* factors8x8;
    float* factors4x4;
    size_t factorsSize;
    int componentSize;
    int bitsPerComponent;
    int mode;

    fdct_idct_func_t mainProc8x8;
    fdct_idct_func_t mainProc4x4;

public:
    DCTFilter(PClip child, double* factor8x8, double* factor4x4, int dcount8x8,
              int dcount4x4, int chroma, int opt, int mode, bool is_plus);
    PVideoFrame __stdcall GetFrame(int n, ise_t* env);
    ~DCTFilter();
    int __stdcall SetCacheHints(int hints, int)
    {
        return hints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
};


DCTFilter::DCTFilter(PClip c, double* f8, double* f4, int dcount8, int dcount4,
                     int ch, int opt, int m, bool ip)
        : GenericVideoFilter(c), chroma(ch), isPlus(ip), mode(m)
{
    if (!vi.IsPlanar()) {
        throw std::runtime_error("input is not planar format.");
    }

    if ((vi.width | vi.height) & 3) {
        throw std::runtime_error(
            "first plane's width and height must be mod 4.");
    }

    numPlanes = (vi.pixel_type & VideoInfo::CS_INTERLEAVED) ? 1 : 3;

    if (numPlanes == 1 || chroma < 0 || chroma > 2) {
        chroma = 2;
    }
    if (chroma != 1) {
        numPlanes = 1;
    }

    if (vi.IsYUV()) {
        planes[0] = PLANAR_Y;
        planes[1] = PLANAR_U;
        planes[2] = PLANAR_V;
    } else {
        planes[0] = PLANAR_G;
        planes[1] = PLANAR_B;
        planes[2] = PLANAR_R;
    }

    componentSize = vi.BytesFromPixels(1);
    bitsPerComponent = componentSize * 8;

    if (chroma == 1) {
        int w = vi.width >> vi.GetPlaneWidthSubsampling(planes[1]);
        int h = vi.height >> vi.GetPlaneHeightSubsampling(planes[1]);
        if ((w | h) & 3) {
            throw std::runtime_error(
                "second plane's width and height must be mod 4.");
        }
    }

    opt = check_opt(opt);

    factorsSize = 8 + 8 + 8 + 64 + (opt == 3 ? 32 : 16);
    size_t size = (factorsSize + (isPlus ? 0 : 128)) * sizeof(float);
    factorsLoad = reinterpret_cast<float*>(_aligned_malloc(size, 32));
    if (!factorsLoad) {
        throw std::runtime_error("failed to create table of factors.");
    }
    factorsStore8x8 = factorsLoad + 8;
    factorsStore4x4 = factorsStore8x8 + 8;
    factors8x8 = factorsStore4x4 + 8;
    factors4x4 = factors8x8 + 64;

    float load, store8, store4;
    if (componentSize == 4) {
        load = 1.0f;
        store8 = 0.125f;
        store4 = 0.25f;
    } else {
        int valmax = (1 << bitsPerComponent) - 1;
        load = 1.0f / valmax;
        store8 = 0.125f * valmax;
        store4 = 0.25f * valmax;
    }
    std::fill_n(factorsLoad, 8, load);
    std::fill_n(factorsStore8x8, 8, store8);
    std::fill_n(factorsStore4x4, 8, store4);

    set_factors(factors8x8, 8, dcount8, f8, 0.1250f, opt);
    set_factors(factors4x4, 4, dcount4, f4, 0.250f, opt);

    mainProc8x8 = get_main_proc_8x8(componentSize, opt);
    mainProc4x4 = get_main_proc_4x4(componentSize, opt);
}


DCTFilter::~DCTFilter()
{
    _aligned_free(factorsLoad);
    factorsLoad = nullptr;
}


PVideoFrame __stdcall DCTFilter::GetFrame(int n, ise_t* env)
{
    auto src = child->GetFrame(n, env);
    auto dst = env->NewVideoFrame(vi, 32);

    float* buff;
    if (!isPlus) {
        buff = factors8x8 + factorsSize;
    } else {
        buff = reinterpret_cast<float*>(static_cast<IScriptEnvironment2*>(
            env)->Allocate(128 * sizeof(float), 32, AVS_POOLED_ALLOC));
        if (!buff) {
            env->ThrowError("DCTFilter: failed to allocate temporal buffer.");
        }
    }

    for (int p = 0; p < numPlanes; ++p) {
        const int plane = planes[p];

        const uint8_t* srcp = src->GetReadPtr(plane);
        uint8_t* dstp = dst->GetWritePtr(plane);
        int spitch = src->GetPitch(plane);
        int dpitch = dst->GetPitch(plane);
        int width = src->GetRowSize(plane) / componentSize;
        int height = src->GetHeight(plane);
        int width8 = width & ~7;
        int height8 = height & ~7;

        if (mode != MODE_4X4) {
            mainProc8x8(srcp, dstp, spitch, dpitch, width8, height8, buff,
                        factors8x8, factorsLoad, factorsStore8x8,
                        bitsPerComponent);
            if (width != width8) {
                int offset = width8 * componentSize;
                mainProc4x4(srcp + offset, dstp + offset, spitch, dpitch,
                            4, height8, buff, factors4x4, factorsLoad,
                            factorsStore4x4, bitsPerComponent);
            }
            srcp += height8 * spitch;
            dstp += height8 * dpitch;
            height -= height8;
        }

        if (height != 0) {
            mainProc4x4(srcp, dstp, spitch, dpitch, width, height, buff,
                        factors4x4, factorsLoad, factorsStore4x4,
                        bitsPerComponent);
        }
    }

    if (chroma == 0) {
        const int dpitch = dst->GetPitch(planes[1]);
        const int spitch = src->GetPitch(planes[1]);
        const int rowsize = dst->GetRowSize(planes[1]);
        const int height = dst->GetHeight(planes[1]);
        env->BitBlt(dst->GetWritePtr(planes[1]), dpitch,
                    src->GetReadPtr(planes[1]), spitch, rowsize, height);
        env->BitBlt(dst->GetWritePtr(planes[2]), dpitch,
                    src->GetReadPtr(planes[2]), spitch, rowsize, height);
    }

    if (isPlus) {
        static_cast<IScriptEnvironment2*>(env)->Free(buff);
    }

    return dst;
}


static AVSValue __cdecl create_8(AVSValue args, void*, ise_t* env)
{
    double f8[8];
    for (int i = 0; i < 8; ++i) {
        f8[i] = args[i + 1].AsFloat();
        if (f8[i] < 0.0 || f8[i] > 1.0) {
            env->ThrowError(
                "DCTFilter8: all scaling factors must be between 0.0 and 1.0.");
        }
    }

    double f4[4];
    for (int i = 0; i < 4; ++i) {
        f4[i] = args[i + 9].AsFloat(1.0f);
        if (f4[i] < 0.0 || f4[i] > 1.0) {
            env->ThrowError(
                "DCTFilter8: all scaling factors must be between 0.0 and 1.0.");
        }
    }

    try {
        return new DCTFilter(
            args[0].AsClip(), f8, f4, 0, 0, args[13].AsInt(1), args[14].AsInt(-1),
            MODE_8X8, env->FunctionExists("SetFilterMTMode"));
    } catch (std::runtime_error& e) {
        env->ThrowError("DCTFilter8: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl create_8d(AVSValue args, void*, ise_t* env)
{
    int diag_count8 = args[1].AsInt();
    if (diag_count8 < 1 || diag_count8 > 14) {
        env->ThrowError("DCTFilter8D: diagonals count must be 1 to 14.");
    }

    int diag_count4 = args[2].AsInt(1);
    if (diag_count4 < 1 || diag_count4 > 6) {
        env->ThrowError("DCTFilter8D: diagonals count for 4x4 must be 1 to 6.");
    }

    try {
        return new DCTFilter(
            args[0].AsClip(), nullptr, nullptr, diag_count8, diag_count4,
            args[3].AsInt(1), args[4].AsInt(-1), MODE_8X8,
            env->FunctionExists("SetFilterMTMode"));
    } catch (std::runtime_error& e) {
        env->ThrowError("DCTFilter8D: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl create_4(AVSValue args, void*, ise_t* env)
{
    double f4[4];
    for (int i = 0; i < 4; ++i) {
        f4[i] = args[i + 1].AsFloat();
        if (f4[i] < 0.0 || f4[i] > 1.0) {
            env->ThrowError(
                "DCTFilter4: all scaling factors must be between 0.0 and 1.0.");
        }
    }

    try {
        return new DCTFilter(
            args[0].AsClip(), nullptr, f4, 0, 0, args[5].AsInt(1), args[6].AsInt(-1),
            MODE_4X4, env->FunctionExists("SetFilterMTMode"));
    } catch (std::runtime_error& e) {
        env->ThrowError("DCTFilter4: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl create_4d(AVSValue args, void*, ise_t* env)
{
    int diag_count4 = args[1].AsInt(1);
    if (diag_count4 < 1 || diag_count4 > 6) {
        env->ThrowError("DCTFilter4D: diagonals count for 4x4 must be 1 to 6.");
    }

    try {
        return new DCTFilter(
            args[0].AsClip(), nullptr, nullptr, 0, diag_count4, args[2].AsInt(1),
            args[3].AsInt(-1), MODE_4X4, env->FunctionExists("SetFilterMTMode"));
    } catch (std::runtime_error& e) {
        env->ThrowError("DCTFilter4D: %s", e.what());
    }
    return 0;
}


static const AVS_Linkage* AVS_linkage = nullptr;


extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(ise_t* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    const char* args8 =
        "c"                         // 0
        "ffffffff"                  // 1 - 8
        "[x40]f[x41]f[x42]f[x43]f"  // 9 -12
        "[chroma]i[opt]i";          // 13, 14

    const char* args8d = "ci[x4]i[chroma]i[opt]i";

    const char* args4 = "cffff[chroma]i[opt]i";

    const char* args4d = "ci[chroma]i[opt]i";

    env->AddFunction("DCTFilter", args8, create_8, nullptr);
    env->AddFunction("DCTFilterD", args8d, create_8d, nullptr);
    env->AddFunction("DCTFilter8", args8, create_8, nullptr);
    env->AddFunction("DCTFilter8D", args8d, create_8d, nullptr);
    env->AddFunction("DCTFilter4", args4, create_4, nullptr);
    env->AddFunction("DCTFilter4D", args4d, create_4d, nullptr);

    return "a rewite of DctFilter ver." DCT_FILTER_VERSION;
}