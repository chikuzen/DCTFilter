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
#include <avisynth.h>
#include <avs/win.h>
#include <avs/alignment.h>

#define DCT_FILTER_VERSION "0.0.0"


typedef IScriptEnvironment ise_t;

typedef void(*dct_idct_func_t)(
    const uint8_t*, uint8_t*, int, int, int, int, float*, const float*, int);

extern dct_idct_func_t get_main_proc(int component_size);


class DCTFilter : public GenericVideoFilter {
    int numPlanes;
    int planes[3];
    int chroma;
    float* factors;

    dct_idct_func_t mainProc;

public:
    DCTFilter(PClip child, double* factor, int chroma);
    PVideoFrame __stdcall GetFrame(int n, ise_t* env);
    ~DCTFilter();
    int __stdcall SetCacheHints(int hints, int)
    {
        return hints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
};


DCTFilter::DCTFilter(PClip c, double* f, int ch)
        : GenericVideoFilter(c), chroma(ch)
{
    if (!vi.IsPlanar()) {
        throw std::runtime_error("input is not planar format.");
    }

    if ((vi.width | vi.height) & 7) {
        throw std::runtime_error(
            "first plane's width and height must be mod 8.");
    }

    numPlanes = (vi.pixel_type & VideoInfo::CS_INTERLEAVED) ? 1 : 3;

    if (numPlanes == 1 || chroma < 0 || chroma > 2) {
        chroma = 2;
    }

    if (chroma == 1) {
        int w = vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
        int h = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);
        if ((w | h) & 7) {
            throw std::runtime_error(
                "second plane's width and height must be mod 8.");
        }
    }

    factors = reinterpret_cast<float*>(avs_malloc(64 * sizeof(float), 32));
    if (!factors) {
        throw std::runtime_error("failed to create table of factors.");
    }
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            factors[y * 8 + x] = static_cast<float>(f[y] * f[x] * 0.1250);
        }
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

    mainProc = get_main_proc(vi.ComponentSize());
}


DCTFilter::~DCTFilter()
{
    avs_free(factors);
    factors = nullptr;
}


PVideoFrame __stdcall DCTFilter::GetFrame(int n, ise_t* env)
{
    auto env2 = static_cast<IScriptEnvironment2*>(env);

    auto src = child->GetFrame(n, env);
    auto dst = env->NewVideoFrame(vi, 32);

    auto buff = reinterpret_cast<float*>(
            env2->Allocate(128 * sizeof(float), 32, AVS_POOLED_ALLOC));
    if (!buff) {
        env->ThrowError("DCTFilter: failed to allocate temporal buffer.");
    }

    for (int p = 0; p < numPlanes; ++p) {
        const int plane = planes[p];

        mainProc(src->GetReadPtr(plane), dst->GetWritePtr(plane),
                 src->GetPitch(plane), dst->GetPitch(plane),
                 src->GetRowSize(plane), src->GetHeight(plane), buff, factors,
                 //vi.BitsPerComponent());
                 vi.ComponentSize() * 8);
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

    env2->Free(buff);

    return dst;
}

static AVSValue __cdecl create(AVSValue args, void*, ise_t* env)
{
    double f[8];
    for (int i = 0; i < 8; ++i) {
        f[i] = args[i + 1].AsFloat();
        if (f[i] < 0.0 || f[i] > 1.0) {
            env->ThrowError(
                "DCTFilter: all scaling factors must be between 0.0 and 1.0.");
        }
    }

    try {
        return new DCTFilter(args[0].AsClip(), f, args[9].AsInt(1));
    } catch (std::runtime_error& e) {
        env->ThrowError("DCTFilter: %s", e.what());
    }
    return AVSValue();
}


static const AVS_Linkage* AVS_linkage = nullptr;


extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(ise_t* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("DCTFilter", "cffffffff[chroma]i", create, nullptr);

    return "a rewite of DctFilter for Avisynth+ ver." DCT_FILTER_VERSION;
}