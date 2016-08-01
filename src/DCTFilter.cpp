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
#include <malloc.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NOGDI
#define VC_EXTRALEAN
#include <windows.h>
#include <avisynth.h>



#define DCT_FILTER_VERSION "0.3.0"


typedef IScriptEnvironment ise_t;

typedef void(*dct_idct_func_t)(
    const uint8_t*, uint8_t*, int, int, int, int, float*, const float*, int);

extern dct_idct_func_t get_main_proc(int component_size, int opt);

extern bool has_sse2();
extern bool has_sse41();
extern bool has_avx2();


class DCTFilter : public GenericVideoFilter {
    int numPlanes;
    int planes[3];
    int chroma;
    bool isPlus;
    float* factors;

    dct_idct_func_t mainProc;

public:
    DCTFilter(PClip child, double* factor, int diag_count, int chroma, int opt,
              bool is_plus);
    PVideoFrame __stdcall GetFrame(int n, ise_t* env);
    ~DCTFilter();
    int __stdcall SetCacheHints(int hints, int)
    {
        return hints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
};


DCTFilter::DCTFilter(PClip c, double* f, int diag_count, int ch, int opt,
                     bool ip)
        : GenericVideoFilter(c), chroma(ch), isPlus(ip)
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

    if (chroma == 1) {
        int w = vi.width >> vi.GetPlaneWidthSubsampling(planes[1]);
        int h = vi.height >> vi.GetPlaneHeightSubsampling(planes[1]);
        if ((w | h) & 7) {
            throw std::runtime_error(
                "second plane's width and height must be mod 8.");
        }
    }

    size_t size = (64 + (isPlus ? 0 : 128)) * sizeof(float);
    factors = reinterpret_cast<float*>(_aligned_malloc(size, 32));
    if (!factors) {
        throw std::runtime_error("failed to create table of factors.");
    }

    if (diag_count == 0) {
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                factors[y * 8 + x] = static_cast<float>(f[y] * f[x] * 0.1250);
            }
        }
    } else {
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                factors[y * 8 + x] = (x + y) > (14 - diag_count) ? 0 : 0.1250f;
            }
        }
    }

    if (opt == 0 || !has_sse2()) {
        opt = 0;
    } else if (opt == 1 || !has_sse41()) {
        opt = 1;
    } else if (opt == 2 || !has_avx2()){
        opt = 2;
    } else {
        opt = 3;
    }

    mainProc = get_main_proc(vi.BytesFromPixels(1), opt);
}


DCTFilter::~DCTFilter()
{
    _aligned_free(factors);
    factors = nullptr;
}


PVideoFrame __stdcall DCTFilter::GetFrame(int n, ise_t* env)
{
    auto src = child->GetFrame(n, env);
    auto dst = env->NewVideoFrame(vi, 32);

    float* buff;
    if (!isPlus) {
        buff = factors + 64;
    } else {
        buff = reinterpret_cast<float*>(static_cast<IScriptEnvironment2*>(
            env)->Allocate(128 * sizeof(float), 32, AVS_POOLED_ALLOC));
        if (!buff) {
            env->ThrowError("DCTFilter: failed to allocate temporal buffer.");
        }
    }

    for (int p = 0; p < numPlanes; ++p) {
        const int plane = planes[p];

        mainProc(src->GetReadPtr(plane), dst->GetWritePtr(plane),
                 src->GetPitch(plane), dst->GetPitch(plane),
                 src->GetRowSize(plane), src->GetHeight(plane), buff, factors,
                 //vi.BitsPerComponent());
                 vi.BytesFromPixels(1) * 8);
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
        return new DCTFilter(
            args[0].AsClip(), f, 0, args[9].AsInt(1), args[10].AsInt(-1),
            env->FunctionExists("SetFilterMTMode"));
    } catch (std::runtime_error& e) {
        env->ThrowError("DCTFilter: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl create_d(AVSValue args, void*, ise_t* env)
{
    int diag_count = args[1].AsInt();
    if (diag_count < 1 || diag_count > 14) {
        env->ThrowError("DCTFilterD: diagonals count must be 1 to 14.");
    }

    try {
        return new DCTFilter(
            args[0].AsClip(), nullptr, diag_count, args[2].AsInt(1),
            args[3].AsInt(-1), env->FunctionExists("SetFilterMTMode"));
    } catch (std::runtime_error& e) {
        env->ThrowError("DCTFilterD: %s", e.what());
    }
    return 0;
}


static const AVS_Linkage* AVS_linkage = nullptr;


extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(ise_t* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("DCTFilter", "cffffffff[chroma]i[opt]i", create, nullptr);
    env->AddFunction("DCTFilterD", "ci[chroma]i[opt]i", create_d, nullptr);

    return "a rewite of DctFilter ver." DCT_FILTER_VERSION;
}