#include "vizion_core.h"

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <emmintrin.h>
#endif

#include <string.h>

/* 圖外已經補進 pad。每個前景位移對一整列做最小或最大。 */

void vizion_plane_minmax_sse2(
    const uint8_t* pad, int pad_width, uint8_t* dst, int width, int height,
    const uint8_t* kernel, int kernel_height, int kernel_width, int dilate)
{
    int ky;
    int kx;
    int row;
    int col;
    memset(dst, dilate ? 0 : 255, (size_t)width * (size_t)height);
    for (ky = 0; ky < kernel_height; ky++) {
        for (kx = 0; kx < kernel_width; kx++) {
            if (!kernel[ky * kernel_width + kx])
                continue;
            for (row = 0; row < height; row++) {
                const uint8_t* line = pad + ((size_t)row + (size_t)ky) * (size_t)pad_width + kx;
                uint8_t* out = dst + (size_t)row * (size_t)width;
                for (col = 0; col + 16 <= width; col += 16) {
                    __m128i sample = _mm_loadu_si128((const __m128i*)(line + col));
                    __m128i cur = _mm_loadu_si128((const __m128i*)(out + col));
                    __m128i mixed = dilate ? _mm_max_epu8(cur, sample) : _mm_min_epu8(cur, sample);
                    _mm_storeu_si128((__m128i*)(out + col), mixed);
                }
                for (; col < width; col++) {
                    uint8_t sample = line[col];
                    if (dilate) {
                        if (sample > out[col])
                            out[col] = sample;
                    } else if (sample < out[col]) {
                        out[col] = sample;
                    }
                }
            }
        }
    }
}

