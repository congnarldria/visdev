#include "vizion_core.h"

#include <string.h>

/* OpenCV COLOR_BGR2GRAY：(R*9798 + G*19235 + B*3735 + 16384) >> 15。記憶體順序是 B、G、R。 */

void vizion_bgr_sse2(const uint8_t* bgr, uint8_t* gray, int count);
void vizion_bgr_avx2(const uint8_t* bgr, uint8_t* gray, int count);

void vizion_bgr_scalar(const uint8_t* bgr, uint8_t* gray, int begin, int count)
{
    int index;
    for (index = begin; index < count; index++) {
        const uint8_t* pixel = bgr + (size_t)index * 3;
        int value = (pixel[2] * 9798 + pixel[1] * 19235 + pixel[0] * 3735 + 16384) >> 15;
        gray[index] = (uint8_t)value;
    }
}

int vizion_bgr_gray(const uint8_t* bgr, int count, uint8_t* gray)
{
    int flags;
    if (!bgr || !gray || count < 1)
        return -1;
    flags = vizion_cpu_flags();
    if (flags & VIZION_CPU_SELECTED_AVX2) {
        vizion_bgr_avx2(bgr, gray, count);
        return 0;
    }
    if (flags & VIZION_CPU_SELECTED_SSE2) {
        vizion_bgr_sse2(bgr, gray, count);
        return 0;
    }
    vizion_bgr_scalar(bgr, gray, 0, count);
    return 0;
}

int vizion_bgr_selftest(void)
{
    uint8_t locked[15] = {
        10, 20, 30,
        255, 128, 0,
        1, 2, 3,
        0, 0, 255,
        68, 0, 153
    };
    uint8_t expect[5] = {22, 104, 2, 76, 53};
    uint8_t got[5];
    uint8_t image[64 * 3];
    uint8_t scalar[64];
    uint8_t fast[64];
    int i;
    int flags = vizion_cpu_flags();
    vizion_bgr_scalar(locked, got, 0, 5);
    if (memcmp(got, expect, 5) != 0)
        return 1;
    for (i = 0; i < 64 * 3; i++)
        image[i] = (uint8_t)((i * 17 + 9) & 255);
    vizion_bgr_scalar(image, scalar, 0, 64);
    if (flags & VIZION_CPU_SSE2) {
        memset(fast, 0, sizeof(fast));
        vizion_bgr_sse2(image, fast, 64);
        if (memcmp(scalar, fast, 64) != 0)
            return 2;
        memset(fast, 0, sizeof(fast));
        vizion_bgr_sse2(image, fast, 13);
        vizion_bgr_scalar(image, scalar, 0, 13);
        if (memcmp(scalar, fast, 13) != 0)
            return 3;
    }
    if (flags & VIZION_CPU_AVX2) {
        vizion_bgr_scalar(image, scalar, 0, 64);
        memset(fast, 0, sizeof(fast));
        vizion_bgr_avx2(image, fast, 64);
        if (memcmp(scalar, fast, 64) != 0)
            return 4;
        memset(fast, 0, sizeof(fast));
        vizion_bgr_avx2(image, fast, 17);
        vizion_bgr_scalar(image, scalar, 0, 17);
        if (memcmp(scalar, fast, 17) != 0)
            return 5;
    }
    if (vizion_bgr_gray(locked, 5, got) != 0 || memcmp(got, expect, 5) != 0)
        return 6;
    return 0;
}
