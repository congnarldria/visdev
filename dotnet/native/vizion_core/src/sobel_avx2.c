#include "vizion_core.h"

#if defined(_MSC_VER)
#  include <immintrin.h>
#else
#  include <immintrin.h>
#endif

void vizion_sobel_pixel(const uint8_t* gray, int width, int height, int row, int column, float* dx, float* dy);
void vizion_sobel_mag(const float* gx, const float* gy, float* mag, int count);

static __m256 vizion_u8_ps8(const uint8_t* pixel)
{
    __m128i bytes = _mm_loadl_epi64((const __m128i*)pixel);
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
}

static void vizion_sobel_row8(const uint8_t* gray, int width, int row, int col, float* gx, float* gy)
{
    const uint8_t* up = gray + (size_t)(row - 1) * (size_t)width + col;
    const uint8_t* mid = gray + (size_t)row * (size_t)width + col;
    const uint8_t* down = gray + (size_t)(row + 1) * (size_t)width + col;
    __m256 two = _mm256_set1_ps(2.f);
    __m256 up_left = vizion_u8_ps8(up - 1);
    __m256 up_mid = vizion_u8_ps8(up);
    __m256 up_right = vizion_u8_ps8(up + 1);
    __m256 mid_left = vizion_u8_ps8(mid - 1);
    __m256 mid_right = vizion_u8_ps8(mid + 1);
    __m256 down_left = vizion_u8_ps8(down - 1);
    __m256 down_mid = vizion_u8_ps8(down);
    __m256 down_right = vizion_u8_ps8(down + 1);
    __m256 dx = _mm256_add_ps(_mm256_sub_ps(up_right, up_left), _mm256_sub_ps(down_right, down_left));
    __m256 dy = _mm256_sub_ps(
        _mm256_add_ps(down_left, _mm256_add_ps(_mm256_mul_ps(down_mid, two), down_right)),
        _mm256_add_ps(up_left, _mm256_add_ps(_mm256_mul_ps(up_mid, two), up_right)));
    dx = _mm256_add_ps(dx, _mm256_mul_ps(_mm256_sub_ps(mid_right, mid_left), two));
    _mm256_storeu_ps(gx + (size_t)row * (size_t)width + col, dx);
    _mm256_storeu_ps(gy + (size_t)row * (size_t)width + col, dy);
}

int vizion_sobel_avx2(const uint8_t* gray, int width, int height, float* gx, float* gy, float* mag)
{
    int row;
    int col;
    for (col = 0; col < width; col++) {
        float dx, dy;
        vizion_sobel_pixel(gray, width, height, 0, col, &dx, &dy);
        gx[col] = dx;
        gy[col] = dy;
        if (height > 1) {
            vizion_sobel_pixel(gray, width, height, height - 1, col, &dx, &dy);
            gx[(size_t)(height - 1) * (size_t)width + col] = dx;
            gy[(size_t)(height - 1) * (size_t)width + col] = dy;
        }
    }
    for (row = 1; row < height - 1; row++) {
        float dx, dy;
        vizion_sobel_pixel(gray, width, height, row, 0, &dx, &dy);
        gx[(size_t)row * (size_t)width] = dx;
        gy[(size_t)row * (size_t)width] = dy;
        if (width > 1) {
            vizion_sobel_pixel(gray, width, height, row, width - 1, &dx, &dy);
            gx[(size_t)row * (size_t)width + (width - 1)] = dx;
            gy[(size_t)row * (size_t)width + (width - 1)] = dy;
        }
    }
    for (row = 1; row < height - 1; row++) {
        col = 1;
        for (; col + 8 <= width - 1; col += 8)
            vizion_sobel_row8(gray, width, row, col, gx, gy);
        for (; col < width - 1; col++) {
            float dx, dy;
            vizion_sobel_pixel(gray, width, height, row, col, &dx, &dy);
            gx[(size_t)row * (size_t)width + col] = dx;
            gy[(size_t)row * (size_t)width + col] = dy;
        }
    }
    vizion_sobel_mag(gx, gy, mag, width * height);
    return 0;
}
