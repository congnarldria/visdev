#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Kirsch、Frei-Chen、Robinson、方向角、局部中距、正弦餘弦正切。
   邊緣遮罩用 OpenCV 的相關、邊界重複邊緣。中距的邊界是 REFLECT_101。 */

int vizion_reflect101(int position, int length);

static uint8_t compass_round(double value)
{
    int whole;
    double frac;
    int rounded;
    if (!(value > 0.0))
        return 0;
    if (value >= 255.0)
        return 255;
    whole = (int)value;
    frac = value - (double)whole;
    if (frac > 0.5)
        rounded = whole + 1;
    else if (frac < 0.5)
        rounded = whole;
    else
        rounded = (whole & 1) ? whole + 1 : whole;
    if (rounded > 255)
        return 255;
    return (uint8_t)rounded;
}

static int compass_replicate(int position, int length)
{
    if (position < 0)
        return 0;
    if (position >= length)
        return length - 1;
    return position;
}

static float compass_conv(const float* src, int width, int height, int row, int col, const float* kernel)
{
    float sum = 0.f;
    int dy;
    for (dy = 0; dy < 3; dy++) {
        int yy = compass_replicate(row + dy - 1, height);
        int dx;
        for (dx = 0; dx < 3; dx++) {
            int xx = compass_replicate(col + dx - 1, width);
            sum += src[(size_t)yy * (size_t)width + xx] * kernel[dy * 3 + dx];
        }
    }
    return sum;
}

static const float compass_kirsch_kernel[8][9] = {
    { 5, 5, 5, -3, 0, -3, -3, -3, -3 },
    { 5, 5, -3, 5, 0, -3, -3, -3, -3 },
    { 5, -3, -3, 5, 0, -3, 5, -3, -3 },
    { -3, -3, -3, 5, 0, -3, 5, 5, -3 },
    { -3, -3, -3, -3, 0, -3, 5, 5, 5 },
    { -3, -3, -3, -3, 0, 5, -3, 5, 5 },
    { -3, -3, 5, -3, 0, 5, -3, -3, 5 },
    { -3, 5, 5, -3, 0, 5, -3, -3, -3 },
};

static const float compass_robinson_kernel[8][9] = {
    { -1, 0, 1, -2, 0, 2, -1, 0, 1 },
    { 0, 1, 2, -1, 0, 1, -2, -1, 0 },
    { 1, 2, 1, 0, 0, 0, -1, -2, -1 },
    { 2, 1, 0, 1, 0, -1, 0, -1, -2 },
    { 1, 0, -1, 2, 0, -2, 1, 0, -1 },
    { 0, -1, -2, 1, 0, -1, 2, 1, 0 },
    { -1, -2, -1, 0, 0, 0, 1, 2, 1 },
    { -2, -1, 0, -1, 0, 1, 0, 1, 2 },
};

static float vizion_frei_kernel[4][9];
static int vizion_frei_ready;

static void vizion_frei_init(void)
{
    static const float pattern[4][9] = {
        { 1, 0, 1, 0, 0, 0, -1, 0, -1 },
        { 1, 0, -1, 0, 0, 0, 1, 0, -1 },
        { 0, -1, 0, 1, 0, -1, 0, 1, 0 },
        { 0, -1, 0, -1, 0, 1, 0, 1, 0 },
    };
    static const int root_slot[4][9] = {
        { 0, 1, 0, 0, 0, 0, 0, -1, 0 },
        { 0, 0, 0, 1, 0, -1, 0, 0, 0 },
        { 0, 0, 1, 0, 0, 0, -1, 0, 0 },
        { 1, 0, 0, 0, 0, 0, 0, 0, -1 },
    };
    float root;
    float scale;
    int mask;
    int tap;
    if (vizion_frei_ready)
        return;
    root = (float)sqrt(2.0);
    scale = (float)(1.0 / (2.0 * sqrt(2.0)));
    for (mask = 0; mask < 4; mask++) {
        for (tap = 0; tap < 9; tap++) {
            float value = pattern[mask][tap];
            if (root_slot[mask][tap] > 0)
                value = root;
            else if (root_slot[mask][tap] < 0)
                value = -root;
            vizion_frei_kernel[mask][tap] = scale * value;
        }
    }
    vizion_frei_ready = 1;
}

static int compass_plane(const uint8_t* src, float* plane, int count)
{
    int index;
    if (!src || !plane || count < 1)
        return -1;
    for (index = 0; index < count; index++)
        plane[index] = (float)src[index];
    return 0;
}

int vizion_kirsch(const uint8_t* src, uint8_t* dst, int width, int height)
{
    float* plane;
    int count;
    int row;
    int col;
    if (!src || !dst || width < 1 || height < 1)
        return -1;
    count = width * height;
    plane = (float*)malloc((size_t)count * sizeof(float));
    if (!plane || compass_plane(src, plane, count) != 0) {
        free(plane);
        return -1;
    }
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float best = compass_conv(plane, width, height, row, col, compass_kirsch_kernel[0]);
            int mask;
            for (mask = 1; mask < 8; mask++) {
                float response = compass_conv(plane, width, height, row, col, compass_kirsch_kernel[mask]);
                if (response > best)
                    best = response;
            }
            dst[(size_t)row * (size_t)width + col] = compass_round((double)best / 15.0);
        }
    }
    free(plane);
    return 0;
}

int vizion_frei(const uint8_t* src, uint8_t* dst, int width, int height)
{
    float* plane;
    int count;
    int row;
    int col;
    if (!src || !dst || width < 1 || height < 1)
        return -1;
    vizion_frei_init();
    count = width * height;
    plane = (float*)malloc((size_t)count * sizeof(float));
    if (!plane) return -1;
    compass_plane(src, plane, count);
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float energy = 0.f;
            int mask;
            for (mask = 0; mask < 4; mask++) {
                float response = compass_conv(plane, width, height, row, col, vizion_frei_kernel[mask]);
                energy += response * response;
            }
            dst[(size_t)row * (size_t)width + col] = compass_round((double)sqrtf(energy));
        }
    }
    free(plane);
    return 0;
}

static void compass_robinson_responses(const float* plane, int width, int height, int row, int col, float out[8])
{
    int mask;
    for (mask = 0; mask < 8; mask++)
        out[mask] = compass_conv(plane, width, height, row, col, compass_robinson_kernel[mask]);
}

int vizion_robinson_amp(const uint8_t* src, uint8_t* dst, int width, int height)
{
    float* plane;
    int count;
    int row;
    int col;
    if (!src || !dst || width < 1 || height < 1)
        return -1;
    count = width * height;
    plane = (float*)malloc((size_t)count * sizeof(float));
    if (!plane) return -1;
    compass_plane(src, plane, count);
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float response[8];
            float best;
            int mask;
            compass_robinson_responses(plane, width, height, row, col, response);
            best = fabsf(response[0]);
            for (mask = 1; mask < 8; mask++) {
                float magnitude = fabsf(response[mask]);
                if (magnitude > best)
                    best = magnitude;
            }
            dst[(size_t)row * (size_t)width + col] = compass_round((double)best);
        }
    }
    free(plane);
    return 0;
}

int vizion_robinson_dir(const uint8_t* src, uint8_t* dst, int width, int height)
{
    float* plane;
    int count;
    int row;
    int col;
    if (!src || !dst || width < 1 || height < 1)
        return -1;
    count = width * height;
    plane = (float*)malloc((size_t)count * sizeof(float));
    if (!plane) return -1;
    compass_plane(src, plane, count);
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float response[8];
            int best_index = 0;
            float best;
            int mask;
            double angle;
            compass_robinson_responses(plane, width, height, row, col, response);
            best = response[0];
            for (mask = 1; mask < 8; mask++) {
                if (response[mask] > best) {
                    best = response[mask];
                    best_index = mask;
                }
            }
            angle = (double)best_index * (3.14159265358979323846 / 4.0);
            if (best < 0.f)
                angle += 3.14159265358979323846;
            angle = atan2(sin(angle), cos(angle));
            dst[(size_t)row * (size_t)width + col] = compass_round((angle + 3.14159265358979323846) / (2.0 * 3.14159265358979323846) * 255.0);
        }
    }
    free(plane);
    return 0;
}

int vizion_atan2_gray(const uint8_t* vertical, const uint8_t* horizontal, uint8_t* dst, int count)
{
    int index;
    if (!vertical || !horizontal || !dst || count < 1)
        return -1;
    for (index = 0; index < count; index++) {
        double angle = atan2((double)vertical[index] - 128.0, (double)horizontal[index] - 128.0);
        dst[index] = compass_round((angle + 3.14159265358979323846) / (2.0 * 3.14159265358979323846) * 255.0);
    }
    return 0;
}

int vizion_midrange(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height)
{
    int anchor_x;
    int anchor_y;
    int row;
    int col;
    if (!src || !dst || width < 1 || height < 1 || mask_width < 1 || mask_height < 1)
        return -1;
    if ((mask_width & 1) == 0 || (mask_height & 1) == 0)
        return -1;
    anchor_x = mask_width / 2;
    anchor_y = mask_height / 2;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            int darkest = 255;
            int brightest = 0;
            int dy;
            for (dy = 0; dy < mask_height; dy++) {
                int yy = vizion_reflect101(row + dy - anchor_y, height);
                int dx;
                for (dx = 0; dx < mask_width; dx++) {
                    int xx = vizion_reflect101(col + dx - anchor_x, width);
                    int value = src[(size_t)yy * (size_t)width + xx];
                    if (value < darkest)
                        darkest = value;
                    if (value > brightest)
                        brightest = value;
                }
            }
            dst[(size_t)row * (size_t)width + col] = compass_round(((double)darkest + (double)brightest) / 2.0);
        }
    }
    return 0;
}

int vizion_wave(const uint8_t* src, uint8_t* dst, int count, int op)
{
    int index;
    if (!src || !dst || count < 1 || op < 0 || op > 2)
        return -1;
    for (index = 0; index < count; index++) {
        double angle = (double)src[index];
        double waved;
        if (op == 0)
            waved = (sin(angle) + 1.0) * 127.5;
        else if (op == 1)
            waved = (cos(angle) + 1.0) * 127.5;
        else {
            waved = tan(angle);
            if (waved < -16.0)
                waved = -16.0;
            if (waved > 16.0)
                waved = 16.0;
            waved = (waved + 16.0) / 32.0 * 255.0;
        }
        dst[index] = compass_round(waved);
    }
    return 0;
}

static int compass_same(const uint8_t* left, const uint8_t* right, int count)
{
    int index;
    for (index = 0; index < count; index++) {
        if (left[index] != right[index])
            return 0;
    }
    return 1;
}

int vizion_compass_selftest(void)
{
    static const uint8_t src[9] = { 1, 9, 3, 8, 2, 7, 4, 6, 5 };
    static const uint8_t kirsch[9] = { 3, 2, 3, 2, 2, 1, 2, 1, 2 };
    static const uint8_t frei[9] = { 8, 8, 5, 7, 4, 5, 4, 4, 3 };
    static const uint8_t robinson[9] = { 18, 7, 13, 7, 4, 3, 8, 3, 3 };
    static const uint8_t direction[9] = { 128, 96, 255, 96, 96, 128, 191, 159, 159 };
    static const uint8_t mid[9] = { 5, 5, 6, 5, 5, 6, 5, 5, 4 };
    static const uint8_t wave_src[6] = { 0, 1, 2, 64, 128, 255 };
    static const uint8_t sine[6] = { 128, 235, 243, 245, 219, 63 };
    static const uint8_t cosine[6] = { 255, 196, 74, 177, 39, 18 };
    static const uint8_t tangent[6] = { 128, 140, 110, 146, 119, 132 };
    uint8_t vertical[4] = { 128, 128, 0, 255 };
    uint8_t horizontal[4] = { 128, 255, 128, 128 };
    static const uint8_t angles[4] = { 128, 128, 64, 191 };
    uint8_t out[9];
    uint8_t wave[6];
    uint8_t turned[4];
    if (vizion_kirsch(src, out, 3, 3) != 0 || !compass_same(out, kirsch, 9))
        return 1;
    if (vizion_frei(src, out, 3, 3) != 0 || !compass_same(out, frei, 9))
        return 2;
    if (vizion_robinson_amp(src, out, 3, 3) != 0 || !compass_same(out, robinson, 9))
        return 3;
    if (vizion_robinson_dir(src, out, 3, 3) != 0 || !compass_same(out, direction, 9))
        return 4;
    if (vizion_midrange(src, out, 3, 3, 3, 3) != 0 || !compass_same(out, mid, 9))
        return 5;
    if (vizion_atan2_gray(vertical, horizontal, turned, 4) != 0 || !compass_same(turned, angles, 4))
        return 6;
    if (vizion_wave(wave_src, wave, 6, 0) != 0 || !compass_same(wave, sine, 6))
        return 7;
    if (vizion_wave(wave_src, wave, 6, 1) != 0 || !compass_same(wave, cosine, 6))
        return 8;
    if (vizion_wave(wave_src, wave, 6, 2) != 0 || !compass_same(wave, tangent, 6))
        return 9;
    return 0;
}
