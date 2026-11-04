#include "vizion_core.h"

#include <math.h>
#include <string.h>

/* 次方、對數、伽瑪和區間拉伸。先用和 NumPy 相同的 float32 或 float64 算，
   再做偶數捨入；區間拉伸則把小數直接丟掉。 */

void vizion_bits_sse2(const uint8_t* a, const uint8_t* b, uint8_t* dst, int count, int op);

int vizion_gray_bitconst(const uint8_t* src, uint8_t* dst, int count, int op, int amount);

static uint8_t map_trunc_f(float value)
{
    if (!(value > 0.f))
        return 0;
    if (value >= 255.f)
        return 255;
    return (uint8_t)value;
}

static uint8_t map_round_f(float value)
{
    int whole;
    float frac;
    int rounded;
    if (!(value > 0.f))
        return 0;
    if (value >= 255.f)
        return 255;
    whole = (int)value;
    frac = value - (float)whole;
    if (frac > 0.5f)
        rounded = whole + 1;
    else if (frac < 0.5f)
        rounded = whole;
    else
        rounded = (whole & 1) ? whole + 1 : whole;
    if (rounded > 255)
        return 255;
    return (uint8_t)rounded;
}

static uint8_t map_round_d(double value)
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

int vizion_gray_sqrt(const uint8_t* src, uint8_t* dst, int count)
{
    int index;
    if (!src || !dst || count < 1)
        return -1;
    for (index = 0; index < count; index++)
        dst[index] = map_round_f(sqrtf((float)src[index]));
    return 0;
}

int vizion_gray_pow(const uint8_t* src, uint8_t* dst, int count, double exponent)
{
    int index;
    float power;
    if (!src || !dst || count < 1 || !(exponent > 0.0))
        return -1;
    power = (float)exponent;
    for (index = 0; index < count; index++)
        dst[index] = map_round_f(powf((float)src[index] / 255.f, power) * 255.f);
    return 0;
}

int vizion_gray_log(const uint8_t* src, uint8_t* dst, int count, int natural)
{
    int index;
    float scale;
    if (!src || !dst || count < 1)
        return -1;
    scale = 255.f / logf(256.f);
    for (index = 0; index < count; index++) {
        if (!natural)
            dst[index] = map_round_f(logf((float)src[index] + 1.f) * scale);
        else if (src[index] < 1)
            dst[index] = 0;
        else
            dst[index] = map_round_f(logf((float)src[index]));
    }
    return 0;
}

int vizion_gray_exp(const uint8_t* src, uint8_t* dst, int count, int natural)
{
    int index;
    if (!src || !dst || count < 1)
        return -1;
    for (index = 0; index < count; index++) {
        if (!natural)
            dst[index] = map_round_d(pow(256.0, (double)src[index] / 255.0) - 1.0);
        else
            dst[index] = map_round_d(exp((double)src[index]));
    }
    return 0;
}

int vizion_gray_abs(const uint8_t* src, uint8_t* dst, int count)
{
    int index;
    if (!src || !dst || count < 1)
        return -1;
    for (index = 0; index < count; index++) {
        int signed_value = (int)src[index] - 128;
        dst[index] = (uint8_t)(signed_value < 0 ? -signed_value : signed_value);
    }
    return 0;
}

int vizion_gray_gamma(
    const uint8_t* src, uint8_t* dst, int count,
    double gamma, double offset, double threshold, double max_gray, int encode)
{
    float power;
    float shift;
    float floor;
    float ceiling;
    float exponent;
    int index;
    if (!src || !dst || count < 1 || !(gamma > 0.0) || !(max_gray > threshold))
        return -1;
    power = (float)gamma;
    shift = (float)offset;
    floor = (float)threshold;
    ceiling = (float)max_gray;
    exponent = encode ? 1.f / power : power;
    for (index = 0; index < count; index++) {
        float sample = (float)src[index];
        float ratio;
        float mapped;
        if (sample < floor) {
            dst[index] = map_round_f(sample);
            continue;
        }
        ratio = (sample - floor) / (ceiling - floor);
        if (ratio < 0.f)
            ratio = 0.f;
        else if (ratio > 1.f)
            ratio = 1.f;
        mapped = shift + (ceiling - shift) * powf(ratio, exponent);
        dst[index] = map_round_f(mapped);
    }
    return 0;
}

int vizion_gray_range(const uint8_t* src, uint8_t* dst, int count, double low, double high)
{
    float floor;
    float span;
    int index;
    if (!src || !dst || count < 1)
        return -1;
    if (fabs(high - low) < 1e-9) {
        memset(dst, 0, (size_t)count);
        return 0;
    }
    floor = (float)low;
    span = (float)high - floor;
    for (index = 0; index < count; index++)
        dst[index] = map_trunc_f(((float)src[index] - floor) / span * 255.f);
    return 0;
}

void vizion_bits_scalar(const uint8_t* a, const uint8_t* b, uint8_t* dst, int begin, int count, int op)
{
    int index;
    for (index = begin; index < count; index++) {
        uint8_t right = b ? b[index] : 0;
        switch (op) {
        case 0:
            dst[index] = (uint8_t)(a[index] & right);
            break;
        case 1:
            dst[index] = (uint8_t)(a[index] | right);
            break;
        case 2:
            dst[index] = (uint8_t)(a[index] ^ right);
            break;
        default:
            dst[index] = (uint8_t)~a[index];
            break;
        }
    }
}

int vizion_gray_bits(const uint8_t* a, const uint8_t* b, uint8_t* dst, int count, int op)
{
    int flags;
    if (!a || !dst || count < 1 || op < 0 || op > 3)
        return -1;
    if (op != 3 && !b)
        return -1;
    flags = vizion_cpu_flags();
    if ((flags & VIZION_CPU_SELECTED_AVX2) || (flags & VIZION_CPU_SELECTED_SSE2)) {
        vizion_bits_sse2(a, b, dst, count, op);
        return 0;
    }
    vizion_bits_scalar(a, b, dst, 0, count, op);
    return 0;
}

static int map_same(const uint8_t* left, const uint8_t* right, int count)
{
    int index;
    for (index = 0; index < count; index++) {
        if (left[index] != right[index])
            return 0;
    }
    return 1;
}

int vizion_map_selftest(void)
{
    static const uint8_t src[8] = { 0, 1, 2, 81, 128, 200, 254, 255 };
    static const uint8_t sqrt_expect[8] = { 0, 1, 1, 9, 11, 14, 16, 16 };
    static const uint8_t pow_expect[8] = { 0, 0, 0, 26, 64, 157, 253, 255 };
    static const uint8_t log_expect[8] = { 0, 32, 51, 203, 223, 244, 255, 255 };
    static const uint8_t logn_expect[8] = { 0, 0, 1, 4, 5, 5, 6, 6 };
    static const uint8_t exp_expect[8] = { 0, 0, 0, 5, 15, 76, 249, 255 };
    static const uint8_t expn_expect[8] = { 1, 3, 7, 255, 255, 255, 255, 255 };
    static const uint8_t abs_expect[8] = { 128, 127, 126, 47, 0, 72, 126, 127 };
    uint8_t out[8];
    uint8_t left[4] = { 0xF0, 0x0A, 0xFF, 0x00 };
    uint8_t right[4] = { 0x3C, 0x0F, 0x0F, 0xFF };
    uint8_t fast[4];
    uint8_t slow[4];
    uint8_t gamma_out[1];
    uint8_t range_out[1];
    if (vizion_gray_sqrt(src, out, 8) != 0 || !map_same(out, sqrt_expect, 8))
        return 1;
    if (vizion_gray_pow(src, out, 8, 2.0) != 0 || !map_same(out, pow_expect, 8))
        return 2;
    if (vizion_gray_log(src, out, 8, 0) != 0 || !map_same(out, log_expect, 8))
        return 3;
    if (vizion_gray_log(src, out, 8, 1) != 0 || !map_same(out, logn_expect, 8))
        return 4;
    if (vizion_gray_exp(src, out, 8, 0) != 0 || !map_same(out, exp_expect, 8))
        return 5;
    if (vizion_gray_exp(src, out, 8, 1) != 0 || !map_same(out, expn_expect, 8))
        return 6;
    if (vizion_gray_abs(src, out, 8) != 0 || !map_same(out, abs_expect, 8))
        return 7;
    if (vizion_gray_gamma(src + 1, gamma_out, 1, 2.0, 0.0, 0.0, 255.0, 1) != 0 || gamma_out[0] != 16)
        return 8;
    if (vizion_gray_range(src + 4, range_out, 1, 0.0, 63.0) != 0 || range_out[0] != 255)
        return 9;
    if (vizion_gray_bits(left, right, fast, 4, 0) != 0 || fast[0] != 0x30 || fast[1] != 0x0A)
        return 10;
    vizion_bits_sse2(left, right, fast, 4, 2);
    vizion_bits_scalar(left, right, slow, 0, 4, 2);
    if (!map_same(fast, slow, 4) || fast[0] != 0xCC)
        return 11;
    if (vizion_gray_pow(src, out, 8, 0.0) == 0)
        return 12;
    if (vizion_gray_bitconst(src, out, 8, 1, 240) != 0 || out[3] != (81 & 240) || out[7] != (255 & 240))
        return 13;
    if (vizion_gray_bitconst(src, out, 8, 2, 3) != 0 || out[4] != (uint8_t)(128 << 3))
        return 14;
    if (vizion_gray_bitconst(src, out, 8, 3, 2) != 0 || out[5] != (uint8_t)(200 >> 2))
        return 15;
    if (vizion_gray_bitconst(src, out, 8, 4, 0) != 0 || out[0] != 0 || out[1] != 255)
        return 16;
    if (vizion_gray_bitconst(src, out, 8, 2, 8) != 0 || out[4] != 0)
        return 17;
    return 0;
}

int vizion_gray_bitconst(const uint8_t* src, uint8_t* dst, int count, int op, int amount)
{
    int index;
    if (!src || !dst || count < 1 || op < 0 || op > 4)
        return -1;
    if (op == 1 && (amount < 0 || amount > 255))
        return -1;
    if ((op == 2 || op == 3) && amount < 0)
        return -1;
    if (op == 4 && (amount < 0 || amount > 7))
        return -1;
    for (index = 0; index < count; index++) {
        uint8_t value = src[index];
        switch (op) {
        case 0:
            dst[index] = (uint8_t)~value;
            break;
        case 1:
            dst[index] = (uint8_t)(value & (uint8_t)amount);
            break;
        case 2:
            dst[index] = amount >= 8 ? 0 : (uint8_t)(value << amount);
            break;
        case 3:
            dst[index] = amount >= 8 ? 0 : (uint8_t)(value >> amount);
            break;
        default:
            dst[index] = (value & (uint8_t)(1 << amount)) ? 255 : 0;
            break;
        }
    }
    return 0;
}
