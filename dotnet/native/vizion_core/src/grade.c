#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int vizion_reflect101(int position, int length);
void vizion_sep_quad_sse2(const float* src, float* dst, int col, const float* kernel, int length, int anchor);

/* 可分離浮點相關，邊界是 REFLECT_101。係數用 float32，乘加不合併成融合乘加。 */

static int grade_bankers(double value)
{
    double whole = floor(value);
    double frac = value - whole;
    int base = (int)whole;
    if (frac > 0.5)
        return base + 1;
    if (frac < 0.5)
        return base;
    return (base & 1) ? base + 1 : base;
}

uint8_t grade_round_d(double value)
{
    int whole;
    double frac;
    int rounded;
    if (!(value > 0.0))
        return 0;
    if (value >= 255.0)
        return 255;
    whole = (int)value;
    frac = value - whole;
    rounded = frac > 0.5 ? whole + 1 : frac < 0.5 ? whole : ((whole & 1) ? whole + 1 : whole);
    return rounded > 255 ? (uint8_t)255 : (uint8_t)rounded;
}

uint8_t grade_round(float value)
{
    return grade_round_d((double)value);
}

static int grade_fast(void)
{
    int flags = vizion_cpu_flags();
    return (flags & VIZION_CPU_SELECTED_AVX2) || (flags & VIZION_CPU_SELECTED_SSE2);
}

static void grade_conv_1d(const float* src, int length, int stride, float* dst, int count, const float* kernel, int ksize, int use_sse)
{
    int anchor = ksize / 2;
    int guard = ksize - 1 - anchor;
    int index = 0;
    if (use_sse && stride == 1) {
        for (; index + 4 <= count; index += 4) {
            if (index < anchor || index + 3 + guard >= count)
                break;
            vizion_sep_quad_sse2(src, dst, index, kernel, ksize, anchor);
        }
    }
    for (; index < count; index++) {
        int begin = index - anchor;
        float acc = 0.f;
        int tap;
        for (tap = 0; tap < ksize; tap++) {
            int sample = vizion_reflect101(begin + tap, length);
            acc += src[(size_t)sample * (size_t)stride] * kernel[tap];
        }
        dst[index] = acc;
    }
}

int grade_sep(const float* src, float* dst, int width, int height, const float* across, int across_n, const float* along, int along_n, int use_sse)
{
    float* horizontal;
    float* column;
    float* filtered;
    int row;
    int col;
    horizontal = (float*)malloc((size_t)width * (size_t)height * sizeof(float));
    column = (float*)malloc((size_t)height * sizeof(float));
    filtered = (float*)malloc((size_t)height * sizeof(float));
    if (!horizontal || !column || !filtered) {
        free(horizontal);
        free(column);
        free(filtered);
        return -1;
    }
    for (row = 0; row < height; row++)
        grade_conv_1d(src + (size_t)row * (size_t)width, width, 1, horizontal + (size_t)row * (size_t)width, width, across, across_n, use_sse);
    for (col = 0; col < width; col++) {
        for (row = 0; row < height; row++)
            column[row] = horizontal[(size_t)row * (size_t)width + col];
        grade_conv_1d(column, height, 1, filtered, height, along, along_n, use_sse);
        for (row = 0; row < height; row++)
            dst[(size_t)row * (size_t)width + col] = filtered[row];
    }
    free(horizontal);
    free(column);
    free(filtered);
    return 0;
}

void grade_gauss_kernel(double sigma, int ksize, float* kernel)
{
    double scale = -0.5 / (sigma * sigma);
    double half = (ksize - 1) * 0.5;
    double sum = 0.0;
    double raw[31];
    int i;
    for (i = 0; i < ksize; i++) {
        double x = (double)i - half;
        raw[i] = exp(scale * x * x);
        sum += raw[i];
    }
    for (i = 0; i < ksize; i++)
        kernel[i] = (float)(raw[i] / sum);
}

int grade_blur_side(double sigma)
{
    int side;
    if (!(sigma > 0.0))
        return 0;
    side = grade_bankers(sigma * 6.0) | 1;
    if (side < 3)
        side = 3;
    if (side > 31)
        side = 31;
    return side;
}

int grade_deriv_kernels(double sigma, float* smooth, float* first, float* second, int* count)
{
    int radius;
    int i;
    double variance;
    double raw[65];
    double sum = 0.0;
    if (!(sigma > 0.0))
        return -1;
    radius = (int)ceil(sigma * 4.0);
    if (radius < 1)
        radius = 1;
    if (radius > 32)
        return -1;
    variance = sigma * sigma;
    *count = radius * 2 + 1;
    for (i = -radius; i <= radius; i++) {
        double x = (double)i;
        raw[i + radius] = exp(-0.5 * x * x / variance);
        sum += raw[i + radius];
    }
    for (i = -radius; i <= radius; i++) {
        double x = (double)i;
        double weight = raw[i + radius] / sum;
        int at = i + radius;
        smooth[at] = (float)weight;
        first[at] = (float)((x / variance) * weight);
        second[at] = (float)(((x * x / (variance * variance)) - (1.0 / variance)) * weight);
    }
    return 0;
}

/* x 沿行、y 沿列的可分離高斯導數。有號結果以 128 為 0，gradient 和 none 不加 128。 */

int vizion_deriv_gauss(const uint8_t* src, uint8_t* dst, int width, int height, double sigma, int component)
{
    float smooth[65];
    float first[65];
    float second[65];
    int count = 0;
    int pixels;
    int index;
    int fast;
    float* gray;
    float* across;
    float* along;
    float* other;
    const float* kx;
    const float* ky;
    if (!src || !dst || width < 1 || height < 1 || component < 0 || component > 7)
        return -1;
    if (grade_deriv_kernels(sigma, smooth, first, second, &count) != 0)
        return -1;
    pixels = width * height;
    gray = (float*)malloc((size_t)pixels * sizeof(float) * 3);
    if (!gray)
        return -1;
    across = gray + pixels;
    along = across + pixels;
    for (index = 0; index < pixels; index++)
        gray[index] = (float)src[index];
    fast = grade_fast();
    kx = smooth;
    ky = smooth;
    if (component == 1 || component == 5 || component == 6)
        kx = first;
    else if (component == 3 || component == 7)
        kx = second;
    if (component == 2 || component == 5 || component == 6)
        ky = first;
    else if (component == 4 || component == 7)
        ky = second;
    if (component == 6) {
        kx = first;
        ky = smooth;
    }
    if (component == 7) {
        kx = second;
        ky = smooth;
    }
    if (grade_sep(gray, across, width, height, kx, count, ky, count, fast) != 0) {
        free(gray);
        return -1;
    }
    if (component == 6 || component == 7) {
        if (component == 6) {
            if (grade_sep(gray, along, width, height, smooth, count, first, count, fast) != 0) {
                free(gray);
                return -1;
            }
        } else if (grade_sep(gray, along, width, height, smooth, count, second, count, fast) != 0) {
            free(gray);
            return -1;
        }
    }
    other = along;
    for (index = 0; index < pixels; index++) {
        float response = across[index];
        if (component == 6)
            response = sqrtf(across[index] * across[index] + other[index] * other[index]);
        else if (component == 7)
            response += other[index];
        if (component != 0 && component != 6)
            response += 128.f;
        dst[index] = grade_round(response);
    }
    free(gray);
    return 0;
}

static const float grade_laws3[3][3] = {
    {1.f, 2.f, 1.f},
    {-1.f, 0.f, 1.f},
    {-1.f, 2.f, -1.f},
};
static const float grade_laws5[5][5] = {
    {1.f, 4.f, 6.f, 4.f, 1.f},
    {-1.f, -2.f, 0.f, 2.f, 1.f},
    {-1.f, 0.f, 2.f, 0.f, -1.f},
    {1.f, -4.f, 6.f, -4.f, 1.f},
    {-1.f, 2.f, 0.f, -2.f, 1.f},
};

/* 第一個字母沿列、第二個沿行。取絕對值後除以兩個核的絕對值和，再除以 2 的 Shift 次方。 */

int vizion_texture_laws(const uint8_t* src, uint8_t* dst, int width, int height, int size, int vertical, int horizontal, int shift)
{
    const float* down;
    const float* side;
    float vkernel[5];
    float hkernel[5];
    float scale;
    float* gray;
    float* response;
    int pixels;
    int index;
    int taps;
    if (!src || !dst || width < 1 || height < 1 || shift < 0)
        return -1;
    if (size == 3) {
        if (vertical < 0 || vertical > 2 || horizontal < 0 || horizontal > 2)
            return -1;
        down = grade_laws3[vertical];
        side = grade_laws3[horizontal];
        taps = 3;
    } else if (size == 5) {
        if (vertical < 0 || vertical > 4 || horizontal < 0 || horizontal > 4)
            return -1;
        down = grade_laws5[vertical];
        side = grade_laws5[horizontal];
        taps = 5;
    } else {
        return -1;
    }
    scale = 0.f;
    {
        float vsum = 0.f;
        float hsum = 0.f;
        for (index = 0; index < taps; index++) {
            vkernel[index] = down[index];
            hkernel[index] = side[index];
            vsum += fabsf(down[index]);
            hsum += fabsf(side[index]);
        }
        scale = vsum * hsum;
    }
    if (!(scale > 0.f))
        return -1;
    {
        double divisor = (double)scale * exp2((double)shift);
    pixels = width * height;
    gray = (float*)malloc((size_t)pixels * sizeof(float) * 2);
    if (!gray)
        return -1;
    response = gray + pixels;
    for (index = 0; index < pixels; index++)
        gray[index] = (float)src[index];
    if (grade_sep(gray, response, width, height, hkernel, taps, vkernel, taps, grade_fast()) != 0) {
        free(gray);
        return -1;
    }
    for (index = 0; index < pixels; index++) {
        double value = response[index];
        if (value < 0.0)
            value = -value;
        dst[index] = grade_round_d(value / divisor);
    }
    free(gray);
    return 0;
    }
}

static int grade_nearest(float value)
{
    return (int)lrint((double)value);
}

/* 第 0 列是圓心。角度 0 朝右，順時針從左到右繞一圈。最近鄰，圖外是 0。 */

int vizion_polar(const uint8_t* src, uint8_t* dst, int width, int height, int channels, double row, double column, int out_width, int out_height)
{
    int y;
    int x;
    int channel;
    double turn;
    if (!src || !dst || width < 1 || height < 1 || out_width < 1 || out_height < 1 || (channels != 1 && channels != 3))
        return -1;
    turn = 2.0 * acos(-1.0) / (double)out_width;
    for (y = 0; y < out_height; y++) {
        float radius = (float)y;
        for (x = 0; x < out_width; x++) {
            float angle = (float)((double)x * turn);
            float cosine = (float)cos((double)angle);
            float sine = (float)sin((double)angle);
            float map_x = (float)(column + (double)(radius * cosine));
            float map_y = (float)(row + (double)(radius * sine));
            int sample_x = grade_nearest(map_x);
            int sample_y = grade_nearest(map_y);
            int inside = (unsigned)sample_x < (unsigned)width && (unsigned)sample_y < (unsigned)height;
            for (channel = 0; channel < channels; channel++) {
                uint8_t value = 0;
                if (inside)
                    value = src[((size_t)sample_y * (size_t)width + sample_x) * (size_t)channels + channel];
                dst[((size_t)y * (size_t)out_width + x) * (size_t)channels + channel] = value;
            }
        }
    }
    return 0;
}

/* 128 是沒變。比 128 大是變亮，比 128 小是變暗。 */

int vizion_enhance_map(const uint8_t* before, const uint8_t* after, uint8_t* dst, int count)
{
    int index;
    if (!before || !after || !dst || count < 1)
        return -1;
    for (index = 0; index < count; index++)
        dst[index] = grade_round(128.f + ((float)after[index] - (float)before[index]));
    return 0;
}

/* 平均和最大都是絕對差。平均用 float32 累加再除以像素數。 */

int vizion_enhance_change(const uint8_t* before, const uint8_t* after, int count, double* mean, double* maximum)
{
    int index;
    float sum = 0.f;
    float peak = 0.f;
    if (!before || !after || !mean || !maximum || count < 1)
        return -1;
    for (index = 0; index < count; index++) {
        float delta = (float)after[index] - (float)before[index];
        if (delta < 0.f)
            delta = -delta;
        sum += delta;
        if (delta > peak)
            peak = delta;
    }
    *mean = (double)(sum / (float)count);
    *maximum = (double)peak;
    return 0;
}

/* 絕對差達到門檻的像素是 255。門檻 0 表示只要有差就算。 */

int vizion_enhance_hot(const uint8_t* before, const uint8_t* after, uint8_t* mask, int count, double limit)
{
    int index;
    if (!before || !after || !mask || count < 1 || limit < 0.0)
        return -1;
    for (index = 0; index < count; index++) {
        float delta = (float)after[index] - (float)before[index];
        if (delta < 0.f)
            delta = -delta;
        if (limit == 0.0)
            mask[index] = delta > 0.f ? 255 : 0;
        else
            mask[index] = (double)delta + 1e-6 >= limit ? 255 : 0;
    }
    return 0;
}

/* 定義域外不算。絕對差達到 MinDiff 的像素是 255。 */

int vizion_diff_mask(const uint8_t* image, const uint8_t* golden, const uint8_t* domain, uint8_t* mask, int count, double minimum)
{
    int index;
    if (!image || !golden || !mask || count < 1 || minimum < 0.0)
        return -1;
    for (index = 0; index < count; index++) {
        int delta;
        int allowed = !domain || domain[index];
        if (!allowed) {
            mask[index] = 0;
            continue;
        }
        delta = (int)image[index] - (int)golden[index];
        if (delta < 0)
            delta = -delta;
        if (minimum == 0.0)
            mask[index] = delta > 0 ? 255 : 0;
        else
            mask[index] = (double)delta + 1e-6 >= minimum ? 255 : 0;
    }
    return 0;
}

/* 只統計定義域內的絕對差。沒有有效像素時平均和最大都是 0。 */

int vizion_diff_amount(const uint8_t* image, const uint8_t* golden, const uint8_t* domain, int count, double* mean, double* maximum)
{
    int index;
    double sum = 0.0;
    double peak = 0.0;
    int used = 0;
    if (!image || !golden || !mean || !maximum || count < 1)
        return -1;
    for (index = 0; index < count; index++) {
        int delta;
        if (domain && !domain[index])
            continue;
        delta = (int)image[index] - (int)golden[index];
        if (delta < 0)
            delta = -delta;
        sum += (double)delta;
        if ((double)delta > peak)
            peak = (double)delta;
        used++;
    }
    *mean = used == 0 ? 0.0 : sum / (double)used;
    *maximum = peak;
    return 0;
}

static int grade_descale(int value, int shift)
{
    return (value + (1 << (shift - 1))) >> shift;
}

static int grade_sat(int value)
{
    if (value < 0)
        return 0;
    if (value > 255)
        return 255;
    return value;
}

void grade_bgr_ycrcb(uint8_t b, uint8_t g, uint8_t r, int* y, int* cr, int* cb)
{
    int yy = grade_descale((int)b * 1868 + (int)g * 9617 + (int)r * 4899, 14);
    *y = yy;
    *cr = grade_sat(grade_descale(((int)r - yy) * 11682 + (128 << 14), 14));
    *cb = grade_sat(grade_descale(((int)b - yy) * 9241 + (128 << 14), 14));
}

void grade_ycrcb_bgr(int y, int cr, int cb, uint8_t* b, uint8_t* g, uint8_t* r)
{
    *b = (uint8_t)grade_sat(y + grade_descale((cb - 128) * 29049, 14));
    *g = (uint8_t)grade_sat(y + grade_descale((cb - 128) * -5636 + (cr - 128) * -11698, 14));
    *r = (uint8_t)grade_sat(y + grade_descale((cr - 128) * 22987, 14));
}

float grade_boost(float sample, float blurred, double factor, double threshold, double limit)
{
    float boost = sample - blurred;
    if (threshold > 0.0 && !(fabs((double)boost) >= threshold))
        boost = 0.f;
    boost *= (float)factor;
    if (limit > 0.0) {
        float cap = (float)limit;
        if (boost > cap)
            boost = cap;
        if (boost < -cap)
            boost = -cap;
    }
    return boost;
}

/* 原圖減模糊的差，太小的起伏不加，超過 Limit 就截斷，再加回亮度。定義域外保持原像素。 */

int vizion_edge_enhance(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double sigma, double factor, double threshold, double limit, const uint8_t* domain)
{
    int pixels;
    int index;
    int side;
    float kernel[31];
    float* plane;
    float* blurred;
    if (!src || !dst || width < 1 || height < 1 || (channels != 1 && channels != 3))
        return -1;
    if (sigma < 0.0 || factor < 0.0 || threshold < 0.0 || limit < 0.0)
        return -1;
    pixels = width * height;
    if (!(sigma > 0.0) || !(factor > 0.0)) {
        memcpy(dst, src, (size_t)pixels * (size_t)channels);
        return 0;
    }
    side = grade_blur_side(sigma);
    grade_gauss_kernel(sigma, side, kernel);
    plane = (float*)malloc((size_t)pixels * sizeof(float) * 2);
    if (!plane)
        return -1;
    blurred = plane + pixels;
    if (channels == 1) {
        for (index = 0; index < pixels; index++)
            plane[index] = (float)src[index];
    } else {
        for (index = 0; index < pixels; index++) {
            int y;
            int cr;
            int cb;
            grade_bgr_ycrcb(src[index * 3], src[index * 3 + 1], src[index * 3 + 2], &y, &cr, &cb);
            plane[index] = (float)y;
        }
    }
    if (grade_sep(plane, blurred, width, height, kernel, side, kernel, side, grade_fast()) != 0) {
        free(plane);
        return -1;
    }
    if (channels == 1) {
        for (index = 0; index < pixels; index++) {
            float boost = grade_boost(plane[index], blurred[index], factor, threshold, limit);
            dst[index] = grade_round(plane[index] + boost);
        }
    } else {
        memcpy(dst, src, (size_t)pixels * 3);
        for (index = 0; index < pixels; index++) {
            float boost = grade_boost(plane[index], blurred[index], factor, threshold, limit);
            int y;
            int cr;
            int cb;
            int next;
            if (!(boost > 1e-6f || boost < -1e-6f))
                continue;
            grade_bgr_ycrcb(src[index * 3], src[index * 3 + 1], src[index * 3 + 2], &y, &cr, &cb);
            next = (int)grade_round(plane[index] + boost);
            grade_ycrcb_bgr(next, cr, cb, &dst[index * 3], &dst[index * 3 + 1], &dst[index * 3 + 2]);
        }
    }
    if (domain) {
        for (index = 0; index < pixels; index++) {
            int channel;
            if (domain[index])
                continue;
            for (channel = 0; channel < channels; channel++)
                dst[index * channels + channel] = src[index * channels + channel];
        }
    }
    free(plane);
    return 0;
}

/* 原圖減去高斯，比門檻小的起伏當成 0，再以 128 為沒有要加的地方。 */

int vizion_edge_residual(const uint8_t* src, uint8_t* dst, int width, int height, double sigma, double threshold)
{
    int side;
    int pixels;
    int index;
    float kernel[31];
    float* gray;
    float* blurred;
    if (!src || !dst || width < 1 || height < 1 || threshold < 0.0)
        return -1;
    pixels = width * height;
    if (!(sigma > 0.0)) {
        for (index = 0; index < pixels; index++)
            dst[index] = 128;
        return 0;
    }
    side = grade_blur_side(sigma);
    grade_gauss_kernel(sigma, side, kernel);
    gray = (float*)malloc((size_t)pixels * sizeof(float) * 2);
    if (!gray)
        return -1;
    blurred = gray + pixels;
    for (index = 0; index < pixels; index++)
        gray[index] = (float)src[index];
    if (grade_sep(gray, blurred, width, height, kernel, side, kernel, side, grade_fast()) != 0) {
        free(gray);
        return -1;
    }
    for (index = 0; index < pixels; index++) {
        float boost = gray[index] - blurred[index];
        if (threshold > 0.0 && !(fabs((double)boost) >= threshold))
            boost = 0.f;
        dst[index] = grade_round(128.f + boost);
    }
    free(gray);
    return 0;
}

int vizion_grade_selftest(void)
{
    static const uint8_t flat[9] = {40, 40, 40, 40, 40, 40, 40, 40, 40};
    static const uint8_t before[4] = {10, 20, 30, 40};
    static const uint8_t after[4] = {10, 25, 20, 40};
    uint8_t out[9];
    uint8_t laws[9];
    uint8_t mapped[4];
    uint8_t hot[4];
    uint8_t polar_src[121];
    uint8_t polar[48];
    double mean = -1.0;
    double maximum = -1.0;
    int index;
    memset(polar_src, 0, sizeof(polar_src));
    polar_src[5 * 11 + 8] = 200;
    if (vizion_texture_laws(flat, laws, 3, 3, 5, 1, 0, 0) != 0)
        return 1;
    for (index = 0; index < 9; index++) {
        if (laws[index] != 0)
            return 1;
    }
    if (vizion_enhance_map(before, after, mapped, 4) != 0 || mapped[0] != 128 || mapped[1] != 133 || mapped[2] != 118 || mapped[3] != 128)
        return 2;
    if (vizion_enhance_change(before, after, 4, &mean, &maximum) != 0 || maximum != 10.0 || !(mean > 3.7 && mean < 3.8))
        return 3;
    if (vizion_enhance_hot(before, after, hot, 4, 10.0) != 0 || hot[0] != 0 || hot[1] != 0 || hot[2] != 255 || hot[3] != 0)
        return 4;
    if (vizion_polar(polar_src, polar, 11, 11, 1, 5.0, 5.0, 8, 6) != 0 || polar[3 * 8] != 200)
        return 5;
    if (vizion_edge_residual(flat, out, 3, 3, 0.0, 0.0) != 0)
        return 6;
    for (index = 0; index < 9; index++) {
        if (out[index] != 128)
            return 6;
    }
    if (vizion_deriv_gauss(flat, out, 3, 3, 1.0, 1) != 0)
        return 7;
    for (index = 0; index < 9; index++) {
        if (out[index] != 128)
            return 7;
    }
    return 0;
}
