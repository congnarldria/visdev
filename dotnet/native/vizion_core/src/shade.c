#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

uint8_t grade_round_d(double value);
uint8_t grade_round(float value);
int grade_sep(const float* src, float* dst, int width, int height, const float* across, int across_n, const float* along, int along_n, int use_sse);
void grade_gauss_kernel(double sigma, int ksize, float* kernel);
int grade_blur_side(double sigma);
int grade_deriv_kernels(double sigma, float* smooth, float* first, float* second, int* count);
void grade_bgr_ycrcb(uint8_t b, uint8_t g, uint8_t r, int* y, int* cr, int* cb);
void grade_ycrcb_bgr(int y, int cr, int cb, uint8_t* b, uint8_t* g, uint8_t* r);
float grade_boost(float sample, float blurred, double factor, double threshold, double limit);
int vizion_reflect101(int position, int length);

static int shade_fast(void)
{
    int flags = vizion_cpu_flags();
    return (flags & VIZION_CPU_SELECTED_AVX2) || (flags & VIZION_CPU_SELECTED_SSE2);
}

static void shade_restore(const uint8_t* src, uint8_t* dst, int pixels, int channels, const uint8_t* domain)
{
    int index;
    int channel;
    if (!domain)
        return;
    for (index = 0; index < pixels; index++) {
        if (domain[index])
            continue;
        for (channel = 0; channel < channels; channel++)
            dst[index * channels + channel] = src[index * channels + channel];
    }
}

static void shade_load_luma(const uint8_t* src, float* luma, int pixels, int channels)
{
    int index;
    if (channels == 1) {
        for (index = 0; index < pixels; index++)
            luma[index] = (float)src[index];
        return;
    }
    for (index = 0; index < pixels; index++) {
        int y;
        int cr;
        int cb;
        grade_bgr_ycrcb(src[index * 3], src[index * 3 + 1], src[index * 3 + 2], &y, &cr, &cb);
        luma[index] = (float)y;
    }
}

static int shade_blur(const float* src, float* dst, int width, int height, double sigma)
{
    int side;
    float kernel[31];
    int pixels = width * height;
    if (!(sigma > 0.0)) {
        memcpy(dst, src, (size_t)pixels * sizeof(float));
        return 0;
    }
    side = grade_blur_side(sigma);
    grade_gauss_kernel(sigma, side, kernel);
    return grade_sep(src, dst, width, height, kernel, side, kernel, side, shade_fast());
}

static void shade_paint_f(
    const uint8_t* src, uint8_t* dst, const float* luma, const float* add,
    int pixels, int channels, const uint8_t* domain)
{
    int index;
    if (channels == 1) {
        for (index = 0; index < pixels; index++)
            dst[index] = grade_round(luma[index] + add[index]);
    } else {
        memcpy(dst, src, (size_t)pixels * 3);
        for (index = 0; index < pixels; index++) {
            int y;
            int cr;
            int cb;
            int next;
            float boost = add[index];
            if (!(boost > 1e-6f || boost < -1e-6f))
                continue;
            grade_bgr_ycrcb(src[index * 3], src[index * 3 + 1], src[index * 3 + 2], &y, &cr, &cb);
            next = (int)grade_round(luma[index] + boost);
            grade_ycrcb_bgr(next, cr, cb, &dst[index * 3], &dst[index * 3 + 1], &dst[index * 3 + 2]);
        }
    }
    shade_restore(src, dst, pixels, channels, domain);
}

static void shade_paint_d(
    const uint8_t* src, uint8_t* dst, const float* luma, const double* add,
    int pixels, int channels, const uint8_t* domain)
{
    int index;
    if (channels == 1) {
        for (index = 0; index < pixels; index++)
            dst[index] = grade_round_d((double)luma[index] + add[index]);
    } else {
        memcpy(dst, src, (size_t)pixels * 3);
        for (index = 0; index < pixels; index++) {
            int y;
            int cr;
            int cb;
            if (!(add[index] > 1e-6 || add[index] < -1e-6))
                continue;
            grade_bgr_ycrcb(src[index * 3], src[index * 3 + 1], src[index * 3 + 2], &y, &cr, &cb);
            grade_ycrcb_bgr(
                (int)grade_round_d((double)luma[index] + add[index]),
                cr, cb, &dst[index * 3], &dst[index * 3 + 1], &dst[index * 3 + 2]);
        }
    }
    shade_restore(src, dst, pixels, channels, domain);
}

static int shade_prepare(const uint8_t* src, int width, int height, int channels, float** luma, float** blurred)
{
    int pixels = width * height;
    float* plane = (float*)malloc((size_t)pixels * sizeof(float) * 2);
    if (!plane)
        return -1;
    *luma = plane;
    *blurred = plane + pixels;
    shade_load_luma(src, plane, pixels, channels);
    return 0;
}

/* 細高斯減粗高斯，太小的起伏不加，再加回亮度。定義域外保持原像素。 */

int vizion_edge_band(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double sigma_fine, double sigma_coarse, double factor, double threshold, const uint8_t* domain)
{
    int pixels;
    int index;
    float* luma;
    float* fine;
    float* coarse;
    float* add;
    if (!src || !dst || width < 1 || height < 1 || (channels != 1 && channels != 3))
        return -1;
    if (sigma_fine < 0.0 || sigma_coarse < 0.0 || factor < 0.0 || threshold < 0.0 || !(sigma_coarse > sigma_fine))
        return -1;
    pixels = width * height;
    if (!(factor > 0.0)) {
        memcpy(dst, src, (size_t)pixels * (size_t)channels);
        return 0;
    }
    luma = (float*)malloc((size_t)pixels * sizeof(float) * 4);
    if (!luma)
        return -1;
    fine = luma + pixels;
    coarse = fine + pixels;
    add = coarse + pixels;
    shade_load_luma(src, luma, pixels, channels);
    if (shade_blur(luma, fine, width, height, sigma_fine) != 0 || shade_blur(luma, coarse, width, height, sigma_coarse) != 0) {
        free(luma);
        return -1;
    }
    for (index = 0; index < pixels; index++)
        add[index] = grade_boost(fine[index], coarse[index], factor, threshold, 0.0);
    shade_paint_f(src, dst, luma, add, pixels, channels, domain);
    free(luma);
    return 0;
}

static int shade_magnitude(const float* src, float* magnitude, int width, int height)
{
    static const float deriv[3] = {-1.f, 0.f, 1.f};
    static const float smooth[3] = {1.f, 2.f, 1.f};
    int pixels = width * height;
    int index;
    float* gx = (float*)malloc((size_t)pixels * sizeof(float) * 2);
    float* gy;
    if (!gx)
        return -1;
    gy = gx + pixels;
    if (grade_sep(src, gx, width, height, deriv, 3, smooth, 3, shade_fast()) != 0
        || grade_sep(src, gy, width, height, smooth, 3, deriv, 3, shade_fast()) != 0) {
        free(gx);
        return -1;
    }
    for (index = 0; index < pixels; index++)
        magnitude[index] = sqrtf(gx[index] * gx[index] + gy[index] * gy[index]);
    free(gx);
    return 0;
}

/* 原圖減模糊的差，乘上梯度佔 Level 的比例，再加回亮度。定義域外保持原像素。 */

int vizion_edge_masked(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double sigma, double factor, double level, const uint8_t* domain)
{
    int pixels;
    int index;
    float* luma;
    float* blurred;
    float* magnitude;
    double* add;
    if (!src || !dst || width < 1 || height < 1 || (channels != 1 && channels != 3))
        return -1;
    if (sigma < 0.0 || factor < 0.0 || !(level > 0.0))
        return -1;
    pixels = width * height;
    if (!(sigma > 0.0) || !(factor > 0.0)) {
        memcpy(dst, src, (size_t)pixels * (size_t)channels);
        return 0;
    }
    if (shade_prepare(src, width, height, channels, &luma, &blurred) != 0)
        return -1;
    magnitude = (float*)malloc((size_t)pixels * sizeof(float));
    add = (double*)malloc((size_t)pixels * sizeof(double));
    if (!magnitude || !add || shade_blur(luma, blurred, width, height, sigma) != 0 || shade_magnitude(blurred, magnitude, width, height) != 0) {
        free(luma);
        free(magnitude);
        free(add);
        return -1;
    }
    for (index = 0; index < pixels; index++) {
        double weight = (double)magnitude[index] / level;
        double diff = (double)luma[index] - (double)blurred[index];
        if (weight < 0.0)
            weight = 0.0;
        if (weight > 1.0)
            weight = 1.0;
        add[index] = diff * weight * factor;
    }
    shade_paint_d(src, dst, luma, add, pixels, channels, domain);
    free(luma);
    free(magnitude);
    free(add);
    return 0;
}

/* 梯度小的地方改用高斯，梯度達到 Level 就留原圖。定義域外保持原像素。 */

int vizion_edge_preserve(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double sigma, double level, const uint8_t* domain)
{
    int pixels;
    int index;
    float* luma;
    float* blurred;
    float* magnitude;
    double* add;
    if (!src || !dst || width < 1 || height < 1 || (channels != 1 && channels != 3))
        return -1;
    if (sigma < 0.0 || !(level > 0.0))
        return -1;
    pixels = width * height;
    if (!(sigma > 0.0)) {
        memcpy(dst, src, (size_t)pixels * (size_t)channels);
        return 0;
    }
    if (shade_prepare(src, width, height, channels, &luma, &blurred) != 0)
        return -1;
    magnitude = (float*)malloc((size_t)pixels * sizeof(float));
    add = (double*)malloc((size_t)pixels * sizeof(double));
    if (!magnitude || !add || shade_blur(luma, blurred, width, height, sigma) != 0 || shade_magnitude(blurred, magnitude, width, height) != 0) {
        free(luma);
        free(magnitude);
        free(add);
        return -1;
    }
    for (index = 0; index < pixels; index++) {
        double weight = (double)magnitude[index] / level;
        double diff = (double)luma[index] - (double)blurred[index];
        if (weight < 0.0)
            weight = 0.0;
        if (weight > 1.0)
            weight = 1.0;
        add[index] = diff * weight - diff;
    }
    shade_paint_d(src, dst, luma, add, pixels, channels, domain);
    free(luma);
    free(magnitude);
    free(add);
    return 0;
}

static int shade_gray(const uint8_t* src, uint8_t* gray, int pixels, int channels)
{
    if (channels == 1) {
        memcpy(gray, src, (size_t)pixels);
        return 0;
    }
    return vizion_bgr_gray(src, pixels, gray);
}

/* 四鄰域 Perona-Malik 或拋物線傳導。最外一圈不動。彩色先轉成亮度。 */

int vizion_anisotropic(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    int mode, double contrast, double theta, int iterations)
{
    int pixels;
    int index;
    int step;
    int row;
    int col;
    double scale;
    uint8_t* gray;
    float* work;
    float* next;
    if (!src || !dst || width < 1 || height < 1 || (channels != 1 && channels != 3))
        return -1;
    if ((mode != 0 && mode != 1) || !(contrast > 0.0) || !(theta > 0.0) || theta > 0.25 || iterations < 1 || iterations > 40)
        return -1;
    pixels = width * height;
    gray = (uint8_t*)malloc((size_t)pixels);
    work = (float*)malloc((size_t)pixels * sizeof(float));
    next = (float*)malloc((size_t)pixels * sizeof(float));
    if (!gray || !work || !next || shade_gray(src, gray, pixels, channels) != 0) {
        free(gray);
        free(work);
        free(next);
        return -1;
    }
    for (index = 0; index < pixels; index++)
        work[index] = (float)gray[index];
    scale = contrast * contrast;
    for (step = 0; step < iterations; step++) {
        memcpy(next, work, (size_t)pixels * sizeof(float));
        if (width < 3 || height < 3) {
            memcpy(work, next, (size_t)pixels * sizeof(float));
            break;
        }
        for (row = 1; row < height - 1; row++) {
            for (col = 1; col < width - 1; col++) {
                int at = row * width + col;
                float center = work[at];
                float north = work[at - width] - center;
                float south = work[at + width] - center;
                float west = work[at - 1] - center;
                float east = work[at + 1] - center;
                double nc;
                double sc;
                double wc;
                double ec;
                double mix;
                if (mode == 0) {
                    nc = exp(-((double)(north * north)) / scale);
                    sc = exp(-((double)(south * south)) / scale);
                    wc = exp(-((double)(west * west)) / scale);
                    ec = exp(-((double)(east * east)) / scale);
                } else {
                    nc = 1.0 / (1.0 + ((double)(north * north)) / scale);
                    sc = 1.0 / (1.0 + ((double)(south * south)) / scale);
                    wc = 1.0 / (1.0 + ((double)(west * west)) / scale);
                    ec = 1.0 / (1.0 + ((double)(east * east)) / scale);
                }
                mix = ((nc * (double)north + sc * (double)south) + wc * (double)west) + ec * (double)east;
                next[at] = (float)((double)center + theta * mix);
            }
        }
        memcpy(work, next, (size_t)pixels * sizeof(float));
    }
    for (index = 0; index < pixels; index++)
        dst[index] = grade_round((float)work[index]);
    free(gray);
    free(work);
    free(next);
    return 0;
}

/* 沿著拉普拉斯或梯度方向的二階導數把邊緣變陡。完全平的地方不變。 */

int vizion_shock(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double theta, int iterations, int mode, double sigma)
{
    int pixels;
    int index;
    int step;
    int row;
    int col;
    uint8_t* gray;
    float* work;
    float* smooth;
    float* gx;
    float* gy;
    float* guide;
    int own_smooth;
    if (!src || !dst || width < 1 || height < 1 || (channels != 1 && channels != 3))
        return -1;
    if (!(theta > 0.0) || theta > 1.0 || iterations < 1 || iterations > 20 || (mode != 0 && mode != 1) || sigma < 0.0)
        return -1;
    pixels = width * height;
    gray = (uint8_t*)malloc((size_t)pixels);
    work = (float*)calloc((size_t)pixels, sizeof(float));
    smooth = (float*)malloc((size_t)pixels * sizeof(float));
    gx = (float*)calloc((size_t)pixels, sizeof(float));
    gy = (float*)calloc((size_t)pixels, sizeof(float));
    guide = (float*)calloc((size_t)pixels, sizeof(float));
    if (!gray || !work || !smooth || !gx || !gy || !guide || shade_gray(src, gray, pixels, channels) != 0) {
        free(gray);
        free(work);
        free(smooth);
        free(gx);
        free(gy);
        free(guide);
        return -1;
    }
    for (index = 0; index < pixels; index++)
        work[index] = (float)gray[index];
    own_smooth = sigma > 0.0;
    for (step = 0; step < iterations; step++) {
        const float* used;
        if (!own_smooth) {
            used = work;
        } else if (shade_blur(work, smooth, width, height, sigma) != 0) {
            free(gray);
            free(work);
            free(smooth);
            free(gx);
            free(gy);
            free(guide);
            return -1;
        } else {
            used = smooth;
        }
        memset(gx, 0, (size_t)pixels * sizeof(float));
        memset(gy, 0, (size_t)pixels * sizeof(float));
        memset(guide, 0, (size_t)pixels * sizeof(float));
        if (width >= 3) {
            for (row = 0; row < height; row++) {
                for (col = 1; col < width - 1; col++) {
                    int at = row * width + col;
                    float right = used[at + 1];
                    float left = used[at - 1];
                    gx[at] = (float)(((double)right - (double)left) * 0.5);
                }
            }
        }
        if (height >= 3) {
            for (row = 1; row < height - 1; row++) {
                for (col = 0; col < width; col++) {
                    int at = row * width + col;
                    float south = used[at + width];
                    float north = used[at - width];
                    gy[at] = (float)(((double)south - (double)north) * 0.5);
                }
            }
        }
        if (width >= 3 && height >= 3 && mode == 0) {
            for (row = 1; row < height - 1; row++) {
                for (col = 1; col < width - 1; col++) {
                    int at = row * width + col;
                    float east = used[at + 1];
                    float west = used[at - 1];
                    float south = used[at + width];
                    float north = used[at - width];
                    float sum = ((east + west) + south) + north;
                    guide[at] = (float)((double)sum - 4.0 * (double)used[at]);
                }
            }
        } else if (width >= 3 && height >= 3) {
            for (row = 1; row < height - 1; row++) {
                for (col = 1; col < width - 1; col++) {
                    int at = row * width + col;
                    float center = used[at];
                    float east = used[at + 1];
                    float west = used[at - 1];
                    float south = used[at + width];
                    float north = used[at - width];
                    float ix = gx[at];
                    float iy = gy[at];
                    float mag2 = ix * ix + iy * iy;
                    double gxx = ((double)east - 2.0 * (double)center) + (double)west;
                    double gyy = ((double)south - 2.0 * (double)center) + (double)north;
                    float pp = used[at + width + 1];
                    float pm = used[at + width - 1];
                    float mp = used[at - width + 1];
                    float mm = used[at - width - 1];
                    float inner = ((pp - pm) - mp) + mm;
                    double gxy = 0.25 * (double)inner;
                    double dir = 0.0;
                    if ((double)mag2 > 1e-6) {
                        double num = (((double)(ix * ix)) * gxx + (2.0 * (double)ix * (double)iy * gxy)) + ((double)(iy * iy)) * gyy;
                        dir = num / (double)mag2;
                    }
                    guide[at] = (float)dir;
                }
            }
        }
        for (index = 0; index < pixels; index++) {
            float mag = sqrtf(gx[index] * gx[index] + gy[index] * gy[index]);
            float sign = guide[index] > 0.f ? 1.f : guide[index] < 0.f ? -1.f : 0.f;
            work[index] = (float)((double)work[index] - theta * (double)sign * (double)mag);
        }
    }
    for (index = 0; index < pixels; index++)
        dst[index] = grade_round(work[index]);
    free(gray);
    free(work);
    free(smooth);
    free(gx);
    free(gy);
    free(guide);
    return 0;
}

static int shade_box_f32(const float* src, float* dst, int width, int height, int radius)
{
    int window = radius * 2 + 1;
    int row;
    int col;
    int tap;
    double* horizontal = (double*)malloc((size_t)width * (size_t)height * sizeof(double));
    if (!horizontal)
        return -1;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            double acc = 0.0;
            for (tap = -radius; tap <= radius; tap++)
                acc += (double)src[(size_t)row * (size_t)width + vizion_reflect101(col + tap, width)];
            horizontal[(size_t)row * (size_t)width + col] = acc / (double)window;
        }
    }
    for (col = 0; col < width; col++) {
        for (row = 0; row < height; row++) {
            double acc = 0.0;
            for (tap = -radius; tap <= radius; tap++)
                acc += horizontal[(size_t)vizion_reflect101(row + tap, height) * (size_t)width + col];
            dst[(size_t)row * (size_t)width + col] = (float)(acc / (double)window);
        }
    }
    free(horizontal);
    return 0;
}

static int shade_box_f64(const double* src, double* dst, int width, int height, int radius)
{
    int window = radius * 2 + 1;
    int row;
    int col;
    int tap;
    double* horizontal = (double*)malloc((size_t)width * (size_t)height * sizeof(double));
    if (!horizontal)
        return -1;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            double acc = 0.0;
            for (tap = -radius; tap <= radius; tap++)
                acc += src[(size_t)row * (size_t)width + vizion_reflect101(col + tap, width)];
            horizontal[(size_t)row * (size_t)width + col] = acc / (double)window;
        }
    }
    for (col = 0; col < width; col++) {
        for (row = 0; row < height; row++) {
            double acc = 0.0;
            for (tap = -radius; tap <= radius; tap++)
                acc += horizontal[(size_t)vizion_reflect101(row + tap, height) * (size_t)width + col];
            dst[(size_t)row * (size_t)width + col] = acc / (double)window;
        }
    }
    free(horizontal);
    return 0;
}

/* 用影像自己的亮度當引導。平坦處變成局部平均，階梯邊緣的兩邊不會被拉到中間。 */

int vizion_guided(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels, int radius, double eps)
{
    int pixels;
    int index;
    int channel;
    uint8_t* gray;
    float* guide;
    float* mean_g;
    float* box_gg;
    float* plane;
    float* mean_s;
    float* product;
    double* coeff;
    double* offset;
    double* box_c;
    double* box_o;
    if (!src || !dst || width < 1 || height < 1 || (channels != 1 && channels != 3) || radius < 1 || !(eps > 0.0))
        return -1;
    pixels = width * height;
    gray = (uint8_t*)malloc((size_t)pixels);
    guide = (float*)malloc((size_t)pixels * sizeof(float));
    mean_g = (float*)malloc((size_t)pixels * sizeof(float));
    box_gg = (float*)malloc((size_t)pixels * sizeof(float));
    plane = (float*)malloc((size_t)pixels * sizeof(float));
    mean_s = (float*)malloc((size_t)pixels * sizeof(float));
    product = (float*)malloc((size_t)pixels * sizeof(float));
    coeff = (double*)malloc((size_t)pixels * sizeof(double));
    offset = (double*)malloc((size_t)pixels * sizeof(double));
    box_c = (double*)malloc((size_t)pixels * sizeof(double));
    box_o = (double*)malloc((size_t)pixels * sizeof(double));
    if (!gray || !guide || !mean_g || !box_gg || !plane || !mean_s || !product || !coeff || !offset || !box_c || !box_o
        || shade_gray(src, gray, pixels, channels) != 0) {
        free(gray);
        free(guide);
        free(mean_g);
        free(box_gg);
        free(plane);
        free(mean_s);
        free(product);
        free(coeff);
        free(offset);
        free(box_c);
        free(box_o);
        return -1;
    }
    for (index = 0; index < pixels; index++) {
        guide[index] = (float)gray[index];
        product[index] = guide[index] * guide[index];
    }
    if (shade_box_f32(guide, mean_g, width, height, radius) != 0 || shade_box_f32(product, box_gg, width, height, radius) != 0) {
        free(gray);
        free(guide);
        free(mean_g);
        free(box_gg);
        free(plane);
        free(mean_s);
        free(product);
        free(coeff);
        free(offset);
        free(box_c);
        free(box_o);
        return -1;
    }
    for (channel = 0; channel < channels; channel++) {
        for (index = 0; index < pixels; index++)
            plane[index] = (float)src[index * channels + channel];
        if (shade_box_f32(plane, mean_s, width, height, radius) != 0) {
            free(gray);
            free(guide);
            free(mean_g);
            free(box_gg);
            free(plane);
            free(mean_s);
            free(product);
            free(coeff);
            free(offset);
            free(box_c);
            free(box_o);
            return -1;
        }
        for (index = 0; index < pixels; index++)
            product[index] = guide[index] * plane[index];
        if (shade_box_f32(product, plane, width, height, radius) != 0) {
            free(gray);
            free(guide);
            free(mean_g);
            free(box_gg);
            free(plane);
            free(mean_s);
            free(product);
            free(coeff);
            free(offset);
            free(box_c);
            free(box_o);
            return -1;
        }
        for (index = 0; index < pixels; index++) {
            float var32 = box_gg[index] - mean_g[index] * mean_g[index];
            double variance = (double)var32;
            float cov = plane[index] - mean_g[index] * mean_s[index];
            if (variance < 0.0)
                variance = 0.0;
            coeff[index] = (double)cov / (variance + eps);
            offset[index] = (double)mean_s[index] - coeff[index] * (double)mean_g[index];
        }
        if (shade_box_f64(coeff, box_c, width, height, radius) != 0 || shade_box_f64(offset, box_o, width, height, radius) != 0) {
            free(gray);
            free(guide);
            free(mean_g);
            free(box_gg);
            free(plane);
            free(mean_s);
            free(product);
            free(coeff);
            free(offset);
            free(box_c);
            free(box_o);
            return -1;
        }
        for (index = 0; index < pixels; index++)
            dst[index * channels + channel] = grade_round_d(box_c[index] * (double)guide[index] + box_o[index]);
    }
    free(gray);
    free(guide);
    free(mean_g);
    free(box_gg);
    free(plane);
    free(mean_s);
    free(product);
    free(coeff);
    free(offset);
    free(box_c);
    free(box_o);
    return 0;
}

static int shade_frangi_scale(const float* gray, float* vessel, int width, int height, double sigma, int dark)
{
    int pixels = width * height;
    int count = 0;
    int index;
    float smooth_k[65];
    float first[65];
    float second[65];
    float* xx;
    float* yy;
    float* xy;
    double* ratio_d;
    double* magnitude_d;
    double* eigen_d;
    double scale;
    double peak = 0.0;
    if (grade_deriv_kernels(sigma, smooth_k, first, second, &count) != 0)
        return -1;
    xx = (float*)malloc((size_t)pixels * sizeof(float) * 3);
    if (!xx)
        return -1;
    yy = xx + pixels;
    xy = yy + pixels;
    if (grade_sep(gray, xx, width, height, second, count, smooth_k, count, shade_fast()) != 0
        || grade_sep(gray, yy, width, height, smooth_k, count, second, count, shade_fast()) != 0
        || grade_sep(gray, xy, width, height, first, count, first, count, shade_fast()) != 0) {
        free(xx);
        return -1;
    }
    scale = sigma * sigma;
    ratio_d = (double*)malloc((size_t)pixels * sizeof(double) * 3);
    if (!ratio_d) {
        free(xx);
        return -1;
    }
    magnitude_d = ratio_d + pixels;
    eigen_d = magnitude_d + pixels;
    for (index = 0; index < pixels; index++) {
        double dxx = (double)xx[index] * scale;
        double dyy = (double)yy[index] * scale;
        double dxy = (double)xy[index] * scale;
        double trace = dxx + dyy;
        double det = dxx * dyy - dxy * dxy;
        double disc = sqrt(fmax(trace * trace * 0.25 - det, 0.0));
        double high = trace * 0.5 + disc;
        double low = trace * 0.5 - disc;
        double first_eigen = fabs(high) < fabs(low) ? high : low;
        double second_eigen = fabs(high) < fabs(low) ? low : high;
        double second_abs = fabs(second_eigen);
        double ratio = second_abs > 1e-8 ? fabs(first_eigen) / second_abs : 0.0;
        double magnitude = sqrt(first_eigen * first_eigen + second_eigen * second_eigen);
        if (magnitude > peak)
            peak = magnitude;
        ratio_d[index] = ratio;
        magnitude_d[index] = magnitude;
        eigen_d[index] = second_eigen;
    }
    if (peak < 1e-8) {
        memset(vessel, 0, (size_t)pixels * sizeof(float));
        free(ratio_d);
        free(xx);
        return 0;
    }
    {
        double half = 0.5 * peak;
        for (index = 0; index < pixels; index++) {
            int keep = dark ? eigen_d[index] > 0.0 : eigen_d[index] < 0.0;
            double value = exp(-(ratio_d[index] * ratio_d[index]) / 0.5)
                * (1.0 - exp(-(magnitude_d[index] * magnitude_d[index]) / (2.0 * half * half)));
            vessel[index] = keep ? (float)value : 0.f;
        }
    }
    free(ratio_d);
    free(xx);
    return 0;
}

/* Frangi 用 Hessian 兩個特徵值把細線加強，方塊內部接近 0。 */

int vizion_frangi(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double sigma, double sigma_max, int dark)
{
    int pixels;
    int index;
    int scale_index;
    int scales_n = 1;
    double scales[3];
    uint8_t* bytes;
    float* gray;
    float* vessel;
    float* best;
    if (!src || !dst || width < 1 || height < 1 || (channels != 1 && channels != 3))
        return -1;
    if (!(sigma > 0.0) || !(sigma_max > 0.0) || sigma_max < sigma || (dark != 0 && dark != 1))
        return -1;
    pixels = width * height;
    scales[0] = sigma;
    if (fabs(sigma_max - sigma) >= 1e-6) {
        scales[1] = (sigma + sigma_max) * 0.5;
        scales[2] = sigma_max;
        scales_n = 3;
    }
    bytes = (uint8_t*)malloc((size_t)pixels);
    gray = (float*)malloc((size_t)pixels * sizeof(float));
    vessel = (float*)malloc((size_t)pixels * sizeof(float));
    best = (float*)calloc((size_t)pixels, sizeof(float));
    if (!bytes || !gray || !vessel || !best || shade_gray(src, bytes, pixels, channels) != 0) {
        free(bytes);
        free(gray);
        free(vessel);
        free(best);
        return -1;
    }
    for (index = 0; index < pixels; index++)
        gray[index] = (float)bytes[index];
    for (scale_index = 0; scale_index < scales_n; scale_index++) {
        if (shade_frangi_scale(gray, vessel, width, height, scales[scale_index], dark) != 0) {
            free(bytes);
            free(gray);
            free(vessel);
            free(best);
            return -1;
        }
        for (index = 0; index < pixels; index++) {
            if (vessel[index] > best[index])
                best[index] = vessel[index];
        }
    }
    for (index = 0; index < pixels; index++)
        dst[index] = grade_round_d((double)best[index] * 255.0);
    free(bytes);
    free(gray);
    free(vessel);
    free(best);
    return 0;
}

static int shade_clahe_plane(const uint8_t* src, uint8_t* dst, int width, int height, double clip, int tiles)
{
    int use_pad = (width % tiles) != 0 || (height % tiles) != 0;
    int bottom = use_pad ? tiles - (height % tiles) : 0;
    int right = use_pad ? tiles - (width % tiles) : 0;
    int pad_w = width + right;
    int pad_h = height + bottom;
    int tile_w = pad_w / tiles;
    int tile_h = pad_h / tiles;
    int tile_area = tile_w * tile_h;
    int clip_limit = 0;
    int ty;
    int tx;
    int x;
    int y;
    float lut_scale = (float)(256 - 1) / (float)tile_area;
    float inv_w = 1.f / (float)tile_w;
    float inv_h = 1.f / (float)tile_h;
    uint8_t* padded = (uint8_t*)malloc((size_t)pad_w * (size_t)pad_h);
    uint8_t* lut = (uint8_t*)malloc((size_t)tiles * (size_t)tiles * 256);
    float* xa = (float*)malloc((size_t)width * sizeof(float));
    float* xa1 = (float*)malloc((size_t)width * sizeof(float));
    int* tx1s = (int*)malloc((size_t)width * sizeof(int));
    int* tx2s = (int*)malloc((size_t)width * sizeof(int));
    if (!padded || !lut || !xa || !xa1 || !tx1s || !tx2s) {
        free(padded);
        free(lut);
        free(xa);
        free(xa1);
        free(tx1s);
        free(tx2s);
        return -1;
    }
    for (y = 0; y < pad_h; y++) {
        int sy = y < height ? y : vizion_reflect101(y, height);
        for (x = 0; x < pad_w; x++) {
            int sx = x < width ? x : vizion_reflect101(x, width);
            padded[y * pad_w + x] = src[sy * width + sx];
        }
    }
    if (clip > 0.0) {
        clip_limit = (int)(clip * (double)tile_area / 256.0);
        if (clip_limit < 1)
            clip_limit = 1;
    }
    for (ty = 0; ty < tiles; ty++) {
        for (tx = 0; tx < tiles; tx++) {
            int hist[256];
            int clipped = 0;
            int bin;
            int sum = 0;
            uint8_t* row = lut + ((size_t)ty * (size_t)tiles + tx) * 256;
            memset(hist, 0, sizeof(hist));
            for (y = 0; y < tile_h; y++) {
                const uint8_t* line = padded + ((size_t)ty * (size_t)tile_h + y) * (size_t)pad_w + (size_t)tx * (size_t)tile_w;
                for (x = 0; x < tile_w; x++)
                    hist[line[x]]++;
            }
            if (clip_limit > 0) {
                int redist;
                int residual;
                int residual_step;
                for (bin = 0; bin < 256; bin++) {
                    if (hist[bin] > clip_limit) {
                        clipped += hist[bin] - clip_limit;
                        hist[bin] = clip_limit;
                    }
                }
                redist = clipped / 256;
                residual = clipped - redist * 256;
                for (bin = 0; bin < 256; bin++)
                    hist[bin] += redist;
                if (residual != 0) {
                    residual_step = 256 / residual;
                    if (residual_step < 1)
                        residual_step = 1;
                    for (bin = 0; bin < 256 && residual > 0; bin += residual_step, residual--)
                        hist[bin]++;
                }
            }
            for (bin = 0; bin < 256; bin++) {
                sum += hist[bin];
                row[bin] = grade_round((float)sum * lut_scale);
            }
        }
    }
    for (x = 0; x < width; x++) {
        float txf = (float)x * inv_w - 0.5f;
        int tx1 = (int)floorf(txf);
        int tx2 = tx1 + 1;
        xa[x] = txf - (float)tx1;
        xa1[x] = 1.f - xa[x];
        if (tx1 < 0)
            tx1 = 0;
        if (tx2 > tiles - 1)
            tx2 = tiles - 1;
        tx1s[x] = tx1;
        tx2s[x] = tx2;
    }
    for (y = 0; y < height; y++) {
        float tyf = (float)y * inv_h - 0.5f;
        int ty1 = (int)floorf(tyf);
        int ty2 = ty1 + 1;
        float ya = tyf - (float)ty1;
        float ya1 = 1.f - ya;
        if (ty1 < 0)
            ty1 = 0;
        if (ty2 > tiles - 1)
            ty2 = tiles - 1;
        for (x = 0; x < width; x++) {
            int value = src[y * width + x];
            float a = lut[((size_t)ty1 * (size_t)tiles + tx1s[x]) * 256 + value];
            float b = lut[((size_t)ty1 * (size_t)tiles + tx2s[x]) * 256 + value];
            float c = lut[((size_t)ty2 * (size_t)tiles + tx1s[x]) * 256 + value];
            float d = lut[((size_t)ty2 * (size_t)tiles + tx2s[x]) * 256 + value];
            float res = ((a * xa1[x] + b * xa[x]) * ya1) + ((c * xa1[x] + d * xa[x]) * ya);
            dst[y * width + x] = grade_round(res);
        }
    }
    free(padded);
    free(lut);
    free(xa);
    free(xa1);
    free(tx1s);
    free(tx2s);
    return 0;
}

/* 切成格子各自等化，相鄰格子再混起來。彩色只處理亮度。 */

int vizion_clahe(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels, double clip, int tiles)
{
    int pixels;
    int index;
    uint8_t* plane;
    if (!src || !dst || width < 1 || height < 1 || (channels != 1 && channels != 3) || !(clip > 0.0) || tiles < 2)
        return -1;
    pixels = width * height;
    if (channels == 1)
        return shade_clahe_plane(src, dst, width, height, clip, tiles);
    plane = (uint8_t*)malloc((size_t)pixels);
    if (!plane)
        return -1;
    for (index = 0; index < pixels; index++) {
        int y;
        int cr;
        int cb;
        grade_bgr_ycrcb(src[index * 3], src[index * 3 + 1], src[index * 3 + 2], &y, &cr, &cb);
        plane[index] = (uint8_t)y;
        dst[index * 3 + 1] = (uint8_t)cr;
        dst[index * 3 + 2] = (uint8_t)cb;
    }
    if (shade_clahe_plane(plane, plane, width, height, clip, tiles) != 0) {
        free(plane);
        return -1;
    }
    for (index = 0; index < pixels; index++)
        grade_ycrcb_bgr((int)plane[index], (int)dst[index * 3 + 1], (int)dst[index * 3 + 2], &dst[index * 3], &dst[index * 3 + 1], &dst[index * 3 + 2]);
    free(plane);
    return 0;
}

/* 樣板或核放在左上角，其餘補 0，做循環相關或循環卷積，再截到 0 到 255。 */

int vizion_circular(
    const uint8_t* image, const uint8_t* kernel, uint8_t* dst,
    int width, int height, int kernel_width, int kernel_height, int correlation)
{
    int row;
    int col;
    if (!image || !kernel || !dst || width < 1 || height < 1 || kernel_width < 1 || kernel_height < 1)
        return -1;
    if (kernel_height > height || kernel_width > width)
        return -1;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            long long acc = 0;
            int i;
            int j;
            for (i = 0; i < kernel_height; i++) {
                int rr = correlation ? row + i : row - i;
                if (rr >= height)
                    rr -= height;
                if (rr < 0)
                    rr += height;
                for (j = 0; j < kernel_width; j++) {
                    int cc = correlation ? col + j : col - j;
                    if (cc >= width)
                        cc -= width;
                    if (cc < 0)
                        cc += width;
                    acc += (long long)kernel[i * kernel_width + j] * image[rr * width + cc];
                }
            }
            if (acc <= 0)
                dst[row * width + col] = 0;
            else if (acc >= 255)
                dst[row * width + col] = 255;
            else
                dst[row * width + col] = (uint8_t)acc;
        }
    }
    return 0;
}

int vizion_shade_selftest(void)
{
    static const uint8_t ramp[64] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    };
    static const uint8_t clahe_expect[64] = {
        16, 32, 48, 48, 40, 40, 48, 64, 80, 96, 112, 112, 104, 104, 112, 128,
        143, 159, 175, 175, 167, 167, 175, 191, 155, 167, 179, 179, 173, 173, 179, 191,
        136, 144, 152, 152, 148, 148, 152, 160, 124, 136, 148, 148, 142, 142, 148, 160,
        143, 159, 175, 175, 167, 167, 175, 191, 207, 223, 239, 239, 231, 231, 239, 255
    };
    uint8_t src[4] = {1, 2, 3, 4};
    uint8_t kernel[1] = {10};
    uint8_t out[64];
    uint8_t flat[9];
    int index;
    memset(flat, 40, sizeof(flat));
    if (vizion_edge_band(flat, out, 3, 3, 1, 0.8, 1.6, 0.0, 2.0, NULL) != 0)
        return 1;
    for (index = 0; index < 9; index++) {
        if (out[index] != 40)
            return 2;
    }
    if (vizion_circular(src, kernel, out, 2, 2, 1, 1, 1) != 0)
        return 3;
    if (out[0] != 10 || out[1] != 20 || out[2] != 30 || out[3] != 40)
        return 4;
    if (vizion_circular(src, kernel, out, 2, 2, 1, 1, 0) != 0)
        return 5;
    if (out[0] != 10 || out[1] != 20 || out[2] != 30 || out[3] != 40)
        return 6;
    if (vizion_clahe(ramp, out, 8, 8, 1, 2.0, 2) != 0)
        return 7;
    if (memcmp(out, clahe_expect, 64) != 0)
        return 8;
    if (vizion_guided(flat, out, 3, 3, 1, 1, 10.0) != 0)
        return 9;
    for (index = 0; index < 9; index++) {
        if (out[index] != 40)
            return 10;
    }
    if (vizion_anisotropic(flat, out, 3, 3, 1, 0, 20.0, 0.2, 2) != 0)
        return 11;
    for (index = 0; index < 9; index++) {
        if (out[index] != 40)
            return 12;
    }
    if (vizion_shock(flat, out, 3, 3, 1, 0.5, 2, 0, 0.0) != 0)
        return 13;
    for (index = 0; index < 9; index++) {
        if (out[index] != 40)
            return 14;
    }
    if (vizion_frangi(flat, out, 3, 3, 1, 1.0, 1.0, 1) != 0)
        return 15;
    for (index = 0; index < 9; index++) {
        if (out[index] != 0)
            return 16;
    }
    return 0;
}
