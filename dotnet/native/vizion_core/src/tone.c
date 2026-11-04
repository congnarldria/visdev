#include "vizion_core.h"

#include <stdlib.h>
#include <string.h>

#include "tone_lab_tabs.h"

/* 直方圖等化、拉滿對比、拆色、OpenCV 8 位元色彩、以及 float 盒狀濾波的照明校正。
   小數截斷和 cvRound 的偶數捨入都對過 cv2。 */

extern int vizion_reflect101(int position, int length);

static uint8_t tone_trunc(double value)
{
    if (value <= 0.0)
        return 0;
    if (value >= 255.0)
        return 255;
    return (uint8_t)value;
}

static int tone_round_pos(double value)
{
    int whole = (int)value;
    double frac = value - (double)whole;
    if (frac > 0.5)
        return whole + 1;
    if (frac < 0.5)
        return whole;
    return (whole & 1) ? whole + 1 : whole;
}

static int tone_round_u8(double value)
{
    int rounded;
    if (value <= 0.0)
        return 0;
    if (value >= 255.0)
        return 255;
    rounded = tone_round_pos(value);
    if (rounded > 255)
        return 255;
    return rounded;
}

static int tone_sat(int value)
{
    if (value < 0)
        return 0;
    if (value > 255)
        return 255;
    return value;
}

static int tone_descale(int value, int shift)
{
    return (value + (1 << (shift - 1))) >> shift;
}

int vizion_equalize(const uint8_t* src, uint8_t* dst, int count)
{
    int hist[256];
    int lut[256];
    int index;
    int first;
    int total;
    int sum;
    double scale;
    if (!src || !dst || count < 1)
        return -1;
    memset(hist, 0, sizeof(hist));
    for (index = 0; index < count; index++)
        hist[src[index]]++;
    first = 0;
    while (first < 256 && hist[first] == 0)
        first++;
    total = count;
    if (first == 256 || hist[first] == total) {
        if (dst != src)
            memcpy(dst, src, (size_t)count);
        return 0;
    }
    scale = 255.0 / (double)(total - hist[first]);
    for (index = 0; index <= first; index++)
        lut[index] = 0;
    sum = 0;
    for (index = first + 1; index < 256; index++) {
        sum += hist[index];
        lut[index] = tone_round_u8((double)sum * scale);
    }
    for (index = 0; index < count; index++)
        dst[index] = (uint8_t)lut[src[index]];
    return 0;
}

static void tone_bgr_to_ycrcb(uint8_t b, uint8_t g, uint8_t r, int* y, int* cr, int* cb)
{
    int yy = tone_descale((int)b * 1868 + (int)g * 9617 + (int)r * 4899, 14);
    *y = yy;
    *cr = tone_sat(tone_descale(((int)r - yy) * 11682 + (128 << 14), 14));
    *cb = tone_sat(tone_descale(((int)b - yy) * 9241 + (128 << 14), 14));
}

static void tone_ycrcb_to_bgr(int y, int cr, int cb, uint8_t* b, uint8_t* g, uint8_t* r)
{
    *b = (uint8_t)tone_sat(y + tone_descale((cb - 128) * 29049, 14));
    *g = (uint8_t)tone_sat(y + tone_descale((cb - 128) * -5636 + (cr - 128) * -11698, 14));
    *r = (uint8_t)tone_sat(y + tone_descale((cr - 128) * 22987, 14));
}

int vizion_equalize_bgr(const uint8_t* src, uint8_t* dst, int count)
{
    uint8_t* y;
    uint8_t* cr;
    uint8_t* cb;
    int index;
    int code;
    if (!src || !dst || count < 1)
        return -1;
    y = (uint8_t*)malloc((size_t)count * 3);
    if (!y)
        return -1;
    cr = y + count;
    cb = cr + count;
    for (index = 0; index < count; index++) {
        int yy, rr, bb;
        tone_bgr_to_ycrcb(src[index * 3], src[index * 3 + 1], src[index * 3 + 2], &yy, &rr, &bb);
        y[index] = (uint8_t)yy;
        cr[index] = (uint8_t)rr;
        cb[index] = (uint8_t)bb;
    }
    code = vizion_equalize(y, y, count);
    if (code != 0) {
        free(y);
        return code;
    }
    for (index = 0; index < count; index++)
        tone_ycrcb_to_bgr(y[index], cr[index], cb[index], &dst[index * 3], &dst[index * 3 + 1], &dst[index * 3 + 2]);
    free(y);
    return 0;
}

int vizion_scale_max(const uint8_t* src, uint8_t* dst, int count, const uint8_t* domain)
{
    int index;
    int lo = 255;
    int hi = 0;
    int any = 0;
    double scale;
    if (!src || !dst || count < 1)
        return -1;
    for (index = 0; index < count; index++) {
        if (domain && !domain[index])
            continue;
        any = 1;
        if (src[index] < lo)
            lo = src[index];
        if (src[index] > hi)
            hi = src[index];
    }
    if (!any || hi <= lo) {
        memset(dst, 0, (size_t)count);
        return 0;
    }
    scale = 255.0 / (double)(hi - lo);
    for (index = 0; index < count; index++) {
        double value;
        if (domain && !domain[index]) {
            dst[index] = 0;
            continue;
        }
        value = ((double)src[index] - (double)lo) * scale;
        dst[index] = tone_trunc(value);
    }
    return 0;
}

int vizion_split_bgr(const uint8_t* bgr, uint8_t* blue, uint8_t* green, uint8_t* red, int count)
{
    int index;
    if (!bgr || !blue || !green || !red || count < 1)
        return -1;
    for (index = 0; index < count; index++) {
        blue[index] = bgr[index * 3];
        green[index] = bgr[index * 3 + 1];
        red[index] = bgr[index * 3 + 2];
    }
    return 0;
}

int vizion_merge_bgr(const uint8_t* blue, const uint8_t* green, const uint8_t* red, uint8_t* bgr, int count)
{
    int index;
    if (!blue || !green || !red || !bgr || count < 1)
        return -1;
    for (index = 0; index < count; index++) {
        bgr[index * 3] = blue[index];
        bgr[index * 3 + 1] = green[index];
        bgr[index * 3 + 2] = red[index];
    }
    return 0;
}

static int vizion_sdiv[256];
static int vizion_hdiv[256];
static int vizion_hsv_ready;

static void vizion_hsv_init(void)
{
    int index;
    if (vizion_hsv_ready)
        return;
    vizion_sdiv[0] = 0;
    vizion_hdiv[0] = 0;
    for (index = 1; index < 256; index++) {
        vizion_sdiv[index] = tone_round_pos((255 << 12) / (double)index);
        vizion_hdiv[index] = tone_round_pos((180 << 12) / (6.0 * (double)index));
    }
    vizion_hsv_ready = 1;
}

static void tone_rgb_hsv(uint8_t r, uint8_t g, uint8_t b, uint8_t* h, uint8_t* s, uint8_t* v)
{
    int vmax = r;
    int vmin = r;
    int diff;
    int vr;
    int vg;
    int hue;
    int sat;
    if (g > vmax)
        vmax = g;
    if (b > vmax)
        vmax = b;
    if (g < vmin)
        vmin = g;
    if (b < vmin)
        vmin = b;
    diff = vmax - vmin;
    vr = vmax == r ? -1 : 0;
    vg = vmax == g ? -1 : 0;
    sat = (diff * vizion_sdiv[vmax] + (1 << 11)) >> 12;
    hue = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
    hue = (hue * vizion_hdiv[diff] + (1 << 11)) >> 12;
    if (hue < 0)
        hue += 180;
    *h = (uint8_t)hue;
    *s = (uint8_t)sat;
    *v = (uint8_t)vmax;
}

static void tone_rgb_yuv(uint8_t r, uint8_t g, uint8_t b, uint8_t* y, uint8_t* u, uint8_t* v)
{
    int yy = tone_descale((int)r * 4899 + (int)g * 9617 + (int)b * 1868, 14);
    int uu = tone_descale(((int)b - yy) * 8061 + (128 << 14), 14);
    int vv = tone_descale(((int)r - yy) * 14369 + (128 << 14), 14);
    *y = (uint8_t)tone_sat(yy);
    *u = (uint8_t)tone_sat(uu);
    *v = (uint8_t)tone_sat(vv);
}

static const int vizion_lab_coeff[9] = { 1777, 1541, 778, 871, 2929, 296, 73, 448, 3575 };

static void tone_rgb_lab(uint8_t r, uint8_t g, uint8_t b, uint8_t* l, uint8_t* a, uint8_t* bb)
{
    int red = vizion_lab_gamma[r];
    int green = vizion_lab_gamma[g];
    int blue = vizion_lab_gamma[b];
    int ix = tone_descale(red * vizion_lab_coeff[0] + green * vizion_lab_coeff[1] + blue * vizion_lab_coeff[2], 12);
    int iy = tone_descale(red * vizion_lab_coeff[3] + green * vizion_lab_coeff[4] + blue * vizion_lab_coeff[5], 12);
    int iz = tone_descale(red * vizion_lab_coeff[6] + green * vizion_lab_coeff[7] + blue * vizion_lab_coeff[8], 12);
    int fx;
    int fy;
    int fz;
    if (ix < 0 || iy < 0 || iz < 0 || ix >= 3072 || iy >= 3072 || iz >= 3072) {
        *l = 0;
        *a = 128;
        *bb = 128;
        return;
    }
    fx = vizion_lab_cbrt[ix];
    fy = vizion_lab_cbrt[iy];
    fz = vizion_lab_cbrt[iz];
    *l = (uint8_t)tone_sat(tone_descale(296 * fy + -1336934, 15));
    *a = (uint8_t)tone_sat(tone_descale(500 * (fx - fy) + 128 * (1 << 15), 15));
    *bb = (uint8_t)tone_sat(tone_descale(200 * (fy - fz) + 128 * (1 << 15), 15));
}

int vizion_rgb_convert(
    const uint8_t* red, const uint8_t* green, const uint8_t* blue,
    uint8_t* first, uint8_t* second, uint8_t* third, int count, int space)
{
    int index;
    if (!red || !green || !blue || !first || !second || !third || count < 1 || space < 0 || space > 2)
        return -1;
    if (space == 0)
        vizion_hsv_init();
    for (index = 0; index < count; index++) {
        if (space == 0)
            tone_rgb_hsv(red[index], green[index], blue[index], &first[index], &second[index], &third[index]);
        else if (space == 1)
            tone_rgb_lab(red[index], green[index], blue[index], &first[index], &second[index], &third[index]);
        else
            tone_rgb_yuv(red[index], green[index], blue[index], &first[index], &second[index], &third[index]);
    }
    return 0;
}

static int tone_box_float(
    const float* src, float* dst, int width, int height, int mask_width, int mask_height)
{
    double* horizontal;
    int row;
    int col;
    int anchor_x = mask_width / 2;
    int anchor_y = mask_height / 2;
    double area = (double)mask_width * (double)mask_height;
    double scale = 1.0 / area;
    horizontal = (double*)malloc((size_t)width * (size_t)height * sizeof(double));
    if (!horizontal)
        return -1;
    for (row = 0; row < height; row++) {
        const float* line = src + (size_t)row * (size_t)width;
        double* out = horizontal + (size_t)row * (size_t)width;
        double sum = 0.0;
        int tap;
        for (tap = 0; tap < mask_width; tap++)
            sum += line[vizion_reflect101(tap - anchor_x, width)];
        out[0] = sum;
        for (col = 1; col < width; col++) {
            sum += line[vizion_reflect101(col + mask_width - 1 - anchor_x, width)];
            sum -= line[vizion_reflect101(col - 1 - anchor_x, width)];
            out[col] = sum;
        }
    }
    for (col = 0; col < width; col++) {
        double sum = 0.0;
        int tap;
        for (tap = 0; tap < mask_height; tap++)
            sum += horizontal[(size_t)vizion_reflect101(tap - anchor_y, height) * (size_t)width + col];
        dst[col] = (float)(sum * scale);
        for (row = 1; row < height; row++) {
            sum += horizontal[(size_t)vizion_reflect101(row + mask_height - 1 - anchor_y, height) * (size_t)width + col];
            sum -= horizontal[(size_t)vizion_reflect101(row - 1 - anchor_y, height) * (size_t)width + col];
            dst[(size_t)row * (size_t)width + col] = (float)(sum * scale);
        }
    }
    free(horizontal);
    return 0;
}

int vizion_illuminate(
    const uint8_t* src, uint8_t* dst, int width, int height,
    int mask_width, int mask_height, double factor, const uint8_t* domain)
{
    int count;
    int index;
    int valid = 0;
    float sum = 0.f;
    float fill;
    float* work;
    float* blur;
    int code;
    if (!src || !dst || width < 1 || height < 1 || mask_width < 1 || mask_height < 1)
        return -1;
    if ((mask_width & 1) == 0 || (mask_height & 1) == 0)
        return -1;
    count = width * height;
    for (index = 0; index < count; index++) {
        if (domain && !domain[index])
            continue;
        sum += (float)src[index];
        valid++;
    }
    fill = valid ? sum / (float)valid : 0.f;
    work = (float*)malloc((size_t)count * sizeof(float) * 2);
    if (!work)
        return -1;
    blur = work + count;
    for (index = 0; index < count; index++)
        work[index] = (domain && !domain[index]) ? fill : (float)src[index];
    code = tone_box_float(work, blur, width, height, mask_width, mask_height);
    if (code != 0) {
        free(work);
        return code;
    }
    for (index = 0; index < count; index++) {
        float diff = (float)src[index] - blur[index];
        float value = diff + blur[index] * (float)factor + fill * (float)(1.0 - factor);
        dst[index] = (domain && !domain[index]) ? 0 : tone_trunc(value);
    }
    free(work);
    return 0;
}

static int tone_same(const uint8_t* left, const uint8_t* right, int count)
{
    int index;
    for (index = 0; index < count; index++) {
        if (left[index] != right[index])
            return 0;
    }
    return 1;
}

int vizion_tone_selftest(void)
{
    static const uint8_t uneven[8] = { 0, 0, 10, 10, 10, 200, 255, 30 };
    static const uint8_t uneven_eq[8] = { 0, 0, 128, 128, 128, 212, 255, 170 };
    static const uint8_t flat[4] = { 7, 7, 7, 7 };
    uint8_t out[8];
    uint8_t h, s, v, y, u, vv, l, a, b;
    uint8_t red = 10, green = 20, blue = 30;
    uint8_t bgr[3] = { 30, 20, 10 };
    uint8_t back[3];
    static const uint8_t plane[20] = {
        1, 9, 3, 8, 2,
        7, 4, 6, 5, 0,
        12, 20, 15, 11, 9,
        3, 3, 3, 40, 2
    };
    uint8_t domain[20];
    static const uint8_t lit_expect[20] = {
        0, 10, 5, 9, 0,
        0, 4, 6, 5, 0,
        0, 20, 14, 10, 0,
        0, 2, 0, 38, 0
    };
    uint8_t lit[20];
    int column;
    uint8_t scaled[4];
    static const uint8_t scale_src[4] = { 10, 20, 30, 40 };
    static const uint8_t scale_dom[4] = { 0, 255, 255, 0 };
    static const uint8_t scale_expect[4] = { 0, 0, 255, 0 };
    if (vizion_equalize(uneven, out, 8) != 0 || !tone_same(out, uneven_eq, 8))
        return 1;
    if (vizion_equalize(flat, out, 4) != 0 || !tone_same(out, flat, 4))
        return 2;
    if (vizion_rgb_convert(&red, &green, &blue, &h, &s, &v, 1, 0) != 0 || h != 105 || s != 170 || v != 30)
        return 3;
    red = 255;
    green = 0;
    blue = 0;
    if (vizion_rgb_convert(&red, &green, &blue, &y, &u, &vv, 1, 2) != 0 || y != 76 || u != 91 || vv != 255)
        return 4;
    red = 19;
    green = 25;
    blue = 129;
    if (vizion_rgb_convert(&red, &green, &blue, &l, &a, &b, 1, 1) != 0 || l != 45 || a != 166 || b != 70)
        return 5;
    if (vizion_equalize_bgr(bgr, back, 1) != 0)
        return 6;
    if (vizion_scale_max(scale_src, scaled, 4, scale_dom) != 0 || !tone_same(scaled, scale_expect, 4))
        return 7;
    memset(domain, 0, sizeof(domain));
    for (column = 0; column < 20; column++) {
        if (column % 5 != 0 && column % 5 != 4)
            domain[column] = 255;
    }
    if (vizion_illuminate(plane, lit, 5, 4, 3, 3, 0.5, domain) != 0 || !tone_same(lit, lit_expect, 20))
        return 8;
    if (vizion_split_bgr(bgr, &blue, &green, &red, 1) != 0 || blue != 30 || green != 20 || red != 10)
        return 9;
    if (vizion_merge_bgr(&blue, &green, &red, back, 1) != 0 || back[0] != 30 || back[1] != 20 || back[2] != 10)
        return 10;
    return 0;
}
