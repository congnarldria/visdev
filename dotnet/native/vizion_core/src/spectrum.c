#include "vizion_core.h"

#include <math.h>
#include <string.h>

static int spectrum_ok(const double* real, const double* imag, int count)
{
    return real && imag && count > 0;
}

/* 整張填同一個複數。 */

int vizion_complex_fill(double* real, double* imag, int count, double re, double im)
{
    int index;
    if (!real || !imag || count < 1)
        return -1;
    for (index = 0; index < count; index++) {
        real[index] = re;
        imag[index] = im;
    }
    return 0;
}

/* 每個頻率乘上同一個實數。 */

int vizion_complex_scale(const double* real, const double* imag, double* dst_real, double* dst_imag, int count, double factor)
{
    int index;
    if (!spectrum_ok(real, imag, count) || !dst_real || !dst_imag)
        return -1;
    for (index = 0; index < count; index++) {
        dst_real[index] = real[index] * factor;
        dst_imag[index] = imag[index] * factor;
    }
    return 0;
}

/* 虛部變號，實部不變。 */

int vizion_complex_conj(const double* real, const double* imag, double* dst_real, double* dst_imag, int count)
{
    int index;
    if (!spectrum_ok(real, imag, count) || !dst_real || !dst_imag)
        return -1;
    for (index = 0; index < count; index++) {
        dst_real[index] = real[index];
        dst_imag[index] = -imag[index];
    }
    return 0;
}

/* 逐點相加。 */

int vizion_complex_add(
    const double* left_real, const double* left_imag, const double* right_real, const double* right_imag,
    double* dst_real, double* dst_imag, int count)
{
    int index;
    if (!spectrum_ok(left_real, left_imag, count) || !right_real || !right_imag || !dst_real || !dst_imag)
        return -1;
    for (index = 0; index < count; index++) {
        dst_real[index] = left_real[index] + right_real[index];
        dst_imag[index] = left_imag[index] + right_imag[index];
    }
    return 0;
}

/* 逐點相減。 */

int vizion_complex_sub(
    const double* left_real, const double* left_imag, const double* right_real, const double* right_imag,
    double* dst_real, double* dst_imag, int count)
{
    int index;
    if (!spectrum_ok(left_real, left_imag, count) || !right_real || !right_imag || !dst_real || !dst_imag)
        return -1;
    for (index = 0; index < count; index++) {
        dst_real[index] = left_real[index] - right_real[index];
        dst_imag[index] = left_imag[index] - right_imag[index];
    }
    return 0;
}

/* 逐點相乘。相乘就是空間上的循環卷積。 */

int vizion_complex_mul(
    const double* left_real, const double* left_imag, const double* right_real, const double* right_imag,
    double* dst_real, double* dst_imag, int count)
{
    int index;
    if (!spectrum_ok(left_real, left_imag, count) || !right_real || !right_imag || !dst_real || !dst_imag)
        return -1;
    for (index = 0; index < count; index++) {
        double ar = left_real[index];
        double ai = left_imag[index];
        double br = right_real[index];
        double bi = right_imag[index];
        dst_real[index] = fma(ar, br, -(ai * bi));
        dst_imag[index] = fma(ar, bi, ai * br);
    }
    return 0;
}

/* 絕對值不大於 1e-12 的除數，那個頻率的商是 0。 */

int vizion_complex_div(
    const double* left_real, const double* left_imag, const double* right_real, const double* right_imag,
    double* dst_real, double* dst_imag, int count)
{
    int index;
    if (!spectrum_ok(left_real, left_imag, count) || !right_real || !right_imag || !dst_real || !dst_imag)
        return -1;
    for (index = 0; index < count; index++) {
        double ar = left_real[index];
        double ai = left_imag[index];
        double br = right_real[index];
        double bi = right_imag[index];
        double abs_br = fabs(br);
        double abs_bi = fabs(bi);
        double rat;
        double scl;
        if (!(hypot(br, bi) > 1e-12)) {
            dst_real[index] = 0.0;
            dst_imag[index] = 0.0;
            continue;
        }
        if (abs_br >= abs_bi) {
            rat = bi / br;
            scl = 1.0 / (br + bi * rat);
            dst_real[index] = (ar + ai * rat) * scl;
            dst_imag[index] = (ai - ar * rat) * scl;
        } else {
            rat = br / bi;
            scl = 1.0 / (bi + br * rat);
            dst_real[index] = (ar * rat + ai) * scl;
            dst_imag[index] = (ai * rat - ar) * scl;
        }
    }
    return 0;
}

static int spectrum_mod(int value, int size)
{
    int wrapped = value % size;
    if (wrapped < 0)
        wrapped += size;
    return wrapped;
}

/* center 為 1 時把低頻搬到中央，為 0 時搬回角落。 */

int vizion_complex_shift(
    const double* real, const double* imag, double* dst_real, double* dst_imag, int width, int height, int center)
{
    int row;
    int col;
    int row_shift;
    int col_shift;
    if (!spectrum_ok(real, imag, width * height) || !dst_real || !dst_imag || width < 1 || height < 1)
        return -1;
    row_shift = center ? height / 2 : -(height / 2);
    col_shift = center ? width / 2 : -(width / 2);
    for (row = 0; row < height; row++) {
        int src_row = spectrum_mod(row - row_shift, height);
        for (col = 0; col < width; col++) {
            int src_col = spectrum_mod(col - col_shift, width);
            int dst = row * width + col;
            int src = src_row * width + src_col;
            dst_real[dst] = real[src];
            dst_imag[dst] = imag[src];
        }
    }
    return 0;
}

/* 只留直流、清掉直流、只留實部，或只留虛部。 */

int vizion_complex_mask(
    const double* real, const double* imag, double* dst_real, double* dst_imag, int width, int height, int mode)
{
    int count;
    int index;
    if (!spectrum_ok(real, imag, width * height) || !dst_real || !dst_imag || width < 1 || height < 1 || mode < 0 || mode > 3)
        return -1;
    count = width * height;
    if (mode == 0) {
        memset(dst_real, 0, (size_t)count * sizeof(double));
        memset(dst_imag, 0, (size_t)count * sizeof(double));
        dst_real[0] = real[0];
        dst_imag[0] = imag[0];
        return 0;
    }
    if (mode == 1) {
        memcpy(dst_real, real, (size_t)count * sizeof(double));
        memcpy(dst_imag, imag, (size_t)count * sizeof(double));
        dst_real[0] = 0.0;
        dst_imag[0] = 0.0;
        return 0;
    }
    if (mode == 2) {
        memcpy(dst_real, real, (size_t)count * sizeof(double));
        memset(dst_imag, 0, (size_t)count * sizeof(double));
        return 0;
    }
    for (index = 0; index < count; index++) {
        dst_real[index] = 0.0;
        dst_imag[index] = imag[index];
    }
    return 0;
}

/* 所有頻率絕對值平方的平均。常數 g 的能量是 g 的平方乘上點數。 */

int vizion_complex_energy(const double* real, const double* imag, int count, double* energy)
{
    int index;
    double sum = 0.0;
    if (!spectrum_ok(real, imag, count) || !energy)
        return -1;
    for (index = 0; index < count; index++) {
        double magnitude = hypot(real[index], imag[index]);
        sum += magnitude * magnitude;
    }
    *energy = sum / (double)count;
    return 0;
}

int vizion_spectrum_selftest(void)
{
    double real[4] = {1.0, 2.0, 3.0, 4.0};
    double imag[4] = {5.0, 6.0, 7.0, 8.0};
    double dst_real[4];
    double dst_imag[4];
    double energy = 0.0;
    if (vizion_complex_shift(real, imag, dst_real, dst_imag, 2, 2, 1) != 0)
        return 1;
    if (dst_real[0] != 4.0 || dst_real[1] != 3.0 || dst_real[2] != 2.0 || dst_real[3] != 1.0)
        return 2;
    if (dst_imag[0] != 8.0 || dst_imag[3] != 5.0)
        return 3;
    if (vizion_complex_mask(real, imag, dst_real, dst_imag, 2, 2, 0) != 0)
        return 4;
    if (dst_real[0] != 1.0 || dst_imag[0] != 5.0 || dst_real[1] != 0.0 || dst_imag[2] != 0.0)
        return 5;
    if (vizion_complex_div(real, imag, real, imag, dst_real, dst_imag, 4) != 0)
        return 6;
    if (dst_real[0] != 1.0 || dst_imag[0] != 0.0)
        return 7;
    {
        double zero_real = 0.0;
        double zero_imag = 0.0;
        double quotient_real = 1.0;
        double quotient_imag = 1.0;
        if (vizion_complex_div(real, imag, &zero_real, &zero_imag, &quotient_real, &quotient_imag, 1) != 0)
            return 8;
        if (quotient_real != 0.0 || quotient_imag != 0.0)
            return 9;
    }
    real[0] = 3.0;
    imag[0] = 4.0;
    if (vizion_complex_energy(real, imag, 1, &energy) != 0 || energy != 25.0)
        return 10;
    return 0;
}
