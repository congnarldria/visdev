#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
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

uint8_t grade_round_d(double value);

static double spectrum_abs(double re, double im)
{
    double ax = fabs(re);
    double ay = fabs(im);
    double ratio;
    if (ax < ay) {
        double swap = ax;
        ax = ay;
        ay = swap;
    }
    if (ax == 0.0)
        return 0.0;
    ratio = ay / ax;
    return ax * sqrt(1.0 + ratio * ratio);
}

static double spectrum_fftfreq(int index, int length)
{
    int half = (length - 1) / 2 + 1;
    double scale = 1.0 / (double)length;
    if (index < half)
        return (double)index * scale;
    return (double)(index - length) * scale;
}

static double spectrum_pairwise(const double* values, int count)
{
    int index;
    if (count < 8) {
        double sum = -0.0;
        for (index = 0; index < count; index++)
            sum += values[index];
        return sum;
    }
    if (count <= 128) {
        double lane[8];
        double sum;
        int end;
        for (index = 0; index < 8; index++)
            lane[index] = values[index];
        end = count - (count % 8);
        for (index = 8; index < end; index += 8) {
            int lane_index;
            for (lane_index = 0; lane_index < 8; lane_index++)
                lane[lane_index] += values[index + lane_index];
        }
        sum = ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7]));
        for (; index < count; index++)
            sum += values[index];
        return sum;
    }
    {
        int mid = count / 2;
        mid -= mid % 8;
        return spectrum_pairwise(values, mid) + spectrum_pairwise(values + mid, count - mid);
    }
}

/* 用相位把影像平移。整數位移時，超出的部分從對面捲回來。 */

int vizion_complex_translate(
    const double* real, const double* imag, double* dst_real, double* dst_imag,
    int width, int height, double row, double column)
{
    int y;
    int x;
    double turn = 2.0 * acos(-1.0);
    if (!real || !imag || !dst_real || !dst_imag || width < 1 || height < 1)
        return -1;
    for (y = 0; y < height; y++) {
        double freq_row = spectrum_fftfreq(y, height);
        for (x = 0; x < width; x++) {
            int index = y * width + x;
            double theta = -turn * (freq_row * row + spectrum_fftfreq(x, width) * column);
            double phase_real = cos(theta);
            double phase_imag = sin(theta);
            double ar = real[index];
            double ai = imag[index];
            dst_real[index] = fma(ar, phase_real, -(ai * phase_imag));
            dst_imag[index] = fma(ar, phase_imag, ai * phase_real);
        }
    }
    return 0;
}

/* 不是 0 的頻率除以自己的絕對值，只留相位。 */

int vizion_complex_normalize(const double* real, const double* imag, double* dst_real, double* dst_imag, int count)
{
    int index;
    if (!real || !imag || !dst_real || !dst_imag || count < 1)
        return -1;
    for (index = 0; index < count; index++) {
        double magnitude = spectrum_abs(real[index], imag[index]);
        double inverse;
        if (!(magnitude > 1e-12)) {
            dst_real[index] = 0.0;
            dst_imag[index] = 0.0;
            continue;
        }
        inverse = 1.0 / magnitude;
        dst_real[index] = real[index] * inverse;
        dst_imag[index] = imag[index] * inverse;
    }
    return 0;
}

static double spectrum_distance(int row, int column, int height, int width)
{
    int row_distance = row < height - row ? row : height - row;
    int column_distance = column < width - column ? column : width - column;
    return hypot((double)row_distance, (double)column_distance);
}

/* low 留下距離不大於半徑的頻率，high 去掉那些。距離從左上角量，對面的低頻也算。 */

int vizion_complex_freq(
    const double* real, const double* imag, double* dst_real, double* dst_imag,
    int width, int height, double radius, int high)
{
    int row;
    int column;
    if (!real || !imag || !dst_real || !dst_imag || width < 1 || height < 1 || radius < 0.0)
        return -1;
    for (row = 0; row < height; row++) {
        for (column = 0; column < width; column++) {
            int index = row * width + column;
            int keep = spectrum_distance(row, column, height, width) <= radius;
            if (high)
                keep = !keep;
            dst_real[index] = keep ? real[index] : 0.0;
            dst_imag[index] = keep ? imag[index] : 0.0;
        }
    }
    return 0;
}

/* 半徑內的絕對值平方加總後，除以整張的點數。 */

int vizion_complex_freq_energy(
    const double* real, const double* imag, int width, int height, double radius, double* energy)
{
    int count;
    int index;
    int row;
    int column;
    double* values;
    if (!real || !imag || !energy || width < 1 || height < 1 || radius < 0.0)
        return -1;
    count = width * height;
    values = (double*)malloc((size_t)count * sizeof(double));
    if (!values)
        return -1;
    index = 0;
    for (row = 0; row < height; row++) {
        for (column = 0; column < width; column++, index++) {
            if (spectrum_distance(row, column, height, width) <= radius) {
                double magnitude = spectrum_abs(real[index], imag[index]);
                values[index] = magnitude * magnitude;
            } else {
                values[index] = 0.0;
            }
        }
    }
    *energy = spectrum_pairwise(values, count) / (double)count;
    free(values);
    return 0;
}

/* real、imag、abs 四捨五入到 0 到 255。log 用 log(1+絕對值) 拉到這張的最大值。 */

int vizion_complex_view(const double* real, const double* imag, uint8_t* dst, int count, int mode)
{
    int index;
    double peak;
    if (!real || !imag || !dst || count < 1 || mode < 0 || mode > 3)
        return -1;
    if (mode != 3) {
        for (index = 0; index < count; index++) {
            double value = mode == 0 ? real[index] : mode == 1 ? imag[index] : spectrum_abs(real[index], imag[index]);
            dst[index] = grade_round_d(value);
        }
        return 0;
    }
    peak = 0.0;
    for (index = 0; index < count; index++) {
        double value = log1p(spectrum_abs(real[index], imag[index]));
        if (value > peak)
            peak = value;
    }
    if (!(peak > 0.0)) {
        memset(dst, 0, (size_t)count);
        return 0;
    }
    for (index = 0; index < count; index++) {
        double value = log1p(spectrum_abs(real[index], imag[index]));
        dst[index] = grade_round_d(value / peak * 255.0);
    }
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
    real[0] = 3.0;
    imag[0] = 4.0;
    if (vizion_complex_normalize(real, imag, dst_real, dst_imag, 1) != 0)
        return 11;
    if (dst_real[0] != 3.0 * (1.0 / 5.0) || dst_imag[0] != 4.0 * (1.0 / 5.0))
        return 12;
    if (vizion_complex_freq(real, imag, dst_real, dst_imag, 2, 2, 0.0, 0) != 0)
        return 13;
    if (dst_real[0] != 3.0 || dst_real[1] != 0.0 || dst_imag[1] != 0.0)
        return 14;
    return 0;
}
