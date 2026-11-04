#include "vizion_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 形狀比對命中之後的姿態精修。半像素、角度、拋物線、沿法線的最小二乘，順序和 C# 相同。 */

static const double vizion_pi = 3.14159265358979323846;

static int vizion_round_even(double value)
{
    double base = floor(value);
    double diff = value - base;
    int whole = (int)base;
    if (diff > 0.5)
        return whole + 1;
    if (diff < 0.5)
        return whole;
    return (whole & 1) ? whole + 1 : whole;
}

static double vizion_wrap(double angle)
{
    while (angle <= -vizion_pi)
        angle += 2 * vizion_pi;
    while (angle > vizion_pi)
        angle -= 2 * vizion_pi;
    return angle;
}

static double vizion_parabola(double left, double center, double right)
{
    double denom = left - 2 * center + right;
    double shift;
    if (fabs(denom) < 1e-6 || center < left || center < right)
        return 0;
    shift = 0.5 * (left - right) / denom;
    if (shift < -0.75)
        return -0.75;
    if (shift > 0.75)
        return 0.75;
    return shift;
}

static void vizion_prepare(
    const float* mx, const float* my, const float* mgx, const float* mgy, int count,
    double angle, int scale, float* rx, float* ry, float* rgx, float* rgy)
{
    float cosine = (float)cos(angle);
    float sine = (float)sin(angle);
    float inverse = (float)(1.0 / scale);
    int i;
    for (i = 0; i < count; i++) {
        rx[i] = mx[i] * cosine + my[i] * sine;
        ry[i] = -mx[i] * sine + my[i] * cosine;
        rgx[i] = mgx[i] * cosine + mgy[i] * sine;
        rgy[i] = -mgx[i] * sine + mgy[i] * cosine;
    }
    if (inverse == 1.f)
        return;
    for (i = 0; i < count; i++) {
        rx[i] *= inverse;
        ry[i] *= inverse;
    }
}

static double vizion_score_pose(
    const float* mx, const float* my, const float* mgx, const float* mgy, int count,
    const float* gx, const float* gy, const float* mag, int width, int height, int scale,
    float* rx, float* ry, float* rgx, float* rgy,
    double column, double row, double angle, double min_contrast)
{
    vizion_prepare(mx, my, mgx, mgy, count, angle, scale, rx, ry, rgx, rgy);
    return vizion_shape_score(gx, gy, mag, width, height, column / (double)scale, row / (double)scale, rx, ry, rgx, rgy, count, min_contrast);
}

static int vizion_sample(const float* image, int width, int height, double x, double y, float* value)
{
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    float wx;
    float wy;
    int index;
    float top;
    float bottom;
    if (x0 < 0 || y0 < 0 || x0 + 1 >= width || y0 + 1 >= height)
        return 0;
    wx = (float)(x - x0);
    wy = (float)(y - y0);
    index = y0 * width + x0;
    top = (1.f - wx) * image[index] + wx * image[index + 1];
    bottom = (1.f - wx) * image[index + width] + wx * image[index + width + 1];
    *value = (1.f - wy) * top + wy * bottom;
    return 1;
}

static int vizion_solve3(
    double a00, double a01, double a02, double a11, double a12, double a22,
    double b0, double b1, double b2, double* x0, double* x1, double* x2)
{
    double rows[3][4] = {
        {a00, a01, a02, b0},
        {a01, a11, a12, b1},
        {a02, a12, a22, b2},
    };
    int col;
    for (col = 0; col < 3; col++) {
        int pivot = col;
        double best = fabs(rows[col][col]);
        int row;
        double div;
        for (row = col + 1; row < 3; row++) {
            double value = fabs(rows[row][col]);
            if (value > best) {
                best = value;
                pivot = row;
            }
        }
        if (best < 1e-12) {
            *x0 = *x1 = *x2 = 0;
            return 0;
        }
        if (pivot != col) {
            int k;
            for (k = col; k < 4; k++) {
                double swap = rows[col][k];
                rows[col][k] = rows[pivot][k];
                rows[pivot][k] = swap;
            }
        }
        div = rows[col][col];
        {
            int k;
            for (k = col; k < 4; k++)
                rows[col][k] /= div;
        }
        for (row = 0; row < 3; row++) {
            double factor;
            int k;
            if (row == col)
                continue;
            factor = rows[row][col];
            if (factor == 0)
                continue;
            for (k = col; k < 4; k++)
                rows[row][k] -= factor * rows[col][k];
        }
    }
    *x0 = rows[0][3];
    *x1 = rows[1][3];
    *x2 = rows[2][3];
    return 1;
}

static int vizion_solve_translation(double a00, double a01, double a11, double b0, double b1, double* d_col, double* d_row)
{
    double det = a00 * a11 - a01 * a01;
    *d_col = 0;
    *d_row = 0;
    if (fabs(det) > 1e-8) {
        *d_col = (a11 * b0 - a01 * b1) / det;
        *d_row = (a00 * b1 - a01 * b0) / det;
        return 1;
    }
    if (a00 > 1e-8) {
        *d_col = b0 / a00;
        return 1;
    }
    if (a11 > 1e-8) {
        *d_row = b1 / a11;
        return 1;
    }
    return 0;
}

static void vizion_snap(
    const float* mx, const float* my, const float* mgx, const float* mgy, int count,
    const float* gx, const float* gy, const float* mag, int width, int height, int scale,
    float* rx, float* ry, float* rgx, float* rgy, double min_contrast,
    double* row, double* column, double* angle, double* score)
{
    int row0 = vizion_round_even(*row);
    int col0 = vizion_round_even(*column);
    int d_row;
    int d_col;
    for (d_row = -2; d_row <= 2; d_row++) {
        for (d_col = -2; d_col <= 2; d_col++) {
            double next_row = row0 + d_row * 0.5;
            double next_col = col0 + d_col * 0.5;
            double next_score;
            if (next_row < 2 || next_col < 2 || next_row >= height - 2 || next_col >= width - 2)
                continue;
            next_score = vizion_score_pose(mx, my, mgx, mgy, count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, next_col, next_row, *angle, min_contrast);
            if (next_score > *score) {
                *row = next_row;
                *column = next_col;
                *score = next_score;
            }
        }
    }
}

static void vizion_polish(
    const float* mx, const float* my, const float* mgx, const float* mgy, int count,
    const float* gx, const float* gy, const float* mag, int width, int height, int scale,
    float* rx, float* ry, float* rgx, float* rgy, double min_contrast,
    double column, double row, double* angle, double* score)
{
    double origin = *angle;
    double step = 0.5 * vizion_pi / 180.0;
    int index;
    for (index = -6; index <= 6; index++) {
        double next_angle = origin + index * step;
        double next_score = vizion_score_pose(mx, my, mgx, mgy, count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, column, row, next_angle, min_contrast);
        if (next_score > *score) {
            *angle = next_angle;
            *score = next_score;
        }
    }
}

static void vizion_parabolic(
    const float* mx, const float* my, const float* mgx, const float* mgy, int count,
    const float* gx, const float* gy, const float* mag, int width, int height, int scale,
    float* rx, float* ry, float* rgx, float* rgy, double min_contrast,
    double* row, double* column, double* angle, double score)
{
    double step = 0.8 * vizion_pi / 180.0;
    const double spacing = 0.5;
    int pass;
    double next_row = *row;
    double next_col = *column;
    double next_angle = *angle;
    for (pass = 0; pass < 2; pass++) {
        double center = vizion_score_pose(mx, my, mgx, mgy, count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, next_col, next_row, next_angle, min_contrast);
        double d_row = vizion_parabola(
            vizion_score_pose(mx, my, mgx, mgy, count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, next_col, next_row - spacing, next_angle, min_contrast),
            center,
            vizion_score_pose(mx, my, mgx, mgy, count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, next_col, next_row + spacing, next_angle, min_contrast));
        double d_col = vizion_parabola(
            vizion_score_pose(mx, my, mgx, mgy, count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, next_col - spacing, next_row, next_angle, min_contrast),
            center,
            vizion_score_pose(mx, my, mgx, mgy, count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, next_col + spacing, next_row, next_angle, min_contrast));
        double d_angle = vizion_parabola(
            vizion_score_pose(mx, my, mgx, mgy, count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, next_col, next_row, next_angle - step, min_contrast),
            center,
            vizion_score_pose(mx, my, mgx, mgy, count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, next_col, next_row, next_angle + step, min_contrast));
        next_row += d_row * spacing;
        next_col += d_col * spacing;
        next_angle += d_angle * step;
        step *= 0.5;
    }
    {
        double next_score = vizion_score_pose(mx, my, mgx, mgy, count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, next_col, next_row, next_angle, min_contrast);
        if (next_score + 1e-4 < score)
            return;
        *row = next_row;
        *column = next_col;
        *angle = next_angle;
    }
}

static void vizion_least_squares(
    const float* mx, const float* my, const float* mgx, const float* mgy, int count,
    const float* mag, int width, int height, double min_contrast,
    double* row, double* column, double* angle)
{
    int step = (int)ceil(count / 80.0);
    double angle_limit = 1.5 * vizion_pi / 180.0;
    int turn;
    if (step < 1)
        step = 1;
    for (turn = 0; turn < 4; turn++) {
        double cosine = cos(*angle);
        double sine = sin(*angle);
        double a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
        double b0 = 0, b1 = 0, b2 = 0;
        int kept = 0;
        int i;
        double d_col = 0;
        double d_row = 0;
        double d_angle = 0;
        int solved;
        for (i = 0; i < count; i += step) {
            double rotated_col = mx[i] * cosine + my[i] * sine;
            double rotated_row = -mx[i] * sine + my[i] * cosine;
            double normal_col = mgx[i] * cosine + mgy[i] * sine;
            double normal_row = -mgx[i] * sine + mgy[i] * cosine;
            double sample_col = *column + rotated_col;
            double sample_row = *row + rotated_row;
            float center;
            float before;
            float after;
            double denom;
            double offset;
            double twist;
            if (!vizion_sample(mag, width, height, sample_col, sample_row, &center))
                continue;
            if (!vizion_sample(mag, width, height, sample_col - normal_col, sample_row - normal_row, &before))
                continue;
            if (!vizion_sample(mag, width, height, sample_col + normal_col, sample_row + normal_row, &after))
                continue;
            denom = before - 2.0 * center + after;
            if (center < before || center < after || fabs(denom) <= 1e-3 || center < min_contrast)
                continue;
            offset = 0.5 * (before - after) / denom;
            if (offset < -0.6)
                offset = -0.6;
            if (offset > 0.6)
                offset = 0.6;
            twist = normal_col * rotated_row - normal_row * rotated_col;
            a00 += normal_col * normal_col;
            a01 += normal_col * normal_row;
            a02 += normal_col * twist;
            a11 += normal_row * normal_row;
            a12 += normal_row * twist;
            a22 += twist * twist;
            b0 += normal_col * offset;
            b1 += normal_row * offset;
            b2 += twist * offset;
            kept++;
        }
        if (kept < 8)
            break;
        solved = vizion_solve3(a00, a01, a02, a11, a12, a22, b0, b1, b2, &d_col, &d_row, &d_angle);
        if (!solved && !vizion_solve_translation(a00, a01, a11, b0, b1, &d_col, &d_row))
            break;
        if (d_col < -0.6)
            d_col = -0.6;
        if (d_col > 0.6)
            d_col = 0.6;
        if (d_row < -0.6)
            d_row = -0.6;
        if (d_row > 0.6)
            d_row = 0.6;
        if (d_angle < -angle_limit)
            d_angle = -angle_limit;
        if (d_angle > angle_limit)
            d_angle = angle_limit;
        if (fabs(d_col) + fabs(d_row) + fabs(d_angle) < 1e-4)
            break;
        *column += d_col;
        *row += d_row;
        *angle += d_angle;
    }
}

int vizion_shape_pose(
    const float* model_x, const float* model_y, const float* model_gx, const float* model_gy, int model_count,
    const float* gx, const float* gy, const float* mag, int width, int height, int scale,
    double row, double column, double angle, double score, double min_contrast,
    double* out_row, double* out_column, double* out_angle, double* out_score)
{
    float* rx;
    float* ry;
    float* rgx;
    float* rgy;
    double posed_score;
    double fitted_score;
    if (!model_x || !model_y || !model_gx || !model_gy || !gx || !gy || !mag || !out_row || !out_column || !out_angle || !out_score)
        return -1;
    if (model_count < 1 || width < 3 || height < 3 || scale < 1)
        return -1;
    rx = (float*)malloc((size_t)model_count * sizeof(float));
    ry = (float*)malloc((size_t)model_count * sizeof(float));
    rgx = (float*)malloc((size_t)model_count * sizeof(float));
    rgy = (float*)malloc((size_t)model_count * sizeof(float));
    if (!rx || !ry || !rgx || !rgy) {
        free(rx);
        free(ry);
        free(rgx);
        free(rgy);
        return -1;
    }
    vizion_snap(model_x, model_y, model_gx, model_gy, model_count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, min_contrast, &row, &column, &angle, &score);
    vizion_polish(model_x, model_y, model_gx, model_gy, model_count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, min_contrast, column, row, &angle, &score);
    vizion_snap(model_x, model_y, model_gx, model_gy, model_count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, min_contrast, &row, &column, &angle, &score);
    vizion_parabolic(model_x, model_y, model_gx, model_gy, model_count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, min_contrast, &row, &column, &angle, score);
    posed_score = vizion_score_pose(model_x, model_y, model_gx, model_gy, model_count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, column, row, angle, min_contrast);
    {
        double posed_row = row;
        double posed_column = column;
        double posed_angle = angle;
        vizion_least_squares(model_x, model_y, model_gx, model_gy, model_count, mag, width, height, min_contrast, &row, &column, &angle);
        fitted_score = vizion_score_pose(model_x, model_y, model_gx, model_gy, model_count, gx, gy, mag, width, height, scale, rx, ry, rgx, rgy, column, row, angle, min_contrast);
        if (fitted_score + 1e-4 < posed_score) {
            row = posed_row;
            column = posed_column;
            angle = posed_angle;
            fitted_score = posed_score;
        }
    }
    *out_row = row;
    *out_column = column;
    *out_angle = vizion_wrap(angle);
    *out_score = fitted_score;
    free(rx);
    free(ry);
    free(rgx);
    free(rgy);
    return 0;
}

int vizion_pose_selftest(void)
{
    float model_x[12];
    float model_y[12];
    float model_gx[12];
    float model_gy[12];
    float field[16 * 16];
    double row;
    double column;
    double angle;
    double score;
    int i;
    for (i = 0; i < 12; i++) {
        model_x[i] = 0;
        model_y[i] = 0;
        model_gx[i] = 1;
        model_gy[i] = 0;
    }
    memset(field, 0, sizeof(field));
    if (vizion_shape_pose(model_x, model_y, model_gx, model_gy, 12, field, field, field, 16, 16, 1, 8, 8, 0, 0, 10, &row, &column, &angle, &score) != 0)
        return 1;
    if (row != 8 || column != 8 || angle != 0)
        return 2;
    return 0;
}
