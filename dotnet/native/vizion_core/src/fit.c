#include "vizion_core.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* 找線、找圓。加總順序和 C# 相同，所以不做會打亂順序的 SIMD 累加。飛點丟掉後再擬，最多四次。 */

static int vizion_compare_double(const void* left, const void* right)
{
    double a = *(const double*)left;
    double b = *(const double*)right;
    if (a < b)
        return -1;
    if (a > b)
        return 1;
    return 0;
}

static double vizion_hypot(double x, double y)
{
    return sqrt(x * x + y * y);
}

static double vizion_fit_mean(const double* values, const double* weight, int count)
{
    double sum = 0;
    double mass = 0;
    int i;
    for (i = 0; i < count; i++) {
        sum += weight[i] * values[i];
        mass += weight[i];
    }
    return mass == 0 ? 0 : sum / mass;
}

static double vizion_median(double* values, int count)
{
    int mid = count / 2;
    qsort(values, (size_t)count, sizeof(double), vizion_compare_double);
    if (count % 2 == 1)
        return values[mid];
    return 0.5 * (values[mid - 1] + values[mid]);
}

static double vizion_max(const double* values, int count)
{
    double far = values[0];
    int i;
    for (i = 1; i < count; i++) {
        if (values[i] > far)
            far = values[i];
    }
    return far;
}

static void vizion_principal(
    const double* cols, const double* rows, int count,
    double center_col, double center_row, const double* weight,
    double* dir_col, double* dir_row)
{
    double xx = 0;
    double yy = 0;
    double xy = 0;
    double gap;
    double larger;
    double length;
    int i;
    for (i = 0; i < count; i++) {
        double scale = sqrt(weight[i]);
        double dc = (cols[i] - center_col) * scale;
        double dr = (rows[i] - center_row) * scale;
        xx += dc * dc;
        yy += dr * dr;
        xy += dc * dr;
    }
    gap = xx - yy;
    larger = 0.5 * (xx + yy + sqrt(gap * gap + 4 * xy * xy));
    if (fabs(xy) > 1e-15) {
        *dir_col = larger - yy;
        *dir_row = xy;
    } else if (xx >= yy) {
        *dir_col = 1;
        *dir_row = 0;
    } else {
        *dir_col = 0;
        *dir_row = 1;
    }
    length = vizion_hypot(*dir_col, *dir_row);
    if (length < 1e-15) {
        *dir_col = 1;
        *dir_row = 0;
        return;
    }
    *dir_col /= length;
    *dir_row /= length;
}

static int vizion_pratt(
    const double* cols, const double* rows, const unsigned char* keep, int count,
    double* out_row, double* out_col, double* out_radius)
{
    int alive = 0;
    int i;
    int step;
    double sum_col = 0;
    double sum_row = 0;
    double origin_col;
    double origin_row;
    double moment_col = 0;
    double moment_row = 0;
    double moment_cross = 0;
    double moment_col_sq = 0;
    double moment_row_sq = 0;
    double moment_sq = 0;
    double moment_sum;
    double covariance;
    double cubic;
    double quad;
    double linear;
    double constant;
    double root = 0;
    double value = 1e20;
    double determinant;
    double center_col;
    double center_row;
    double radius2;
    for (i = 0; i < count; i++) {
        if (!keep[i])
            continue;
        alive++;
        sum_col += cols[i];
        sum_row += rows[i];
    }
    if (alive < 3)
        return -2;
    origin_col = sum_col / alive;
    origin_row = sum_row / alive;
    for (i = 0; i < count; i++) {
        double shifted_col;
        double shifted_row;
        double square;
        if (!keep[i])
            continue;
        shifted_col = cols[i] - origin_col;
        shifted_row = rows[i] - origin_row;
        square = shifted_col * shifted_col + shifted_row * shifted_row;
        moment_col += shifted_col * shifted_col;
        moment_row += shifted_row * shifted_row;
        moment_cross += shifted_col * shifted_row;
        moment_col_sq += shifted_col * square;
        moment_row_sq += shifted_row * square;
        moment_sq += square * square;
    }
    moment_col /= alive;
    moment_row /= alive;
    moment_cross /= alive;
    moment_col_sq /= alive;
    moment_row_sq /= alive;
    moment_sq /= alive;
    moment_sum = moment_col + moment_row;
    covariance = moment_col * moment_row - moment_cross * moment_cross;
    cubic = 4 * moment_sum;
    quad = -3 * moment_sum * moment_sum - moment_sq;
    linear = moment_sq * moment_sum + 4 * covariance * moment_sum
        - moment_col_sq * moment_col_sq - moment_row_sq * moment_row_sq - moment_sum * moment_sum * moment_sum;
    constant = moment_col_sq * moment_col_sq * moment_row + moment_row_sq * moment_row_sq * moment_col
        - moment_sq * covariance - 2 * moment_col_sq * moment_row_sq * moment_cross
        + moment_sum * moment_sum * covariance;
    for (step = 0; step < 20; step++) {
        double previous = value;
        double slope;
        double updated;
        value = constant + root * (linear + root * (quad + root * cubic));
        if (fabs(value) > fabs(previous)) {
            root = 0;
            break;
        }
        slope = linear + root * (2 * quad + 3 * root * cubic);
        if (fabs(slope) < 1e-18)
            break;
        updated = root - value / slope;
        if (updated < 0) {
            root = 0;
            break;
        }
        if (fabs(updated) > 1e-12 && fabs((updated - root) / updated) < 1e-12) {
            root = updated;
            break;
        }
        root = updated;
    }
    determinant = root * root - root * moment_sum + covariance;
    if (fabs(determinant) < 1e-15)
        return -2;
    center_col = (moment_col_sq * (moment_row - root) - moment_row_sq * moment_cross) / determinant / 2.0;
    center_row = (moment_row_sq * (moment_col - root) - moment_col_sq * moment_cross) / determinant / 2.0;
    radius2 = center_col * center_col + center_row * center_row + moment_sum + 2 * root;
    if (radius2 <= 1e-8)
        return -2;
    *out_row = center_row + origin_row;
    *out_col = center_col + origin_col;
    *out_radius = sqrt(radius2);
    return 0;
}

int vizion_fit_line(
    const double* cols, const double* rows, int count,
    double* row_begin, double* col_begin, double* row_end, double* col_end)
{
    double* weight;
    double* distance;
    double* sorted;
    double center_col;
    double center_row;
    double dir_col;
    double dir_row;
    double min_proj;
    double max_proj;
    int kept;
    int step;
    int i;
    if (!cols || !rows || !row_begin || !col_begin || !row_end || !col_end || count < 2)
        return -1;
    weight = (double*)malloc((size_t)count * sizeof(double));
    distance = (double*)malloc((size_t)count * sizeof(double));
    sorted = (double*)malloc((size_t)count * sizeof(double));
    if (!weight || !distance || !sorted) {
        free(weight);
        free(distance);
        free(sorted);
        return -1;
    }
    for (i = 0; i < count; i++)
        weight[i] = 1;
    center_col = vizion_fit_mean(cols, weight, count);
    center_row = vizion_fit_mean(rows, weight, count);
    vizion_principal(cols, rows, count, center_col, center_row, weight, &dir_col, &dir_row);
    for (step = 0; step < 4; step++) {
        double normal_col = -dir_row;
        double normal_row = dir_col;
        double typical;
        double far;
        double limit;
        double sum = 0;
        for (i = 0; i < count; i++) {
            double dc = cols[i] - center_col;
            double dr = rows[i] - center_row;
            distance[i] = fabs(dc * normal_col + dr * normal_row);
        }
        memcpy(sorted, distance, (size_t)count * sizeof(double));
        typical = vizion_median(sorted, count);
        far = vizion_max(distance, count);
        if (typical < 1e-8 && far < 1e-6)
            break;
        limit = typical > 0.4 ? 2.5 * typical : 1.0;
        for (i = 0; i < count; i++) {
            weight[i] = distance[i] <= limit ? 1 : 0;
            sum += weight[i];
        }
        if (sum < 2)
            break;
        center_col = vizion_fit_mean(cols, weight, count);
        center_row = vizion_fit_mean(rows, weight, count);
        vizion_principal(cols, rows, count, center_col, center_row, weight, &dir_col, &dir_row);
    }
    min_proj = HUGE_VAL;
    max_proj = -HUGE_VAL;
    kept = 0;
    for (i = 0; i < count; i++) {
        double dc;
        double dr;
        double projection;
        if (weight[i] <= 0.05)
            continue;
        kept++;
        dc = cols[i] - center_col;
        dr = rows[i] - center_row;
        projection = dc * dir_col + dr * dir_row;
        if (projection < min_proj)
            min_proj = projection;
        if (projection > max_proj)
            max_proj = projection;
    }
    if (kept < 2) {
        min_proj = HUGE_VAL;
        max_proj = -HUGE_VAL;
        for (i = 0; i < count; i++) {
            double dc = cols[i] - center_col;
            double dr = rows[i] - center_row;
            double projection = dc * dir_col + dr * dir_row;
            if (projection < min_proj)
                min_proj = projection;
            if (projection > max_proj)
                max_proj = projection;
        }
    }
    *row_begin = center_row + dir_row * min_proj;
    *col_begin = center_col + dir_col * min_proj;
    *row_end = center_row + dir_row * max_proj;
    *col_end = center_col + dir_col * max_proj;
    free(weight);
    free(distance);
    free(sorted);
    return 0;
}

int vizion_fit_circle(
    const double* cols, const double* rows, int count,
    double* row, double* col, double* radius)
{
    unsigned char* keep;
    double* distance;
    double* sorted;
    double fit_row;
    double fit_col;
    double fit_radius;
    int step;
    int code;
    int i;
    if (!cols || !rows || !row || !col || !radius || count < 3)
        return -1;
    keep = (unsigned char*)malloc((size_t)count);
    distance = (double*)malloc((size_t)count * sizeof(double));
    sorted = (double*)malloc((size_t)count * sizeof(double));
    if (!keep || !distance || !sorted) {
        free(keep);
        free(distance);
        free(sorted);
        return -1;
    }
    for (i = 0; i < count; i++)
        keep[i] = 1;
    code = vizion_pratt(cols, rows, keep, count, &fit_row, &fit_col, &fit_radius);
    if (code != 0) {
        free(keep);
        free(distance);
        free(sorted);
        return code;
    }
    for (step = 0; step < 4; step++) {
        double typical;
        double far;
        double limit;
        int alive = 0;
        for (i = 0; i < count; i++)
            distance[i] = fabs(vizion_hypot(cols[i] - fit_col, rows[i] - fit_row) - fit_radius);
        memcpy(sorted, distance, (size_t)count * sizeof(double));
        typical = vizion_median(sorted, count);
        far = vizion_max(distance, count);
        if (typical < 1e-8 && far < 1e-6)
            break;
        limit = typical > 0.4 ? 2.5 * typical : 1.0;
        for (i = 0; i < count; i++) {
            keep[i] = distance[i] <= limit ? 1 : 0;
            if (keep[i])
                alive++;
        }
        if (alive < 3)
            break;
        code = vizion_pratt(cols, rows, keep, count, &fit_row, &fit_col, &fit_radius);
        if (code != 0) {
            free(keep);
            free(distance);
            free(sorted);
            return code;
        }
    }
    *row = fit_row;
    *col = fit_col;
    *radius = fit_radius;
    free(keep);
    free(distance);
    free(sorted);
    return 0;
}

static int vizion_fit_close(double left, double right)
{
    double diff = left - right;
    if (diff < 0)
        diff = -diff;
    return diff <= 1e-6;
}

int vizion_fit_selftest(void)
{
    double line_rows[5] = {0, 0, 8, 0, 0};
    double line_cols[5] = {0, 10, 15, 20, 30};
    double row_begin, col_begin, row_end, col_end;
    double circle_rows[13] = {
        50, 60, 67.3205080757, 70, 67.3205080757, 60, 50, 40, 32.6794919243, 30, 32.6794919243, 40, 50
    };
    double circle_cols[13] = {
        100, 97.3205080757, 90, 80, 70, 62.6794919243, 60, 62.6794919243, 70, 80, 90, 97.3205080757, 108
    };
    double row, col, radius;
    int forward;
    int backward;
    if (vizion_fit_line(line_cols, line_rows, 5, &row_begin, &col_begin, &row_end, &col_end) != 0)
        return 1;
    forward = vizion_fit_close(row_begin, 0) && vizion_fit_close(col_begin, 0) && vizion_fit_close(row_end, 0) && vizion_fit_close(col_end, 30);
    backward = vizion_fit_close(row_begin, 0) && vizion_fit_close(col_begin, 30) && vizion_fit_close(row_end, 0) && vizion_fit_close(col_end, 0);
    if (!forward && !backward)
        return 2;
    if (vizion_fit_circle(circle_cols, circle_rows, 13, &row, &col, &radius) != 0)
        return 3;
    if (!vizion_fit_close(row, 50) || !vizion_fit_close(col, 80) || !vizion_fit_close(radius, 20))
        return 4;
    return 0;
}
