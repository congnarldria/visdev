/*
 * HALCON-style NCC and shape matching.
 *
 * create_* builds a pyramid and the rotated templates for AngleStart/Extent/Step.
 * find_* scans only the coarsest level (8-wide AVX2 FMA), then refines candidates.
 * The hot score kernels are AVX2; MSVC x64 has no inline asm, so the CPU sees the
 * same vfmadd231ps / vmovups sequence the intrinsics lower to.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <immintrin.h>
#include <windows.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LEVELS 6
#define MAX_ANGLES 720
#define PI 3.14159265358979323846

typedef struct {
    float row, col, angle, score;
} Pose;

typedef struct {
    int n;
    float* dx;
    float* dy;
    float* val;
    float norm;
    int nangles;
    float step;
    float* angles;
    float* rdx;
    float* rdy;
    int16_t* ix;
    int16_t* iy;
    int min_x, max_x, min_y, max_y;
} NccLevel;

typedef struct {
    int nlevels;
    int width, height;
    float radius;
    float angle_start, angle_extent, angle_step;
    int ignore;
    NccLevel levels[MAX_LEVELS];
    float* dense_dx;
    float* dense_dy;
    float* dense_val;
    int dense_n;
    float dense_norm;
} NccNative;

typedef struct {
    int n;
    float* dx;
    float* dy;
    float* gx;
    float* gy;
    int nangles;
    float step;
    float* angles;
    float* rdx;
    float* rdy;
    float* rgx;
    float* rgy;
    int16_t* ix;
    int16_t* iy;
    int min_x, max_x, min_y, max_y;
} ShapeLevel;

typedef struct {
    int nlevels;
    float radius;
    float angle_start, angle_extent, angle_step;
    float min_contrast;
    int ignore;
    ShapeLevel levels[MAX_LEVELS];
} ShapeNative;

static float* alloc_f(size_t n) {
    return (float*)malloc(n * sizeof(float));
}

static int format_code(const char* fmt) {
    if (!fmt) return 0;
    while (*fmt == '<' || *fmt == '>' || *fmt == '!' || *fmt == '=' || *fmt == '@') fmt++;
    return (unsigned char)*fmt;
}

static int read_1d(PyObject* obj, float** out, int* n_out) {
    Py_buffer view;
    int i, code, count;
    float* dst;
    if (PyObject_GetBuffer(obj, &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) return -1;
    if (view.ndim != 1) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "point array must be 1-D");
        return -1;
    }
    count = (int)view.shape[0];
    code = format_code(view.format);
    dst = alloc_f((size_t)count);
    if (!dst) {
        PyBuffer_Release(&view);
        PyErr_NoMemory();
        return -1;
    }
    if (code == 'f') {
        memcpy(dst, view.buf, (size_t)count * sizeof(float));
    } else if (code == 'd') {
        const double* src = (const double*)view.buf;
        for (i = 0; i < count; i++) dst[i] = (float)src[i];
    } else {
        free(dst);
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "point array must be float32 or float64");
        return -1;
    }
    PyBuffer_Release(&view);
    *out = dst;
    *n_out = count;
    return 0;
}

static int read_image(PyObject* obj, float** out, int* w_out, int* h_out) {
    Py_buffer view;
    int code, w, h, i, n;
    float* dst;
    if (PyObject_GetBuffer(obj, &view, PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) return -1;
    if (view.ndim != 2) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "image must be 2-D");
        return -1;
    }
    h = (int)view.shape[0];
    w = (int)view.shape[1];
    n = w * h;
    code = format_code(view.format);
    dst = alloc_f((size_t)n);
    if (!dst) {
        PyBuffer_Release(&view);
        PyErr_NoMemory();
        return -1;
    }
    if (code == 'f') {
        memcpy(dst, view.buf, (size_t)n * sizeof(float));
    } else if (code == 'd') {
        const double* src = (const double*)view.buf;
        for (i = 0; i < n; i++) dst[i] = (float)src[i];
    } else if (code == 'B') {
        const unsigned char* src = (const unsigned char*)view.buf;
        for (i = 0; i < n; i++) dst[i] = (float)src[i];
    } else {
        free(dst);
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "image must be uint8 or float");
        return -1;
    }
    PyBuffer_Release(&view);
    *out = dst;
    *w_out = w;
    *h_out = h;
    return 0;
}

static int reflect101(int p, int n) {
    if (n <= 1) return 0;
    while (p < 0 || p >= n) {
        if (p < 0) p = -p;
        else p = (n - 1) * 2 - p;
    }
    return p;
}

static void pyrdown_f(const float* src, int sw, int sh, float* dst, int dw, int dh) {
    static const float k[5] = {1.f, 4.f, 6.f, 4.f, 1.f};
    int y, x, ky, kx;
    for (y = 0; y < dh; y++) {
        int sy = y * 2;
        for (x = 0; x < dw; x++) {
            int sx = x * 2;
            float acc = 0.f;
            for (ky = -2; ky <= 2; ky++) {
                int yy = reflect101(sy + ky, sh);
                float wy = k[ky + 2];
                const float* row = src + yy * sw;
                for (kx = -2; kx <= 2; kx++) {
                    int xx = reflect101(sx + kx, sw);
                    acc += row[xx] * wy * k[kx + 2];
                }
            }
            dst[y * dw + x] = acc * (1.f / 256.f);
        }
    }
}

static void sobel3(const float* src, int w, int h, float* gx, float* gy, float* mag) {
    int y, x;
    for (y = 0; y < h; y++) {
        int ym = reflect101(y - 1, h);
        int yp = reflect101(y + 1, h);
        const float* rm = src + ym * w;
        const float* r0 = src + y * w;
        const float* rp = src + yp * w;
        for (x = 0; x < w; x++) {
            int xm = reflect101(x - 1, w);
            int xp = reflect101(x + 1, w);
            float gxv = (rm[xp] + 2.f * r0[xp] + rp[xp]) - (rm[xm] + 2.f * r0[xm] + rp[xm]);
            float gyv = (rp[xm] + 2.f * rp[x] + rp[xp]) - (rm[xm] + 2.f * rm[x] + rm[xp]);
            gx[y * w + x] = gxv;
            gy[y * w + x] = gyv;
            mag[y * w + x] = sqrtf(gxv * gxv + gyv * gyv);
        }
    }
}

static float angle_gap(float a, float b) {
    float d = fmodf(a - b + (float)PI, (float)(2.0 * PI));
    if (d < 0.f) d += (float)(2.0 * PI);
    d -= (float)PI;
    return fabsf(d);
}

static float level_step(float base, float radius, int level, float widest_k) {
    float scale = ldexpf(1.f, level);
    float rad = radius / scale;
    float widest, grown, left, right, step;
    if (rad < 1.f) rad = 1.f;
    widest = widest_k / rad;
    grown = base * scale;
    left = grown > base ? grown : base;
    right = widest > base ? widest : base;
    step = left < right ? left : right;
    return step > base ? step : base;
}

static int build_angles(float start, float extent, float step, float* out, int cap) {
    int n = 0;
    double angle, stop;
    if (cap < 1) return 0;
    if (!(extent > 0.f) || !(step > 0.f)) {
        out[0] = start;
        return 1;
    }
    if (extent / step > cap) step = extent / (float)cap;
    angle = start;
    stop = (double)start + (double)extent;
    while (angle < stop - 1e-6 && n < cap) {
        out[n++] = (float)angle;
        angle += step;
    }
    if (!n) {
        out[0] = start;
        return 1;
    }
    return n;
}

static int cmp_pose(const void* a, const void* b) {
    float d = ((const Pose*)b)->score - ((const Pose*)a)->score;
    if (d > 0.f) return 1;
    if (d < 0.f) return -1;
    return 0;
}

static int suppress_poses(Pose* src, int n, Pose* dst, int max_out, float min_dist, float angle_gate) {
    int i, j, count = 0;
    float min2 = min_dist * min_dist;
    if (max_out < 1) return 0;
    for (i = 0; i < n && count < max_out; i++) {
        int blocked = 0;
        for (j = 0; j < count; j++) {
            float dr = src[i].row - dst[j].row;
            float dc = src[i].col - dst[j].col;
            if (dr * dr + dc * dc >= min2) continue;
            if (angle_gate > 0.f && angle_gap(src[i].angle, dst[j].angle) > angle_gate) continue;
            blocked = 1;
            break;
        }
        if (!blocked) dst[count++] = src[i];
    }
    return count;
}

static float parabola(float left, float center, float right) {
    float denom = left - 2.f * center + right;
    float delta;
    if (fabsf(denom) < 1e-6f || center < left || center < right) return 0.f;
    delta = 0.5f * (left - right) / denom;
    if (delta > 0.75f) return 0.75f;
    if (delta < -0.75f) return -0.75f;
    return delta;
}

static int angle_in_range(float angle, float start, float extent) {
    float end, a;
    if (!(extent > 0.f)) return angle_gap(angle, start) < 1e-4f;
    end = start + extent;
    a = angle;
    if (a < start - 1e-5f) a += (float)(2.0 * PI);
    if (a < start - 1e-5f) a += (float)(2.0 * PI);
    return a >= start - 1e-4f && a < end + 1e-4f;
}

static void zero_mean(float* val, int n, float* norm_out) {
    int i;
    double sum = 0.0, energy = 0.0, mean;
    for (i = 0; i < n; i++) sum += val[i];
    mean = sum / (n > 0 ? n : 1);
    for (i = 0; i < n; i++) {
        double v = val[i] - mean;
        val[i] = (float)v;
        energy += v * v;
    }
    *norm_out = (float)sqrt(energy);
}

static int collect_points(const float* gray, const unsigned char* mask, int w, int h, int limit,
                          float** dx, float** dy, float** val, int* n_out, float* norm) {
    int stride = 1, y, x, count = 0, cap, i;
    float ox, oy;
    while (stride < 32 && (h / stride) * (w / stride) > limit) stride++;
    cap = ((h + stride - 1) / stride) * ((w + stride - 1) / stride);
    *dx = alloc_f((size_t)cap);
    *dy = alloc_f((size_t)cap);
    *val = alloc_f((size_t)cap);
    if (!*dx || !*dy || !*val) return -1;
    ox = (h - 1) * 0.5f;
    oy = (w - 1) * 0.5f;
    for (y = 0; y < h; y += stride) {
        for (x = 0; x < w; x += stride) {
            if (mask && mask[y * w + x] == 0) continue;
            (*dy)[count] = (float)y - ox;
            (*dx)[count] = (float)x - oy;
            (*val)[count] = gray[y * w + x];
            count++;
        }
    }
    if (count < 16) return -1;
    zero_mean(*val, count, norm);
    *n_out = count;
    (void)i;
    return 0;
}

static void rotate_points(const float* dx, const float* dy, int n, float angle, float* ox, float* oy) {
    float c = cosf(angle);
    float s = sinf(angle);
    int i;
    for (i = 0; i < n; i++) {
        ox[i] = dx[i] * c + dy[i] * s;
        oy[i] = -dx[i] * s + dy[i] * c;
    }
}

static void rotate_dirs(const float* dx, const float* dy, const float* gx, const float* gy, int n,
                        float angle, float* ox, float* oy, float* ogx, float* ogy) {
    float c = cosf(angle);
    float s = sinf(angle);
    int i;
    for (i = 0; i < n; i++) {
        ox[i] = dx[i] * c + dy[i] * s;
        oy[i] = -dx[i] * s + dy[i] * c;
        ogx[i] = gx[i] * c + gy[i] * s;
        ogy[i] = -gx[i] * s + gy[i] * c;
    }
}

static void bounds_from(const float* dx, const float* dy, int n, int16_t* ix, int16_t* iy,
                        int* min_x, int* max_x, int* min_y, int* max_y) {
    int i;
    int mnx = 0, mxx = 0, mny = 0, mxy = 0;
    for (i = 0; i < n; i++) {
        int x = (int)(dx[i] + (dx[i] >= 0.f ? 0.5f : -0.5f));
        int y = (int)(dy[i] + (dy[i] >= 0.f ? 0.5f : -0.5f));
        if (x > 32767) x = 32767;
        if (x < -32768) x = -32768;
        if (y > 32767) y = 32767;
        if (y < -32768) y = -32768;
        ix[i] = (int16_t)x;
        iy[i] = (int16_t)y;
        if (i == 0 || x < mnx) mnx = x;
        if (i == 0 || x > mxx) mxx = x;
        if (i == 0 || y < mny) mny = y;
        if (i == 0 || y > mxy) mxy = y;
    }
    *min_x = mnx;
    *max_x = mxx;
    *min_y = mny;
    *max_y = mxy;
}

static void ncc_level_free(NccLevel* level) {
    free(level->dx);
    free(level->dy);
    free(level->val);
    free(level->angles);
    free(level->rdx);
    free(level->rdy);
    free(level->ix);
    free(level->iy);
    memset(level, 0, sizeof(*level));
}

static void ncc_free(NccNative* model) {
    int i;
    if (!model) return;
    for (i = 0; i < model->nlevels; i++) ncc_level_free(&model->levels[i]);
    free(model->dense_dx);
    free(model->dense_dy);
    free(model->dense_val);
    free(model);
}

static void shape_level_free(ShapeLevel* level) {
    free(level->dx);
    free(level->dy);
    free(level->gx);
    free(level->gy);
    free(level->angles);
    free(level->rdx);
    free(level->rdy);
    free(level->rgx);
    free(level->rgy);
    free(level->ix);
    free(level->iy);
    memset(level, 0, sizeof(*level));
}

static void shape_free(ShapeNative* model) {
    int i;
    if (!model) return;
    for (i = 0; i < model->nlevels; i++) shape_level_free(&model->levels[i]);
    free(model);
}

static void free_ncc_capsule(PyObject* capsule) {
    ncc_free((NccNative*)PyCapsule_GetPointer(capsule, "visdev.ncc"));
}

static void free_shape_capsule(PyObject* capsule) {
    shape_free((ShapeNative*)PyCapsule_GetPointer(capsule, "visdev.shape"));
}

static int fill_ncc_level(NccLevel* level, const float* gray, const unsigned char* mask, int w, int h,
                          int limit, float radius, float angle_start, float angle_extent, float base_step,
                          int level_index) {
    float* angles;
    int a, nangles;
    float step;
    if (collect_points(gray, mask, w, h, limit, &level->dx, &level->dy, &level->val, &level->n, &level->norm) < 0)
        return -1;
    step = level_step(base_step, radius, level_index, 2.2f);
    angles = alloc_f(MAX_ANGLES);
    if (!angles) return -1;
    nangles = build_angles(angle_start, angle_extent, step, angles, MAX_ANGLES);
    level->angles = angles;
    level->nangles = nangles;
    level->step = step;
    level->rdx = alloc_f((size_t)nangles * (size_t)level->n);
    level->rdy = alloc_f((size_t)nangles * (size_t)level->n);
    level->ix = (int16_t*)malloc((size_t)nangles * (size_t)level->n * sizeof(int16_t));
    level->iy = (int16_t*)malloc((size_t)nangles * (size_t)level->n * sizeof(int16_t));
    if (!level->rdx || !level->rdy || !level->ix || !level->iy) return -1;
    level->min_x = 0;
    level->max_x = 0;
    level->min_y = 0;
    level->max_y = 0;
    for (a = 0; a < nangles; a++) {
        float* ox = level->rdx + (size_t)a * level->n;
        float* oy = level->rdy + (size_t)a * level->n;
        int16_t* ix = level->ix + (size_t)a * level->n;
        int16_t* iy = level->iy + (size_t)a * level->n;
        int mnx, mxx, mny, mxy;
        rotate_points(level->dx, level->dy, level->n, angles[a], ox, oy);
        bounds_from(ox, oy, level->n, ix, iy, &mnx, &mxx, &mny, &mxy);
        if (a == 0 || mnx < level->min_x) level->min_x = mnx;
        if (a == 0 || mxx > level->max_x) level->max_x = mxx;
        if (a == 0 || mny < level->min_y) level->min_y = mny;
        if (a == 0 || mxy > level->max_y) level->max_y = mxy;
    }
    return 0;
}

static PyObject* ncc_create(PyObject* self, PyObject* args) {
    PyObject *gray_obj, *mask_obj, *capsule;
    int num_levels, ignore, w, h, level, i;
    double angle_start, angle_extent, angle_step, radius;
    float* gray = NULL;
    float* mask_f = NULL;
    unsigned char* mask = NULL;
    int mw = 0, mh = 0;
    NccNative* model;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOidddi", &gray_obj, &mask_obj, &num_levels, &angle_start, &angle_extent,
                          &angle_step, &ignore))
        return NULL;
    if (read_image(gray_obj, &gray, &w, &h) < 0) return NULL;
    if (mask_obj != Py_None) {
        if (read_image(mask_obj, &mask_f, &mw, &mh) < 0) {
            free(gray);
            return NULL;
        }
        if (mw != w || mh != h) {
            free(gray);
            free(mask_f);
            PyErr_SetString(PyExc_ValueError, "mask size must match the template");
            return NULL;
        }
        mask = (unsigned char*)malloc((size_t)w * (size_t)h);
        if (!mask) {
            free(gray);
            free(mask_f);
            return PyErr_NoMemory();
        }
        for (i = 0; i < w * h; i++) mask[i] = mask_f[i] > 0.f ? 255 : 0;
        free(mask_f);
    }
    if (num_levels < 1) num_levels = 1;
    if (num_levels > MAX_LEVELS) num_levels = MAX_LEVELS;
    if (!(angle_step > 0.0)) angle_step = 0.5 * PI / 180.0;
    radius = hypot((w - 1) * 0.5, (h - 1) * 0.5);
    if (radius < 1.0) radius = 1.0;
    model = (NccNative*)calloc(1, sizeof(NccNative));
    if (!model) {
        free(gray);
        free(mask);
        return PyErr_NoMemory();
    }
    model->nlevels = num_levels;
    model->width = w;
    model->height = h;
    model->radius = (float)radius;
    model->angle_start = (float)angle_start;
    model->angle_extent = (float)angle_extent;
    model->angle_step = (float)angle_step;
    model->ignore = ignore;
    {
        float* level_gray = gray;
        unsigned char* level_mask = mask;
        int lw = w, lh = h;
        int own_gray = 0, own_mask = 0;
        for (level = 0; level < num_levels; level++) {
            int limit = (lw * lh <= 24 * 24) ? lw * lh : (level == 0 ? 768 : 480);
            if (fill_ncc_level(&model->levels[level], level_gray, level_mask, lw, lh, limit, model->radius,
                               model->angle_start, model->angle_extent, model->angle_step, level) < 0) {
                if (own_gray) free(level_gray);
                if (own_mask) free(level_mask);
                free(gray);
                free(mask);
                ncc_free(model);
                PyErr_SetString(PyExc_ValueError, "NCC template does not have enough pixels");
                return NULL;
            }
            if (level + 1 < num_levels) {
                int nw = (lw + 1) / 2;
                int nh = (lh + 1) / 2;
                float* next = alloc_f((size_t)nw * (size_t)nh);
                unsigned char* next_mask = (unsigned char*)malloc((size_t)nw * (size_t)nh);
                float* mask_src;
                float* mask_blur;
                if (!next || !next_mask) {
                    free(next);
                    free(next_mask);
                    if (own_gray) free(level_gray);
                    if (own_mask) free(level_mask);
                    free(gray);
                    free(mask);
                    ncc_free(model);
                    return PyErr_NoMemory();
                }
                pyrdown_f(level_gray, lw, lh, next, nw, nh);
                mask_src = alloc_f((size_t)lw * (size_t)lh);
                mask_blur = alloc_f((size_t)nw * (size_t)nh);
                for (i = 0; i < lw * lh; i++) mask_src[i] = level_mask ? (level_mask[i] ? 255.f : 0.f) : 255.f;
                pyrdown_f(mask_src, lw, lh, mask_blur, nw, nh);
                for (i = 0; i < nw * nh; i++) next_mask[i] = mask_blur[i] > 127.f ? 255 : 0;
                free(mask_src);
                free(mask_blur);
                if (own_gray) free(level_gray);
                if (own_mask) free(level_mask);
                level_gray = next;
                level_mask = next_mask;
                own_gray = 1;
                own_mask = 1;
                lw = nw;
                lh = nh;
            }
        }
        if (own_gray) free(level_gray);
        if (own_mask) free(level_mask);
    }
    if (collect_points(gray, mask, w, h, 2500, &model->dense_dx, &model->dense_dy, &model->dense_val, &model->dense_n,
                       &model->dense_norm) < 0) {
        free(gray);
        free(mask);
        ncc_free(model);
        PyErr_SetString(PyExc_ValueError, "NCC template does not have enough pixels");
        return NULL;
    }
    free(gray);
    free(mask);
    capsule = PyCapsule_New(model, "visdev.ncc", free_ncc_capsule);
    if (!capsule) {
        ncc_free(model);
        return NULL;
    }
    return capsule;
}

static float bilinear(const float* img, int w, int h, float x, float y) {
    int x0, y0;
    float wx, wy, top, bot;
    const float* r0;
    const float* r1;
    if (x < 0.f || y < 0.f || x >= (float)(w - 1) || y >= (float)(h - 1)) return 0.f;
    x0 = (int)x;
    y0 = (int)y;
    wx = x - (float)x0;
    wy = y - (float)y0;
    r0 = img + y0 * w;
    r1 = r0 + w;
    top = r0[x0] * (1.f - wx) + r0[x0 + 1] * wx;
    bot = r1[x0] * (1.f - wx) + r1[x0 + 1] * wx;
    return top * (1.f - wy) + bot * wy;
}

static float ncc_bilinear(const float* img, int w, int h, const float* dx, const float* dy, const float* val,
                          int n, float norm, float row, float col, int ignore) {
    int i, inside = 0;
    double dot = 0.0, sum = 0.0, sumsq = 0.0;
    float mean, energy, score;
    if (n < 16 || norm < 1e-6f) return 0.f;
    for (i = 0; i < n; i++) {
        float x = col + dx[i];
        float y = row + dy[i];
        float pix;
        if (x < 0.f || y < 0.f || x >= (float)(w - 1) || y >= (float)(h - 1)) continue;
        pix = bilinear(img, w, h, x, y);
        inside++;
        dot += (double)val[i] * pix;
        sum += pix;
        sumsq += (double)pix * pix;
    }
    if (inside < 16 || inside * 20 < n * 13) return 0.f;
    mean = (float)(sum / inside);
    energy = (float)(sumsq - sum * mean);
    if (energy < 1e-6f) return 0.f;
    score = (float)(dot / (norm * sqrtf(energy)));
    if (ignore) score = fabsf(score);
    if (score > 1.f) score = 1.f;
    if (score < -1.f) score = -1.f;
    return score;
}

static void ncc_score_avx(const float* img, int w, const int16_t* ix, const int16_t* iy, const float* val, int n,
                          float norm, int ignore, int row, int col, float* out8) {
    __m256 dot = _mm256_setzero_ps();
    __m256 sum = _mm256_setzero_ps();
    __m256 sumsq = _mm256_setzero_ps();
    __m256 energy, denom, score;
    int i;
    for (i = 0; i < n; i++) {
        const float* pix = img + ((row + iy[i]) * w + (col + ix[i]));
        __m256 p = _mm256_loadu_ps(pix);
        __m256 t = _mm256_set1_ps(val[i]);
        dot = _mm256_fmadd_ps(p, t, dot);
        sum = _mm256_add_ps(sum, p);
        sumsq = _mm256_fmadd_ps(p, p, sumsq);
    }
    energy = _mm256_sub_ps(sumsq, _mm256_mul_ps(sum, _mm256_mul_ps(sum, _mm256_set1_ps(1.f / (float)n))));
    energy = _mm256_max_ps(energy, _mm256_set1_ps(1e-6f));
    denom = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_sqrt_ps(energy));
    score = _mm256_div_ps(dot, denom);
    if (ignore) {
        __m256 sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        score = _mm256_and_ps(score, sign);
    }
    _mm256_storeu_ps(out8, score);
}

static float ncc_score_scalar(const float* img, int w, const int16_t* ix, const int16_t* iy, const float* val, int n,
                              float norm, int ignore, int row, int col) {
    int i;
    double dot = 0.0, sum = 0.0, sumsq = 0.0, mean, energy, score;
    for (i = 0; i < n; i++) {
        float pix = img[(row + iy[i]) * w + (col + ix[i])];
        dot += (double)val[i] * pix;
        sum += pix;
        sumsq += (double)pix * pix;
    }
    mean = sum / n;
    energy = sumsq - sum * mean;
    if (energy < 1e-6 || norm < 1e-6f) return 0.f;
    score = dot / (norm * sqrt(energy));
    if (ignore) score = fabs(score);
    if (score > 1.0) score = 1.0;
    if (score < -1.0) score = -1.0;
    return (float)score;
}

typedef struct {
    const float* img;
    int w, h;
    const NccLevel* level;
    float min_score;
    int ignore;
    int a0, a1;
    Pose* peaks;
    int npeaks;
    float* map;
} NccJob;

static DWORD WINAPI ncc_worker(LPVOID param) {
    NccJob* job = (NccJob*)param;
    const NccLevel* level = job->level;
    int a;
    job->npeaks = 0;
    for (a = job->a0; a < job->a1; a++) {
        const int16_t* ix = level->ix + (size_t)a * level->n;
        const int16_t* iy = level->iy + (size_t)a * level->n;
        int mnx = 32767, mxx = -32768, mny = 32767, mxy = -32768;
        int i, r0, r1, c0, c1, row, col;
        float angle = level->angles[a];
        for (i = 0; i < level->n; i++) {
            int x = ix[i], y = iy[i];
            if (x < mnx) mnx = x;
            if (x > mxx) mxx = x;
            if (y < mny) mny = y;
            if (y > mxy) mxy = y;
        }
        r0 = mny < 0 ? -mny : 0;
        c0 = mnx < 0 ? -mnx : 0;
        r1 = job->h - mxy;
        c1 = job->w - mxx;
        if (r1 <= r0 || c1 <= c0) continue;
        for (row = r0; row < r1; row++) {
            float* dst = job->map + row * job->w;
            col = c0;
            for (; col + 7 < c1; col += 8) {
                ncc_score_avx(job->img, job->w, ix, iy, level->val, level->n, level->norm, job->ignore, row, col,
                              dst + col);
            }
            for (; col < c1; col++) {
                dst[col] = ncc_score_scalar(job->img, job->w, ix, iy, level->val, level->n, level->norm, job->ignore,
                                            row, col);
            }
        }
        for (row = r0 + 2; row < r1 - 2; row++) {
            for (col = c0 + 2; col < c1 - 2; col++) {
                float s = job->map[row * job->w + col];
                int ok = 1, dy, dx;
                if (s < job->min_score) continue;
                for (dy = -2; dy <= 2 && ok; dy++) {
                    for (dx = -2; dx <= 2; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        if (s + 1e-6f < job->map[(row + dy) * job->w + (col + dx)]) {
                            ok = 0;
                            break;
                        }
                    }
                }
                if (!ok || job->npeaks >= 512) continue;
                job->peaks[job->npeaks].row = (float)row;
                job->peaks[job->npeaks].col = (float)col;
                job->peaks[job->npeaks].angle = angle;
                job->peaks[job->npeaks].score = s;
                job->npeaks++;
            }
        }
    }
    return 0;
}

static int local_peaks_from_jobs(NccJob* jobs, int njobs, Pose* dst, int cap) {
    int j, i, n = 0;
    for (j = 0; j < njobs; j++) {
        for (i = 0; i < jobs[j].npeaks && n < cap; i++) dst[n++] = jobs[j].peaks[i];
    }
    return n;
}

static float ncc_at_rotated(const float* img, int w, int h, const float* dx, const float* dy, const float* val, int n,
                            float norm, float row, float col, int ignore) {
    return ncc_bilinear(img, w, h, dx, dy, val, n, norm, row, col, ignore);
}

static PyObject* poses_to_lists(Pose* poses, int n) {
    PyObject *rows, *cols, *angles, *scores;
    int i;
    rows = PyList_New(n);
    cols = PyList_New(n);
    angles = PyList_New(n);
    scores = PyList_New(n);
    if (!rows || !cols || !angles || !scores) {
        Py_XDECREF(rows);
        Py_XDECREF(cols);
        Py_XDECREF(angles);
        Py_XDECREF(scores);
        return NULL;
    }
    for (i = 0; i < n; i++) {
        PyList_SET_ITEM(rows, i, PyFloat_FromDouble(poses[i].row));
        PyList_SET_ITEM(cols, i, PyFloat_FromDouble(poses[i].col));
        PyList_SET_ITEM(angles, i, PyFloat_FromDouble(poses[i].angle));
        PyList_SET_ITEM(scores, i, PyFloat_FromDouble(poses[i].score));
    }
    return Py_BuildValue("(NNNN)", rows, cols, angles, scores);
}

static PyObject* ncc_find(PyObject* self, PyObject* args) {
    PyObject *cap, *image_obj, *result;
    NccNative* model;
    double angle_start, angle_extent, min_score, max_overlap;
    int num_matches, subpixel, num_levels;
    float* scene = NULL;
    int sw = 0, sh = 0;
    float* pyr[MAX_LEVELS];
    int pw[MAX_LEVELS], ph[MAX_LEVELS];
    int used, coarse, a, t, nthreads, alive;
    Pose* found = NULL;
    Pose* refined = NULL;
    int nfound = 0;
    NccJob jobs[8];
    HANDLE threads[8];
    (void)self;
    if (!PyArg_ParseTuple(args, "OOdddidii", &cap, &image_obj, &angle_start, &angle_extent, &min_score, &num_matches,
                          &max_overlap, &subpixel, &num_levels))
        return NULL;
    model = (NccNative*)PyCapsule_GetPointer(cap, "visdev.ncc");
    if (!model) return NULL;
    if (read_image(image_obj, &scene, &sw, &sh) < 0) return NULL;
    used = model->nlevels;
    if (num_levels > 0 && num_levels < used) used = num_levels;
    pyr[0] = scene;
    pw[0] = sw;
    ph[0] = sh;
    for (a = 1; a < used; a++) {
        pw[a] = (pw[a - 1] + 1) / 2;
        ph[a] = (ph[a - 1] + 1) / 2;
        pyr[a] = alloc_f((size_t)pw[a] * (size_t)ph[a]);
        if (!pyr[a]) {
            PyErr_NoMemory();
            goto fail;
        }
        pyrdown_f(pyr[a - 1], pw[a - 1], ph[a - 1], pyr[a], pw[a], ph[a]);
    }
    coarse = used - 1;
    found = (Pose*)malloc(sizeof(Pose) * 8192);
    refined = (Pose*)malloc(sizeof(Pose) * 8192);
    if (!found || !refined) {
        PyErr_NoMemory();
        goto fail;
    }
    memset(jobs, 0, sizeof(jobs));
    nthreads = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 8) nthreads = 8;
    if (nthreads > model->levels[coarse].nangles) nthreads = model->levels[coarse].nangles;
    if (nthreads < 1) nthreads = 1;
    {
        int total = model->levels[coarse].nangles;
        int base = total / nthreads;
        int extra = total % nthreads;
        int cursor = 0;
        for (t = 0; t < nthreads; t++) {
            int count = base + (t < extra ? 1 : 0);
            jobs[t].img = pyr[coarse];
            jobs[t].w = pw[coarse];
            jobs[t].h = ph[coarse];
            jobs[t].level = &model->levels[coarse];
            jobs[t].min_score = (float)min_score * 0.62f;
            jobs[t].ignore = model->ignore;
            jobs[t].a0 = cursor;
            jobs[t].a1 = cursor + count;
            cursor += count;
            jobs[t].peaks = (Pose*)malloc(sizeof(Pose) * 512);
            jobs[t].map = alloc_f((size_t)pw[coarse] * (size_t)ph[coarse]);
            if (!jobs[t].peaks || !jobs[t].map) {
                PyErr_NoMemory();
                goto fail;
            }
            memset(jobs[t].map, 0, (size_t)pw[coarse] * (size_t)ph[coarse] * sizeof(float));
        }
        Py_BEGIN_ALLOW_THREADS
        alive = 0;
        for (t = 0; t < nthreads; t++) {
            if (jobs[t].a0 >= jobs[t].a1) {
                threads[t] = NULL;
                continue;
            }
            threads[t] = CreateThread(NULL, 0, ncc_worker, &jobs[t], 0, NULL);
            if (threads[t]) alive++;
            else ncc_worker(&jobs[t]);
        }
        for (t = 0; t < nthreads; t++) {
            if (threads[t]) {
                WaitForSingleObject(threads[t], INFINITE);
                CloseHandle(threads[t]);
            }
        }
        Py_END_ALLOW_THREADS
        nfound = local_peaks_from_jobs(jobs, nthreads, found, 8192);
    }
    qsort(found, (size_t)nfound, sizeof(Pose), cmp_pose);
    {
        float dist = model->radius * (float)ph[coarse] / (float)ph[0];
        if (dist < 4.f) dist = 4.f;
        else dist *= 0.45f;
        nfound = suppress_poses(found, nfound, refined, 12, dist, (float)(18.0 * PI / 180.0));
        memcpy(found, refined, (size_t)nfound * sizeof(Pose));
    }
    for (a = coarse - 1; a >= 0; a--) {
        float scale_r = (float)ph[a] / (float)ph[a + 1];
        float scale_c = (float)pw[a] / (float)pw[a + 1];
        float window = model->levels[a + 1].step;
        int rad = (a == coarse - 1) ? 3 : 2;
        int p, nn = 0;
        NccLevel* level = &model->levels[a];
        float* tdx = alloc_f((size_t)level->n);
        float* tdy = alloc_f((size_t)level->n);
        int16_t* ix = (int16_t*)malloc((size_t)level->n * sizeof(int16_t));
        int16_t* iy = (int16_t*)malloc((size_t)level->n * sizeof(int16_t));
        if (!tdx || !tdy || !ix || !iy) {
            free(tdx);
            free(tdy);
            free(ix);
            free(iy);
            PyErr_NoMemory();
            goto fail;
        }
        for (p = 0; p < nfound; p++) {
            float base_r = found[p].row * scale_r;
            float base_c = found[p].col * scale_c;
            float base_a = found[p].angle;
            float best_s = -2.f, best_r = base_r, best_c = base_c, best_a_out = base_a;
            int sa, dy, dx;
            int origin_r = (int)(base_r + (base_r >= 0.f ? 0.5f : -0.5f));
            int origin_c = (int)(base_c + (base_c >= 0.f ? 0.5f : -0.5f));
            for (sa = -1; sa <= 1; sa++) {
                float ang = base_a + window * (float)sa;
                int mnx, mxx, mny, mxy;
                rotate_points(level->dx, level->dy, level->n, ang, tdx, tdy);
                bounds_from(tdx, tdy, level->n, ix, iy, &mnx, &mxx, &mny, &mxy);
                for (dy = -rad; dy <= rad; dy++) {
                    for (dx = -rad; dx <= rad; dx++) {
                        int rr = origin_r + dy;
                        int cc = origin_c + dx;
                        float s;
                        if (rr + mny < 0 || cc + mnx < 0 || rr + mxy >= ph[a] || cc + mxx >= pw[a]) continue;
                        s = ncc_score_scalar(pyr[a], pw[a], ix, iy, level->val, level->n, level->norm, model->ignore, rr,
                                             cc);
                        if (s > best_s) {
                            best_s = s;
                            best_r = (float)rr;
                            best_c = (float)cc;
                            best_a_out = ang;
                        }
                    }
                }
            }
            if (best_s >= (float)min_score * 0.75f && nn < 8192) {
                refined[nn].row = best_r;
                refined[nn].col = best_c;
                refined[nn].angle = best_a_out;
                refined[nn].score = best_s;
                nn++;
            }
        }
        qsort(refined, (size_t)nn, sizeof(Pose), cmp_pose);
        {
            float dist = model->radius * (float)ph[a] / (float)ph[0] * 0.5f;
            if (dist < 3.f) dist = 3.f;
            nfound = suppress_poses(refined, nn, found, 32, dist, (float)(18.0 * PI / 180.0));
        }
        free(tdx);
        free(tdy);
        free(ix);
        free(iy);
        if (!nfound) break;
    }
    if (nfound && subpixel) {
        int p, nn = 0;
        float* tdx = alloc_f((size_t)model->dense_n);
        float* tdy = alloc_f((size_t)model->dense_n);
        float* adx = alloc_f((size_t)model->dense_n);
        float* ady = alloc_f((size_t)model->dense_n);
        if (!tdx || !tdy || !adx || !ady) {
            free(tdx);
            free(tdy);
            free(adx);
            free(ady);
            PyErr_NoMemory();
            goto fail;
        }
        for (p = 0; p < nfound; p++) {
            float row = found[p].row;
            float col = found[p].col;
            float angle = found[p].angle;
            float step = model->angle_step;
            int iter;
            if (step < (float)(0.6 * PI / 180.0)) step = (float)(0.6 * PI / 180.0);
            for (iter = 0; iter < 2; iter++) {
                float sc, su, sd, sl, sr, sa, sb;
                rotate_points(model->dense_dx, model->dense_dy, model->dense_n, angle, tdx, tdy);
                sc = ncc_at_rotated(pyr[0], sw, sh, tdx, tdy, model->dense_val, model->dense_n, model->dense_norm, row,
                                    col, model->ignore);
                su = ncc_at_rotated(pyr[0], sw, sh, tdx, tdy, model->dense_val, model->dense_n, model->dense_norm,
                                    row - 1.f, col, model->ignore);
                sd = ncc_at_rotated(pyr[0], sw, sh, tdx, tdy, model->dense_val, model->dense_n, model->dense_norm,
                                    row + 1.f, col, model->ignore);
                row += parabola(su, sc, sd);
                sl = ncc_at_rotated(pyr[0], sw, sh, tdx, tdy, model->dense_val, model->dense_n, model->dense_norm, row,
                                    col - 1.f, model->ignore);
                sc = ncc_at_rotated(pyr[0], sw, sh, tdx, tdy, model->dense_val, model->dense_n, model->dense_norm, row,
                                    col, model->ignore);
                sr = ncc_at_rotated(pyr[0], sw, sh, tdx, tdy, model->dense_val, model->dense_n, model->dense_norm, row,
                                    col + 1.f, model->ignore);
                col += parabola(sl, sc, sr);
                rotate_points(model->dense_dx, model->dense_dy, model->dense_n, angle - step, adx, ady);
                sa = ncc_at_rotated(pyr[0], sw, sh, adx, ady, model->dense_val, model->dense_n, model->dense_norm, row,
                                    col, model->ignore);
                rotate_points(model->dense_dx, model->dense_dy, model->dense_n, angle + step, adx, ady);
                sb = ncc_at_rotated(pyr[0], sw, sh, adx, ady, model->dense_val, model->dense_n, model->dense_norm, row,
                                    col, model->ignore);
                rotate_points(model->dense_dx, model->dense_dy, model->dense_n, angle, tdx, tdy);
                sc = ncc_at_rotated(pyr[0], sw, sh, tdx, tdy, model->dense_val, model->dense_n, model->dense_norm, row,
                                    col, model->ignore);
                angle += parabola(sa, sc, sb) * step;
                step *= 0.5f;
            }
            rotate_points(model->dense_dx, model->dense_dy, model->dense_n, angle, tdx, tdy);
            {
                float score = ncc_at_rotated(pyr[0], sw, sh, tdx, tdy, model->dense_val, model->dense_n, model->dense_norm,
                                             row, col, model->ignore);
                if (score >= (float)min_score && nn < 8192) {
                    refined[nn].row = row;
                    refined[nn].col = col;
                    refined[nn].angle = angle;
                    refined[nn].score = score;
                    nn++;
                }
            }
        }
        free(tdx);
        free(tdy);
        free(adx);
        free(ady);
        nfound = nn;
        memcpy(found, refined, (size_t)nfound * sizeof(Pose));
    } else {
        int p, nn = 0;
        for (p = 0; p < nfound; p++) {
            if (found[p].score >= (float)min_score) refined[nn++] = found[p];
        }
        nfound = nn;
        memcpy(found, refined, (size_t)nfound * sizeof(Pose));
    }
    qsort(found, (size_t)nfound, sizeof(Pose), cmp_pose);
    {
        float overlap = (float)max_overlap;
        float dist;
        int keep = num_matches > 0 ? num_matches : 100;
        if (overlap < 0.f) overlap = 0.f;
        if (overlap > 1.f) overlap = 1.f;
        dist = model->radius * (0.25f > (1.f - overlap) ? 0.25f : (1.f - overlap));
        nfound = suppress_poses(found, nfound, refined, keep, dist, 0.f);
    }
    result = poses_to_lists(refined, nfound);
    goto cleanup;
fail:
    result = NULL;
cleanup:
    for (t = 0; t < 8; t++) {
        free(jobs[t].peaks);
        free(jobs[t].map);
    }
    for (a = 1; a < used; a++) free(pyr[a]);
    free(scene);
    free(found);
    free(refined);
    return result;
}

static int shape_keep_step(int n, float greediness) {
    int keep, step;
    if (!(greediness > 0.f) || n <= 30) return 1;
    keep = (int)(n * (1.f - 0.55f * greediness) + 0.5f);
    if (keep < 30) keep = 30;
    if (keep > n) keep = n;
    step = (n + keep - 1) / keep;
    return step < 1 ? 1 : step;
}

static int fill_shape_level(ShapeLevel* level, const float* dx, const float* dy, const float* gx, const float* gy,
                            int n, float radius, float angle_start, float angle_extent, float base_step, int index) {
    int a;
    float step = level_step(base_step, radius, index, 2.1f);
    level->n = n;
    level->dx = alloc_f((size_t)n);
    level->dy = alloc_f((size_t)n);
    level->gx = alloc_f((size_t)n);
    level->gy = alloc_f((size_t)n);
    level->angles = alloc_f(MAX_ANGLES);
    if (!level->dx || !level->dy || !level->gx || !level->gy || !level->angles) return -1;
    memcpy(level->dx, dx, (size_t)n * sizeof(float));
    memcpy(level->dy, dy, (size_t)n * sizeof(float));
    memcpy(level->gx, gx, (size_t)n * sizeof(float));
    memcpy(level->gy, gy, (size_t)n * sizeof(float));
    level->nangles = build_angles(angle_start, angle_extent, step, level->angles, MAX_ANGLES);
    level->step = step;
    level->rdx = alloc_f((size_t)level->nangles * (size_t)n);
    level->rdy = alloc_f((size_t)level->nangles * (size_t)n);
    level->rgx = alloc_f((size_t)level->nangles * (size_t)n);
    level->rgy = alloc_f((size_t)level->nangles * (size_t)n);
    level->ix = (int16_t*)malloc((size_t)level->nangles * (size_t)n * sizeof(int16_t));
    level->iy = (int16_t*)malloc((size_t)level->nangles * (size_t)n * sizeof(int16_t));
    if (!level->rdx || !level->rdy || !level->rgx || !level->rgy || !level->ix || !level->iy) return -1;
    for (a = 0; a < level->nangles; a++) {
        size_t off = (size_t)a * (size_t)n;
        int mnx, mxx, mny, mxy;
        rotate_dirs(dx, dy, gx, gy, n, level->angles[a], level->rdx + off, level->rdy + off, level->rgx + off,
                    level->rgy + off);
        bounds_from(level->rdx + off, level->rdy + off, n, level->ix + off, level->iy + off, &mnx, &mxx, &mny, &mxy);
        if (a == 0 || mnx < level->min_x) level->min_x = mnx;
        if (a == 0 || mxx > level->max_x) level->max_x = mxx;
        if (a == 0 || mny < level->min_y) level->min_y = mny;
        if (a == 0 || mxy > level->max_y) level->max_y = mxy;
    }
    return 0;
}

static PyObject* shape_create(PyObject* self, PyObject* args) {
    PyObject *levels_obj, *capsule;
    int ignore, nlevels, i;
    double angle_start, angle_extent, angle_step, min_contrast, radius;
    ShapeNative* model;
    (void)self;
    if (!PyArg_ParseTuple(args, "Oddddid", &levels_obj, &angle_start, &angle_extent, &angle_step, &min_contrast,
                          &ignore, &radius))
        return NULL;
    if (!PyList_Check(levels_obj)) {
        PyErr_SetString(PyExc_TypeError, "levels must be a list");
        return NULL;
    }
    nlevels = (int)PyList_Size(levels_obj);
    if (nlevels < 1 || nlevels > MAX_LEVELS) {
        PyErr_SetString(PyExc_ValueError, "shape model needs 1 to 6 pyramid levels");
        return NULL;
    }
    model = (ShapeNative*)calloc(1, sizeof(ShapeNative));
    if (!model) return PyErr_NoMemory();
    model->nlevels = nlevels;
    model->angle_start = (float)angle_start;
    model->angle_extent = (float)angle_extent;
    model->angle_step = (float)angle_step > 0.f ? (float)angle_step : (float)(0.6 * PI / 180.0);
    model->min_contrast = (float)min_contrast;
    model->ignore = ignore;
    model->radius = (float)(radius > 1.0 ? radius : 1.0);
    for (i = 0; i < nlevels; i++) {
        PyObject* item = PyList_GetItem(levels_obj, i);
        PyObject* c0;
        PyObject* c1;
        PyObject* c2;
        PyObject* c3;
        float *dx, *dy, *gx, *gy;
        int n0, n1, n2, n3;
        if (!PySequence_Check(item) || PySequence_Size(item) < 4) {
            shape_free(model);
            PyErr_SetString(PyExc_TypeError, "each shape level is (rows, cols, gx, gy)");
            return NULL;
        }
        c0 = PySequence_GetItem(item, 0);
        c1 = PySequence_GetItem(item, 1);
        c2 = PySequence_GetItem(item, 2);
        c3 = PySequence_GetItem(item, 3);
        dx = dy = gx = gy = NULL;
        n0 = n1 = n2 = n3 = 0;
        if (!c0 || !c1 || !c2 || !c3 || read_1d(c0, &dy, &n0) < 0 || read_1d(c1, &dx, &n1) < 0 ||
            read_1d(c2, &gx, &n2) < 0 || read_1d(c3, &gy, &n3) < 0 || n0 != n1 || n0 != n2 || n0 != n3 || n0 < 8) {
            Py_XDECREF(c0);
            Py_XDECREF(c1);
            Py_XDECREF(c2);
            Py_XDECREF(c3);
            free(dx);
            free(dy);
            free(gx);
            free(gy);
            shape_free(model);
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "shape level points do not line up");
            return NULL;
        }
        Py_DECREF(c0);
        Py_DECREF(c1);
        Py_DECREF(c2);
        Py_DECREF(c3);
        if (fill_shape_level(&model->levels[i], dx, dy, gx, gy, n0, model->radius, model->angle_start,
                             model->angle_extent, model->angle_step, i) < 0) {
            free(dx);
            free(dy);
            free(gx);
            free(gy);
            shape_free(model);
            return PyErr_NoMemory();
        }
        free(dx);
        free(dy);
        free(gx);
        free(gy);
    }
    capsule = PyCapsule_New(model, "visdev.shape", free_shape_capsule);
    if (!capsule) {
        shape_free(model);
        return NULL;
    }
    return capsule;
}

static float shape_bilinear(const float* gx, const float* gy, const float* mag, int w, int h, const float* dx,
                            const float* dy, const float* mgx, const float* mgy, int n, float row, float col,
                            float min_contrast, int ignore) {
    int i, valid = 0;
    float sum = 0.f;
    int need = n > 20 ? (int)(0.4f * n) : 8;
    if (need < 8) need = 8;
    for (i = 0; i < n; i++) {
        float x = col + dx[i];
        float y = row + dy[i];
        float ix, iy, m, dot;
        if (x < 0.f || y < 0.f || x >= (float)(w - 1) || y >= (float)(h - 1)) continue;
        m = bilinear(mag, w, h, x, y);
        if (m < min_contrast) continue;
        ix = bilinear(gx, w, h, x, y);
        iy = bilinear(gy, w, h, x, y);
        dot = (ix * mgx[i] + iy * mgy[i]) / (m > 1e-6f ? m : 1e-6f);
        if (ignore) dot = fabsf(dot);
        sum += dot;
        valid++;
    }
    if (valid < need) return 0.f;
    return sum / (float)valid;
}

typedef struct {
    const float* ux;
    const float* uy;
    const float* valid;
    int w, h;
    const ShapeLevel* level;
    int step;
    float min_score;
    float min_contrast;
    int ignore;
    int a0, a1;
    Pose* peaks;
    int npeaks;
    float* map;
} ShapeJob;

static void shape_score_avx(const float* ux, const float* uy, const float* valid, int w, const int16_t* ix,
                            const int16_t* iy, const float* gx, const float* gy, int n, int step, int ignore,
                            float min_score, int row, int col, float* out8) {
    __m256 acc = _mm256_setzero_ps();
    __m256 cnt = _mm256_setzero_ps();
    __m256 sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 floor = _mm256_set1_ps(min_score);
    int i, seen = 0, total = 0;
    for (i = 0; i < n; i += step) total++;
    for (i = 0; i < n; i += step) {
        const float* px = ux + ((row + iy[i]) * w + (col + ix[i]));
        const float* py = uy + ((row + iy[i]) * w + (col + ix[i]));
        const float* mk = valid + ((row + iy[i]) * w + (col + ix[i]));
        __m256 dots = _mm256_fmadd_ps(_mm256_loadu_ps(px), _mm256_set1_ps(gx[i]),
                                      _mm256_mul_ps(_mm256_loadu_ps(py), _mm256_set1_ps(gy[i])));
        __m256 m = _mm256_loadu_ps(mk);
        if (ignore) dots = _mm256_and_ps(dots, sign);
        acc = _mm256_fmadd_ps(dots, m, acc);
        cnt = _mm256_add_ps(cnt, m);
        seen++;
        if ((seen & 7) == 0 && seen >= 16 && seen < total) {
            __m256 possible = _mm256_div_ps(_mm256_add_ps(acc, _mm256_set1_ps((float)(total - seen))),
                                            _mm256_set1_ps((float)total));
            if (_mm256_movemask_ps(_mm256_cmp_ps(possible, floor, _CMP_LT_OQ)) == 255) break;
        }
    }
    {
        __m256 need = _mm256_set1_ps((float)(seen > 20 ? (int)(0.4f * seen) : 8));
        __m256 ok = _mm256_cmp_ps(cnt, need, _CMP_GE_OQ);
        __m256 score = _mm256_div_ps(acc, _mm256_max_ps(cnt, _mm256_set1_ps(1.f)));
        score = _mm256_and_ps(score, ok);
        _mm256_storeu_ps(out8, score);
    }
}

static float shape_int(const float* ux, const float* uy, const float* valid, int w, int h, const int16_t* ix,
                       const int16_t* iy, const float* gx, const float* gy, int n, int row, int col, int ignore,
                       int mnx, int mxx, int mny, int mxy) {
    int i, seen = 0, okc = 0;
    float sum = 0.f;
    if (row + mny < 0 || col + mnx < 0 || row + mxy >= h || col + mxx >= w) return 0.f;
    for (i = 0; i < n; i++) {
        int yy = row + iy[i];
        int xx = col + ix[i];
        float dot;
        seen++;
        if (valid[yy * w + xx] < 0.5f) continue;
        dot = ux[yy * w + xx] * gx[i] + uy[yy * w + xx] * gy[i];
        if (ignore) dot = fabsf(dot);
        sum += dot;
        okc++;
    }
    if (okc < (seen > 20 ? (int)(0.4f * seen) : 8)) return 0.f;
    return sum / (float)okc;
}

static DWORD WINAPI shape_worker(LPVOID param) {
    ShapeJob* job = (ShapeJob*)param;
    const ShapeLevel* level = job->level;
    int a;
    job->npeaks = 0;
    for (a = job->a0; a < job->a1; a++) {
        size_t off = (size_t)a * (size_t)level->n;
        const int16_t* ix = level->ix + off;
        const int16_t* iy = level->iy + off;
        const float* gx = level->rgx + off;
        const float* gy = level->rgy + off;
        int mnx = 32767, mxx = -32768, mny = 32767, mxy = -32768;
        int i, r0, r1, c0, c1, row, col;
        for (i = 0; i < level->n; i += job->step) {
            int x = ix[i], y = iy[i];
            if (x < mnx) mnx = x;
            if (x > mxx) mxx = x;
            if (y < mny) mny = y;
            if (y > mxy) mxy = y;
        }
        r0 = mny < 0 ? -mny : 0;
        c0 = mnx < 0 ? -mnx : 0;
        r1 = job->h - mxy;
        c1 = job->w - mxx;
        if (r1 <= r0 || c1 <= c0) continue;
        for (row = r0; row < r1; row++) {
            col = c0;
            for (; col + 7 < c1; col += 8) {
                shape_score_avx(job->ux, job->uy, job->valid, job->w, ix, iy, gx, gy, level->n, job->step, job->ignore,
                                job->min_score, row, col, job->map + row * job->w + col);
            }
            for (; col < c1; col++) {
                int seen = 0, okc = 0;
                float sum = 0.f;
                for (i = 0; i < level->n; i += job->step) {
                    int yy = row + iy[i];
                    int xx = col + ix[i];
                    float m = job->valid[yy * job->w + xx];
                    seen++;
                    if (m < 0.5f) continue;
                    {
                        float dot = job->ux[yy * job->w + xx] * gx[i] + job->uy[yy * job->w + xx] * gy[i];
                        if (job->ignore) dot = fabsf(dot);
                        sum += dot;
                        okc++;
                    }
                }
                if (okc >= (seen > 20 ? (int)(0.4f * seen) : 8)) job->map[row * job->w + col] = sum / (float)okc;
                else job->map[row * job->w + col] = 0.f;
            }
        }
        for (row = r0 + 2; row < r1 - 2; row++) {
            for (col = c0 + 2; col < c1 - 2; col++) {
                float s = job->map[row * job->w + col];
                int ok = 1, dy, dx;
                if (s < job->min_score) continue;
                for (dy = -2; dy <= 2 && ok; dy++) {
                    for (dx = -2; dx <= 2; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        if (s + 1e-6f < job->map[(row + dy) * job->w + (col + dx)]) {
                            ok = 0;
                            break;
                        }
                    }
                }
                if (!ok || job->npeaks >= 512) continue;
                job->peaks[job->npeaks].row = (float)row;
                job->peaks[job->npeaks].col = (float)col;
                job->peaks[job->npeaks].angle = level->angles[a];
                job->peaks[job->npeaks].score = s;
                job->npeaks++;
            }
        }
    }
    return 0;
}

static PyObject* shape_find(PyObject* self, PyObject* args) {
    PyObject *cap, *images, *result = NULL;
    ShapeNative* model;
    double angle_start, angle_extent, min_score, max_overlap, greediness;
    int num_matches, subpixel, i, used, coarse, t, nthreads;
    float* gray[MAX_LEVELS];
    float* gx[MAX_LEVELS];
    float* gy[MAX_LEVELS];
    float* mag[MAX_LEVELS];
    float* ux[MAX_LEVELS];
    float* uy[MAX_LEVELS];
    float* valid[MAX_LEVELS];
    int pw[MAX_LEVELS], ph[MAX_LEVELS];
    Pose *found = NULL, *refined = NULL;
    int nfound = 0;
    ShapeJob jobs[8];
    HANDLE threads[8];
    (void)self;
    memset(gray, 0, sizeof(gray));
    memset(gx, 0, sizeof(gx));
    memset(gy, 0, sizeof(gy));
    memset(mag, 0, sizeof(mag));
    memset(ux, 0, sizeof(ux));
    memset(uy, 0, sizeof(uy));
    memset(valid, 0, sizeof(valid));
    memset(jobs, 0, sizeof(jobs));
    if (!PyArg_ParseTuple(args, "OOdddidid", &cap, &images, &angle_start, &angle_extent, &min_score, &num_matches,
                          &max_overlap, &subpixel, &greediness))
        return NULL;
    model = (ShapeNative*)PyCapsule_GetPointer(cap, "visdev.shape");
    if (!model) return NULL;
    if (!PyList_Check(images)) {
        PyErr_SetString(PyExc_TypeError, "pyramid must be a list of images");
        return NULL;
    }
    used = (int)PyList_Size(images);
    if (used < 1) used = 1;
    if (used > model->nlevels) used = model->nlevels;
    for (i = 0; i < used; i++) {
        int n, p;
        if (read_image(PyList_GetItem(images, i), &gray[i], &pw[i], &ph[i]) < 0) goto fail;
        n = pw[i] * ph[i];
        gx[i] = alloc_f((size_t)n);
        gy[i] = alloc_f((size_t)n);
        mag[i] = alloc_f((size_t)n);
        ux[i] = alloc_f((size_t)n);
        uy[i] = alloc_f((size_t)n);
        valid[i] = alloc_f((size_t)n);
        if (!gx[i] || !gy[i] || !mag[i] || !ux[i] || !uy[i] || !valid[i]) {
            PyErr_NoMemory();
            goto fail;
        }
        sobel3(gray[i], pw[i], ph[i], gx[i], gy[i], mag[i]);
        for (p = 0; p < n; p++) {
            float m = mag[i][p];
            valid[i][p] = m >= model->min_contrast ? 1.f : 0.f;
            if (m >= 1e-3f) {
                ux[i][p] = gx[i][p] / m;
                uy[i][p] = gy[i][p] / m;
            } else {
                ux[i][p] = 0.f;
                uy[i][p] = 0.f;
            }
        }
    }
    coarse = used - 1;
    found = (Pose*)malloc(sizeof(Pose) * 8192);
    refined = (Pose*)malloc(sizeof(Pose) * 8192);
    if (!found || !refined) {
        PyErr_NoMemory();
        goto fail;
    }
    nthreads = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 8) nthreads = 8;
    if (nthreads > model->levels[coarse].nangles) nthreads = model->levels[coarse].nangles;
    if (nthreads < 1) nthreads = 1;
    {
        int total = model->levels[coarse].nangles;
        int base = total / nthreads;
        int extra = total % nthreads;
        int cursor = 0;
        int step = shape_keep_step(model->levels[coarse].n, (float)greediness);
        float coarse_min = (float)min_score * (greediness < 0.85 ? 0.72f : 0.82f);
        for (t = 0; t < nthreads; t++) {
            int count = base + (t < extra ? 1 : 0);
            jobs[t].ux = ux[coarse];
            jobs[t].uy = uy[coarse];
            jobs[t].valid = valid[coarse];
            jobs[t].w = pw[coarse];
            jobs[t].h = ph[coarse];
            jobs[t].level = &model->levels[coarse];
            jobs[t].step = step;
            jobs[t].min_score = coarse_min;
            jobs[t].min_contrast = model->min_contrast;
            jobs[t].ignore = model->ignore;
            jobs[t].a0 = cursor;
            jobs[t].a1 = cursor + count;
            cursor += count;
            jobs[t].peaks = (Pose*)malloc(sizeof(Pose) * 512);
            jobs[t].map = alloc_f((size_t)pw[coarse] * (size_t)ph[coarse]);
            if (!jobs[t].peaks || !jobs[t].map) {
                PyErr_NoMemory();
                goto fail;
            }
        }
        Py_BEGIN_ALLOW_THREADS
        for (t = 0; t < nthreads; t++) {
            if (jobs[t].a0 >= jobs[t].a1) {
                threads[t] = NULL;
                continue;
            }
            threads[t] = CreateThread(NULL, 0, shape_worker, &jobs[t], 0, NULL);
            if (!threads[t]) shape_worker(&jobs[t]);
        }
        for (t = 0; t < nthreads; t++) {
            if (threads[t]) {
                WaitForSingleObject(threads[t], INFINITE);
                CloseHandle(threads[t]);
            }
        }
        Py_END_ALLOW_THREADS
        nfound = 0;
        for (t = 0; t < nthreads; t++) {
            int p;
            for (p = 0; p < jobs[t].npeaks && nfound < 8192; p++) found[nfound++] = jobs[t].peaks[p];
        }
    }
    qsort(found, (size_t)nfound, sizeof(Pose), cmp_pose);
    {
        float dist = model->radius * (float)ph[coarse] / (float)ph[0];
        if (dist * 0.45f < 4.f) dist = 4.f;
        else dist *= 0.45f;
        nfound = suppress_poses(found, nfound, refined, 16, dist, (float)(20.0 * PI / 180.0));
        memcpy(found, refined, (size_t)nfound * sizeof(Pose));
    }
    for (i = coarse - 1; i >= 0; i--) {
        float scale_r = (float)ph[i] / (float)ph[i + 1];
        float scale_c = (float)pw[i] / (float)pw[i + 1];
        float window = model->levels[i + 1].step;
        float fine = model->angle_step;
        int p, nn = 0;
        ShapeLevel* level = &model->levels[i];
        float* tdx = alloc_f((size_t)level->n);
        float* tdy = alloc_f((size_t)level->n);
        float* tgx = alloc_f((size_t)level->n);
        float* tgy = alloc_f((size_t)level->n);
        int16_t* ix = (int16_t*)malloc((size_t)level->n * sizeof(int16_t));
        int16_t* iy = (int16_t*)malloc((size_t)level->n * sizeof(int16_t));
        if (!tdx || !tdy || !tgx || !tgy || !ix || !iy) {
            free(tdx);
            free(tdy);
            free(tgx);
            free(tgy);
            free(ix);
            free(iy);
            PyErr_NoMemory();
            goto fail;
        }
        if (window * 0.5f > fine) fine = window * 0.5f;
        for (p = 0; p < nfound; p++) {
            float base_r = found[p].row * scale_r;
            float base_c = found[p].col * scale_c;
            float base_a = found[p].angle;
            float best_s = -1.f, best_r = base_r, best_c = base_c, best_a = base_a;
            int sa, dy, dx;
            int origin_r = (int)(base_r + (base_r >= 0.f ? 0.5f : -0.5f));
            int origin_c = (int)(base_c + (base_c >= 0.f ? 0.5f : -0.5f));
            for (sa = -1; sa <= 1; sa++) {
                float ang = base_a + window * (float)sa;
                int mnx, mxx, mny, mxy;
                rotate_dirs(level->dx, level->dy, level->gx, level->gy, level->n, ang, tdx, tdy, tgx, tgy);
                bounds_from(tdx, tdy, level->n, ix, iy, &mnx, &mxx, &mny, &mxy);
                for (dy = -2; dy <= 2; dy++) {
                    for (dx = -2; dx <= 2; dx++) {
                        int rr = origin_r + dy;
                        int cc = origin_c + dx;
                        float s = shape_int(ux[i], uy[i], valid[i], pw[i], ph[i], ix, iy, tgx, tgy, level->n, rr, cc,
                                            model->ignore, mnx, mxx, mny, mxy);
                        if (s > best_s) {
                            best_s = s;
                            best_r = (float)rr;
                            best_c = (float)cc;
                            best_a = ang;
                        }
                    }
                }
            }
            if (best_s >= (float)min_score * 0.8f) {
                float nudged_a = best_a;
                float nudged_s = best_s;
                int k;
                int mnx, mxx, mny, mxy;
                int rr = (int)(best_r + (best_r >= 0.f ? 0.5f : -0.5f));
                int cc = (int)(best_c + (best_c >= 0.f ? 0.5f : -0.5f));
                for (k = -1; k <= 1; k++) {
                    float ang = best_a + fine * (float)k;
                    float s;
                    rotate_dirs(level->dx, level->dy, level->gx, level->gy, level->n, ang, tdx, tdy, tgx, tgy);
                    bounds_from(tdx, tdy, level->n, ix, iy, &mnx, &mxx, &mny, &mxy);
                    s = shape_int(ux[i], uy[i], valid[i], pw[i], ph[i], ix, iy, tgx, tgy, level->n, rr, cc,
                                  model->ignore, mnx, mxx, mny, mxy);
                    if (s > nudged_s) {
                        nudged_s = s;
                        nudged_a = ang;
                    }
                }
                if (nn < 8192) {
                    refined[nn].row = best_r;
                    refined[nn].col = best_c;
                    refined[nn].angle = nudged_a;
                    refined[nn].score = nudged_s;
                    nn++;
                }
            }
        }
        free(tdx);
        free(tdy);
        free(tgx);
        free(tgy);
        free(ix);
        free(iy);
        qsort(refined, (size_t)nn, sizeof(Pose), cmp_pose);
        {
            float dist = model->radius * (float)ph[i] / (float)ph[0] * 0.55f;
            if (dist < 3.f) dist = 3.f;
            nfound = suppress_poses(refined, nn, found, 80, dist, (float)(20.0 * PI / 180.0));
        }
        if (!nfound) break;
    }
    if (nfound && subpixel) {
        int p, nn = 0;
        ShapeLevel* level = &model->levels[0];
        for (p = 0; p < nfound; p++) {
            float row = found[p].row;
            float col = found[p].col;
            float angle = found[p].angle;
            float step = model->angle_step;
            int iter;
            if (step < (float)(0.8 * PI / 180.0)) step = (float)(0.8 * PI / 180.0);
            for (iter = 0; iter < 2; iter++) {
                float s_c, s_u, s_d, s_l, s_r, s_a, s_b;
                float* tdx = alloc_f((size_t)level->n);
                float* tdy = alloc_f((size_t)level->n);
                float* tgx = alloc_f((size_t)level->n);
                float* tgy = alloc_f((size_t)level->n);
                if (!tdx || !tdy || !tgx || !tgy) {
                    free(tdx);
                    free(tdy);
                    free(tgx);
                    free(tgy);
                    break;
                }
                rotate_dirs(level->dx, level->dy, level->gx, level->gy, level->n, angle, tdx, tdy, tgx, tgy);
                s_c = shape_bilinear(gx[0], gy[0], mag[0], pw[0], ph[0], tdx, tdy, tgx, tgy, level->n, row, col,
                                     model->min_contrast, model->ignore);
                s_u = shape_bilinear(gx[0], gy[0], mag[0], pw[0], ph[0], tdx, tdy, tgx, tgy, level->n, row - 1.f, col,
                                     model->min_contrast, model->ignore);
                s_d = shape_bilinear(gx[0], gy[0], mag[0], pw[0], ph[0], tdx, tdy, tgx, tgy, level->n, row + 1.f, col,
                                     model->min_contrast, model->ignore);
                s_l = shape_bilinear(gx[0], gy[0], mag[0], pw[0], ph[0], tdx, tdy, tgx, tgy, level->n, row, col - 1.f,
                                     model->min_contrast, model->ignore);
                s_r = shape_bilinear(gx[0], gy[0], mag[0], pw[0], ph[0], tdx, tdy, tgx, tgy, level->n, row, col + 1.f,
                                     model->min_contrast, model->ignore);
                rotate_dirs(level->dx, level->dy, level->gx, level->gy, level->n, angle - step, tdx, tdy, tgx, tgy);
                s_a = shape_bilinear(gx[0], gy[0], mag[0], pw[0], ph[0], tdx, tdy, tgx, tgy, level->n, row, col,
                                     model->min_contrast, model->ignore);
                rotate_dirs(level->dx, level->dy, level->gx, level->gy, level->n, angle + step, tdx, tdy, tgx, tgy);
                s_b = shape_bilinear(gx[0], gy[0], mag[0], pw[0], ph[0], tdx, tdy, tgx, tgy, level->n, row, col,
                                     model->min_contrast, model->ignore);
                free(tdx);
                free(tdy);
                free(tgx);
                free(tgy);
                row += parabola(s_u, s_c, s_d);
                col += parabola(s_l, s_c, s_r);
                angle += parabola(s_a, s_c, s_b) * step;
                step *= 0.5f;
            }
            {
                float* tdx = alloc_f((size_t)level->n);
                float* tdy = alloc_f((size_t)level->n);
                float* tgx = alloc_f((size_t)level->n);
                float* tgy = alloc_f((size_t)level->n);
                float score = 0.f;
                if (tdx && tdy && tgx && tgy) {
                    rotate_dirs(level->dx, level->dy, level->gx, level->gy, level->n, angle, tdx, tdy, tgx, tgy);
                    score = shape_bilinear(gx[0], gy[0], mag[0], pw[0], ph[0], tdx, tdy, tgx, tgy, level->n, row, col,
                                           model->min_contrast, model->ignore);
                }
                free(tdx);
                free(tdy);
                free(tgx);
                free(tgy);
                if (score >= (float)min_score && nn < 8192) {
                    refined[nn].row = row;
                    refined[nn].col = col;
                    refined[nn].angle = angle;
                    refined[nn].score = score;
                    nn++;
                }
            }
        }
        nfound = nn;
        memcpy(found, refined, (size_t)nfound * sizeof(Pose));
    } else {
        int p, nn = 0;
        for (p = 0; p < nfound; p++) {
            if (found[p].score >= (float)min_score) refined[nn++] = found[p];
        }
        nfound = nn;
        memcpy(found, refined, (size_t)nfound * sizeof(Pose));
    }
    qsort(found, (size_t)nfound, sizeof(Pose), cmp_pose);
    {
        float overlap = (float)max_overlap;
        float dist;
        int keep = num_matches > 0 ? num_matches : 100;
        if (overlap < 0.f) overlap = 0.f;
        if (overlap > 1.f) overlap = 1.f;
        dist = model->radius * (0.2f > (1.f - overlap) ? 0.2f : (1.f - overlap));
        nfound = suppress_poses(found, nfound, refined, keep, dist, 0.f);
    }
    result = poses_to_lists(refined, nfound);
    goto cleanup;
fail:
    result = NULL;
cleanup:
    for (t = 0; t < 8; t++) {
        free(jobs[t].peaks);
        free(jobs[t].map);
    }
    for (i = 0; i < MAX_LEVELS; i++) {
        free(gray[i]);
        free(gx[i]);
        free(gy[i]);
        free(mag[i]);
        free(ux[i]);
        free(uy[i]);
        free(valid[i]);
    }
    free(found);
    free(refined);
    return result;
}

static PyObject* backend(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    return PyUnicode_FromString("c-avx2");
}

static PyMethodDef methods[] = {
    {"ncc_create", ncc_create, METH_VARARGS, "Build a rotated NCC pyramid."},
    {"ncc_find", ncc_find, METH_VARARGS, "Search an NCC model."},
    {"shape_create", shape_create, METH_VARARGS, "Build rotated shape models."},
    {"shape_find", shape_find, METH_VARARGS, "Search a shape model."},
    {"backend", backend, METH_NOARGS, "Native backend name."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "visdev_match", "AVX2 NCC and shape matching.", -1, methods,
};

PyMODINIT_FUNC PyInit_visdev_match(void) {
    return PyModule_Create(&moduledef);
}
