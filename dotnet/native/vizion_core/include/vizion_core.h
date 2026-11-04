#ifndef VIZION_CORE_H
#define VIZION_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(VIZION_BUILD)
#    define VIZION_EXPORT __declspec(dllexport)
#  else
#    define VIZION_EXPORT __declspec(dllimport)
#  endif
#  if defined(_MSC_VER) && defined(_M_IX86)
#    define VIZION_CALL __cdecl
#  else
#    define VIZION_CALL
#  endif
#else
#  define VIZION_EXPORT __attribute__((visibility("default")))
#  define VIZION_CALL
#endif

/* bit0 SSE2 可用，bit1 AVX2 可用，bit2 目前走 AVX2，bit3 走 SSE2，bit4 走純 C。 */
#define VIZION_CPU_SSE2 1
#define VIZION_CPU_AVX2 2
#define VIZION_CPU_SELECTED_AVX2 4
#define VIZION_CPU_SELECTED_SSE2 8
#define VIZION_CPU_SELECTED_SCALAR 16

VIZION_EXPORT int VIZION_CALL vizion_cpu_flags(void);

/* 灰階落在 min 與 max（含）的像素收成水平行程。
   rows 傳 NULL 時只回傳段數。成功回 0，參數錯誤 -1，緩衝不夠 -2。 */
VIZION_EXPORT int VIZION_CALL vizion_threshold_runs(
    const uint8_t* gray, int width, int height, int min_gray, int max_gray,
    int* rows, int* columns, int* lengths, int capacity, int* written);

VIZION_EXPORT int VIZION_CALL vizion_threshold_selftest(void);

/* 八連通。輸出行程依物件排好，物件順序是掃描時第一次遇到的那段。回傳物件數，失敗 -1。 */
VIZION_EXPORT int VIZION_CALL vizion_connection8(
    const int* rows, const int* columns, const int* lengths, int count, int height,
    int* out_rows, int* out_columns, int* out_lengths, int* out_offsets, int* out_counts);

VIZION_EXPORT int VIZION_CALL vizion_area_center(
    const int* rows, const int* columns, const int* lengths, int count,
    int* area, double* row, double* column);

VIZION_EXPORT int VIZION_CALL vizion_bounds(
    const int* rows, const int* columns, const int* lengths, int count,
    int* row1, int* column1, int* row2, int* column2);

VIZION_EXPORT int VIZION_CALL vizion_region_selftest(void);

/* 灰階矩形原點是 k/2，不把偶數補成奇數。dilate 非 0 表示膨脹，圖外膨脹當 0、侵蝕當 255。 */
VIZION_EXPORT int VIZION_CALL vizion_gray_rect(
    const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int mask_width, int dilate);

VIZION_EXPORT int VIZION_CALL vizion_gray_ellipse(
    const uint8_t* src, uint8_t* dst, int width, int height, int radius, int dilate);

/* 區域矩形偶數寬高會加 1。out 傳 NULL 時只回傳段數。 */
VIZION_EXPORT int VIZION_CALL vizion_region_rect(
    const int* rows, const int* columns, const int* lengths, int count,
    int width, int height, int kernel_width, int kernel_height, int dilate,
    int* out_rows, int* out_columns, int* out_lengths, int capacity, int* written);

VIZION_EXPORT int VIZION_CALL vizion_region_disk(
    const int* rows, const int* columns, const int* lengths, int count,
    int width, int height, double radius, int dilate,
    int* out_rows, int* out_columns, int* out_lengths, int capacity, int* written);

VIZION_EXPORT int VIZION_CALL vizion_morph_selftest(void);

/* 平均遮罩至少 1。高斯 Size 是大於等於 3 的奇數。邊界是 REFLECT_101。 */
VIZION_EXPORT int VIZION_CALL vizion_mean(
    const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height);

VIZION_EXPORT int VIZION_CALL vizion_gauss(
    const uint8_t* src, uint8_t* dst, int width, int height, int size);

VIZION_EXPORT int VIZION_CALL vizion_filter_selftest(void);

/* 形狀比對與正規相關在金字塔每一層的分數。雙線性取樣用 SIMD。 */
VIZION_EXPORT double VIZION_CALL vizion_shape_score(
    const float* gx, const float* gy, const float* mag, int width, int height,
    double column, double row, const float* mx, const float* my, const float* mgx, const float* mgy,
    int count, double min_contrast);

VIZION_EXPORT double VIZION_CALL vizion_ncc_correlate(
    const float* gray, int width, int height, const float* x, const float* y, const float* value,
    int count, double column, double row, int absolute);

VIZION_EXPORT int VIZION_CALL vizion_score_selftest(void);

/* 沿一列量亮條。找不到回 1。左緣、右緣是半像素加拋物線偏差。 */
VIZION_EXPORT int VIZION_CALL vizion_measure_row(const uint8_t* row, int width, double* left, double* right);

/* transition：0 全部、1 由暗到亮、2 由亮到暗。座標是行、列，和影像 x、y 相同。 */
VIZION_EXPORT int VIZION_CALL vizion_measure_scan(
    const uint8_t* gray, int width, int height,
    double x0, double y0, double x1, double y1,
    double sigma, double threshold, int transition,
    double* out_rows, double* out_columns, double* out_amps, double* out_dists,
    int capacity, int* written);

VIZION_EXPORT int VIZION_CALL vizion_measure_selftest(void);

/* 可分離高斯導數、梯度模和法線上的非極大抑制。核由呼叫端算好。成功回 0。 */
VIZION_EXPORT int VIZION_CALL vizion_subpix(
    const float* gray, int width, int height,
    const float* smooth, int smooth_len, const float* deriv, int deriv_len,
    float* dx, float* dy, float* magnitude, float* normal_x, float* normal_y,
    float* shift, uint8_t* center);

/* 套用已建好的去畸變對照表。雙線性，圖外 0，四捨五入遠離 0。 */
VIZION_EXPORT int VIZION_CALL vizion_remap(
    const uint8_t* src, int width, int height,
    const double* map_x, const double* map_y, uint8_t* dst);

VIZION_EXPORT int VIZION_CALL vizion_edge_selftest(void);

/* 3x3 Sobel。邊界把 -1 映到 0。gx、gy、模長和 C# 的 float 相同。 */
VIZION_EXPORT int VIZION_CALL vizion_sobel(
    const uint8_t* gray, int width, int height, float* gx, float* gy, float* mag);

VIZION_EXPORT int VIZION_CALL vizion_sobel_selftest(void);

/* 5 點高斯 [1,4,6,4,1] 後隔點取樣。輸出寬高是 (n+1)/2。 */
VIZION_EXPORT int VIZION_CALL vizion_pyrdown(const uint8_t* src, int width, int height, uint8_t* dst);

VIZION_EXPORT int VIZION_CALL vizion_pyrdown_selftest(void);

/* BGR 轉灰階。count 是像素數，來源是 count*3。和 OpenCV 的整數式相同。 */
VIZION_EXPORT int VIZION_CALL vizion_bgr_gray(const uint8_t* bgr, int count, uint8_t* gray);

VIZION_EXPORT int VIZION_CALL vizion_bgr_selftest(void);

/* 逐像素灰階。op：0 反相、1 乘加、2 相加、3 相減、4 相乘、5 絕對差、6 較小、7 較大、
   8 相除（分母 0 得到 0）、9 強化 a+(a-b)*mult、10 (a-b)+b*mult+add。
   先用 float 算，再截到 0..255，小數直接丟掉，和 NumPy 的 uint8 相同。b 在反相和乘加時可為 NULL。 */
VIZION_EXPORT int VIZION_CALL vizion_gray_arith(
    const uint8_t* a, const uint8_t* b, uint8_t* dst, int count,
    int op, double mult, double add);

/* 單通道中值。ksize 是正奇數。邊界重複邊緣像素，和 cv2.medianBlur 相同。 */
VIZION_EXPORT int VIZION_CALL vizion_median_image(
    const uint8_t* src, uint8_t* dst, int width, int height, int ksize);

VIZION_EXPORT int VIZION_CALL vizion_arith_selftest(void);

/* 單通道直方圖等化，和 cv2.equalizeHist 相同。 */
VIZION_EXPORT int VIZION_CALL vizion_equalize(const uint8_t* src, uint8_t* dst, int count);

/* 彩色只等化 YCrCb 的 Y，和 cv2.equalizeHist 套在亮度上相同。count 是像素數。 */
VIZION_EXPORT int VIZION_CALL vizion_equalize_bgr(const uint8_t* src, uint8_t* dst, int count);

/* 拉平照明。遮罩是正奇數。domain 非 0 才算平均，算完定義域外寫 0。domain 可為 NULL。 */
VIZION_EXPORT int VIZION_CALL vizion_illuminate(
    const uint8_t* src, uint8_t* dst, int width, int height,
    int mask_width, int mask_height, double factor, const uint8_t* domain);

/* 最小值拉到 0、最大值拉到 255。domain 非 0 才參與統計；domain 為 0 的像素輸出 0。domain 可為 NULL。 */
VIZION_EXPORT int VIZION_CALL vizion_scale_max(
    const uint8_t* src, uint8_t* dst, int count, const uint8_t* domain);

/* BGR 拆成三張、或由藍紅綠合成。count 是像素數。 */
VIZION_EXPORT int VIZION_CALL vizion_split_bgr(
    const uint8_t* bgr, uint8_t* blue, uint8_t* green, uint8_t* red, int count);
VIZION_EXPORT int VIZION_CALL vizion_merge_bgr(
    const uint8_t* blue, const uint8_t* green, const uint8_t* red, uint8_t* bgr, int count);

/* 記憶體是紅、綠、藍。space：0 hsv、1 lab、2 yuv。數值和 OpenCV 的 8 位元轉換相同。 */
VIZION_EXPORT int VIZION_CALL vizion_rgb_convert(
    const uint8_t* red, const uint8_t* green, const uint8_t* blue,
    uint8_t* first, uint8_t* second, uint8_t* third, int count, int space);

VIZION_EXPORT int VIZION_CALL vizion_tone_selftest(void);

/* 次方先除 255 再乘回來。log/exp 的 natural 非 0 表示自然對數或 e 的次方。
   gamma 的 encode 非 0 表示次方用 1/gamma。range 把 low..high 拉到 0..255，小數直接丟掉。
   bits：0 且、1 或、2 互斥或、3 否定。否定時 b 可為 NULL。 */
VIZION_EXPORT int VIZION_CALL vizion_gray_sqrt(const uint8_t* src, uint8_t* dst, int count);
VIZION_EXPORT int VIZION_CALL vizion_gray_pow(const uint8_t* src, uint8_t* dst, int count, double exponent);
VIZION_EXPORT int VIZION_CALL vizion_gray_log(const uint8_t* src, uint8_t* dst, int count, int natural);
VIZION_EXPORT int VIZION_CALL vizion_gray_exp(const uint8_t* src, uint8_t* dst, int count, int natural);
VIZION_EXPORT int VIZION_CALL vizion_gray_abs(const uint8_t* src, uint8_t* dst, int count);
VIZION_EXPORT int VIZION_CALL vizion_gray_gamma(
    const uint8_t* src, uint8_t* dst, int count,
    double gamma, double offset, double threshold, double max_gray, int encode);
VIZION_EXPORT int VIZION_CALL vizion_gray_range(const uint8_t* src, uint8_t* dst, int count, double low, double high);
VIZION_EXPORT int VIZION_CALL vizion_gray_bits(const uint8_t* a, const uint8_t* b, uint8_t* dst, int count, int op);
/* op：0 否定、1 遮罩、2 左移、3 右移、4 抽出一個位元。移 8 位以上變成 0。 */
VIZION_EXPORT int VIZION_CALL vizion_gray_bitconst(const uint8_t* src, uint8_t* dst, int count, int op, int amount);
VIZION_EXPORT int VIZION_CALL vizion_map_selftest(void);

/* 高通用奇數遮罩。標準差允許偶數遮罩。灰階差的遮罩是高、寬。拉普拉斯 ksize 是 3、5 或 7。 */
VIZION_EXPORT int VIZION_CALL vizion_highpass(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height);
VIZION_EXPORT int VIZION_CALL vizion_deviation(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height);
VIZION_EXPORT int VIZION_CALL vizion_gray_span(const uint8_t* src, uint8_t* dst, int width, int height, int mask_height, int mask_width);
VIZION_EXPORT int VIZION_CALL vizion_laplace(const uint8_t* src, uint8_t* dst, int width, int height, int ksize, int eight, int absolute);
VIZION_EXPORT int VIZION_CALL vizion_local_selftest(void);

/* 填灰、查表、0.299/0.587/0.114 四捨五入。塗灰時 mask 為 0 的像素保留底圖，mask 可為 NULL。
   rank 的 border：0 重複邊緣、1 REFLECT_101、2 循環。遮罩是 1 到 7 的奇數。 */
VIZION_EXPORT int VIZION_CALL vizion_fill(uint8_t* dst, int count, int value);
VIZION_EXPORT int VIZION_CALL vizion_lut(const uint8_t* src, uint8_t* dst, int count, const uint8_t* table);
VIZION_EXPORT int VIZION_CALL vizion_rgb3_gray(const uint8_t* red, const uint8_t* green, const uint8_t* blue, uint8_t* gray, int count);
VIZION_EXPORT int VIZION_CALL vizion_paint(
    const uint8_t* base, const uint8_t* source, uint8_t* dst,
    int width, int height, int base_channels, int source_channels, const uint8_t* mask);
VIZION_EXPORT int VIZION_CALL vizion_rank(
    const uint8_t* src, uint8_t* dst, int width, int height,
    int mask_width, int mask_height, int rank, int border);
VIZION_EXPORT int VIZION_CALL vizion_prewitt(const uint8_t* src, uint8_t* dst, int width, int height);
VIZION_EXPORT int VIZION_CALL vizion_roberts(const uint8_t* src, uint8_t* dst, int width, int height, int sum_abs);
VIZION_EXPORT int VIZION_CALL vizion_binomial(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height);
VIZION_EXPORT int VIZION_CALL vizion_paint_selftest(void);

/* 羅盤邊緣、兩個方向圖的角度、局部中距。wave：0 正弦、1 餘弦、2 正切、3 反正弦、4 反餘弦、5 反正切。 */
VIZION_EXPORT int VIZION_CALL vizion_kirsch(const uint8_t* src, uint8_t* dst, int width, int height);
VIZION_EXPORT int VIZION_CALL vizion_frei(const uint8_t* src, uint8_t* dst, int width, int height);
VIZION_EXPORT int VIZION_CALL vizion_robinson_amp(const uint8_t* src, uint8_t* dst, int width, int height);
VIZION_EXPORT int VIZION_CALL vizion_robinson_dir(const uint8_t* src, uint8_t* dst, int width, int height);
VIZION_EXPORT int VIZION_CALL vizion_atan2_gray(const uint8_t* vertical, const uint8_t* horizontal, uint8_t* dst, int count);
VIZION_EXPORT int VIZION_CALL vizion_midrange(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height);
VIZION_EXPORT int VIZION_CALL vizion_wave(const uint8_t* src, uint8_t* dst, int count, int op);
VIZION_EXPORT int VIZION_CALL vizion_compass_selftest(void);

/* 斜坡、左右或上下拼接、卷積、帶通、加邊、金字塔。金字塔 mode：0 高斯、1 隔點、2 最小、3 最大。
   卷積 border：0 重複邊緣、1 循環、2 REFLECT_101。係數和接近 0 時結果加 128。 */
VIZION_EXPORT int VIZION_CALL vizion_gray_ramp(
    uint8_t* dst, int width, int height, double alpha, double beta, double gamma, double row, double column);
VIZION_EXPORT int VIZION_CALL vizion_tile(
    const uint8_t* left, int left_width, int left_height,
    const uint8_t* right, int right_width, int right_height,
    uint8_t* dst, int channels, int vertical);
VIZION_EXPORT int VIZION_CALL vizion_convol(
    const uint8_t* src, uint8_t* dst, int width, int height,
    const float* kernel, int kernel_height, int kernel_width, int border);
VIZION_EXPORT int VIZION_CALL vizion_bandpass(const uint8_t* src, uint8_t* dst, int width, int height, int inner, int outer);
VIZION_EXPORT int VIZION_CALL vizion_make_border(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels, int border, int fill);
VIZION_EXPORT int VIZION_CALL vizion_pyramid(const uint8_t* src, uint8_t* dst, int width, int height, int channels, int mode);
VIZION_EXPORT int VIZION_CALL vizion_entropy(const uint8_t* src, uint8_t* dst, int width, int height, int mask_width, int mask_height);
VIZION_EXPORT int VIZION_CALL vizion_stage_selftest(void);

/* 圓形排序半徑 1 到 5，邊界重複邊緣。灰階內部是八鄰域最低瓶頸。矩形等化含邊界。 */
VIZION_EXPORT int VIZION_CALL vizion_rank_disk(const uint8_t* src, uint8_t* dst, int width, int height, int radius, int rank);
VIZION_EXPORT int VIZION_CALL vizion_gray_inside(const uint8_t* src, uint8_t* dst, int width, int height);
VIZION_EXPORT int VIZION_CALL vizion_equalize_rect(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    int top, int left, int bottom, int right);
VIZION_EXPORT int VIZION_CALL vizion_region_mean(
    const uint8_t* gray, uint8_t* dst, int width,
    const int* rows, const int* columns, const int* lengths, int runs);
VIZION_EXPORT int VIZION_CALL vizion_surface(
    uint8_t* dst, int width, int height, double alpha, double beta, double gamma, double row, double column);
VIZION_EXPORT int VIZION_CALL vizion_surface_quad(
    uint8_t* dst, int width, int height,
    double alpha, double beta, double gamma, double delta, double epsilon, double zeta,
    double row, double column);
VIZION_EXPORT int VIZION_CALL vizion_inside_selftest(void);

/* 二值核取最大或最小，圖外重複邊緣。mode：0 膨脹、1 侵蝕、2 開運算、3 閉運算。 */
VIZION_EXPORT int VIZION_CALL vizion_gray_kernel(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    const uint8_t* kernel, int kernel_height, int kernel_width, int mode);

/* 前景權重相等，圖外補 0。列首融合乘加，列尾分開乘加，再四捨五入到偶數。 */
VIZION_EXPORT int VIZION_CALL vizion_mean_kernel(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    const uint8_t* kernel, int kernel_height, int kernel_width);

/* 索引 1 是紅、2 是綠、3 是藍。灰階只留索引 1，其餘變 0。 */
VIZION_EXPORT int VIZION_CALL vizion_select_gray(
    const uint8_t* src, const uint8_t* index, uint8_t* dst, int width, int height, int channels);

/* 矩形、菱形，或和 OpenCV 橢圓結構元素相同的八角形。shape：0 矩形、1 菱形、2 八角形。 */
VIZION_EXPORT int VIZION_CALL vizion_shape_kernel(uint8_t* kernel, int height, int width, int shape);

VIZION_EXPORT int VIZION_CALL vizion_gshape_selftest(void);

/* 高斯導數。component：0 none、1 x、2 y、3 xx、4 yy、5 xy、6 gradient、7 laplace。有號結果以 128 為 0。 */
VIZION_EXPORT int VIZION_CALL vizion_deriv_gauss(
    const uint8_t* src, uint8_t* dst, int width, int height, double sigma, int component);
/* Laws 紋理。字母 0=l、1=e、2=s、3=r、4=w。第一個沿列，第二個沿行。 */
VIZION_EXPORT int VIZION_CALL vizion_texture_laws(
    const uint8_t* src, uint8_t* dst, int width, int height, int size, int vertical, int horizontal, int shift);
/* 極座標最近鄰。角度 0 朝右，順時針繞一圈，圖外是 0。 */
VIZION_EXPORT int VIZION_CALL vizion_polar(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double row, double column, int out_width, int out_height);
/* 128 是沒變。比 128 大是變亮。 */
VIZION_EXPORT int VIZION_CALL vizion_enhance_map(const uint8_t* before, const uint8_t* after, uint8_t* dst, int count);
/* 平均用 float32 累加絕對差，再除以像素數。 */
VIZION_EXPORT int VIZION_CALL vizion_enhance_change(
    const uint8_t* before, const uint8_t* after, int count, double* mean, double* maximum);
/* 絕對差達到門檻的像素是 255。門檻 0 表示只要有差就算。 */
VIZION_EXPORT int VIZION_CALL vizion_enhance_hot(
    const uint8_t* before, const uint8_t* after, uint8_t* mask, int count, double limit);
/* 定義域外不算。絕對差達到 MinDiff 的像素是 255。 */
VIZION_EXPORT int VIZION_CALL vizion_diff_mask(
    const uint8_t* image, const uint8_t* golden, const uint8_t* domain, uint8_t* mask, int count, double minimum);
/* 只統計定義域內的絕對差。沒有有效像素時平均和最大都是 0。 */
VIZION_EXPORT int VIZION_CALL vizion_diff_amount(
    const uint8_t* image, const uint8_t* golden, const uint8_t* domain, int count, double* mean, double* maximum);
/* 原圖減去高斯，比門檻小的起伏當成 0，再以 128 為 0。 */
VIZION_EXPORT int VIZION_CALL vizion_edge_residual(
    const uint8_t* src, uint8_t* dst, int width, int height, double sigma, double threshold);
/* 原圖減模糊的差加回亮度。太小不加，超過 Limit 截斷。定義域外保持原像素。 */
VIZION_EXPORT int VIZION_CALL vizion_edge_enhance(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double sigma, double factor, double threshold, double limit, const uint8_t* domain);
VIZION_EXPORT int VIZION_CALL vizion_grade_selftest(void);

/* 細高斯減粗高斯，太小的起伏不加，再加回亮度。定義域外保持原像素。 */
VIZION_EXPORT int VIZION_CALL vizion_edge_band(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double sigma_fine, double sigma_coarse, double factor, double threshold, const uint8_t* domain);
/* 原圖減模糊的差，乘上梯度佔 Level 的比例，再加回亮度。定義域外保持原像素。 */
VIZION_EXPORT int VIZION_CALL vizion_edge_masked(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double sigma, double factor, double level, const uint8_t* domain);
/* 梯度小的地方改用高斯，梯度達到 Level 就留原圖。定義域外保持原像素。 */
VIZION_EXPORT int VIZION_CALL vizion_edge_preserve(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double sigma, double level, const uint8_t* domain);
/* 四鄰域 Perona-Malik 或拋物線傳導。最外一圈不動。彩色先轉成亮度。mode：0 perona-malik、1 parabolic。 */
VIZION_EXPORT int VIZION_CALL vizion_anisotropic(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    int mode, double contrast, double theta, int iterations);
/* 沿著拉普拉斯或梯度方向的二階導數把邊緣變陡。完全平的地方不變。mode：0 laplace、1 canny。 */
VIZION_EXPORT int VIZION_CALL vizion_shock(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double theta, int iterations, int mode, double sigma);
/* 用影像自己的亮度當引導。平坦處變成局部平均，階梯邊緣的兩邊不會被拉到中間。 */
VIZION_EXPORT int VIZION_CALL vizion_guided(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels, int radius, double eps);
/* Frangi 用 Hessian 兩個特徵值把細線加強，方塊內部接近 0。dark 為 1 時加強暗線。 */
VIZION_EXPORT int VIZION_CALL vizion_frangi(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    double sigma, double sigma_max, int dark);
/* 切成格子各自等化，相鄰格子再混起來。彩色只處理亮度。 */
VIZION_EXPORT int VIZION_CALL vizion_clahe(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels, double clip, int tiles);
/* 樣板或核放在左上角，其餘補 0，做循環相關或循環卷積，再截到 0 到 255。correlation 為 1 是相關。 */
VIZION_EXPORT int VIZION_CALL vizion_circular(
    const uint8_t* image, const uint8_t* kernel, uint8_t* dst,
    int width, int height, int kernel_width, int kernel_height, int correlation);
VIZION_EXPORT int VIZION_CALL vizion_shade_selftest(void);

/* 區域行程上的平均和母體標準差。沒有像素時兩者都是 0。 */
VIZION_EXPORT int VIZION_CALL vizion_intensity(
    const uint8_t* gray, int width,
    const int* rows, const int* columns, const int* lengths, int runs,
    double* mean, double* deviation);

VIZION_EXPORT int VIZION_CALL vizion_intensity_selftest(void);

/* 繞影像中心最近鄰旋轉。正弦餘弦由呼叫端算好。圖外是 0。 */
VIZION_EXPORT int VIZION_CALL vizion_rotate_nearest(
    const uint8_t* src, uint8_t* dst, int width, int height, double cosine, double sine);

VIZION_EXPORT int VIZION_CALL vizion_rotate_selftest(void);

/* 次像素邊緣的遲滯和串鏈。列、行是 float 乘加後再存成 double。 */
VIZION_EXPORT int VIZION_CALL vizion_edge_chains(
    const unsigned char* center, const float* magnitude, double low, double high,
    int width, int height,
    const float* shift, const float* normal_x, const float* normal_y,
    double* rows, double* columns, int* offsets, int* counts,
    int point_capacity, int chain_capacity, int* point_count, int* chain_count);

VIZION_EXPORT int VIZION_CALL vizion_edge_link_selftest(void);

/* 找線、找圓。cols 是行、rows 是列。圓擬合不出來回 -2。 */
VIZION_EXPORT int VIZION_CALL vizion_fit_line(
    const double* cols, const double* rows, int count,
    double* row_begin, double* col_begin, double* row_end, double* col_end);

VIZION_EXPORT int VIZION_CALL vizion_fit_circle(
    const double* cols, const double* rows, int count,
    double* row, double* col, double* radius);

VIZION_EXPORT int VIZION_CALL vizion_fit_selftest(void);

/* 形狀命中之後的半像素、角度、拋物線和最小二乘。分數仍走既有的 SIMD 取樣。 */
VIZION_EXPORT int VIZION_CALL vizion_shape_pose(
    const float* model_x, const float* model_y, const float* model_gx, const float* model_gy, int model_count,
    const float* gx, const float* gy, const float* mag, int width, int height, int scale,
    double row, double column, double angle, double score, double min_contrast,
    double* out_row, double* out_column, double* out_angle, double* out_score);

VIZION_EXPORT int VIZION_CALL vizion_pose_selftest(void);

#ifdef __cplusplus
}
#endif

#endif
