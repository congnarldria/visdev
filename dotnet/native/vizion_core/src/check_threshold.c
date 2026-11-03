#include "vizion_core.h"

#include <stdio.h>

int main(void)
{
    int flags = vizion_cpu_flags();
    int code = vizion_threshold_selftest();
    if (code != 0) {
        fprintf(stderr, "閾值自我檢查失敗 %d，flags %d\n", code, flags);
        return 1;
    }
    printf("閾值自我檢查通過 flags %d\n", flags);
    code = vizion_region_selftest();
    if (code != 0) {
        fprintf(stderr, "區域自我檢查失敗 %d\n", code);
        return 1;
    }
    printf("區域自我檢查通過\n");
    code = vizion_morph_selftest();
    if (code != 0) {
        fprintf(stderr, "型態學自我檢查失敗 %d\n", code);
        return 1;
    }
    printf("型態學自我檢查通過\n");
    code = vizion_filter_selftest();
    if (code != 0) {
        fprintf(stderr, "濾波自我檢查失敗 %d\n", code);
        return 1;
    }
    printf("濾波自我檢查通過\n");
    code = vizion_score_selftest();
    if (code != 0) {
        fprintf(stderr, "分數自我檢查失敗 %d\n", code);
        return 1;
    }
    printf("分數自我檢查通過\n");
    code = vizion_measure_selftest();
    if (code != 0) {
        fprintf(stderr, "量測自我檢查失敗 %d\n", code);
        return 1;
    }
    printf("量測自我檢查通過\n");
    code = vizion_edge_selftest();
    if (code != 0) {
        fprintf(stderr, "邊緣自我檢查失敗 %d\n", code);
        return 1;
    }
    printf("邊緣自我檢查通過\n");
    code = vizion_sobel_selftest();
    if (code != 0) {
        fprintf(stderr, "梯度自我檢查失敗 %d\n", code);
        return 1;
    }
    printf("梯度自我檢查通過\n");
    code = vizion_pyrdown_selftest();
    if (code != 0) {
        fprintf(stderr, "金字塔自我檢查失敗 %d\n", code);
        return 1;
    }
    printf("金字塔自我檢查通過\n");
    code = vizion_bgr_selftest();
    if (code != 0) {
        fprintf(stderr, "灰階自我檢查失敗 %d\n", code);
        return 1;
    }
    printf("灰階自我檢查通過\n");
    return 0;
}
