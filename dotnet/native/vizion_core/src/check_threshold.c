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
    return 0;
}
