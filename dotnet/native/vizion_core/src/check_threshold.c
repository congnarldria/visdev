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
    return 0;
}
