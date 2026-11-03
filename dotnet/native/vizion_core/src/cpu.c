#include "vizion_core.h"
#include "threshold_internal.h"

#if defined(_MSC_VER)
#  include <immintrin.h>
#  include <intrin.h>
#elif defined(__i386__) || defined(__x86_64__)
#  include <cpuid.h>
#endif

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <pthread.h>
#endif

static int g_flags;
static vizion_threshold_fn g_threshold = vizion_threshold_scalar;

#if defined(_MSC_VER)
static void vizion_cpuid(int leaf, int sub, int out[4])
{
    __cpuidex(out, leaf, sub);
}

static unsigned long long vizion_xcr0(void)
{
    return _xgetbv(0);
}
#elif defined(__i386__) || defined(__x86_64__)
static void vizion_cpuid(int leaf, int sub, int out[4])
{
    unsigned int a, b, c, d;
    __cpuid_count((unsigned)leaf, (unsigned)sub, a, b, c, d);
    out[0] = (int)a;
    out[1] = (int)b;
    out[2] = (int)c;
    out[3] = (int)d;
}

static unsigned long long vizion_xcr0(void)
{
    unsigned int eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long)edx << 32) | eax;
}
#else
static void vizion_cpuid(int leaf, int sub, int out[4])
{
    (void)leaf;
    (void)sub;
    out[0] = out[1] = out[2] = out[3] = 0;
}

static unsigned long long vizion_xcr0(void)
{
    return 0;
}
#endif

static void vizion_detect(void)
{
    int leaf0[4];
    int leaf1[4];
    int flags = 0;
    int sse2 = 0;
    int avx2 = 0;
    vizion_cpuid(0, 0, leaf0);
    if (leaf0[0] >= 1) {
        int osxsave;
        int avx;
        vizion_cpuid(1, 0, leaf1);
        sse2 = (leaf1[3] & (1 << 26)) != 0;
        osxsave = (leaf1[2] & (1 << 27)) != 0;
        avx = (leaf1[2] & (1 << 28)) != 0;
        if (sse2)
            flags |= VIZION_CPU_SSE2;
        if (osxsave && avx && leaf0[0] >= 7) {
            int leaf7[4];
            unsigned long long xcr;
            vizion_cpuid(7, 0, leaf7);
            xcr = vizion_xcr0();
            if ((leaf7[1] & (1 << 5)) != 0 && (xcr & 6ull) == 6ull)
                avx2 = 1;
        }
    }
    if (avx2) {
        flags |= VIZION_CPU_AVX2 | VIZION_CPU_SELECTED_AVX2;
        g_threshold = vizion_threshold_avx2;
    } else if (sse2) {
        flags |= VIZION_CPU_SELECTED_SSE2;
        g_threshold = vizion_threshold_sse2;
    } else {
        flags |= VIZION_CPU_SELECTED_SCALAR;
        g_threshold = vizion_threshold_scalar;
    }
    g_flags = flags;
}

#if defined(_WIN32)
static INIT_ONCE g_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK vizion_init_once(PINIT_ONCE once, PVOID param, PVOID* context)
{
    (void)once;
    (void)param;
    (void)context;
    vizion_detect();
    return TRUE;
}

static void vizion_ensure(void)
{
    InitOnceExecuteOnce(&g_once, vizion_init_once, 0, 0);
}
#else
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void vizion_ensure(void)
{
    pthread_once(&g_once, vizion_detect);
}
#endif

int vizion_cpu_flags(void)
{
    vizion_ensure();
    return g_flags;
}

vizion_threshold_fn vizion_threshold_selected(void)
{
    vizion_ensure();
    return g_threshold;
}
