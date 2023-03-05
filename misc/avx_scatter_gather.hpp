#include <x86intrin.h>

#define _MM256_SCATTER_PD(double* addr0, double* addr1, \
                          double* addr2, double* addr3, \
                          __m256d vec) {                \
                                                        \
_m128d tmp0 = _mm256_extractf128_pd((vec), 0);          \
_m128d tmp1 = _mm256_extractf128_pd((vec), 1);          \
                                                        \
_mm_store_sd ((addr0), tmp0);                           \
_mm_storeh_pd((addr1), tmp0);                           \
_mm_store_sd ((addr2), tmp1);                           \
_mm_storeh_pd((addr3), tmp1);                           \
}

void _mm256_scatter_pd(double* addr0, double* addr1,
                       double* addr2, double* addr3,
                       __m256d vec)
{
    _m128d tmp0 = _mm256_extractf128_pd(vec, 0);
    _m128d tmp1 = _mm256_extractf128_pd(vec, 1);

    _mm_store_sd (addr0, tmp0);
    _mm_storeh_pd(addr1, tmp0);
    _mm_store_sd (addr2, tmp1);
    _mm_storeh_pd(addr3, tmp1);
}

__m256d _mm256_gather_pd(double* addr0, double* addr1,
                         double* addr2, double* addr3)
{
    __m256d tmp0 = _mm256_loadu_pd(addr0);
    __m256d tmp1 = _mm256_loadu_pd(addr1 - 1);
    __m256d tmp2 = _mm256_loadu_pd(addr2 - 2);
    __m256d tmp3 = _mm256_loadu_pd(addr3 - 3);

    __m256d tmp4 = _mm256_blend_pd(tmp0, tmp2, 0x2);
    __m256d tmp5 = _mm256_blend_pd(tmp1, tmp3, 0x8);

    return _mm256_blend_pd(tmp4, tmp5, 0xC);
}

