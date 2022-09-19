#include <include/sgemm.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <x86intrin.h> // _AVX512

using namespace std;

#define A(i, j) a[i * lda + j]
#define B(i, j) b[i * ldb + j]
#define C(i, j) c[i * ldc + j]

namespace lib {

void sgemm(
    int m, int n, int k, 
    float *a, int lda,
    float *b, int ldb,
    float *c, int ldc
) {
    for (int kk = 0; kk < k; ++kk) {
    for (int ii = 0; ii < m; ++ii) {
    for (int jj = 0; jj < n; ++jj) {
        C(ii, jj) += A(ii, kk) * B(kk, jj);
    }}}
}

} // lib