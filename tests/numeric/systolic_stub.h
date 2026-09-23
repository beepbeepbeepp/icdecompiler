#ifndef SYSTOLIC_STUB_H
#define SYSTOLIC_STUB_H

/*
 * Host-side stub of the systolic array driver.
 * Computes the same matmul in plain C so generated code can be checked
 * for numerical correctness without RTL or hardware.
 */

#include <stdint.h>
#include <stddef.h>

static inline void systolic_multiply_large(int M, int K, int N,
                                           int16_t *A, int16_t *B,
                                           int32_t *Res) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            Res[i * N + j] = acc;
        }
    }
}

static inline int systolic_wait_done(int timeout_limit) {
    (void)timeout_limit;
    return 0;
}

#endif