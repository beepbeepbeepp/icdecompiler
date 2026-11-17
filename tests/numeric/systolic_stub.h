/*
 * Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
 * Author: Muhammad Faudzan Abdullah
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SYSTOLIC_STUB_H
#define SYSTOLIC_STUB_H

/*
 * Host-side stub of the systolic array driver.
 * Computes the same matmul in plain C so generated code can be checked
 * for numerical correctness without RTL or hardware.
 */

#include <stdint.h>
#include <stddef.h>

/* Same codes and limit as the real driver, so the stub fails where it would. */
#define SYSTOLIC_HWPE_OK            0
#define SYSTOLIC_HWPE_ERR_TIMEOUT  -2
#define SYSTOLIC_HWPE_ERR_BAD_K    -3
#define SYSTOLIC_HWPE_ERR_BAD_ARG  -4
#define SYSTOLIC_HWPE_RTL_MAX_K   128

/* Build with -DSYSTOLIC_STUB_FAIL_CALL=n to make the n-th call time out. */
#ifndef SYSTOLIC_STUB_FAIL_CALL
#define SYSTOLIC_STUB_FAIL_CALL 0
#endif
static int systolic_stub_calls = 0;

static inline int systolic_multiply_large_hwpe(int M, int K, int N,
                                               int16_t *A, int16_t *B,
                                               int32_t *Res) {
    if (M <= 0 || K <= 0 || N <= 0 || !A || !B || !Res)
        return SYSTOLIC_HWPE_ERR_BAD_ARG;
    if (K > SYSTOLIC_HWPE_RTL_MAX_K)
        return SYSTOLIC_HWPE_ERR_BAD_K;
    if (++systolic_stub_calls == SYSTOLIC_STUB_FAIL_CALL)
        return SYSTOLIC_HWPE_ERR_TIMEOUT;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            Res[i * N + j] = acc;
        }
    }
    return SYSTOLIC_HWPE_OK;
}

/* The older entry point, which drops the status. */
static inline void systolic_multiply_large(int M, int K, int N,
                                           int16_t *A, int16_t *B,
                                           int32_t *Res) {
    (void)systolic_multiply_large_hwpe(M, K, N, A, B, Res);
}

static inline int systolic_wait_done(int timeout_limit) {
    (void)timeout_limit;
    return 0;
}

#endif