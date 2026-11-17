/*
 * Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
 * Author: Muhammad Faudzan Abdullah
 * SPDX-License-Identifier: Apache-2.0
 */

/* A failed accelerator call must stop the model: the function returns the
   driver's code and nothing after the failed call runs. Built with
   -DSYSTOLIC_STUB_FAIL_CALL=2, so the first call works and the second times
   out. */

#include <stdint.h>
#include <stdio.h>
#include "systolic_stub.h"

int32_t two_calls(int16_t A[8][8], int16_t B[8][8],
                  int32_t C[8][8], int32_t D[8][8]);

#define SENTINEL 0x5A5A5A5A

int main(void)
{
    int16_t A[8][8], B[8][8];
    int32_t C[8][8], D[8][8];

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            A[i][j] = (int16_t)(i * 8 + j - 20);
            B[i][j] = (int16_t)(3 * i - 2 * j + 1);
            C[i][j] = 0;
            D[i][j] = SENTINEL;
        }

    int32_t rc = two_calls(A, B, C, D);
    if (rc != SYSTOLIC_HWPE_ERR_TIMEOUT) {
        printf("FAIL: returned %d, expected %d\n", rc, SYSTOLIC_HWPE_ERR_TIMEOUT);
        return 1;
    }

    /* the first call finished, so C holds A x B */
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            int32_t acc = 0;
            for (int k = 0; k < 8; k++)
                acc += (int32_t)A[i][k] * (int32_t)B[k][j];
            if (C[i][j] != acc) {
                printf("FAIL: C[%d][%d] = %d, expected %d\n", i, j, C[i][j], acc);
                return 1;
            }
        }

    /* the second call failed, so D was never written */
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            if (D[i][j] != SENTINEL) {
                printf("FAIL: D[%d][%d] was written after the failed call\n", i, j);
                return 1;
            }

    printf("OK: a failed accelerator call stops the model and returns its code\n");
    return 0;
}
