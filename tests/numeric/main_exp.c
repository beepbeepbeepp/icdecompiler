/*
 * Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
 * Author: Muhammad Faudzan Abdullah
 * SPDX-License-Identifier: Apache-2.0
 */

/* exp and sigmoid lower to libm calls; check the numbers against libm directly. */

#include <math.h>
#include <stdio.h>

void op(float a[1][8][4], float out[1][8][4]);

int main(void)
{
    float a[1][8][4], out[1][8][4];
    for (int i = 0; i < 8; i++)
        for (int c = 0; c < 4; c++)
            a[0][i][c] = (float)(i * 4 + c - 16) * 0.25f;

    op(a, out);

    int bad = 0;
    for (int i = 0; i < 8; i++)
        for (int c = 0; c < 4; c++) {
            float x = a[0][i][c];
#ifdef SIGMOID
            float ref = 1.0f / (1.0f + expf(-x));
#else
            float ref = expf(x);
#endif
            float err = fabsf(out[0][i][c] - ref);
            if (err > 1e-5f * (fabsf(ref) + 1.0f)) {
                if (bad < 5)
                    printf("mismatch [%d][%d]: x=%f got %f expected %f\n",
                           i, c, x, out[0][i][c], ref);
                bad++;
            }
        }

    if (bad == 0) {
#ifdef SIGMOID
        printf("OK: sigmoid matches libm reference\n");
#else
        printf("OK: exp matches libm reference\n");
#endif
        return 0;
    }
    printf("FAIL: %d mismatches\n", bad);
    return 1;
}