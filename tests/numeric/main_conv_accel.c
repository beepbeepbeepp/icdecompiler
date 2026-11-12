/*
 * Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
 * Author: Muhammad Faudzan Abdullah
 * SPDX-License-Identifier: Apache-2.0
 */

/* Compare the accelerator path against the CPU path for one convolution.
   Both are compiled into this program, so any difference is in the lowering. */

#include <stdint.h>
#include <stdio.h>

void conv_cpu(int16_t in[1][1][8][2], int16_t w[4][1][3][2], int32_t b[4],
              int32_t out[1][1][8][4]);
void conv_acc(int16_t in[1][1][8][2], int16_t w[4][1][3][2], int32_t b[4],
              int32_t out[1][1][8][4]);

int main(void)
{
    int16_t in[1][1][8][2];
    int16_t w[4][1][3][2];
    int32_t bias[4];
    int32_t out_cpu[1][1][8][4] = {{{{0}}}};
    int32_t out_acc[1][1][8][4] = {{{{0}}}};

    for (int i = 0; i < 8; i++)
        for (int c = 0; c < 2; c++)
            in[0][0][i][c] = (int16_t)((i * 3 + c * 5) % 11 - 5);

    for (int f = 0; f < 4; f++) {
        for (int k = 0; k < 3; k++)
            for (int c = 0; c < 2; c++)
                w[f][0][k][c] = (int16_t)((f * 7 + k * 3 + c) % 9 - 4);
        bias[f] = f * 100;
    }

    conv_cpu(in, w, bias, out_cpu);
    conv_acc(in, w, bias, out_acc);

    int bad = 0;
    for (int i = 0; i < 8; i++)
        for (int f = 0; f < 4; f++)
            if (out_cpu[0][0][i][f] != out_acc[0][0][i][f]) {
                if (bad < 8)
                    printf("mismatch [%d][%d]: cpu=%d accel=%d\n",
                           i, f, out_cpu[0][0][i][f], out_acc[0][0][i][f]);
                bad++;
            }

    if (bad == 0) {
        printf("OK: accelerator path matches CPU path\n");
        return 0;
    }
    printf("FAIL: %d of 32 differ\n", bad);
    return 1;
}