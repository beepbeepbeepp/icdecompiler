/*
 * Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
 * Author: Muhammad Faudzan Abdullah
 * SPDX-License-Identifier: Apache-2.0
 */

/* Host wrapper for the depthwise + residual test model.
   Reads the int8 input from stdin, prints the four int8 outputs. */

#include <stdio.h>
#include <stdint.h>
#include "dw_weights.h"

void dw(int8_t input[1][128][3],
        int8_t w1[8][1][5][3],  int32_t b1[8],
        int8_t wd[1][3][8][1],  int32_t bd[8],
        int8_t w2[8][1][1][8],  int32_t b2[8],
        int8_t wf[1][8][4],
        int8_t out[1][4]);

int main(void)
{
    static int8_t input[1][128][3];
    int8_t out[1][4];

    for (int i = 0; i < 128; i++)
        for (int c = 0; c < 3; c++) {
            int v;
            if (scanf("%d", &v) != 1)
                return 1;
            input[0][i][c] = (int8_t)v;
        }

    dw(input,
       (int8_t (*)[1][5][3])dw_conv1_w,  (int32_t *)dw_conv1_b,
       (int8_t (*)[3][8][1])dw_dwconv1_w, (int32_t *)dw_dwconv1_b,
       (int8_t (*)[1][1][8])dw_conv2_w,  (int32_t *)dw_conv2_b,
       (int8_t (*)[8][4])dw_fc1_w,
       out);

    printf("%d %d %d %d\n", out[0][0], out[0][1], out[0][2], out[0][3]);
    return 0;
}