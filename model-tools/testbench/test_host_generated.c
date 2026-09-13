/* Host wrapper for compiler-generated inference.
   Reads the quantized input from stdin, prints the output logits. */

#include <stdio.h>
#include <stdint.h>
#include "har_weights_int8.h"

void har_model(int8_t input[1][512][3],
               int8_t w1[16][1][15][3], int32_t b1[16],
               int8_t w2[32][1][7][16],  int32_t b2[32],
               int8_t w3[64][1][3][32],  int32_t b3[64],
               int8_t wf1[1][64][32],    int32_t bf1[32],
               int8_t wf2[1][32][4],     int32_t bf2[4],
               int8_t out[1][4]);

int main(void)
{
    static int8_t input[1][512][3];
    int8_t out[1][4];

    for (int i = 0; i < 512; i++)
        for (int c = 0; c < 3; c++) {
            int v;
            if (scanf("%d", &v) != 1)
                return 1;
            input[0][i][c] = (int8_t)v;
        }

    har_model(input,
              (int8_t (*)[1][15][3])har_conv1_w, (int32_t *)har_conv1_b,
              (int8_t (*)[1][7][16])har_conv2_w, (int32_t *)har_conv2_b,
              (int8_t (*)[1][3][32])har_conv3_w, (int32_t *)har_conv3_b,
              (int8_t (*)[64][32])har_fc1_w,     (int32_t *)har_fc1_b,
              (int8_t (*)[32][4])har_fc2_w,      (int32_t *)har_fc2_b,
              out);

    printf("%d %d %d %d\n", out[0][0], out[0][1], out[0][2], out[0][3]);
    return 0;
}