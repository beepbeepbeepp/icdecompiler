/* Host wrapper for the tiny-cnn test model, INT8 by default or Q8.8 with
   -DQ88. Reads the input from stdin and prints the four outputs. */

#include <stdio.h>
#include <stdint.h>
#include "tiny_weights.h"

#ifdef Q88
typedef int16_t T;
#else
typedef int8_t T;
#endif

/* Arrays decay to pointers, so this flat prototype matches the generated
   function in both modes. */
void tiny(T *in,
          const T *w1, const int32_t *b1,
          const T *w2, const int32_t *b2,
          const T *w3, const int32_t *b3,
          const T *wf1, const int32_t *bf1,
          const T *wf2, const int32_t *bf2,
          T *out);

int main(void)
{
    static T in[64 * 3];
    T out[4];

    for (int i = 0; i < 64 * 3; i++) {
        int v;
        if (scanf("%d", &v) != 1)
            return 1;
        in[i] = (T)v;
    }

    tiny(in,
         (const T *)tiny_conv1_w, tiny_conv1_b,
         (const T *)tiny_conv2_w, tiny_conv2_b,
         (const T *)tiny_conv3_w, tiny_conv3_b,
         (const T *)tiny_fc1_w,   tiny_fc1_b,
         (const T *)tiny_fc2_w,   tiny_fc2_b,
         out);

    printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
    return 0;
}