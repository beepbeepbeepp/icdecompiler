/* Host wrapper: read quantized input from stdin, print logits.
   Used by verify_c.py to compare against the TFLite interpreter. */

#include <stdio.h>
#include <stdint.h>

void har_int8(const int8_t *input, int8_t *logits);

int main(void)
{
    static int8_t input[512 * 3];
    int8_t logits[4];

    for (int i = 0; i < 512 * 3; i++) {
        int v;
        if (scanf("%d", &v) != 1)
            return 1;
        input[i] = (int8_t)v;
    }

    har_int8(input, logits);
    printf("%d %d %d %d\n", logits[0], logits[1], logits[2], logits[3]);
    return 0;
}