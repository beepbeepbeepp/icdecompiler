/* reduce_max over axis 1 went through a reinterpret_cast before this was
   fixed, so check the rewritten indexing gives the right numbers. */

#include <stdint.h>
#include <stdio.h>

void op(int8_t a[1][8][4], int8_t out[1][1][4]);

int main(void)
{
    int8_t a[1][8][4];
    int8_t out[1][1][4] = {{{0}}};

    for (int i = 0; i < 8; i++)
        for (int c = 0; c < 4; c++)
            a[0][i][c] = (int8_t)(((i * 7 + c * 13) % 23) - 11);

    op(a, out);

    int bad = 0;
    for (int c = 0; c < 4; c++) {
        int8_t ref = a[0][0][c];
        for (int i = 1; i < 8; i++)
            if (a[0][i][c] > ref) ref = a[0][i][c];
        if (out[0][0][c] != ref) {
            printf("mismatch [%d]: got %d, expected %d\n", c, out[0][0][c], ref);
            bad++;
        }
    }

    if (bad == 0) {
        printf("OK: reduce_max through reinterpret_cast matches reference\n");
        return 0;
    }
    printf("FAIL: %d mismatches\n", bad);
    return 1;
}