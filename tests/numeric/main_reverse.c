/*
 * Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
 * Author: Muhammad Faudzan Abdullah
 * SPDX-License-Identifier: Apache-2.0
 */

/* A buffer reversed through a temporary. If the compiler removes the
   temporary and writes in place, the second half reads elements that were
   already overwritten and the result is 4 3 3 4. */

#include <stdint.h>
#include <stdio.h>

int32_t reverse(int32_t buf[4]);

int main(void)
{
    int32_t buf[4] = {1, 2, 3, 4};
    const int32_t want[4] = {4, 3, 2, 1};

    int32_t status = reverse(buf);

    int ok = (status == 0);
    for (int i = 0; i < 4; i++)
        if (buf[i] != want[i])
            ok = 0;

    if (!ok) {
        printf("FAIL: reverse: got %d %d %d %d, expected 4 3 2 1\n",
               (int)buf[0], (int)buf[1], (int)buf[2], (int)buf[3]);
        return 1;
    }
    printf("OK: reverse through a temporary keeps its copy\n");
    return 0;
}
