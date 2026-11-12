/*
 * Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
 * Author: Muhammad Faudzan Abdullah
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * expf for RISC-V cores without an FPU.
 *
 * The libm shipped with the PULP toolchain is built with the F extension, so
 * its expf uses hardware float instructions and traps on a core without one.
 * Compiling this file with the application's flags (-march without f) makes
 * every float operation here a soft-float call instead. Because the symbol is
 * defined in an object file, the linker uses it and never pulls expf from libm.
 *
 * Method: x = k*ln2 + r with |r| <= ln2/2, exp(r) from a degree-6 polynomial,
 * then scale by 2^k by building the exponent bits directly. Relative error is
 * around 1e-7.
 */

#include <stdint.h>

typedef union { float f; uint32_t u; } f32bits;

static float pow2i(int k)
{
    f32bits s;
    s.u = (uint32_t)(k + 127) << 23;   /* valid for k in [-126, 127] */
    return s.f;
}

float expf(float x)
{
    if (x != x)                        /* NaN passes through */
        return x;
    if (x > 88.72283f) {               /* overflows float */
        f32bits inf;
        inf.u = 0x7f800000u;
        return inf.f;
    }
    if (x < -103.97208f)               /* underflows to zero */
        return 0.0f;

    const float inv_ln2 = 1.44269504f;
    const float ln2_hi  = 6.93145752e-1f;   /* ln2 split in two for accuracy */
    const float ln2_lo  = 1.42860677e-6f;

    float kf = x * inv_ln2;
    int k = (int)(kf + (kf >= 0.0f ? 0.5f : -0.5f));
    float r = x - (float)k * ln2_hi - (float)k * ln2_lo;

    float p = 1.0f + r * (1.0f + r * (0.5f + r * (1.6666667e-1f +
              r * (4.1666668e-2f + r * (8.3333338e-3f + r * 1.3888889e-3f)))));

    if (k > 127)                       /* x near the top: 2^k in two steps */
        return p * pow2i(127) * pow2i(k - 127);
    if (k < -126)                      /* subnormal result: same trick */
        return p * pow2i(k + 100) * pow2i(-100);
    return p * pow2i(k);
}