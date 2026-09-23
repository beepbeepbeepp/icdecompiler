/* Compare the soft-float expf in runtime/ against the host libm.
   The runtime file is compiled with -Dexpf=icdec_expf so both can coexist. */

#include <math.h>
#include <stdio.h>

float icdec_expf(float x);

int main(void)
{
    int bad = 0, n = 0;
    float worst = 0.0f, worst_x = 0.0f;

    for (float x = -100.0f; x <= 88.0f; x += 0.0137f) {
        float ref = expf(x);
        float got = icdec_expf(x);
        n++;

        float rel = (ref != 0.0f) ? fabsf(got - ref) / ref : fabsf(got);
        if (rel > worst) { worst = rel; worst_x = x; }
        if (rel > 1e-6f && fabsf(got - ref) > 1.5e-45f) {
            if (bad < 5)
                printf("mismatch x=%g: got %g expected %g (rel %g)\n",
                       x, got, ref, rel);
            bad++;
        }
    }

    /* edges */
    if (icdec_expf(0.0f) != 1.0f)                { printf("exp(0) wrong\n"); bad++; }
    if (!isinf(icdec_expf(100.0f)))              { printf("overflow wrong\n"); bad++; }
    if (icdec_expf(-120.0f) != 0.0f)             { printf("underflow wrong\n"); bad++; }
    if (!isnan(icdec_expf(NAN)))                 { printf("nan wrong\n"); bad++; }

    printf("checked %d points, worst relative error %g at x=%g\n", n, worst, worst_x);
    if (bad == 0) {
        printf("OK: runtime expf matches libm\n");
        return 0;
    }
    printf("FAIL: %d mismatches\n", bad);
    return 1;
}