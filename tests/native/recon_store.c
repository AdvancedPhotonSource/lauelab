/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "liblaue.h"

/* More than one STORE_BLOCK, with an incomplete last block. */
enum { DEPTHS = 33, PIXELS = 4097 };

int main(void)
{
    double *values = malloc(DEPTHS * PIXELS * sizeof(double));
    uint16_t *stored = malloc(DEPTHS * PIXELS * sizeof(uint16_t));
    int64_t *depth = malloc(DEPTHS * sizeof(int64_t));
    int64_t *projection = malloc(PIXELS * sizeof(int64_t));
    int failed = 0;
    if (!values || !stored || !depth || !projection) return 2;
    for (int d = 0; d < DEPTHS; ++d)
        for (int p = 0; p < PIXELS; ++p) values[d * PIXELS + p] = d + 1;
    for (int repeat = 0; repeat < 5; ++repeat) {
        if (laue_recon_store_stripe(values, DEPTHS, PIXELS, 1, LAUE_PIXEL_U16,
                                   stored, depth, projection, 4) != LAUE_OK) failed = 1;
        for (int d = 0; d < DEPTHS; ++d)
            if (depth[d] != (d + 1) * PIXELS) failed = 1;
        for (int p = 0; p < PIXELS; ++p)
            if (projection[p] != DEPTHS * (DEPTHS + 1) / 2) failed = 1;
        if (failed) break;
    }
    free(values);
    free(stored);
    free(depth);
    free(projection);
    if (failed) fprintf(stderr, "stored reductions disagree with known pixel sums\n");
    return failed;
}
