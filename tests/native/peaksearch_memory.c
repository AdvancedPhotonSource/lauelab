/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "liblaue.h"

enum {
    IMAGE_WIDTH = 80,
    IMAGE_HEIGHT = 64,
    REPEAT_COUNT = 5
};

static void add_gaussian(unsigned short *pixels, int center_x, int center_y)
{
    int x;
    int y;

    for (y = 0; y < IMAGE_HEIGHT; ++y) {
        for (x = 0; x < IMAGE_WIDTH; ++x) {
            double dx = x - center_x;
            double dy = y - center_y;
            unsigned short signal = (unsigned short)(2000.0 * exp(-(dx * dx + dy * dy) / 8.0));
            pixels[y * IMAGE_WIDTH + x] += signal;
        }
    }
}

static void add_saturated_plateau(unsigned short *pixels, int center_x, int center_y)
{
    int x;
    int y;

    for (y = center_y - 5; y <= center_y + 5; ++y) {
        for (x = center_x - 5; x <= center_x + 5; ++x) {
            pixels[y * IMAGE_WIDTH + x] = 65535;
        }
    }
}

int main(void)
{
    unsigned short pixels[IMAGE_WIDTH * IMAGE_HEIGHT];
    static int32_t int32_pixels[IMAGE_WIDTH * IMAGE_HEIGHT];
    static double double_pixels[IMAGE_WIDTH * IMAGE_HEIGHT];
    laue_peak_params params = {0};
    int iteration;
    int index;

    for (index = 0; index < IMAGE_WIDTH * IMAGE_HEIGHT; ++index) pixels[index] = 10;
    add_gaussian(pixels, 12, 12);
    add_gaussian(pixels, 38, 14);
    add_gaussian(pixels, 20, 46);
    add_saturated_plateau(pixels, 66, 48);

    params.boxsize = 6;
    params.max_rfactor = 1.0;
    params.min_size = 2;
    params.min_separation = 5;
    params.threshold = 100.0;
    params.threshold_ratio = 4.0;
    params.peak_shape = 0;
    params.max_peaks = 100;

    for (iteration = 0; iteration < REPEAT_COUNT; ++iteration) {
        laue_frame_result result = {0};
        int status = laue_find_peaks(pixels, IMAGE_WIDTH, IMAGE_HEIGHT, &params, &result);

        if (status != LAUE_OK) {
            fprintf(stderr, "peak search %d failed: %s\n", iteration, result.message);
            laue_frame_result_free(&result);
            return status;
        }
        if (result.n_peaks != 3) {
            fprintf(stderr, "peak search %d returned %d peaks, expected 3\n",
                    iteration, result.n_peaks);
            laue_frame_result_free(&result);
            return 1;
        }
        laue_frame_result_free(&result);
    }

    /* Unlimited peak mode (max_peaks == 0) and the typed entry point with
       int32 and double frames must find the same three peaks. */
    for (index = 0; index < IMAGE_WIDTH * IMAGE_HEIGHT; ++index) {
        int32_pixels[index] = pixels[index];
        double_pixels[index] = pixels[index];
    }
    params.max_peaks = 0;
    for (iteration = 0; iteration < 3; ++iteration) {
        laue_frame_result result = {0};
        const void *frame = iteration == 0 ? (const void *)pixels
                          : iteration == 1 ? (const void *)int32_pixels
                          : (const void *)double_pixels;
        int type = iteration == 0 ? LAUE_PIXEL_U16
                 : iteration == 1 ? LAUE_PIXEL_I32 : LAUE_PIXEL_F64;
        int status = laue_find_peaks_typed(frame, type, IMAGE_WIDTH, IMAGE_HEIGHT,
                                           &params, &result);

        if (status != LAUE_OK) {
            fprintf(stderr, "typed peak search %d failed: %s\n", iteration, result.message);
            laue_frame_result_free(&result);
            return status;
        }
        if (result.n_peaks != 3) {
            fprintf(stderr, "typed peak search %d returned %d peaks, expected 3\n",
                    iteration, result.n_peaks);
            laue_frame_result_free(&result);
            return 1;
        }
        laue_frame_result_free(&result);
    }

    return 0;
}
