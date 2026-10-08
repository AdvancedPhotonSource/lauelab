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

/* The 24 q vectors in tests/data/synthetic/baseline/p2q/p2q_synthetic_ni_grain_a.txt.
   The euler CLI indexes all 24 as one Ni grain. */
static const double GRAIN_A_QHAT[][3] = {
    {-0.1342000, 0.5768306, -0.8057647},
    {0.2233914, 0.7493781, -0.6233207},
    {0.0909368, 0.7003385, -0.7079947},
    {-0.2850511, 0.7233420, -0.6289056},
    {0.0285809, 0.6711168, -0.7408005},
    {-0.2449899, 0.6878527, -0.6832559},
    {-0.0746038, 0.8048531, -0.5887663},
    {0.1980575, 0.6108218, -0.7665963},
    {-0.0070593, 0.6527004, -0.7575833},
    {0.1430009, 0.7217765, -0.6771923},
    {-0.2214997, 0.6658232, -0.7124727},
    {-0.0897702, 0.7571957, -0.6469900},
    {0.3043418, 0.7699775, -0.5608126},
    {0.0699723, 0.8360835, -0.5441216},
    {-0.0299962, 0.6401918, -0.7676292},
    {-0.2061572, 0.6509665, -0.7305763},
    {0.1416239, 0.6098923, -0.7797269},
    {-0.0989078, 0.7253992, -0.6811852},
    {0.2645284, 0.6535216, -0.7091786},
    {0.0282047, 0.7941544, -0.6070612},
    {-0.0459690, 0.6311766, -0.7742757},
    {-0.1765903, 0.8066854, -0.5639809},
    {0.0548782, 0.6839099, -0.7274996},
    {0.1010929, 0.6079771, -0.7874923},
};

enum { GRAIN_A_COUNT = sizeof(GRAIN_A_QHAT) / sizeof(GRAIN_A_QHAT[0]) };

/* Index a peak array allocated by the caller, as index_orientations does. The
   caller sets result.peaks to NULL before laue_frame_result_free, so that the
   call frees only the patterns. */
static int index_caller_owned_peaks(const laue_crystal *crystal, int zero_peak)
{
    static laue_peak peaks[GRAIN_A_COUNT];
    laue_index_params params = {0};
    laue_frame_result result = {0};
    int index;
    int axis;
    int status;

    params.kev_max_calc = 17.2;
    params.kev_max_test = 35.0;
    params.angle_tolerance_deg = 0.1;
    params.cone_deg = 72.0;
    params.hkl_prefer[2] = 1;
    params.max_data = 250;
    for (index = 0; index < GRAIN_A_COUNT; ++index) {
        for (axis = 0; axis < 3; ++axis) peaks[index].qhat[axis] = GRAIN_A_QHAT[index][axis];
    }
    if (zero_peak >= 0) {
        for (axis = 0; axis < 3; ++axis) peaks[zero_peak].qhat[axis] = 0.0;
    }
    result.n_peaks = GRAIN_A_COUNT;
    result.peaks = peaks;
    status = laue_index(crystal, &params, &result);
    result.peaks = NULL;
    result.n_peaks = 0;
    if (zero_peak >= 0) {
        laue_frame_result_free(&result);
        if (status != LAUE_INVALID_ARGUMENT) {
            fprintf(stderr, "indexing with a zero q vector returned %d\n", status);
            return 1;
        }
        return 0;
    }
    if (status != LAUE_OK || result.n_patterns != 1 || result.n_indexed != GRAIN_A_COUNT) {
        fprintf(stderr, "indexing returned status %d, %d patterns, %d indexed: %s\n",
                status, result.n_patterns, result.n_indexed, result.message);
        laue_frame_result_free(&result);
        return 1;
    }
    laue_frame_result_free(&result);
    return 0;
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
    params.min_size = 2.5;
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

    {
        laue_atom nickel = {"Ni", 0.0, 0.0, 0.0, 1.0};
        char error[256];
        laue_crystal *crystal = laue_crystal_create(
            "Ni", 225, 3.5238, 3.5238, 3.5238, 90.0, 90.0, 90.0, &nickel, 1, error, sizeof(error)
        );

        if (!crystal) {
            fprintf(stderr, "crystal creation failed: %s\n", error);
            return 1;
        }
        for (iteration = 0; iteration < 3; ++iteration) {
            if (index_caller_owned_peaks(crystal, -1) || index_caller_owned_peaks(crystal, 7)) {
                laue_crystal_free(crystal);
                return 1;
            }
        }
        laue_crystal_free(crystal);
    }

    return 0;
}
