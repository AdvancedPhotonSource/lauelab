/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "liblaue.h"

enum { ROWS = 16, COLS = 32, IMAGES = 6, REPEATS = 5 };

#define REQUIRE(condition) do { \
    if (!(condition)) { fprintf(stderr, "check failed at line %d\n", __LINE__); return 1; } \
} while (0)

int main(int argc, char **argv)
{
    laue_geometry *geometry;
    laue_recon *recon;
    laue_recon_params params = {0};
    laue_wire_info wire_info;
    unsigned short images[IMAGES * ROWS * COLS];
    unsigned char mask[ROWS * COLS];
    double wire_positions[(IMAGES + 1) * 3];
    double *output;
    void *stored, *depth_sums, *pixel_sums;
    double elapsed;
    char error[256];
    int n_depths;
    int index;

    REQUIRE(argc == 2);
    REQUIRE(laue_geometry_from_file(NULL, error, sizeof(error)) == NULL);
    geometry = laue_geometry_from_file(argv[1], error, sizeof(error));
    REQUIRE(geometry != NULL);
    REQUIRE(laue_geometry_wire_info(NULL, &wire_info, error, sizeof(error)) != LAUE_OK);
    REQUIRE(laue_geometry_wire_info(geometry, NULL, error, sizeof(error)) != LAUE_OK);
    REQUIRE(laue_geometry_wire_info(geometry, &wire_info, error, sizeof(error)) == LAUE_OK);
    REQUIRE(wire_info.has_wire);

    params.depth_start_um = -2;
    params.depth_end_um = 2;
    params.resolution_um = 1;
    params.wire_edge = LAUE_RECON_EDGE_LEADING;
    params.nx_full = 2048;
    params.ny_full = 2048;
    params.bin_i = 64;
    params.bin_j = 128;
    params.n_rows_total = ROWS;
    params.n_cols = COLS;
    REQUIRE(laue_recon_create(NULL, 0, &params, error, sizeof(error)) == NULL);
    REQUIRE(laue_recon_create(geometry, 0, NULL, error, sizeof(error)) == NULL);
    params.resolution_um = 0;
    REQUIRE(laue_recon_create(geometry, 0, &params, error, sizeof(error)) == NULL);
    params.resolution_um = 1;
    params.nx_full /= 2;
    REQUIRE(laue_recon_create(geometry, 0, &params, error, sizeof(error)) == NULL);
    REQUIRE(strcmp(error, "image dimensions do not match the detector") == 0);
    params.nx_full *= 2;
    params.start_i = params.nx_full;
    REQUIRE(laue_recon_create(geometry, 0, &params, error, sizeof(error)) == NULL);
    REQUIRE(strcmp(error, "image ROI is outside the detector") == 0);
    params.start_i = 0;
    params.start_j = params.ny_full;
    REQUIRE(laue_recon_create(geometry, 0, &params, error, sizeof(error)) == NULL);
    REQUIRE(strcmp(error, "image ROI is outside the detector") == 0);
    params.start_j = 0;
    recon = laue_recon_create(geometry, 0, &params, error, sizeof(error));
    REQUIRE(recon != NULL);
    REQUIRE(laue_recon_n_depths(NULL) == 0);
    REQUIRE(laue_recon_n_depths(recon) == 5);
    REQUIRE(laue_recon_depth_um(NULL, 0) != laue_recon_depth_um(NULL, 0));
    REQUIRE(laue_recon_depth_um(recon, -1) != laue_recon_depth_um(recon, -1));
    REQUIRE(strcmp(laue_recon_last_error(NULL), "reconstruction context is NULL") == 0);

    for (index = 0; index < IMAGES * ROWS * COLS; ++index) images[index] = (unsigned short)(index % 1000);
    memset(mask, 1, sizeof(mask));
    for (index = 0; index <= IMAGES; ++index) {
        wire_positions[3*index] = 0;
        wire_positions[3*index+1] = 1000;
        wire_positions[3*index+2] = -6 + 2*index;
    }
    REQUIRE(laue_recon_set_wire_positions(NULL, wire_positions, IMAGES + 1, LAUE_POSITIONER_NONE) != LAUE_OK);
    REQUIRE(laue_recon_set_wire_positions(recon, NULL, IMAGES + 1, LAUE_POSITIONER_NONE) != LAUE_OK);
    REQUIRE(laue_recon_set_wire_positions(recon, wire_positions, 1, LAUE_POSITIONER_NONE) != LAUE_OK);
    REQUIRE(laue_recon_set_wire_positions(recon, wire_positions, IMAGES + 1, 99) != LAUE_OK);
    wire_positions[1] = wire_positions[2] = -1e-20;
    REQUIRE(laue_recon_set_wire_positions(recon, wire_positions, IMAGES + 1, LAUE_POSITIONER_PM500) == LAUE_OK);
    wire_positions[1] = 1000;
    wire_positions[2] = -6;
    REQUIRE(laue_recon_set_wire_positions(recon, wire_positions, IMAGES + 1, LAUE_POSITIONER_NONE) == LAUE_OK);

    n_depths = laue_recon_n_depths(recon);
    output = calloc((size_t)n_depths * ROWS * COLS, sizeof(*output));
    REQUIRE(output != NULL);
    REQUIRE(laue_recon_stripe(NULL, images, LAUE_PIXEL_U16, IMAGES, 0, ROWS,
                              NULL, NULL, mask, output, 1, &elapsed) != LAUE_OK);
    REQUIRE(laue_recon_stripe(recon, NULL, LAUE_PIXEL_U16, IMAGES, 0, ROWS,
                              NULL, NULL, mask, output, 1, &elapsed) != LAUE_OK);
    REQUIRE(laue_recon_stripe(recon, images, 99, IMAGES, 0, ROWS,
                              NULL, NULL, mask, output, 1, &elapsed) != LAUE_OK);
    REQUIRE(laue_recon_stripe(recon, images, LAUE_PIXEL_U16, IMAGES - 1, 0, ROWS,
                              NULL, NULL, mask, output, 1, &elapsed) != LAUE_OK);
    REQUIRE(laue_recon_stripe(recon, images, LAUE_PIXEL_U16, IMAGES, ROWS, 1,
                              NULL, NULL, mask, output, 1, &elapsed) != LAUE_OK);
    REQUIRE(laue_recon_stripe(recon, images, LAUE_PIXEL_U16, IMAGES, 0, ROWS,
                              NULL, NULL, NULL, output, 1, &elapsed) != LAUE_OK);
    REQUIRE(laue_recon_stripe(recon, images, LAUE_PIXEL_U16, IMAGES, 0, ROWS,
                              NULL, NULL, mask, NULL, 1, &elapsed) != LAUE_OK);
    REQUIRE(laue_recon_stripe(recon, images, LAUE_PIXEL_U16, IMAGES, 0, ROWS,
                              NULL, NULL, mask, output, 1, NULL) != LAUE_OK);
    REQUIRE(laue_recon_stripe(recon, images, LAUE_PIXEL_U16, IMAGES, 0, ROWS,
                              NULL, NULL, mask, output, 0, &elapsed) == LAUE_INVALID_ARGUMENT);
    REQUIRE(strcmp(laue_recon_last_error(recon), "n_threads must be at least 1") == 0);
    for (index = 0; index < REPEATS; ++index) {
        memset(output, 0, (size_t)n_depths * ROWS * COLS * sizeof(*output));
        REQUIRE(laue_recon_stripe(recon, images, LAUE_PIXEL_U16, IMAGES, 0, ROWS,
                                  NULL, NULL, mask, output, 2, &elapsed) == LAUE_OK);
    }

    stored = malloc((size_t)n_depths * ROWS * COLS * sizeof(double));
    depth_sums = malloc((size_t)n_depths * sizeof(double));
    pixel_sums = malloc(ROWS * COLS * sizeof(double));
    REQUIRE(stored != NULL && depth_sums != NULL && pixel_sums != NULL);
    output[0] = 1e300;
    output[1] = -1e300;
    output[2] = 0.0 / 0.0;
    REQUIRE(laue_recon_store_stripe(NULL, (size_t)n_depths, ROWS * COLS, 1.0, LAUE_PIXEL_U16,
                                    stored, depth_sums, pixel_sums, 2) == LAUE_INVALID_ARGUMENT);
    REQUIRE(laue_recon_store_stripe(output, (size_t)n_depths, ROWS * COLS, 1.0, LAUE_PIXEL_U16,
                                    NULL, depth_sums, pixel_sums, 2) == LAUE_INVALID_ARGUMENT);
    REQUIRE(laue_recon_store_stripe(output, 0, ROWS * COLS, 1.0, LAUE_PIXEL_U16,
                                    stored, depth_sums, pixel_sums, 2) == LAUE_INVALID_ARGUMENT);
    REQUIRE(laue_recon_store_stripe(output, (size_t)n_depths, ROWS * COLS, 1.0, 99,
                                    stored, depth_sums, pixel_sums, 2) == LAUE_INVALID_ARGUMENT);
    REQUIRE(laue_recon_store_stripe(output, (size_t)n_depths, ROWS * COLS, 1.0, LAUE_PIXEL_U16,
                                    stored, depth_sums, pixel_sums, 0) == LAUE_INVALID_ARGUMENT);
    for (index = 0; index < 7; ++index) {
        static const int types[] = {LAUE_PIXEL_U16, LAUE_PIXEL_F64, LAUE_PIXEL_I32, LAUE_PIXEL_F32,
                                    LAUE_PIXEL_I16, LAUE_PIXEL_U8, LAUE_PIXEL_I8};
        REQUIRE(laue_recon_store_stripe(output, (size_t)n_depths, ROWS * COLS, 255.0, types[index],
                                        stored, depth_sums, pixel_sums, 2) == LAUE_OK);
        REQUIRE(laue_recon_store_stripe(output, (size_t)n_depths, ROWS * COLS, 1.0, types[index],
                                        stored, NULL, NULL, 1) == LAUE_OK);
    }
    free(stored);
    free(depth_sums);
    free(pixel_sums);

    free(output);
    laue_recon_free(recon);
    laue_recon_free(NULL);
    laue_geometry_free(geometry);
    return 0;
}
