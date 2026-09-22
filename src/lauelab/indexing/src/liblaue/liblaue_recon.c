/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#include "liblaue_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include "../../../reconstruct/source/recon_cpu/source/cosmicFilter.h"

#ifndef MAX
#define MAX(X,Y) (((X)<(Y)) ? (Y) : (X))
#endif
#ifndef MIN
#define MIN(X,Y) (((X)>(Y)) ? (Y) : (X))
#endif

typedef struct {
    double x, y, z;
} point_xyz;

struct laue_recon {
    laue_recon_params params;
    double pixel_size_i, pixel_size_j;
    double detector_pixels_i, detector_pixels_j;
    double detector_rotation[3][3];
    point_xyz detector_translation;
    double wire_diameter;
    point_xyz wire_origin;
    double wire_rotation[3][3];
    double wire_rho[3][3];
    point_xyz wire_ki;
    point_xyz *wire_positions;
    size_t n_wire_positions;
    double *scratch;
    size_t scratch_count;
    int n_depths;
    char error[256];
};

static void set_recon_error(laue_recon *recon, const char *message)
{
    if (recon) snprintf(recon->error, sizeof(recon->error), "%s", message);
}

static point_xyz multiply31(const double matrix[3][3], point_xyz value)
{
    point_xyz result;
    double x = value.x, y = value.y, z = value.z;
    result.x = matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]*z;
    result.y = matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]*z;
    result.z = matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]*z;
    return result;
}

static int make_wire_rho(const struct wireGeometry *wire, double rho[3][3], point_xyz *ki)
{
    point_xyz vector = {0, wire->axisR[2], -wire->axisR[1]};
    double theta = sqrt(vector.x*vector.x + vector.y*vector.y + vector.z*vector.z);
    double length, angle;
    double nx, ny, nz, sine, cosine, one_minus_cosine;

    if (theta == 0.0) {
        memset(rho, 0, 9 * sizeof(double));
        rho[0][0] = rho[1][1] = rho[2][2] = 1.0;
    } else {
        double scale = asin(theta) / theta;
        vector.x *= scale;
        vector.y *= scale;
        vector.z *= scale;
        length = sqrt(vector.x*vector.x + vector.y*vector.y + vector.z*vector.z);
        angle = length;
        nx = vector.x / length;
        ny = vector.y / length;
        nz = vector.z / length;
        sine = sin(angle);
        cosine = cos(angle);
        one_minus_cosine = 1.0 - cosine;
        rho[0][0] = cosine+nx*nx*one_minus_cosine;
        rho[0][1] = nx*ny*one_minus_cosine-nz*sine;
        rho[0][2] = nx*nz*one_minus_cosine+ny*sine;
        rho[1][0] = nx*ny*one_minus_cosine+nz*sine;
        rho[1][1] = cosine+ny*ny*one_minus_cosine;
        rho[1][2] = ny*nz*one_minus_cosine-nx*sine;
        rho[2][0] = nx*nz*one_minus_cosine-ny*sine;
        rho[2][1] = ny*nz*one_minus_cosine+nx*sine;
        rho[2][2] = cosine+nz*nz*one_minus_cosine;
    }
    *ki = multiply31(rho, (point_xyz){0, 0, 1});
    return 0;
}

static double correct_pm500_y(double value)
{
    static const double correction[] = {
        0.190672,0.122487,0.0709713,0.0194531,-0.0620639,-0.180244,-0.26843,
        -0.353281,-0.491464,-0.519646,-0.467831,-0.242511,-0.0873608,0.117789,
        0.292936,0.413085,0.433239,0.388385,0.353539,0.298687,0.218837
    };
    double relative = fmod(value, 20.0);
    long index;
    if (relative < 0) relative += 20.0;
    index = MIN((long)relative, 19);
    return value + (fmod(relative, 1.0) * (correction[index+1] - correction[index]) + correction[index]);
}

static double correct_pm500_z(double value)
{
    static const double correction[] = {
        0.989107,1.00074,0.823196,0.532596,0.160087,-0.205875,-0.556123,-0.794466,
        -0.938048,-0.903535,-0.771879,-0.562128,-0.363329,-0.19691,-0.0523967,
        0.0983067,0.240439,0.405843,0.562625,0.701474,0.786522,0.772175,0.632715,
        0.376184,0.0351449,-0.313989,-0.629154,-0.850827,-0.961081,-0.93489,
        -0.764372,-0.544069,-0.331984,-0.118546,0.0546623,0.223944,0.393703,
        0.549969,0.710416,0.890823,0.989107
    };
    double relative = fmod(value, 40.0);
    long index;
    if (relative < 0) relative += 40.0;
    index = MIN((long)relative, 39);
    return value + (fmod(relative, 1.0) * (correction[index+1] - correction[index]) + correction[index]);
}

static point_xyz wire_to_beamline(const laue_recon *recon, const double *raw, int positioner)
{
    point_xyz wire;
    double x = raw[0], y = raw[1], z = raw[2];
    if (positioner == LAUE_POSITIONER_PM500) {
        y = correct_pm500_y(y);
        z = correct_pm500_z(z);
    }
    x -= recon->wire_origin.x;
    y -= recon->wire_origin.y;
    z -= recon->wire_origin.z;
    wire.x = recon->wire_rotation[0][0]*x + recon->wire_rotation[0][1]*y + recon->wire_rotation[0][2]*z;
    wire.y = recon->wire_rotation[1][0]*x + recon->wire_rotation[1][1]*y + recon->wire_rotation[1][2]*z;
    wire.z = recon->wire_rotation[2][0]*x + recon->wire_rotation[2][1]*y + recon->wire_rotation[2][2]*z;
    return multiply31(recon->wire_rho, wire);
}

static point_xyz pixel_to_point(const laue_recon *recon, double row, double column)
{
    double corrected_i = column * recon->params.bin_i + recon->params.start_i;
    double corrected_j = row * recon->params.bin_j + recon->params.start_j;
    double x, y, z;
    point_xyz result;
    corrected_i += (recon->params.bin_i-1)/2.;
    corrected_j += (recon->params.bin_j-1)/2.;
    x = (corrected_i - 0.5*(recon->detector_pixels_i - 1)) * recon->pixel_size_i;
    y = (corrected_j - 0.5*(recon->detector_pixels_j - 1)) * recon->pixel_size_j;
    x += recon->detector_translation.x;
    y += recon->detector_translation.y;
    z = recon->detector_translation.z;
    result.x = recon->detector_rotation[0][0]*x + recon->detector_rotation[0][1]*y + recon->detector_rotation[0][2]*z;
    result.y = recon->detector_rotation[1][0]*x + recon->detector_rotation[1][1]*y + recon->detector_rotation[1][2]*z;
    result.z = recon->detector_rotation[2][0]*x + recon->detector_rotation[2][1]*y + recon->detector_rotation[2][2]*z;
    return result;
}

static double pixel_depth(const laue_recon *recon, point_xyz pixel, point_xyz wire, int leading)
{
    point_xyz intersection;
    double dy = wire.y - pixel.y;
    double dz = wire.z - pixel.z;
    double length_squared = dy*dy + dz*dz;
    double radius = recon->wire_diameter / 2;
    double tan_phi0 = dz / dy;
    double tan_dphi = radius / sqrt(length_squared - radius*radius);
    double numerator = leading ? (tan_phi0-tan_dphi) : (tan_phi0+tan_dphi);
    double denominator = leading ? (1+tan_phi0*tan_dphi) : (1-tan_phi0*tan_dphi);
    double tan_phi = numerator / denominator;
    double intercept = pixel.z - pixel.y * tan_phi;
    intersection.z = intercept / (1-tan_phi * recon->wire_ki.y / recon->wire_ki.z);
    intersection.y = recon->wire_ki.y / recon->wire_ki.z * intersection.z;
    intersection.x = recon->wire_ki.x / recon->wire_ki.z * intersection.z;
    return recon->wire_ki.x*intersection.x + recon->wire_ki.y*intersection.y
           + recon->wire_ki.z*intersection.z;
}

static double trapezoid_height(double partial_start, double partial_end,
                               double full_start, double full_end, double depth)
{
    if (depth <= partial_start || depth >= partial_end) return 0;
    if (depth < full_start) return (depth - partial_start) / (full_start - partial_start);
    if (depth > full_end) return (partial_end - depth) / (partial_end - full_end);
    return 1;
}

static void deposit(const laue_recon *recon, double value, double partial_end,
                    double partial_start, double full_start, double full_end, double *pixel_depths)
{
    double resolution = recon->params.resolution_um;
    double max_depth = resolution*(recon->n_depths-1) + recon->params.depth_start_um;
    double area;
    long maximum = recon->n_depths - 1;
    long start, end, depth_index;

    if (value==0 || partial_end < recon->params.depth_start_um || partial_start > max_depth) return;
    if (full_end < full_start) {
        double swap = full_end;
        full_end = full_start;
        full_start = swap;
    }
    area = (full_end + partial_end - full_end - partial_start) / 2;
    if (area < 0 || isnan(area)) return;
    start = (long)floor((partial_start - recon->params.depth_start_um) / resolution);
    start = MAX(0,start);
    start = MIN(maximum,start);
    end = (long)ceil((partial_end - recon->params.depth_start_um) / resolution);
    end = MAX(start,end);
    end = MIN(maximum,end);

    for (depth_index = start; depth_index <= end; ++depth_index) {
        double area_in_range = 0;
        double bin_start = depth_index * resolution + recon->params.depth_start_um - (resolution*0.5);
        double bin_end = bin_start + resolution;
        double depth_1, depth_2, height_1, height_2;
        if (full_start > bin_start && partial_start < bin_end) {
            depth_1 = MAX(bin_start,partial_start);
            depth_2 = MIN(bin_end,full_start);
            height_1 = trapezoid_height(partial_start, partial_end, full_start, full_end, depth_1);
            height_2 = trapezoid_height(partial_start, partial_end, full_start, full_end, depth_2);
            area_in_range += ((height_1 + height_2) / 2 * (depth_2 - depth_1));
        }
        if (full_end > bin_start && full_start < bin_end) {
            depth_1 = MAX(bin_start,full_start);
            depth_2 = MIN(bin_end,full_end);
            area_in_range += (depth_2 - depth_1);
        }
        if (partial_end > bin_start && full_end < bin_end) {
            depth_1 = MAX(bin_start,full_end);
            depth_2 = MIN(bin_end,partial_end);
            height_1 = trapezoid_height(partial_start, partial_end, full_start, full_end, depth_1);
            height_2 = trapezoid_height(partial_start, partial_end, full_start, full_end, depth_2);
            area_in_range += ((height_1 + height_2) / 2 * (depth_2 - depth_1));
        }
        if (area_in_range>0) pixel_depths[depth_index] += value * (area_in_range / area);
    }
}

laue_recon *laue_recon_create(const laue_geometry *geometry, int detector_index,
                              const laue_recon_params *params, char *err, size_t errlen)
{
    laue_recon *recon;
    const struct detectorGeometry *detector;
    const struct wireGeometry *wire;
    double start, end;

    if (!geometry || !params || !geometry->has_wire || detector_index < 0
        || detector_index >= MAX_Ndetectors || !geometry->value.d[detector_index].used
        || !isfinite(params->depth_start_um) || !isfinite(params->depth_end_um)
        || !(params->resolution_um > 0) || params->depth_end_um < params->depth_start_um
        || (params->wire_edge != LAUE_RECON_EDGE_LEADING
            && params->wire_edge != LAUE_RECON_EDGE_TRAILING
            && params->wire_edge != LAUE_RECON_EDGE_BOTH)
        || (params->cosmic_filter != 0 && params->cosmic_filter != 1)
        || params->nx_full < 1 || params->ny_full < 1 || params->start_i < 0
        || params->start_j < 0 || params->bin_i < 1 || params->bin_j < 1
        || params->n_rows_total < 1 || params->n_cols < 1) {
        if (err && errlen) snprintf(err, errlen, "%s", "invalid reconstruction parameters");
        return NULL;
    }
    detector = &geometry->value.d[detector_index];
    if (params->nx_full != detector->Nx || params->ny_full != detector->Ny) {
        if (err && errlen) snprintf(err, errlen, "%s", "image dimensions do not match the detector");
        return NULL;
    }
    if (params->start_i >= detector->Nx || params->start_j >= detector->Ny
        || (size_t)params->n_rows_total > ((size_t)params->ny_full - params->start_j) / params->bin_j
        || (size_t)params->n_cols > ((size_t)params->nx_full - params->start_i) / params->bin_i) {
        if (err && errlen) snprintf(err, errlen, "%s", "image ROI is outside the detector");
        return NULL;
    }
    recon = calloc(1, sizeof(*recon));
    if (!recon) {
        if (err && errlen) snprintf(err, errlen, "%s", "unable to allocate reconstruction context");
        return NULL;
    }
    recon->params = *params;
    start = round(params->depth_start_um / params->resolution_um) * params->resolution_um;
    end = round(params->depth_end_um / params->resolution_um) * params->resolution_um;
    recon->params.depth_start_um = start;
    recon->params.depth_end_um = end;
    recon->n_depths = (int)round((end - start) / params->resolution_um + 1.0);
    if (recon->n_depths < 1) {
        free(recon);
        if (err && errlen) snprintf(err, errlen, "%s", "reconstruction has no output depths");
        return NULL;
    }
    recon->detector_pixels_i = detector->Nx;
    recon->detector_pixels_j = detector->Ny;
    recon->pixel_size_i = detector->sizeX / detector->Nx;
    recon->pixel_size_j = detector->sizeY / detector->Ny;
    recon->detector_translation = (point_xyz){detector->P[0], detector->P[1], detector->P[2]};
    recon->detector_rotation[0][0] = detector->rho00;
    recon->detector_rotation[0][1] = detector->rho01;
    recon->detector_rotation[0][2] = detector->rho02;
    recon->detector_rotation[1][0] = detector->rho10;
    recon->detector_rotation[1][1] = detector->rho11;
    recon->detector_rotation[1][2] = detector->rho12;
    recon->detector_rotation[2][0] = detector->rho20;
    recon->detector_rotation[2][1] = detector->rho21;
    recon->detector_rotation[2][2] = detector->rho22;
    wire = &geometry->value.wire;
    recon->wire_diameter = wire->dia;
    recon->wire_origin = (point_xyz){wire->origin[0], wire->origin[1], wire->origin[2]};
    recon->wire_rotation[0][0] = wire->R00;
    recon->wire_rotation[0][1] = wire->R01;
    recon->wire_rotation[0][2] = wire->R02;
    recon->wire_rotation[1][0] = wire->R10;
    recon->wire_rotation[1][1] = wire->R11;
    recon->wire_rotation[1][2] = wire->R12;
    recon->wire_rotation[2][0] = wire->R20;
    recon->wire_rotation[2][1] = wire->R21;
    recon->wire_rotation[2][2] = wire->R22;
    make_wire_rho(wire, recon->wire_rho, &recon->wire_ki);
    if (err && errlen) err[0] = '\0';
    return recon;
}

int laue_recon_set_wire_positions(laue_recon *recon, const double *xyz_raw,
                                  size_t n, int positioner)
{
    point_xyz *positions;
    size_t index;
    if (!recon || !xyz_raw || n < 2 || (positioner != LAUE_POSITIONER_NONE
        && positioner != LAUE_POSITIONER_PM500 && positioner != LAUE_POSITIONER_ALIO)) {
        set_recon_error(recon, "invalid wire positions");
        return LAUE_INVALID_ARGUMENT;
    }
    if (n > SIZE_MAX / sizeof(*positions)) {
        set_recon_error(recon, "wire position count is too large");
        return LAUE_INVALID_ARGUMENT;
    }
    positions = malloc(n * sizeof(*positions));
    if (!positions) {
        set_recon_error(recon, "unable to allocate wire positions");
        return LAUE_OUT_OF_MEMORY;
    }
    for (index = 0; index < n; ++index) {
        if (!isfinite(xyz_raw[3*index]) || !isfinite(xyz_raw[3*index+1]) || !isfinite(xyz_raw[3*index+2])) {
            free(positions);
            set_recon_error(recon, "wire positions must be finite");
            return LAUE_INVALID_ARGUMENT;
        }
        positions[index] = wire_to_beamline(recon, xyz_raw + 3*index, positioner);
    }
    free(recon->wire_positions);
    recon->wire_positions = positions;
    recon->n_wire_positions = n;
    recon->error[0] = '\0';
    return LAUE_OK;
}

int laue_recon_stripe(laue_recon *recon, const void *images, int pixel_type,
                      size_t n_images, size_t row0, size_t nrows,
                      const double *scale, const double *norm_plane,
                      const unsigned char *mask, double *out,
                      int n_threads, double *seconds_elapsed)
{
    size_t image_pixels, stripe_pixels, scratch_per_thread, scratch_count, row;
    double started;

    if (n_threads < 1) {
        set_recon_error(recon, "n_threads must be at least 1");
        return LAUE_INVALID_ARGUMENT;
    }
    if (!recon || !images || !mask || !out || !seconds_elapsed
        || (pixel_type != LAUE_PIXEL_U16 && pixel_type != LAUE_PIXEL_F64)
        || n_images < 3 || recon->n_wire_positions != n_images + 1
        || nrows < 1 || row0 > (size_t)recon->params.n_rows_total
        || nrows > (size_t)recon->params.n_rows_total - row0
        || n_images > SIZE_MAX / (size_t)recon->params.n_rows_total
        || n_images * (size_t)recon->params.n_rows_total > SIZE_MAX / (size_t)recon->params.n_cols
        || nrows > SIZE_MAX / (size_t)recon->params.n_cols
        || (size_t)recon->n_depths > SIZE_MAX / (nrows * (size_t)recon->params.n_cols)) {
        set_recon_error(recon, "invalid reconstruction stripe");
        return LAUE_INVALID_ARGUMENT;
    }
    image_pixels = nrows * (size_t)recon->params.n_cols;
    stripe_pixels = image_pixels;
    started = omp_get_wtime();
    scratch_per_thread = n_images + 4 * (n_images + 1) + (size_t)recon->n_depths;
    if ((size_t)n_threads > SIZE_MAX / scratch_per_thread) {
        set_recon_error(recon, "reconstruction scratch size is too large");
        return LAUE_INVALID_ARGUMENT;
    }
    scratch_count = (size_t)n_threads * scratch_per_thread;
    if (recon->scratch_count < scratch_count) {
        double *scratch = realloc(recon->scratch, scratch_count * sizeof(*scratch));
        if (!scratch) {
            set_recon_error(recon, "unable to allocate reconstruction scratch storage");
            return LAUE_OUT_OF_MEMORY;
        }
        recon->scratch = scratch;
        recon->scratch_count = scratch_count;
    }

#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (row = 0; row < nrows; ++row) {
        double *trace = recon->scratch + (size_t)omp_get_thread_num() * scratch_per_thread;
        double *depth_back = trace + n_images;
        double *depth_front = depth_back + 2 * (n_images + 1);
        double *pixel_depths = depth_front + 2 * (n_images + 1);
        size_t column;
        for (column = 0; column < (size_t)recon->params.n_cols; ++column) {
            size_t local_pixel = row * (size_t)recon->params.n_cols + column;
            size_t source_pixel = local_pixel;
            point_xyz back, front;
            size_t image_index, depth_index;
            if (!mask[local_pixel]) continue;
            memset(pixel_depths, 0, (size_t)recon->n_depths * sizeof(*pixel_depths));
            back = multiply31(recon->wire_rho, pixel_to_point(recon, row0 + row, (double)column - 0.5));
            front = multiply31(recon->wire_rho, pixel_to_point(recon, row0 + row, (double)column + 0.5));
            for (image_index = 0; image_index <= n_images; ++image_index) {
                int edge = recon->params.wire_edge < 0 ? 1 : recon->params.wire_edge;
                depth_back[image_index] = pixel_depth(recon, back, recon->wire_positions[image_index], edge);
                depth_front[image_index] = pixel_depth(recon, front, recon->wire_positions[image_index], edge);
                if (recon->params.wire_edge < 0) {
                    depth_back[n_images+1+image_index] = pixel_depth(recon, back, recon->wire_positions[image_index], 0);
                    depth_front[n_images+1+image_index] = pixel_depth(recon, front, recon->wire_positions[image_index], 0);
                }
            }
            for (image_index = 0; image_index < n_images; ++image_index) {
                double raw = pixel_type == LAUE_PIXEL_U16
                    ? ((const uint16_t *)images)[image_index * image_pixels + source_pixel]
                    : ((const double *)images)[image_index * image_pixels + source_pixel];
                trace[image_index] = raw;
                if (scale) trace[image_index] *= scale[image_index];
                if (norm_plane) trace[image_index] *= norm_plane[local_pixel];
            }
            if (recon->params.cosmic_filter) cosmic_filter(trace, n_images);
            for (image_index = 0; image_index < n_images - 2; ++image_index) {
                double difference = trace[image_index] - trace[image_index+1];
                if (difference == 0) continue;
                if (recon->params.wire_edge < 0) {
                    deposit(recon, difference, depth_back[image_index+1], depth_front[image_index],
                            depth_back[image_index], depth_front[image_index+1], pixel_depths);
                    deposit(recon, -difference, depth_back[n_images+1+image_index+1],
                            depth_front[n_images+1+image_index], depth_back[n_images+1+image_index],
                            depth_front[n_images+1+image_index+1], pixel_depths);
                } else {
                    deposit(recon, recon->params.wire_edge ? difference : -difference,
                            depth_back[image_index+1], depth_front[image_index],
                            depth_back[image_index], depth_front[image_index+1], pixel_depths);
                }
            }
            for (depth_index = 0; depth_index < (size_t)recon->n_depths; ++depth_index) {
                if (pixel_depths[depth_index] != 0)
                    out[depth_index * stripe_pixels + local_pixel] += pixel_depths[depth_index];
            }
        }
    }
    *seconds_elapsed = omp_get_wtime() - started;
    recon->error[0] = '\0';
    return LAUE_OK;
}

/* Pixels summed into one partial before partials combine, so that a double
   total's rounding error stays near the pairwise bound at any stripe size. */
#define STORE_BLOCK 1024

#define STORE_STRIPE(T, SUM, CONVERT)                                          \
    do {                                                                       \
        T *out = stored;                                                       \
        SUM *dsum = depth_sums;                                                \
        SUM *psum = pixel_sums;                                                \
        size_t d, b;                                                           \
        _Pragma("omp parallel for schedule(static) num_threads(n_threads)")    \
        for (d = 0; d < n_depths; ++d) {                                       \
            const double *row = values + d * n_pixels;                         \
            T *orow = out + d * n_pixels;                                      \
            SUM total = 0;                                                     \
            size_t p0;                                                         \
            for (p0 = 0; p0 < n_pixels; p0 += STORE_BLOCK) {                   \
                size_t p, p1 = p0 + STORE_BLOCK < n_pixels ? p0 + STORE_BLOCK : n_pixels; \
                SUM partial = 0;                                               \
                for (p = p0; p < p1; ++p) {                                    \
                    double v = row[p] * rescale;                               \
                    T s;                                                       \
                    CONVERT;                                                   \
                    orow[p] = s;                                               \
                    partial += s;                                              \
                }                                                              \
                total += partial;                                              \
            }                                                                  \
            if (dsum) dsum[d] = total;                                         \
        }                                                                      \
        if (!psum) break;                                                      \
        _Pragma("omp parallel for schedule(static) num_threads(n_threads)")    \
        for (b = 0; b < (n_pixels + STORE_BLOCK - 1) / STORE_BLOCK; ++b) {     \
            size_t d, p, p0 = b * STORE_BLOCK;                                 \
            size_t p1 = p0 + STORE_BLOCK < n_pixels ? p0 + STORE_BLOCK : n_pixels; \
            for (p = p0; p < p1; ++p) psum[p] = 0;                             \
            for (d = 0; d < n_depths; ++d) {                                   \
                const T *orow = out + d * n_pixels;                            \
                for (p = p0; p < p1; ++p) psum[p] += orow[p];                  \
            }                                                                  \
        }                                                                      \
    } while (0)

/* Saturate at the type limits, then let the cast truncate toward zero. */
#define CONVERT_INT(T, MIN, MAX)                                               \
    if (v != v) s = 0;                                                         \
    else if (v >= (double)(MAX)) s = (MAX);                                    \
    else if (v <= (double)(MIN)) s = (MIN);                                    \
    else s = (T)v

int laue_recon_store_stripe(const double *values, size_t n_depths, size_t n_pixels,
                            double rescale, int stored_type, void *stored,
                            void *depth_sums, void *pixel_sums, int n_threads)
{
    if (!values || !stored || n_threads < 1 || n_depths < 1 || n_pixels < 1
        || n_depths > SIZE_MAX / n_pixels)
        return LAUE_INVALID_ARGUMENT;
    switch (stored_type) {
    case LAUE_PIXEL_F64: STORE_STRIPE(double, double, s = v); break;
    case LAUE_PIXEL_F32: STORE_STRIPE(float, double, s = (float)v); break;
    case LAUE_PIXEL_I32: STORE_STRIPE(int32_t, int64_t, CONVERT_INT(int32_t, INT32_MIN, INT32_MAX)); break;
    case LAUE_PIXEL_I16: STORE_STRIPE(int16_t, int64_t, CONVERT_INT(int16_t, INT16_MIN, INT16_MAX)); break;
    case LAUE_PIXEL_U16: STORE_STRIPE(uint16_t, int64_t, CONVERT_INT(uint16_t, 0, UINT16_MAX)); break;
    case LAUE_PIXEL_I8: STORE_STRIPE(int8_t, int64_t, CONVERT_INT(int8_t, INT8_MIN, INT8_MAX)); break;
    case LAUE_PIXEL_U8: STORE_STRIPE(uint8_t, int64_t, CONVERT_INT(uint8_t, 0, UINT8_MAX)); break;
    default: return LAUE_INVALID_ARGUMENT;
    }
    return LAUE_OK;
}

int laue_recon_n_depths(const laue_recon *recon)
{
    return recon ? recon->n_depths : 0;
}

double laue_recon_depth_um(const laue_recon *recon, int index)
{
    if (!recon || index < 0 || index >= recon->n_depths) return NAN;
    return index * recon->params.resolution_um + recon->params.depth_start_um;
}

const char *laue_recon_last_error(const laue_recon *recon)
{
    return recon ? recon->error : "reconstruction context is NULL";
}

void laue_recon_free(laue_recon *recon)
{
    if (!recon) return;
    free(recon->scratch);
    free(recon->wire_positions);
    free(recon);
}
