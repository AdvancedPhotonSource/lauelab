/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#include "liblaue.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_errno.h>

#include "liblaue_internal.h"
#include "peaksearch.h"
#include "Euler.h"

__thread FILE *fout;

/*
 * GSL is linked statically, and its only abort reference is in gsl_error.
 * initialize_liblaue installs gsl_set_error_handler_off before any GSL call,
 * so that path is unreachable. Keep abort out of the library's imports, but
 * terminate rather than continue if this invariant is ever broken.
 */
__attribute__((noreturn)) void __wrap_abort(void)
{
    __builtin_trap();
}

struct laue_crystal {
    struct crystalStructure value;
};

static void set_error(char *err, size_t errlen, const char *message)
{
    if (err && errlen) {
        snprintf(err, errlen, "%s", message);
    }
}

static void delete_point_value(void *value)
{
    point_delete((Point *)value);
}

static void delete_peak_value(void *value)
{
    peak_delete((Peak *)value);
}

/*
 * Euler's indexing code stores a*, b*, and c* as columns in 1/Angstrom.
 * The liblaue result API exports basis vectors as rows in 1/nm so callers can
 * calculate reciprocal vectors with q = hkl * recip.
 */
static void export_reciprocal_rows(
    double destination[3][3], const double source_columns[3][3])
{
    int basis;
    int component;

    for (basis = 0; basis < 3; ++basis) {
        for (component = 0; component < 3; ++component) {
            destination[basis][component] =
                source_columns[component][basis] * 10.0;
        }
    }
}

/* GSL aborts on errors by default, which is unsafe in an embedded library. */
#if defined(__GNUC__)
__attribute__((constructor))
#endif
static void initialize_liblaue(void)
{
    gsl_set_error_handler_off();
}

laue_geometry *laue_geometry_from_file(const char *path, char *err, size_t errlen)
{
    laue_geometry *geometry;

    if (!path) {
        set_error(err, errlen, "geometry path is NULL");
        return NULL;
    }

    geometry = calloc(1, sizeof(*geometry));
    if (!geometry) {
        set_error(err, errlen, "unable to allocate geometry");
        return NULL;
    }
    if (readGeoForLibrary((char *)path, &geometry->value, &geometry->has_wire)) {
        free(geometry);
        set_error(err, errlen, "unable to read detector geometry file");
        return NULL;
    }
    if (geometry->value.Ndetectors < 1) {
        free(geometry);
        set_error(err, errlen, "geometry contains no active detectors");
        return NULL;
    }
    {
        int i;
        for (i = 0; i < MAX_Ndetectors; ++i) {
            if (geometry->value.d[i].used && DetectorBad(&geometry->value.d[i])) {
                free(geometry);
                set_error(err, errlen, "geometry contains invalid detector parameters");
                return NULL;
            }
        }
    }
    set_error(err, errlen, "");
    return geometry;
}

void laue_geometry_free(laue_geometry *geometry)
{
    free(geometry);
}

laue_crystal *laue_crystal_create(const char *name, int space_group,
                                   double a, double b, double c,
                                   double alpha_deg, double beta_deg, double gamma_deg,
                                   const laue_atom *atoms, size_t n_atoms,
                                   char *err, size_t errlen)
{
    laue_crystal *crystal;
    size_t i;

    if (space_group < 1 || space_group > 230 || !(a > 0) || !(b > 0) || !(c > 0) ||
        !(alpha_deg > 0 && alpha_deg < 180) || !(beta_deg > 0 && beta_deg < 180) ||
        !(gamma_deg > 0 && gamma_deg < 180) || (n_atoms && !atoms)) {
        set_error(err, errlen, "invalid crystal parameters");
        return NULL;
    }
    crystal = calloc(1, sizeof(*crystal));
    if (!crystal) {
        set_error(err, errlen, "unable to allocate crystal");
        return NULL;
    }
    InitCleanCrystalStructure(&crystal->value);
    snprintf(crystal->value.desc, sizeof(crystal->value.desc), "%s", name ? name : "");
    crystal->value.a = a;
    crystal->value.b = b;
    crystal->value.c = c;
    crystal->value.alpha = alpha_deg * M_PI / 180.0;
    crystal->value.beta = beta_deg * M_PI / 180.0;
    crystal->value.gamma = gamma_deg * M_PI / 180.0;
    crystal->value.lengthUnits = 1.e10;
    crystal->value.SpaceGroup = space_group;
    if (n_atoms) {
        crystal->value.atomType = calloc(n_atoms, sizeof(*crystal->value.atomType));
        if (!crystal->value.atomType) {
            free(crystal);
            set_error(err, errlen, "unable to allocate crystal atoms");
            return NULL;
        }
        crystal->value.Ntype = n_atoms;
        for (i = 0; i < n_atoms; ++i) {
            snprintf(crystal->value.atomType[i].name,
                     sizeof(crystal->value.atomType[i].name), "%s", atoms[i].name);
            crystal->value.atomType[i].x = atoms[i].x;
            crystal->value.atomType[i].y = atoms[i].y;
            crystal->value.atomType[i].z = atoms[i].z;
            crystal->value.atomType[i].occ = atoms[i].occupancy;
            crystal->value.atomType[i].Zatom = atomicNumber((char *)atoms[i].name);
        }
    }
    if (ForceLatticeToStructure(&crystal->value)) {
        freeCrystalStructure(&crystal->value);
        free(crystal);
        set_error(err, errlen, "unable to initialize crystal");
        return NULL;
    }
    set_error(err, errlen, "");
    return crystal;
}

void laue_crystal_free(laue_crystal *crystal)
{
    if (!crystal) return;
    freeCrystalStructure(&crystal->value);
    free(crystal);
}

int laue_crystal_reciprocal(const laue_crystal *crystal, double recip[3][3])
{
    if (!crystal || !recip) return LAUE_INVALID_ARGUMENT;
    export_reciprocal_rows(recip, crystal->value.recip);
    return LAUE_OK;
}

int laue_geometry_detector_count(const laue_geometry *geometry)
{
    return geometry ? geometry->value.Ndetectors : 0;
}

int laue_geometry_find_detector(const laue_geometry *geometry, const char *detector_id)
{
    int i;

    if (!geometry || !detector_id) return -1;
    for (i = 0; i < MAX_Ndetectors; ++i) {
        if (geometry->value.d[i].used && !strcmp(geometry->value.d[i].detectorID, detector_id)) {
            return i;
        }
    }
    return -1;
}

int laue_geometry_detector_info(const laue_geometry *geometry, int detector_index,
                                laue_detector_info *info, char *err, size_t errlen)
{
    const struct detectorGeometry *detector;
    int i;

    if (!geometry || !info || detector_index < 0 || detector_index >= MAX_Ndetectors ||
        !geometry->value.d[detector_index].used) {
        set_error(err, errlen, "invalid detector index");
        return 1;
    }
    detector = &geometry->value.d[detector_index];
    if (DetectorBad((struct detectorGeometry *)detector)) {
        set_error(err, errlen, "invalid detector geometry");
        return 1;
    }
    info->nx = detector->Nx;
    info->ny = detector->Ny;
    info->size_x = detector->sizeX;
    info->size_y = detector->sizeY;
    for (i = 0; i < 3; ++i) {
        info->translation[i] = detector->P[i];
        info->rotation_vector[i] = detector->R[i];
    }
    info->rotation[0][0] = detector->rho00;
    info->rotation[0][1] = detector->rho01;
    info->rotation[0][2] = detector->rho02;
    info->rotation[1][0] = detector->rho10;
    info->rotation[1][1] = detector->rho11;
    info->rotation[1][2] = detector->rho12;
    info->rotation[2][0] = detector->rho20;
    info->rotation[2][1] = detector->rho21;
    info->rotation[2][2] = detector->rho22;
    snprintf(info->detector_id, sizeof(info->detector_id), "%s", detector->detectorID);
    set_error(err, errlen, "");
    return 0;
}

int laue_geometry_wire_info(const laue_geometry *geometry, laue_wire_info *info,
                            char *err, size_t errlen)
{
    const struct wireGeometry *wire;

    if (!geometry || !info) {
        set_error(err, errlen, "invalid wire geometry output");
        return LAUE_INVALID_ARGUMENT;
    }
    wire = &geometry->value.wire;
    memset(info, 0, sizeof(*info));
    info->dia = wire->dia;
    info->F = wire->F;
    memcpy(info->origin, wire->origin, sizeof(info->origin));
    memcpy(info->axis, wire->axis, sizeof(info->axis));
    memcpy(info->axisR, wire->axisR, sizeof(info->axisR));
    info->R[0][0] = wire->R00;
    info->R[0][1] = wire->R01;
    info->R[0][2] = wire->R02;
    info->R[1][0] = wire->R10;
    info->R[1][1] = wire->R11;
    info->R[1][2] = wire->R12;
    info->R[2][0] = wire->R20;
    info->R[2][1] = wire->R21;
    info->R[2][2] = wire->R22;
    info->Rmag = wire->Rmag;
    info->has_wire = geometry->has_wire;
    set_error(err, errlen, "");
    return LAUE_OK;
}

static int pixel_type_supported(int pixel_type)
{
    switch (pixel_type) {
    case LAUE_PIXEL_U16:
    case LAUE_PIXEL_F64:
    case LAUE_PIXEL_I32:
    case LAUE_PIXEL_F32:
    case LAUE_PIXEL_I16:
    case LAUE_PIXEL_U8:
    case LAUE_PIXEL_I8:
        return 1;
    default:
        return 0;
    }
}

/* Exact conversion of one borrowed pixel to double; every supported element
   type is representable in a double without rounding. */
static double pixel_value(const void *pixels, int pixel_type, size_t index)
{
    switch (pixel_type) {
    case LAUE_PIXEL_U16: return ((const uint16_t *)pixels)[index];
    case LAUE_PIXEL_F64: return ((const double *)pixels)[index];
    case LAUE_PIXEL_I32: return ((const int32_t *)pixels)[index];
    case LAUE_PIXEL_F32: return ((const float *)pixels)[index];
    case LAUE_PIXEL_I16: return ((const int16_t *)pixels)[index];
    case LAUE_PIXEL_U8: return ((const uint8_t *)pixels)[index];
    case LAUE_PIXEL_I8: return ((const int8_t *)pixels)[index];
    default: return NAN;
    }
}

int laue_find_peaks(const unsigned short *pixels, int nx, int ny,
                    const laue_peak_params *params, laue_frame_result *result)
{
    return laue_find_peaks_typed(pixels, LAUE_PIXEL_U16, nx, ny, params, result);
}

int laue_find_peaks_typed(const void *pixels, int pixel_type, int nx, int ny,
                          const laue_peak_params *params, laue_frame_result *result)
{
    WinViewImage image;
    Grid grid;
    Genfileinf *ginf = NULL;
    List *blobs = NULL;
    List *peaks = NULL;
    ListNode *node;
    double *values = NULL;
    double threshold;
    double average;
    double sum = 0.0;
    double sum_above = 0.0;
    size_t above = 0;
    size_t used = 0;
    double sum_squares = 0.0;
    size_t count;
    size_t i;

    if (!result) return LAUE_INVALID_ARGUMENT;
    result->status = 0;
    result->message[0] = '\0';
    result->n_peaks = 0;
    result->peaks = NULL;
    if (!pixels || !params || nx < 1 || ny < 1) {
        result->status = LAUE_INVALID_ARGUMENT;
        snprintf(result->message, sizeof(result->message), "invalid peak-search input");
        return result->status;
    }
    if (!pixel_type_supported(pixel_type)) {
        result->status = LAUE_INVALID_ARGUMENT;
        snprintf(result->message, sizeof(result->message), "unsupported pixel type %d", pixel_type);
        return result->status;
    }
    /* max_peaks == 0 means no limit; see laue_peak_params. */
    if (params->boxsize < 1 || params->min_size < 1 || params->max_peaks < 0 ||
        params->min_separation < 1 || (params->peak_shape != 0 && params->peak_shape != 1)) {
        result->status = LAUE_INVALID_ARGUMENT;
        snprintf(result->message, sizeof(result->message), "invalid or unsupported peak-search parameters");
        return result->status;
    }

    /* n_peaks is an int and each peak comes from a distinct blob, so the pixel
       count must fit an int for an unlimited search to be exactly that. */
    if ((size_t)nx > SIZE_MAX / (size_t)ny ||
        (size_t)nx * (size_t)ny > SIZE_MAX / sizeof(*values) ||
        (size_t)nx * (size_t)ny > (size_t)INT_MAX) {
        result->status = LAUE_INVALID_ARGUMENT;
        snprintf(result->message, sizeof(result->message), "image dimensions are too large");
        return result->status;
    }
    count = (size_t)nx * (size_t)ny;
    values = malloc(count * sizeof(*values));
    if (!values) goto allocation_error;
    for (i = 0; i < count; ++i) values[i] = pixel_value(pixels, pixel_type, i);

    grid.values = values;
    grid.width = nx;
    grid.height = ny;
    image.data = &grid;
    image.header = NULL;
    image.type = HDF5_FILE;

    if (params->smooth && grid_smooth_gauss(&grid, 2)) goto allocation_error;
    threshold = params->threshold;
    if (isnan(threshold)) {
        double sigma;
        for (i = 0; i < count; ++i) {
            if ((!params->mask || !params->mask[i]) && values[i] != 0.0) {
                sum += values[i];
                sum_squares += values[i] * values[i];
                ++used;
            }
        }
        if (!used) {
            result->nx = nx;
            result->ny = ny;
            result->threshold_used = NAN;
            result->total_sum = 0.0;
            result->sum_above_threshold = 0.0;
            result->num_above_threshold = 0;
            result->peak_minwidth = params->min_size / 4.0;
            result->peak_maxwidth = params->boxsize * 1.5;
            result->peak_max_cent_to_fit = params->boxsize;
            result->peak_boxsize = params->boxsize;
            goto cleanup;
        }
        average = sum / used;
        sigma = sqrt((sum_squares - 2.0 * sum * average + used * average * average) / used);
        threshold = average + (isnan(params->threshold_ratio * sigma)
                               ? 5.0 * average : params->threshold_ratio * sigma);
        if (average < 0.0) average = 0.0;
    } else {
        average = threshold - fabs(0.99 * threshold);
    }

    sum = 0.0;
    for (i = 0; i < count; ++i) {
        if (params->mask && params->mask[i]) {
            values[i] = average;
        } else {
            double raw = pixel_value(pixels, pixel_type, i);
            sum += raw;
            if (raw > threshold) {
                sum_above += raw;
                ++above;
            }
        }
    }

    ginf = default_genfileinf();
    if (!ginf) goto allocation_error;
    ginf->boxsize = params->boxsize;
    ginf->maxCentToFit = params->boxsize;
    ginf->maxwidth = params->boxsize * 1.5;
    ginf->minwidth = params->min_size / 4.0;
    ginf->maxRfactor = params->max_rfactor;
    ginf->peakShape = params->peak_shape;
    ginf->CCDFilename[0] = '\0';
    result->peak_minwidth = ginf->minwidth;
    result->peak_maxwidth = ginf->maxwidth;
    result->peak_max_cent_to_fit = ginf->maxCentToFit;
    result->peak_boxsize = ginf->boxsize;

    {
        int helper_status = 0;
        blobs = blobsearch(&grid, threshold, params->min_size, true, &helper_status);
        if (!blobs || helper_status) goto allocation_error;
        if (sorListPoints(blobs)) goto allocation_error;
        /* processBlobs treats a non-positive limit as no limit. */
        peaks = processBlobs(
            blobs, &image, ginf, params->max_peaks, grid_get_average(&grid), &helper_status
        );
        list_delete_with_values(blobs, delete_point_value);
        blobs = NULL;
        if (!peaks || helper_status == GSL_ENOMEM) goto allocation_error;
        if (helper_status != GSL_SUCCESS) {
            result->status = LAUE_NUMERICAL_ERROR;
            snprintf(result->message, sizeof(result->message), "peak fitting failed: %s", gsl_strerror(helper_status));
            goto cleanup;
        }
    }
    peaks = removeNearbyPeaks(peaks, params->min_separation);

    result->peaks = calloc((size_t)peaks->size, sizeof(*result->peaks));
    if (peaks->size && !result->peaks) goto allocation_error;
    result->n_peaks = peaks->size;
    result->nx = nx;
    result->ny = ny;
    result->threshold_used = threshold;
    result->total_sum = sum;
    result->sum_above_threshold = sum_above;
    result->num_above_threshold = (long)above;

    node = peaks->head;
    for (i = 0; i < (size_t)peaks->size; ++i, node = node->next) {
        Peak *peak = node->value;
        result->peaks[i].fit_x = peak->fitX - 1.0;
        result->peaks[i].fit_y = peak->fitY - 1.0;
        result->peaks[i].intens = peak->intens;
        result->peaks[i].integral = peak->integrIntens;
        result->peaks[i].hwhm_x = peak->fitPeakWidthX;
        result->peaks[i].hwhm_y = peak->fitPeakWidthY;
        result->peaks[i].tilt = peak->fitTilt;
        result->peaks[i].chisq = peak->chisq;
        result->peaks[i].background = peak->fitBackground;
    }
    goto cleanup;

allocation_error:
    result->status = LAUE_OUT_OF_MEMORY;
    snprintf(result->message, sizeof(result->message), "unable to allocate peak-search storage");
cleanup:
    if (blobs) list_delete_with_values(blobs, delete_point_value);
    if (peaks) list_delete_with_values(peaks, delete_peak_value);
    delete_genfileinf(ginf);
    free(values);
    return result->status;
}

static int pixel_to_q(const struct detectorGeometry *detector, double px, double py,
                      double depth, double qhat[3])
{
    double xp = (px - 0.5 * (detector->Nx - 1)) * detector->sizeX / detector->Nx;
    double yp = (py - 0.5 * (detector->Ny - 1)) * detector->sizeY / detector->Ny;
    double zp = detector->P[2];
    double x;
    double y;
    double z;
    double length;

    xp += detector->P[0];
    yp += detector->P[1];
    x = detector->rho00 * xp + detector->rho01 * yp + detector->rho02 * zp;
    y = detector->rho10 * xp + detector->rho11 * yp + detector->rho12 * zp;
    z = detector->rho20 * xp + detector->rho21 * yp + detector->rho22 * zp;
    if (!isnan(depth)) z -= depth;

    length = sqrt(x * x + y * y + z * z);
    if (!(length > 0.0)) return 1;
    x /= length;
    y /= length;
    z = z / length - 1.0;
    length = sqrt(x * x + y * y + z * z);
    if (!(length > 0.0)) return 1;

    qhat[0] = x / length;
    qhat[1] = y / length;
    qhat[2] = z / length;
    return 0;
}

static void free_grain(struct patternOfOneGrain *pattern)
{
    if (!pattern) return;
    free(pattern->hkls);
    free(pattern->Ghat);
    free(pattern->intens);
    free(pattern->pkIndex);
    free(pattern->err);
    freeCrystalStructure(&pattern->xtal);
    memset(pattern, 0, sizeof(*pattern));
}

void laue_frame_result_free(laue_frame_result *result)
{
    int i;

    if (!result) return;
    free(result->peaks);
    result->peaks = NULL;
    result->n_peaks = 0;
    for (i = 0; i < result->n_patterns; ++i) {
        free(result->patterns[i].hkl);
        free(result->patterns[i].pk_index);
        free(result->patterns[i].err_deg);
        free(result->patterns[i].energy_kev);
        free(result->patterns[i].pred_intens);
    }
    free(result->patterns);
    result->patterns = NULL;
    result->n_patterns = 0;
    result->n_indexed = 0;
}

int laue_pixels_to_q(const laue_geometry *geometry, int detector_index, laue_frame_result *result)
{
    const struct detectorGeometry *detector;
    int i;

    if (!result) return LAUE_INVALID_ARGUMENT;
    result->status = 0;
    result->message[0] = '\0';
    if (!geometry) {
        result->status = LAUE_INVALID_ARGUMENT;
        snprintf(result->message, sizeof(result->message), "geometry is NULL");
        return result->status;
    }
    if (detector_index < 0 || detector_index >= MAX_Ndetectors ||
        !geometry->value.d[detector_index].used) {
        result->status = LAUE_INVALID_ARGUMENT;
        snprintf(result->message, sizeof(result->message), "invalid detector index %d", detector_index);
        return result->status;
    }
    if (result->n_peaks < 0 || (result->n_peaks && !result->peaks)) {
        result->status = LAUE_INVALID_ARGUMENT;
        snprintf(result->message, sizeof(result->message), "invalid peak storage");
        return result->status;
    }

    detector = &geometry->value.d[detector_index];
    for (i = 0; i < result->n_peaks; ++i) {
        double px = result->startx + result->peaks[i].fit_x * result->groupx
                    + (result->groupx - 1) / 2.0;
        double py = result->starty + result->peaks[i].fit_y * result->groupy
                    + (result->groupy - 1) / 2.0;
        if (pixel_to_q(detector, px, py, result->depth, result->peaks[i].qhat)) {
            result->status = LAUE_NUMERICAL_ERROR;
            snprintf(result->message, sizeof(result->message), "cannot convert peak %d", i);
            return result->status;
        }
    }
    return 0;
}

int laue_index(const laue_crystal *crystal, const laue_index_params *params,
               laue_frame_result *result)
{
    struct patternOfOneGrain found[MAX_GRAINS_PER_PATTERN];
    struct patternOfOneGrain fit;
    double (*qhats)[3] = NULL;
    int *indices = NULL;
    long n_found = 0;
    int limit;
    int i;
    int j;
    int status;

    if (!result) return LAUE_INVALID_ARGUMENT;
    if (!crystal || !params || result->n_peaks < 2 || !result->peaks || params->max_data < 2) {
        result->status = LAUE_INVALID_ARGUMENT;
        snprintf(result->message, sizeof(result->message), "invalid indexing input");
        return result->status;
    }
    limit = result->n_peaks < params->max_data ? result->n_peaks : params->max_data;
    if (limit < 2) {
        result->status = LAUE_INVALID_ARGUMENT;
        snprintf(result->message, sizeof(result->message), "fewer than two peaks available for indexing");
        return result->status;
    }

    qhats = calloc((size_t)limit, sizeof(*qhats));
    indices = calloc((size_t)limit, sizeof(*indices));
    if (!qhats || !indices) goto allocation_error;
    for (i = 0; i < limit; ++i) {
        double length = sqrt(result->peaks[i].qhat[0] * result->peaks[i].qhat[0]
                           + result->peaks[i].qhat[1] * result->peaks[i].qhat[1]
                           + result->peaks[i].qhat[2] * result->peaks[i].qhat[2]);
        if (!(length > 0.0)) {
            result->status = LAUE_INVALID_ARGUMENT;
            snprintf(result->message, sizeof(result->message), "peak %d has an invalid q vector", i);
            goto cleanup;
        }
        qhats[i][0] = result->peaks[i].qhat[0] / length;
        qhats[i][1] = result->peaks[i].qhat[1] / length;
        qhats[i][2] = result->peaks[i].qhat[2] / length;
        indices[i] = i;
    }

    memset(&fit, 0, sizeof(fit));
    InitCleanCrystalStructure(&fit.xtal);
    for (i = 0; i < MAX_GRAINS_PER_PATTERN; ++i) {
        memset(&found[i], 0, sizeof(found[i]));
        InitCleanCrystalStructure(&found[i].xtal);
    }
    fout = stderr;
    status = OrientFast(41, fabs(params->kev_max_calc), fabs(params->kev_max_test),
                        fabs(params->angle_tolerance_deg) * M_PI / 180.0,
                        (int *)params->hkl_prefer, fabs(params->cone_deg) * M_PI / 180.0,
                        (struct crystalStructure *)&crystal->value, limit, qhats, indices,
                        &n_found, found);
    if (status) {
        result->status = LAUE_INTERNAL_ERROR;
        snprintf(result->message, sizeof(result->message), "orientation search failed");
        goto cleanup_found;
    }

    result->patterns = calloc((size_t)n_found, sizeof(*result->patterns));
    if (n_found && !result->patterns) goto allocation_error_found;
    result->n_patterns = (int)n_found;
    result->n_indexed = 0;

    for (j = 0; j < n_found; ++j) {
        laue_pattern *output = &result->patterns[j];
        double rms = 0.0;
        double rotation[3][3];
        double reciprocal[3][3];
        double start_step;

        for (i = 0; i < found[j].Ni; ++i) rms += found[j].err[i] * found[j].err[i];
        rms = sqrt(rms / found[j].Ni);

        if (copyCrystalStructure(&fit.xtal, &found[j].xtal)) goto allocation_error_found;
        fit.goodness = found[j].goodness;
        fit.alpha = found[j].alpha;
        fit.beta = found[j].beta;
        fit.gamma = found[j].gamma;
        fit.Ni = found[j].Ni;
        fit.hkls = found[j].hkls;
        fit.intens = found[j].intens;
        fit.pkIndex = found[j].pkIndex;
        fit.err = found[j].err;
        fit.Ghat = calloc((size_t)limit, sizeof(*fit.Ghat));
        if (!fit.Ghat) goto allocation_error_found;

        MatrixCopy33(reciprocal, fit.xtal.recip);
        EulerMatrix(fit.alpha, fit.beta, fit.gamma, rotation);
        MatrixMultiply33(rotation, reciprocal, reciprocal);
        for (i = 0; i < fit.Ni; ++i) {
            long nearest = FindClosestMeasuredG(reciprocal, fit.hkls[i], (size_t)limit, qhats);
            if (nearest < 0) {
                result->status = LAUE_INTERNAL_ERROR;
                snprintf(result->message, sizeof(result->message), "cannot match indexed peak");
                goto cleanup_found;
            }
            memcpy(fit.Ghat[i], qhats[nearest], sizeof(fit.Ghat[i]));
        }
        start_step = 10.0 * rms;
        status = optimizeEulerAngles(start_step, 1e-6, 100, &fit);
        if (status != 0 && status != -2) {
            result->status = status == GSL_ENOMEM ? LAUE_OUT_OF_MEMORY : LAUE_NUMERICAL_ERROR;
            snprintf(result->message, sizeof(result->message), "orientation optimization failed");
            goto cleanup_found;
        }
        free(fit.Ghat);
        fit.Ghat = NULL;
        found[j].alpha = fit.alpha;
        found[j].beta = fit.beta;
        found[j].gamma = fit.gamma;

        output->n_indexed = (int)found[j].Ni;
        output->goodness = found[j].goodness;
        output->hkl = calloc((size_t)output->n_indexed * 3, sizeof(*output->hkl));
        output->pk_index = calloc((size_t)output->n_indexed, sizeof(*output->pk_index));
        output->err_deg = calloc((size_t)output->n_indexed, sizeof(*output->err_deg));
        output->energy_kev = calloc((size_t)output->n_indexed, sizeof(*output->energy_kev));
        output->pred_intens = calloc((size_t)output->n_indexed, sizeof(*output->pred_intens));
        if (!output->hkl || !output->pk_index || !output->err_deg ||
            !output->energy_kev || !output->pred_intens) goto allocation_error_found;

        rms = 0.0;
        output->euler_deg[0] = found[j].alpha * 180.0 / M_PI;
        output->euler_deg[1] = found[j].beta * 180.0 / M_PI;
        output->euler_deg[2] = found[j].gamma * 180.0 / M_PI;
        EulerMatrix(found[j].alpha, found[j].beta, found[j].gamma, output->rotation);
        MatrixMultiply33(output->rotation, found[j].xtal.recip, reciprocal);
        export_reciprocal_rows(output->recip, reciprocal);
        for (i = 0; i < output->n_indexed; ++i) {
            long hkl[3] = {found[j].hkls[i][0], found[j].hkls[i][1], found[j].hkls[i][2]};
            double vector[3];
            double sin_theta;
            double energy;

            lowestAllowedHKL(hkl, &found[j].xtal);
            vector[0] = (double)hkl[0];
            vector[1] = (double)hkl[1];
            vector[2] = (double)hkl[2];
            MatrixMultiply31(found[j].xtal.recip, vector, vector);
            sin_theta = -found[j].Ghat[i][2];
            energy = hc * sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
                     / (4.0 * M_PI * sin_theta);
            output->hkl[3 * i] = (int)hkl[0];
            output->hkl[3 * i + 1] = (int)hkl[1];
            output->hkl[3 * i + 2] = (int)hkl[2];
            output->pk_index[i] = found[j].pkIndex[i];
            output->err_deg[i] = found[j].err[i] * 180.0 / M_PI;
            output->energy_kev[i] = energy;
            output->pred_intens[i] = found[j].intens[i];
            rms += found[j].err[i] * found[j].err[i];
        }
        output->rms_error_deg = sqrt(rms / output->n_indexed) * 180.0 / M_PI;
        result->n_indexed += output->n_indexed;
        freeCrystalStructure(&fit.xtal);
        InitCleanCrystalStructure(&fit.xtal);
    }
    result->status = 0;
    result->message[0] = '\0';
    goto cleanup_found;

allocation_error_found:
    result->status = LAUE_OUT_OF_MEMORY;
    snprintf(result->message, sizeof(result->message), "unable to allocate indexing storage");
cleanup_found:
    free(fit.Ghat);
    freeCrystalStructure(&fit.xtal);
    for (i = 0; i < n_found; ++i) free_grain(&found[i]);
cleanup:
    free(qhats);
    free(indices);
    return result->status;
allocation_error:
    result->status = LAUE_OUT_OF_MEMORY;
    snprintf(result->message, sizeof(result->message), "unable to allocate indexing input");
    goto cleanup;
}

const char *laue_version(void)
{
    return "0.2.0";
}
