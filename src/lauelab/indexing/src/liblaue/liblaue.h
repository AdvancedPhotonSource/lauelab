/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#ifndef LIBLAUE_H
#define LIBLAUE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define LAUE_API __attribute__((visibility("default")))
#else
#define LAUE_API
#endif

typedef struct laue_geometry laue_geometry;
typedef struct laue_crystal laue_crystal;
typedef struct laue_recon laue_recon;

typedef enum {
    LAUE_OK = 0,
    LAUE_INVALID_ARGUMENT = 1,
    LAUE_OUT_OF_MEMORY = 2,
    LAUE_NUMERICAL_ERROR = 3,
    LAUE_INTERNAL_ERROR = 4
} laue_status;

typedef struct {
    char name[60];
    double x, y, z, occupancy;
} laue_atom;

typedef struct {
    int nx, ny;
    double size_x, size_y;
    double translation[3];
    double rotation_vector[3];
    double rotation[3][3];
    char detector_id[256];
} laue_detector_info;

typedef struct {
    double dia;
    double F;
    double origin[3];
    double axis[3];
    double axisR[3];
    double R[3][3];
    double Rmag;
    int has_wire;
} laue_wire_info;

typedef enum {
    LAUE_RECON_EDGE_LEADING = 1,
    LAUE_RECON_EDGE_TRAILING = 0,
    LAUE_RECON_EDGE_BOTH = -1
} laue_recon_edge;

typedef enum {
    LAUE_POSITIONER_NONE = 0,
    LAUE_POSITIONER_PM500 = 1,
    LAUE_POSITIONER_ALIO = 2
} laue_positioner;

/* Element type of a borrowed pixel buffer. laue_recon_stripe accepts
   LAUE_PIXEL_U16 and LAUE_PIXEL_F64; laue_find_peaks_typed accepts every member.
   Each element is converted to double exactly; nothing is scaled or clipped. */
typedef enum {
    LAUE_PIXEL_U16 = 0,
    LAUE_PIXEL_F64 = 1,
    LAUE_PIXEL_I32 = 2,
    LAUE_PIXEL_F32 = 3,
    LAUE_PIXEL_I16 = 4,
    LAUE_PIXEL_U8 = 5,
    LAUE_PIXEL_I8 = 6
} laue_pixel_type;

typedef struct {
    double depth_start_um;
    double depth_end_um;
    double resolution_um;
    int wire_edge;
    int cosmic_filter;
    int nx_full;
    int ny_full;
    int start_i;
    int start_j;
    int bin_i;
    int bin_j;
    int n_rows_total;
    int n_cols;
} laue_recon_params;

typedef struct {
    int boxsize;
    double max_rfactor;
    int min_size;
    int min_separation;
    double threshold;
    double threshold_ratio;
    int peak_shape;
    /* Positive limit on the number of fitted peaks, or 0 for no limit: every
       blob above threshold is fitted. Negative values are rejected. */
    int max_peaks;
    int smooth;
    const unsigned char *mask;
} laue_peak_params;

typedef struct {
    double kev_max_calc;
    double kev_max_test;
    double angle_tolerance_deg;
    double cone_deg;
    int hkl_prefer[3];
    int max_data;
} laue_index_params;

typedef struct {
    double fit_x;
    double fit_y;
    double intens;
    double integral;
    double hwhm_x;
    double hwhm_y;
    double tilt;
    double chisq;
    double background;
    double qhat[3];
} laue_peak;

typedef struct {
    double euler_deg[3];
    double rotation[3][3];
    /* Rows are a*, b*, c* in 1/nm; q = hkl * recip. */
    double recip[3][3];
    double goodness;
    double rms_error_deg;
    int n_indexed;
    int *hkl;
    int *pk_index;
    double *err_deg;
    double *energy_kev;
    double *pred_intens;
} laue_pattern;

typedef struct {
    int nx;
    int ny;
    int startx;
    int starty;
    int groupx;
    int groupy;
    double depth;
    double threshold_used;
    double peak_minwidth;
    double peak_maxwidth;
    double peak_max_cent_to_fit;
    int peak_boxsize;
    double total_sum;
    double sum_above_threshold;
    long num_above_threshold;
    int n_peaks;
    laue_peak *peaks;
    int n_patterns;
    int n_indexed;
    laue_pattern *patterns;
    int status;
    char message[256];
} laue_frame_result;

LAUE_API laue_geometry *laue_geometry_from_file(const char *path, char *err, size_t errlen);
LAUE_API void laue_geometry_free(laue_geometry *geometry);
LAUE_API laue_crystal *laue_crystal_create(const char *name, int space_group,
                                            double a, double b, double c,
                                            double alpha_deg, double beta_deg, double gamma_deg,
                                            const laue_atom *atoms, size_t n_atoms,
                                            char *err, size_t errlen);
LAUE_API void laue_crystal_free(laue_crystal *crystal);
LAUE_API int laue_crystal_reciprocal(const laue_crystal *crystal, double recip[3][3]);
LAUE_API int laue_geometry_detector_count(const laue_geometry *geometry);
LAUE_API int laue_geometry_find_detector(const laue_geometry *geometry, const char *detector_id);
LAUE_API int laue_geometry_detector_info(const laue_geometry *geometry, int detector_index,
                                         laue_detector_info *info, char *err, size_t errlen);
LAUE_API int laue_geometry_wire_info(const laue_geometry *geometry, laue_wire_info *info,
                                     char *err, size_t errlen);
/* Create a reconstruction handle from borrowed geometry and params. detector_index
   is zero-based. err points to errlen writable bytes and receives setup errors.
   The returned handle owns its working memory; calls using one handle must not overlap. */
LAUE_API laue_recon *laue_recon_create(const laue_geometry *geometry, int detector_index,
                                       const laue_recon_params *params,
                                       char *err, size_t errlen);
/* Set n raw wire positions, laid out row-major as n x 3 doubles. Reconstruction
   requires n = n_images + 1. positioner selects correction of the borrowed input. */
LAUE_API int laue_recon_set_wire_positions(laue_recon *recon, const double *xyz_raw,
                                           size_t n, int positioner);
/* Reconstruct one detector stripe. images is borrowed row-major storage with shape
   n_images x nrows x params->n_cols and elements selected by pixel_type. row0 is
   the stripe's zero-based row in the full reconstruction. scale is NULL or
   n_images doubles; norm_plane is NULL or nrows x params->n_cols doubles. mask
   is nrows x params->n_cols bytes, where 1 processes a pixel and 0 skips it. out
   is depth-major n_depths x nrows x params->n_cols double storage and must be
   zero-filled by the caller. n_threads must be at least 1. seconds_elapsed may
   be NULL or point to one writable double. Calls on the same recon must not overlap. */
LAUE_API int laue_recon_stripe(laue_recon *recon, const void *images, int pixel_type,
                               size_t n_images, size_t row0, size_t nrows,
                               const double *scale, const double *norm_plane,
                               const unsigned char *mask, double *out,
                               int n_threads, double *seconds_elapsed);
/* Convert one reconstructed stripe to its stored form and reduce it. values is
   the n_depths x n_pixels double output of laue_recon_stripe. Each value is
   multiplied by rescale and stored as an HDF5 dataset of stored_type would
   store it: a floating type receives the rounded value; an integer type
   receives the value truncated toward zero and saturated at the type's limits,
   and NaN as zero. stored receives n_depths x n_pixels elements of stored_type.
   depth_sums receives n_depths totals over pixels and pixel_sums n_pixels totals
   over depths of the stored values, as int64_t for an integer stored_type and
   double otherwise; either may be NULL. Both totals are independent of
   n_threads, which must be at least 1. */
LAUE_API int laue_recon_store_stripe(const double *values, size_t n_depths, size_t n_pixels,
                                     double rescale, int stored_type, void *stored,
                                     void *depth_sums, void *pixel_sums, int n_threads);
/* Return the number of output depths for recon, or 0 for NULL. */
LAUE_API int laue_recon_n_depths(const laue_recon *recon);
/* Return output depth index in micrometres, or NaN for invalid input. */
LAUE_API double laue_recon_depth_um(const laue_recon *recon, int index);
/* Return the last error owned by recon; the pointer remains valid until its next call. */
LAUE_API const char *laue_recon_last_error(const laue_recon *recon);
/* Free recon and its owned memory. NULL is accepted. */
LAUE_API void laue_recon_free(laue_recon *recon);
/* Peak search on a borrowed row-major nx x ny frame. pixel_type selects the
   element type of pixels (any laue_pixel_type member). Frame statistics and
   fitting use the exact double value of each element. Floating-point frames
   must be finite; the caller is responsible for that check. */
LAUE_API int laue_find_peaks_typed(const void *pixels, int pixel_type, int nx, int ny,
                                   const laue_peak_params *params, laue_frame_result *result);
/* Convenience form of laue_find_peaks_typed for LAUE_PIXEL_U16 frames. */
LAUE_API int laue_find_peaks(const unsigned short *pixels, int nx, int ny,
                             const laue_peak_params *params, laue_frame_result *result);
LAUE_API int laue_pixels_to_q(const laue_geometry *geometry, int detector_index, laue_frame_result *result);
LAUE_API int laue_index(const laue_crystal *crystal, const laue_index_params *params,
                        laue_frame_result *result);
LAUE_API void laue_frame_result_free(laue_frame_result *result);
LAUE_API const char *laue_version(void);

#ifdef __cplusplus
}
#endif

#endif
