# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Shared ABI-mode bindings for ``liblaue.so``."""

from importlib import resources
from os import fspath

from cffi import FFI

ffi = FFI()
ffi.cdef(
    """
    typedef struct laue_geometry laue_geometry;
    typedef struct laue_crystal laue_crystal;
    typedef struct laue_recon laue_recon;
    enum {
        LAUE_OK = 0, LAUE_INVALID_ARGUMENT = 1, LAUE_OUT_OF_MEMORY = 2,
        LAUE_NUMERICAL_ERROR = 3, LAUE_INTERNAL_ERROR = 4,
        LAUE_RECON_EDGE_LEADING = 1, LAUE_RECON_EDGE_TRAILING = 0,
        LAUE_RECON_EDGE_BOTH = -1,
        LAUE_POSITIONER_NONE = 0, LAUE_POSITIONER_PM500 = 1,
        LAUE_POSITIONER_ALIO = 2,
        LAUE_PIXEL_U16 = 0, LAUE_PIXEL_F64 = 1, LAUE_PIXEL_I32 = 2,
        LAUE_PIXEL_F32 = 3, LAUE_PIXEL_I16 = 4, LAUE_PIXEL_U8 = 5, LAUE_PIXEL_I8 = 6
    };
    typedef struct { char name[60]; double x, y, z, occupancy; } laue_atom;
    typedef struct {
        int nx, ny; double size_x, size_y; double translation[3];
        double rotation_vector[3]; double rotation[3][3]; char detector_id[256];
    } laue_detector_info;
    typedef struct {
        double dia, F; double origin[3], axis[3], axisR[3], R[3][3], Rmag;
        int has_wire;
    } laue_wire_info;
    typedef struct {
        double depth_start_um, depth_end_um, resolution_um;
        int wire_edge, cosmic_filter, nx_full, ny_full, start_i, start_j;
        int bin_i, bin_j, n_rows_total, n_cols;
    } laue_recon_params;
    typedef struct {
        int boxsize; double max_rfactor; int min_size, min_separation;
        double threshold, threshold_ratio; int peak_shape, max_peaks, smooth;
        const unsigned char *mask;
    } laue_peak_params;
    typedef struct {
        double kev_max_calc, kev_max_test, angle_tolerance_deg, cone_deg;
        int hkl_prefer[3]; int max_data;
    } laue_index_params;
    typedef struct {
        double fit_x, fit_y, intens, integral, hwhm_x, hwhm_y, tilt, chisq, background;
        double qhat[3];
    } laue_peak;
    typedef struct {
        double euler_deg[3], rotation[3][3], recip[3][3];
        double goodness, rms_error_deg; int n_indexed;
        int *hkl, *pk_index; double *err_deg, *energy_kev, *pred_intens;
    } laue_pattern;
    typedef struct {
        int nx, ny, startx, starty, groupx, groupy; double depth;
        double threshold_used, peak_minwidth, peak_maxwidth, peak_max_cent_to_fit;
        int peak_boxsize; double total_sum, sum_above_threshold;
        long num_above_threshold; int n_peaks; laue_peak *peaks;
        int n_patterns, n_indexed; laue_pattern *patterns; int status;
        char message[256];
    } laue_frame_result;
    laue_geometry *laue_geometry_from_file(const char *, char *, size_t);
    void laue_geometry_free(laue_geometry *);
    laue_crystal *laue_crystal_create(const char *, int, double, double, double,
                                       double, double, double, const laue_atom *, size_t,
                                       char *, size_t);
    void laue_crystal_free(laue_crystal *);
    int laue_crystal_reciprocal(const laue_crystal *, double [3][3]);
    int laue_geometry_detector_count(const laue_geometry *);
    int laue_geometry_find_detector(const laue_geometry *, const char *);
    int laue_geometry_detector_info(const laue_geometry *, int, laue_detector_info *, char *, size_t);
    int laue_geometry_wire_info(const laue_geometry *, laue_wire_info *, char *, size_t);
    laue_recon *laue_recon_create(const laue_geometry *, int, const laue_recon_params *, char *, size_t);
    int laue_recon_set_wire_positions(laue_recon *, const double *, size_t, int);
    int laue_recon_stripe(laue_recon *, const void *, int, size_t, size_t, size_t,
                          const double *, const double *, const unsigned char *, double *, int, double *);
    int laue_recon_n_depths(const laue_recon *);
    double laue_recon_depth_um(const laue_recon *, int);
    const char *laue_recon_last_error(const laue_recon *);
    void laue_recon_free(laue_recon *);
    int laue_find_peaks_typed(const void *, int, int, int, const laue_peak_params *, laue_frame_result *);
    int laue_find_peaks(const unsigned short *, int, int, const laue_peak_params *, laue_frame_result *);
    int laue_pixels_to_q(const laue_geometry *, int, laue_frame_result *);
    int laue_index(const laue_crystal *, const laue_index_params *, laue_frame_result *);
    void laue_frame_result_free(laue_frame_result *);
    const char *laue_version(void);
    """
)


def _load_library():
    library = resources.files("lauelab.indexing.bin") / "liblaue.so"
    try:
        return ffi.dlopen(fspath(library))
    except OSError as error:
        raise ImportError(
            "liblaue.so is unavailable; rebuild lauelab to use native lauelab operations"
        ) from error


_lib = None


def get_library():
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib
