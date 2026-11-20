/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#ifndef WSDT
#define WSDT 1

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {	/* a matrix of doubles, element (i,j) is data[i*tda + j] */
	size_t	size1;			/* number of rows */
	size_t	size2;			/* number of columns */
	size_t	tda;			/* distance between rows, equal to size2 */
	double	*data;
} dmatrix;

typedef struct {	/* a matrix of floats, element (i,j) is data[i*tda + j] */
	size_t	size1;
	size_t	size2;
	size_t	tda;
	float	*data;
} fmatrix;

/* An index outside a matrix, two matrices of different sizes or a failed
   allocation is a fatal error: print it and stop with abort(). */
static inline void matrix_error(const char *message) {
	fprintf(stderr, "ERROR -- %s\n", message);
	abort();
}

static inline dmatrix *dmatrix_calloc(size_t n1, size_t n2) {	/* all elements zero */
	dmatrix *m = malloc(sizeof(dmatrix));
	if (!m || !(m->data = calloc(n1 * n2 ? n1 * n2 : 1, sizeof(double)))) matrix_error("cannot allocate space for a matrix");
	m->size1 = n1;
	m->size2 = m->tda = n2;
	return m;
}

static inline void dmatrix_free(dmatrix *m) {
	if (!m) return;
	free(m->data);
	free(m);
}

static inline double *dmatrix_ptr(dmatrix *m, size_t i, size_t j) {
	if (i >= m->size1 || j >= m->size2) matrix_error("matrix index out of range");
	return m->data + i * m->tda + j;
}

static inline double dmatrix_get(const dmatrix *m, size_t i, size_t j) {
	if (i >= m->size1 || j >= m->size2) matrix_error("matrix index out of range");
	return m->data[i * m->tda + j];
}

static inline void dmatrix_set(dmatrix *m, size_t i, size_t j, double x) {
	*dmatrix_ptr(m, i, j) = x;
}

static inline void dmatrix_set_zero(dmatrix *m) {
	size_t i, j;
	for (i = 0; i < m->size1; i++) for (j = 0; j < m->size2; j++) m->data[i * m->tda + j] = 0.0;
}

static inline void dmatrix_scale(dmatrix *a, double x) {		/* a(i,j) *= x */
	size_t i, j, n1 = a->size1, n2 = a->size2;
	for (i = 0; i < n1; i++) {
		double *row = a->data + i * a->tda;
		#pragma omp simd
		for (j = 0; j < n2; j++) row[j] *= x;
	}
}

static inline void dmatrix_mul_elements(dmatrix *a, const dmatrix *b) {	/* a(i,j) *= b(i,j) */
	size_t i, j;
	if (a->size1 != b->size1 || a->size2 != b->size2) matrix_error("matrix sizes are not conformant");
	for (i = 0; i < a->size1; i++) {
		double *row = a->data + i * a->tda;
		const double *other = b->data + i * b->tda;
		size_t n2 = a->size2;
		#pragma omp simd
		for (j = 0; j < n2; j++) row[j] *= other[j];
	}
}

static inline void dmatrix_sub(dmatrix *a, const dmatrix *b) {	/* a(i,j) -= b(i,j) */
	size_t i, j;
	if (a->size1 != b->size1 || a->size2 != b->size2) matrix_error("matrix sizes are not conformant");
	for (i = 0; i < a->size1; i++) {
		double *row = a->data + i * a->tda;
		const double *other = b->data + i * b->tda;
		size_t n2 = a->size2;
		#pragma omp simd
		for (j = 0; j < n2; j++) row[j] -= other[j];
	}
}

static inline double dmatrix_max(const dmatrix *m) {	/* NaN if any element is NaN */
	double max = m->data[0], x;
	size_t i, j;
	for (i = 0; i < m->size1; i++) for (j = 0; j < m->size2; j++) {
		x = m->data[i * m->tda + j];
		if (x > max) max = x;
		if (isnan(x)) return x;
	}
	return max;
}

static inline double dmatrix_min(const dmatrix *m) {	/* NaN if any element is NaN */
	double min = m->data[0], x;
	size_t i, j;
	for (i = 0; i < m->size1; i++) for (j = 0; j < m->size2; j++) {
		x = m->data[i * m->tda + j];
		if (x < min) min = x;
		if (isnan(x)) return x;
	}
	return min;
}

static inline fmatrix *fmatrix_alloc(size_t n1, size_t n2) {	/* elements not initialized */
	fmatrix *m = malloc(sizeof(fmatrix));
	if (!m || !(m->data = malloc((n1 * n2 ? n1 * n2 : 1) * sizeof(float)))) matrix_error("cannot allocate space for a matrix");
	m->size1 = n1;
	m->size2 = m->tda = n2;
	return m;
}

static inline float fmatrix_get(const fmatrix *m, size_t i, size_t j) {
	if (i >= m->size1 || j >= m->size2) matrix_error("matrix index out of range");
	return m->data[i * m->tda + j];
}

static inline void fmatrix_set(fmatrix *m, size_t i, size_t j, float x) {
	if (i >= m->size1 || j >= m->size2) matrix_error("matrix index out of range");
	m->data[i * m->tda + j] = x;
}

typedef struct {	/* standard R3 coordinate system. */
	double x;
	double y;
	double z;
} point_xyz;

typedef struct {	/* detector coordinate system */
	double i;
	double j;
} point_ccd;

typedef struct		/* a structured vector of doubles */
{
	size_t	size;			/* length of vector, how much space is allocated */
	size_t	alloc;			/* length of vector allocated, alloc >= size */
	double	*v;				/* pointer to the vector data */
} dvector;

typedef struct		/* a structured vector of point_xyz */
{
	size_t		size;		/* length of vector, how much space is allocated */
	size_t		alloc;		/* length of vector allocated, alloc >= size */
	point_xyz	*v;			/* pointer to the vector data */
} xyzvector;

typedef struct		/* a generic void vector, be careful with this one */
{
	size_t	size;			/* length of vector, how much space is allocated */
	size_t	alloc;			/* length of vector allocated, alloc >= size */
	void	**v;			/* pointer to the vector data */
} vvector;


/*typedef struct {
 *	HDF5_Header image_header;
 *	point_xyz wire_position;
 *} wire_scan_data; */

typedef struct {
	vvector	wire_scanned;				/* a vector of dmatrix raw wire scanned images (maybe cropped) */
	xyzvector wire_positions;			/* wire location in a wire scan - same size as wire_scanned for each wire_scanned */
	vvector	depth_resolved;				/* a vector of dmatrix depth-resolved images */
	dvector	depth_intensity;			/* sum of the intensity at each depth */
} ws_image_set;



typedef struct {
	point_xyz centre_at_si_xyz;			/* PM500 coords that put wire center on the Si position (micron) */
	double rotation[3][3];				/* rotation matrix for wire PM500 (from R00 through R22) */
	double rho[3][3];					/* entire rotation matrix for wire, puts wire along {1,0,0} */
	double diameter;					/* wire diameter (micron) */
	point_xyz axis;						/* normalized vector, direction along wire axis.  Should be close to {1,0,0} */
	point_xyz axisR;					/* similar to axis, but rotated by R, now direction of wire in beam line frame */
	point_xyz ki;						/* rho x {0,0,1} */
	double F;							/* F of the wire (PM500 coords, micron) */
} ws_wire_calibration;


typedef struct {
	double pixel_size_i;				/* physical size of one un-binned pixel (micron) */
	double pixel_size_j;				/*   just sizeX/Nx and sizeY/Ny */

	double ccd_pixels_i;				/* number of un-binned pixels on detector on x & y axes */
	double ccd_pixels_j;				/*  just Nx and Ny */

	double ccd_dimension_i;				/* physical size of the detector along x & y (micron) */
	double ccd_dimension_j;				/*  just sizeX and sizeY */

	double detector_rotation[3][3];		/* rotation matrix for detector (from rho00 through rho22) */
	point_xyz P;						/* vector, translation vector defining detector position */

	point_xyz ki;						/* vector, direction of incident beam (should always be 001) */

	ws_wire_calibration wire;
} ws_calibration;

typedef struct {
	int nROI_j;							/* number of binned pixels along the y direction of input image */
	int nROI_i;							/* number of binned pixels along the x direction of input image */

	/* the following four numbers are needed because the image in the file may be only a sub-image (i.e. ROI) of the full detector (and binned) */
	/* the ROI is defined in term of full-chip un-binned zero based pixels */
	int starti;							/* first pixel of ROI along x direction (un-binned pixels) */
	int endi;							/* last pixel of ROI  along x direction (un-binned pixels) */
	int startj;							/* first pixel of ROI along z direction (un-binned pixels) */
	int endj;							/* last pixel of ROI  along z direction (un-binned pixels) */

	/* these two numbers describe the binning of the stored ROI in the file, defined by [startx,endx][startz,endz] */
	/* note, nROI_i = (endx-starx+1)/binx,   and    nROI_j = (endz-statrz+1)/binz */
	int bini;							/* the hardware binning along x direction. Note, (endx-startx+1) is divisible by binx */
	int binj;							/* the hardware binning along z direction. Note, (endz-startz+1) is divisible by binz */

	int current_selection_start;		/* i - indicates which section of the selection is being processed. */
	int current_selection_end;
	size_t rows_at_one_time;			/* number of rows that can be processed at one time, the maximum width of one stripe */

	int NinputImages;					/* number of input images taken during a single wire scan */

	int in_pixel_bytes;					/* number of bytes used to specify one input pixel (bytes) */
	int in_pixel_type;					/* type (e.g. float, int, ...) of type input pixel */
										/* 0=float, 1=long, 2=short, 3=ushort, 4=char, 5=double, 6=signed char, 7=uchar */

	point_xyz wire_first_xyz;			/* wire position of first image in scan */
	point_xyz wire_last_xyz;			/* wire position of last imae in scan */
} ws_imaging_parameters;

typedef struct {
	double depth_resolution;			/* depth difference between adjacent reconstructed images (mciron) */
	
	double depth_start;					/* depth (relative to Si) of the first reconstructed image (micron) */
	double depth_end;					/* depth (relative to Si) of the last reconstructed image (micron) */
	int	NoutputDepths;					/* number of output depths (number of output images) */

	int out_pixel_type;					/* type (e.g. float, int, ...) of a pixel value for the output images, usually output_pixel_type == imaging_parameters.pixel_type */
	int wireEdge;						/* flag, 1=leading edge of wire, 0=trailing edge of wire, -1=both edges */
} ws_user_preferences; 

#endif
