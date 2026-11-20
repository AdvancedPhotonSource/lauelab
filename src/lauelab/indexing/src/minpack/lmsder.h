/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
/* Scaled Levenberg-Marquardt least squares: MINPACK lmder with mode 1 (internal
   variable scaling), advanced one outer iteration per call to lmsder_iterate().
   lmsder.c is derived from lmder.c of cminpack; see README.md in this folder. */
#ifndef LMSDER_H
#define LMSDER_H

#include <stddef.h>

/* Status values.  The fit functions return them to the peak search; the
   numbers are GSL's codes for the same conditions. */
#define LMSDER_SUCCESS   0
#define LMSDER_CONTINUE  (-2)
#define LMSDER_ENOMEM    8		/* allocation failed */
#define LMSDER_EBADLEN   19		/* the fit functions: fewer data points than parameters; lmsder_set(): a function with another number of data points */
#define LMSDER_ENOPROG   27		/* 10 trial steps in one iteration, none accepted */
#define LMSDER_ETOLF     29		/* no further reduction of the sum of squares is possible */
#define LMSDER_ETOLX     30		/* no further improvement of x is possible */
#define LMSDER_ETOLG     31		/* the residuals are orthogonal to the Jacobian to machine precision */

typedef struct {
	/* residuals f[i], i < n, at the parameters x[j], j < p */
	int (*f)(const double *x, size_t p, void *params, double *f);
	/* Jacobian at x, column-major: J[i + j*n] = d f[i] / d x[j] */
	int (*df)(const double *x, size_t p, void *params, double *J);
	size_t n;
	size_t p;
	void *params;
} lmsder_function;

typedef struct {
	size_t n, p;
	double *x;				/* current parameters, length p */
	double *dx;				/* last trial step, length p */
	double *f;				/* residuals at x, length n */
	double *J;				/* Jacobian at x, n x p, column-major */
	const lmsder_function *fn;
	double *r, *diag, *qtf, *wa1, *wa2, *wa3, *wa4;
	int *ipvt;
	double fnorm, xnorm, delta, par;
	int scaled;				/* the scaling, xnorm and the initial step bound are set */
	int first;				/* no successful step yet: the step bound is clipped to the step norm */
} lmsder_solver;

/* Returns NULL when p is 0, when n < p, or when the solver cannot be allocated.
   In the peaksearch program (built without LIBLAUE_BUILD) a failed allocation
   prints a message and calls abort() instead. */
lmsder_solver *lmsder_alloc(size_t n, size_t p);
int lmsder_set(lmsder_solver *s, const lmsder_function *fn, const double *x);
int lmsder_iterate(lmsder_solver *s);
int lmsder_test_delta(const lmsder_solver *s, double epsabs, double epsrel);
void lmsder_free(lmsder_solver *s);

#endif
