/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
/* Nelder-Mead simplex minimization, advanced one iteration per call. */
#ifndef NMSIMPLEX_H
#define NMSIMPLEX_H

#include <stddef.h>

/* Status values.  optimizeEulerAngles() returns them and the euler program
   prints them; the numbers are GSL's codes for the same conditions. */
#define NMSIMPLEX_SUCCESS   0
#define NMSIMPLEX_CONTINUE  (-2)
#define NMSIMPLEX_EFAILED   5		/* a shrink of the simplex reached a non-finite value */
#define NMSIMPLEX_ENOMEM    8		/* allocation failed */
#define NMSIMPLEX_EBADFUNC  9		/* non-finite value at a vertex of the initial simplex */

typedef struct {
	size_t n;					/* number of variables */
	double *x;					/* best vertex after a successful iteration, length n; the starting point before */
	double fval;				/* function value at x after a successful iteration; 0 before */
	double *vertex;				/* n+1 vertices, vertex i at vertex[i*n] */
	double *value;				/* function value at each vertex */
	double *centroid, *trial, *trial2;
	double (*f)(const double *x, void *params);
	void *params;
} nmsimplex;

nmsimplex *nmsimplex_alloc(size_t n);
int nmsimplex_set(nmsimplex *s, double (*f)(const double *x, void *params), void *params,
	const double *x, const double *step);
int nmsimplex_iterate(nmsimplex *s);
double nmsimplex_size(nmsimplex *s);
void nmsimplex_free(nmsimplex *s);

#endif
