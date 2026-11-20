/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
/* Nelder-Mead simplex minimization (J. A. Nelder and R. Mead, Computer Journal
   7, 308-313 (1965)), advanced one iteration per call to nmsimplex_iterate().

   The initial simplex is x and x + step[i]*e_i.  Each iteration moves the worst
   vertex h (the first one with the largest value) along the line through the
   centroid m of the other vertices, to m - c*(m - x_h): reflection c = -1,
   expansion c = -2, contraction c = 0.5.  When neither works, every vertex
   moves halfway towards the best vertex (the first one with the smallest
   value).  With f_r, f_e and f_c the values at the reflection, the expansion
   and the contraction: the expansion is tried when f_r is below the best value
   and kept when f_e is below the best value too, otherwise the reflection is
   kept; else the reflection is kept when f_r is not above the second-worst
   value; else the reflection replaces the worst vertex when f_r is not above
   the worst value, and the contraction from that vertex is kept when f_c is
   not above its value.  A trial point whose value is not finite is never
   accepted.  The size of the simplex is the mean distance of the vertices from
   their center.

   With these rules and this order of the arithmetic, optimizeEulerAngles()
   returns the same angles, bit for bit, as it did with GSL's nmsimplex and
   GSL's own BLAS, which is how liblaue.so was built; keep them when changing
   this file.  The euler program was linked to OpenBLAS, whose norm can differ
   from scaled_norm() in the last bit; where the size falls within that bit of
   the tolerance, euler stops one iteration earlier or later than it did. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nmsimplex.h"

nmsimplex *nmsimplex_alloc(size_t n)
{
	nmsimplex *s;

	if (n == 0) return NULL;
	s = calloc(1, sizeof(*s));
	if (!s) return NULL;
	s->vertex = calloc((n + 1) * n + (n + 1) + 4 * n, sizeof(double));
	if (!s->vertex) {
		free(s);
		return NULL;
	}
	s->n = n;
	s->value = s->vertex + (n + 1) * n;
	s->x = s->value + (n + 1);
	s->centroid = s->x + n;
	s->trial = s->centroid + n;
	s->trial2 = s->trial + n;
	return s;
}

void nmsimplex_free(nmsimplex *s)
{
	if (!s) return;
	free(s->vertex);
	free(s);
}

static size_t first_lowest(const nmsimplex *s)
{
	size_t i, lo = 0;

	for (i = 1; i <= s->n; ++i) {
		if (s->value[i] < s->value[lo]) lo = i;
	}
	return lo;
}

static void set_best(nmsimplex *s)
{
	size_t lo = first_lowest(s);

	memcpy(s->x, s->vertex + lo * s->n, s->n * sizeof(double));
	s->fval = s->value[lo];
}

int nmsimplex_set(nmsimplex *s, double (*f)(const double *x, void *params), void *params,
	const double *x, const double *step)
{
	size_t i, n = s->n;
	double *v;

	s->f = f;
	s->params = params;
	/* the first successful iteration publishes the best vertex; until then x is the starting point */
	memcpy(s->x, x, n * sizeof(double));
	s->fval = 0.0;
	for (i = 0; i <= n; ++i) {
		v = s->vertex + i * n;
		memcpy(v, x, n * sizeof(double));
		if (i > 0) v[i - 1] = x[i - 1] + step[i - 1];
		s->value[i] = f(v, params);
		if (!isfinite(s->value[i])) return NMSIMPLEX_EBADFUNC;
	}
	return NMSIMPLEX_SUCCESS;
}

/* point = m - coeff*(m - x_h), m the centroid of the vertices other than h */
static void move_worst(nmsimplex *s, size_t h, double coeff, double *point)
{
	size_t j, n = s->n;
	const double *xh = s->vertex + h * n;

	for (j = 0; j < n; ++j) {
		double m = s->centroid[j];
		point[j] = m - coeff * (m - xh[j]);
	}
}

static void replace(nmsimplex *s, size_t h, const double *point, double value)
{
	memcpy(s->vertex + h * s->n, point, s->n * sizeof(double));
	s->value[h] = value;
}

int nmsimplex_iterate(nmsimplex *s)
{
	size_t i, j, n = s->n, hi = 0, lo, sec;
	double vr, ve, vc;
	int finite;

	for (i = 1; i <= n; ++i) {
		if (s->value[i] > s->value[hi]) hi = i;
	}
	lo = first_lowest(s);
	sec = (hi == 0) ? 1 : 0;
	for (i = sec + 1; i <= n; ++i) {
		if (i != hi && s->value[i] > s->value[sec]) sec = i;
	}
	for (j = 0; j < n; ++j) {
		double sum = 0.0;
		for (i = 0; i <= n; ++i) {
			if (i != hi) sum += s->vertex[i * n + j];
		}
		s->centroid[j] = sum / n;
	}

	move_worst(s, hi, -1.0, s->trial);						/* reflection */
	vr = s->f(s->trial, s->params);
	if (isfinite(vr) && vr < s->value[lo]) {
		move_worst(s, hi, -2.0, s->trial2);					/* expansion */
		ve = s->f(s->trial2, s->params);
		if (isfinite(ve) && ve < s->value[lo]) replace(s, hi, s->trial2, ve);
		else replace(s, hi, s->trial, vr);
	}
	else if (isfinite(vr) && vr <= s->value[sec]) {
		replace(s, hi, s->trial, vr);
	}
	else {
		if (isfinite(vr) && vr <= s->value[hi]) replace(s, hi, s->trial, vr);
		move_worst(s, hi, 0.5, s->trial2);					/* contraction */
		vc = s->f(s->trial2, s->params);
		if (isfinite(vc) && vc <= s->value[hi]) {
			replace(s, hi, s->trial2, vc);
		}
		else {												/* shrink towards the best vertex */
			const double *best = s->vertex + lo * n;
			finite = 1;
			for (i = 0; i <= n; ++i) {
				double *v = s->vertex + i * n;
				if (i == lo) continue;
				for (j = 0; j < n; ++j) v[j] = 0.5 * (v[j] + best[j]);
				s->value[i] = s->f(v, s->params);
				if (!isfinite(s->value[i])) finite = 0;
			}
			if (!finite) return NMSIMPLEX_EFAILED;
		}
	}
	set_best(s);
	return NMSIMPLEX_SUCCESS;
}

/* Euclidean norm, accumulated with scaling as the reference BLAS dnrm2 does */
static double scaled_norm(const double *v, size_t n)
{
	double scale = 0.0, ssq = 1.0;
	size_t i;

	for (i = 0; i < n; ++i) {
		if (v[i] != 0.0) {
			double a = fabs(v[i]);
			if (scale < a) {
				ssq = 1.0 + ssq * (scale / a) * (scale / a);
				scale = a;
			}
			else {
				ssq += (a / scale) * (a / scale);
			}
		}
	}
	return scale * sqrt(ssq);
}

/* mean distance of the vertices from their center */
double nmsimplex_size(nmsimplex *s)
{
	size_t i, j, n = s->n;
	double *d = s->trial2, sum = 0.0;

	for (j = 0; j < n; ++j) {
		double c = 0.0;
		for (i = 0; i <= n; ++i) c += s->vertex[i * n + j];
		s->centroid[j] = c / (n + 1);
	}
	for (i = 0; i <= n; ++i) {
		for (j = 0; j < n; ++j) d[j] = s->vertex[i * n + j] - s->centroid[j];
		sum += scaled_norm(d, n);
	}
	return sum / (n + 1);
}
