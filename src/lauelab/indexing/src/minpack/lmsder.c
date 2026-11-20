/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
/* Derived from lmder.c of cminpack 1.3.11, the C translation of MINPACK's
   lmder (Argonne National Laboratory, MINPACK project, March 1980; Burton S.
   Garbow, Kenneth E. Hillstrom, Jorge J. More).  Minpack Copyright Notice
   (1999) University of Chicago; see CopyrightMINPACK.txt in this folder.

   lmder runs its outer iteration to convergence.  Here one call to
   lmsder_iterate() runs one outer iteration: it factors the Jacobian at x and
   tries steps until one is accepted (at most MAX_TRIES), so the
   caller can apply its own stopping test between iterations.  The arithmetic
   of an iteration is that of lmder with mode 1 and factor 100; the convergence
   tests on ftol, xtol and gtol and the limit maxfev are not used: the caller
   stops on the step size and on an iteration count. */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lmsder.h"
#include "cminpack.h"
#include "cminpackP.h"

#define p1 ((real).1)
#define p5 ((real).5)
#define p25 ((real).25)
#define p75 .75
#define p0001 1e-4
#define FACTOR 100.			/* initial step bound: FACTOR times the norm of diag*x */
#define MAX_TRIES 10		/* trial steps per iteration */

/* The peaksearch program stops when the solver cannot be allocated;
   liblaue.so returns NULL, which its callers report as LMSDER_ENOMEM. */
static lmsder_solver *alloc_failed(void)
{
#ifndef LIBLAUE_BUILD
	fprintf(stderr, "ERROR -- lmsder_alloc(), cannot allocate the least-squares solver\n");
	abort();
#endif
	return NULL;
}

lmsder_solver *lmsder_alloc(size_t n, size_t p)
{
	lmsder_solver *s;
	double *block;

	if (p == 0 || n < p) return NULL;
	if (n > (size_t)INT_MAX / p) return alloc_failed();
	s = calloc(1, sizeof(*s));
	if (!s) return alloc_failed();
	block = calloc(7 * p + 2 * n + 2 * n * p, sizeof(double));
	s->ipvt = calloc(p, sizeof(int));
	if (!block || !s->ipvt) {
		free(block);
		free(s->ipvt);
		free(s);
		return alloc_failed();
	}
	s->n = n;
	s->p = p;
	s->x = block;
	s->dx = s->x + p;
	s->diag = s->dx + p;
	s->qtf = s->diag + p;
	s->wa1 = s->qtf + p;
	s->wa2 = s->wa1 + p;
	s->wa3 = s->wa2 + p;
	s->f = s->wa3 + p;
	s->wa4 = s->f + n;
	s->J = s->wa4 + n;
	s->r = s->J + n * p;
	return s;
}

void lmsder_free(lmsder_solver *s)
{
	if (!s) return;
	free(s->x);
	free(s->ipvt);
	free(s);
}

/* evaluate f and J at x */
int lmsder_set(lmsder_solver *s, const lmsder_function *fn, const double *x)
{
	int status;

	if (fn->n != s->n) return LMSDER_EBADLEN;
	s->fn = fn;
	memcpy(s->x, x, s->p * sizeof(double));
	memset(s->dx, 0, s->p * sizeof(double));
	status = fn->f(s->x, s->p, fn->params, s->f);
	if (status) return status;
	status = fn->df(s->x, s->p, fn->params, s->J);
	if (status) return status;
	s->fnorm = __cminpack_func__(enorm)((int)s->n, s->f);
	s->xnorm = 0.;
	s->delta = 0.;
	s->par = 0.;
	s->scaled = 0;
	s->first = 1;
	return LMSDER_SUCCESS;
}

/* one outer iteration of lmder; on success x, f and J are those of the accepted step */
int lmsder_iterate(lmsder_solver *s)
{
	const int m = (int)s->n, n = (int)s->p;
	const lmsder_function *fn = s->fn;
	double *x = s->x, *fvec = s->f, *fjac = s->r, *diag = s->diag, *qtf = s->qtf;
	double *wa1 = s->wa1, *wa2 = s->wa2, *wa3 = s->wa3, *wa4 = s->wa4;
	int *ipvt = s->ipvt;
	real d1, d2;
	int i, j, l, tries, status;
	real sum, temp, temp1, temp2, ratio, pnorm, fnorm1, actred, prered, dirder, gnorm;
	real epsmch = __cminpack_func__(dpmpar)(1);

	if (s->fnorm == 0.) return LMSDER_SUCCESS;			/* exact fit, no step to take */

/*        compute the qr factorization of the jacobian. */

	memcpy(fjac, s->J, (size_t)m * (size_t)n * sizeof(double));
	__cminpack_func__(qrfac)(m, n, fjac, m, TRUE_, ipvt, n, wa1, wa2, wa3);

/*        on the first iteration, scale according to the norms of the */
/*        columns of the initial jacobian, calculate the norm of the scaled */
/*        x and initialize the step bound delta. */

	if (!s->scaled) {
		for (j = 0; j < n; ++j) {
			diag[j] = wa2[j];
			if (wa2[j] == 0.) {
				diag[j] = 1.;
			}
		}
		for (j = 0; j < n; ++j) {
			wa3[j] = diag[j] * x[j];
		}
		s->xnorm = __cminpack_func__(enorm)(n, wa3);
		s->delta = FACTOR * s->xnorm;
		if (s->delta == 0.) {
			s->delta = FACTOR;
		}
		s->scaled = 1;
	}

/*        form (q transpose)*fvec and store the first n components in */
/*        qtf. */

	for (i = 0; i < m; ++i) {
		wa4[i] = fvec[i];
	}
	for (j = 0; j < n; ++j) {
		if (fjac[j + j * m] != 0.) {
			sum = 0.;
			for (i = j; i < m; ++i) {
				sum += fjac[i + j * m] * wa4[i];
			}
			temp = -sum / fjac[j + j * m];
			for (i = j; i < m; ++i) {
				wa4[i] += fjac[i + j * m] * temp;
			}
		}
		fjac[j + j * m] = wa1[j];
		qtf[j] = wa4[j];
	}

/*        compute the norm of the scaled gradient. */

	gnorm = 0.;
	if (s->fnorm != 0.) {
		for (j = 0; j < n; ++j) {
			l = ipvt[j]-1;
			if (wa2[l] != 0.) {
				sum = 0.;
				for (i = 0; i <= j; ++i) {
					sum += fjac[i + j * m] * (qtf[i] / s->fnorm);
				}
				d1 = fabs(sum / wa2[l]);
				gnorm = max(gnorm,d1);
			}
		}
	}

/*        rescale if necessary. */

	for (j = 0; j < n; ++j) {
		d1 = diag[j], d2 = wa2[j];
		diag[j] = max(d1,d2);
	}

/*        trial steps, until one is accepted. */

	for (tries = 1; ; ++tries) {

/*           determine the levenberg-marquardt parameter. */

		__cminpack_func__(lmpar)(n, fjac, m, ipvt, diag, qtf, s->delta,
			&s->par, wa1, wa2, wa3, wa4);

/*           store the direction p and x + p. calculate the norm of p. */

		for (j = 0; j < n; ++j) {
			wa1[j] = -wa1[j];
			wa2[j] = x[j] + wa1[j];
			wa3[j] = diag[j] * wa1[j];
		}
		pnorm = __cminpack_func__(enorm)(n, wa3);
		memcpy(s->dx, wa1, (size_t)n * sizeof(double));

/*           until the first successful step, adjust the initial step bound. */

		if (s->first) {
			s->delta = min(s->delta,pnorm);
		}

/*           evaluate the function at x + p and calculate its norm. */

		status = fn->f(wa2, s->p, fn->params, wa4);
		if (status) return status;
		fnorm1 = __cminpack_func__(enorm)(m, wa4);

/*           compute the scaled actual reduction. */

		actred = -1.;
		if (p1 * fnorm1 < s->fnorm) {
			d1 = fnorm1 / s->fnorm;
			actred = 1 - d1 * d1;
		}

/*           compute the scaled predicted reduction and */
/*           the scaled directional derivative. */

		for (j = 0; j < n; ++j) {
			wa3[j] = 0.;
			l = ipvt[j]-1;
			temp = wa1[l];
			for (i = 0; i <= j; ++i) {
				wa3[i] += fjac[i + j * m] * temp;
			}
		}
		temp1 = __cminpack_func__(enorm)(n, wa3) / s->fnorm;
		temp2 = (sqrt(s->par) * pnorm) / s->fnorm;
		prered = temp1 * temp1 + temp2 * temp2 / p5;
		dirder = -(temp1 * temp1 + temp2 * temp2);

/*           compute the ratio of the actual to the predicted */
/*           reduction. */

		ratio = 0.;
		if (prered != 0.) {
			ratio = actred / prered;
		}

/*           update the step bound. */

		if (ratio <= p25) {
			if (actred >= 0.) {
				temp = p5;
			} else {
				temp = p5 * dirder / (dirder + p5 * actred);
			}
			if (p1 * fnorm1 >= s->fnorm || temp < p1) {
				temp = p1;
			}
			d1 = pnorm / p1;
			s->delta = temp * min(s->delta,d1);
			s->par /= temp;
		} else {
			if (s->par == 0. || ratio >= p75) {
				s->delta = pnorm / p5;
				s->par = p5 * s->par;
			}
		}

/*           successful iteration. update x, fvec, their norms and the jacobian. */

		if (ratio >= p0001) {
			for (j = 0; j < n; ++j) {
				x[j] = wa2[j];
				wa2[j] = diag[j] * x[j];
			}
			for (i = 0; i < m; ++i) {
				fvec[i] = wa4[i];
			}
			s->xnorm = __cminpack_func__(enorm)(n, wa2);
			s->fnorm = fnorm1;
			s->first = 0;
			return fn->df(x, s->p, fn->params, s->J);
		}

/*           tests for termination and stringent tolerances. */

		if (fabs(actred) <= epsmch && prered <= epsmch && p5 * ratio <= 1.) {
			return LMSDER_ETOLF;
		}
		if (s->delta <= epsmch * s->xnorm) {
			return LMSDER_ETOLX;
		}
		if (gnorm <= epsmch) {
			return LMSDER_ETOLG;
		}
		if (tries == MAX_TRIES) {
			return LMSDER_ENOPROG;
		}
	}
}

/* LMSDER_SUCCESS when every |dx[i]| < epsabs + epsrel*|x[i]|, else LMSDER_CONTINUE */
int lmsder_test_delta(const lmsder_solver *s, double epsabs, double epsrel)
{
	size_t i;

	for (i = 0; i < s->p; ++i) {
		if (!(fabs(s->dx[i]) < epsabs + epsrel * fabs(s->x[i]))) return LMSDER_CONTINUE;
	}
	return LMSDER_SUCCESS;
}
