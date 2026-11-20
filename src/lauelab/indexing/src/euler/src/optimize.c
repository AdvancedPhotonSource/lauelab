/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef _MSC_VER				/* identifies this as a Microsoft compiler */
#define _USE_MATH_DEFINES	/* added RX2011 */
#endif
#include <math.h>
#include <time.h>
#include <limits.h>
#include "nmsimplex.h"
#include "Euler.h"

#ifdef DEBUG_ON
#define DEBUG 1				/* a value of 5 gets everything */
#endif

#define MIN(A,B)	( (A)<(B)?(A):(B) )
#define MAX(A,B)	( (A)>(B)?(A):(B) )

double my_f (const double *v, void *params);

/* When the minimizer cannot allocate memory or meets a non-finite function
   value, the euler program prints a message and stops with abort(), as it
   always has; liblaue returns the status to its caller instead. */
static int solver_error(int status)
{
#ifndef LIBLAUE_BUILD
	if (status == NMSIMPLEX_ENOMEM || status == NMSIMPLEX_EBADFUNC || status == NMSIMPLEX_EFAILED) {
		fprintf(stderr,"ERROR -- optimizeEulerAngles(), %s\n", status == NMSIMPLEX_ENOMEM ? "cannot allocate the minimizer" : "non-finite function value encountered");
		abort();
	}
#endif
	return status;
}




/******************************************************************************************
 * return the error between a lattice rotated by Euler angles and the measured spots
 * thus the returned value is a minimum for the optimimal Euler angles
 */
double my_f(
const double *v,		/* vector with current value of Euler angles */
void	*pattern)		/* pointer to the parameters, in this case a pointer to the pattern structure */
{
	double  G[3];						/* calculated G vector for each hkl */
	double  recip[3][3];				/* reciprocal space */
	double  M_Euler[3][3];				/* an Euler matrix */
	size_t  N;							/* local copy of number of measured spots to check */
	double a,b,g;						/* local copy of the Euler angles */
	struct patternOfOneGrain *p;		/* local value of pattern */
	double  rms;						/* rms value of error, the returned value */
	size_t  i;							/* loop index */
	double  hkl[3];						/* float value of hkl */
	double  dot;

	/* p = pattern; */
	p = (struct patternOfOneGrain *)pattern;
	N = p->Ni;											/* local value for convenience */
	a = v[0];										/* the Euler angles */
	b = v[1];
	g = v[2];
	MatrixCopy33(recip,p->xtal.recip);					/* copy unrotated reciprocal lattice into recip */

	EulerMatrix(a,b,g,M_Euler);							/* make the rotation matrix M_Euler from Euler angles */
	MatrixMultiply33(M_Euler,recip,recip);				/* rotate recip by Euler angles */
	for (i=0,rms=0.0;i<N;i++) {
		VECTOR_COPY3(hkl,p->hkls[i]);					/* convert integer hkl to float for following MatrixMultiply31 */
if (hkl[0]!=hkl[0]) {
fprintf(stderr,"hkl[0] is NaN\n");
}
		MatrixMultiply31(recip,hkl,G);
if (G[0]!=G[0]) {
fprintf(stderr,"G[0] is NaN\n");
}
		normalize3(G);									/* normalized gvec of hkl[i] rotated by Euler angles */
		dot = dot3(G,p->Ghat[i]);
		dot = MIN(dot,1.0);
		rms += 1. - dot;								/* (1-dot) ~ 0.5*(delta angle)^2*/
	}
	rms = sqrt(rms/(double)N);							/* rms of angle errors */
	return rms;
}





/* When calling optimizeEulerAngles(), be sure to set pattern.xtal:

	e.g.
	pattern.xtal.a = pattern.xtal.b = pattern.xtal.c = 4.05;		// just default to Al for now
	pattern.xtal.alpha = pattern.xtal.beta = pattern.xtal.gamma = M_PI/2.0;

	then call:
		FillRecipInLattice(&(pattern.lattice));
	to set the reciprocal lattice in the lattice structure
*/
/* this routine optimizes approximate Euler angles using the Nelder-Mead simplex method. */
int optimizeEulerAngles(
double  startStep,					/* starting step size (radians), make this a bit larger than the error */
double  epsAbs,						/* tolerance on the answer (=0.001) */
long	maxIter,					/* maximum number of iterations, stop after this many regardless, perhaps 100 */
struct	patternOfOneGrain *pattern) /* provides G^'s and hkl's for one fitted pattern, and the lattice parameters */
{
	size_t np = 3;					/* number of dimensions of the function, 3 Euler angles */

	nmsimplex *s = NULL;					/* this is the minimizer */
	double ss[3], x[3];						/* ss is stepsize, x is point */
	size_t iter=0, i;
	int status;								/* status of returned simplex function, 0 is OK */
	double size;							/* size of minimizer, sort of maximum error */

	double  G[3];							/* calculated G vector for each hkl */
	double  recip[3][3];					/* reciprocal space */
	double  M_Euler[3][3];					/* an Euler matrix */
	double  hkl[3];							/* float value of hkl */
	double  dot;

	if (!pattern) { fprintf(stderr,"pattern in NULL on entry to optimizeEulerAngles()\n"); return 1; }
	if (pattern->Ni<3) { fprintf(stderr,"less than 3 pairs of Ghat & hkls on entry to optimizeEulerAngles()\n"); return 1; }
	if (!(pattern->Ghat) || !(pattern->hkls)) { fprintf(stderr,"Ghat or hkls are NULL on entry to optimizeEulerAngles()\n"); return 1; }
	if ((pattern->xtal.a)<=0 || (pattern->xtal.b)<=0 || (pattern->xtal.c)<=0) { fprintf(stderr,"invalid lattice in optimizeEulerAngles()\n"); return 1; }

	/* Set all step sizes to startStep, the starting step size */
	ss[0] = ss[1] = ss[2] = startStep;		/* set ss[0] = ss[1]  = ss[2] = startStep (radian) */

	/* Starting point */
	x[0] = pattern->alpha;					/* set to starting guess, input (alpha,beta,gamma) */
	x[1] = pattern->beta;
	x[2] = pattern->gamma;

	s = nmsimplex_alloc(np);				/* allocate for a 3d simplex minimizer named 's' */
	if (!s) return solver_error(NMSIMPLEX_ENOMEM);

	/* Set starting point for the minimizer:
		for the minimizer 's',
		working on the function my_f() with the pattern
		starting from the initial guess x[3]
		using step sizes of ss[3]  */
	status = nmsimplex_set(s, &my_f, (void *)pattern, x, ss);
	if (status != NMSIMPLEX_SUCCESS) {
		nmsimplex_free(s);
		return solver_error(status);
	}

#if (DEBUG)		/******************************************************************************/
	fprintf(fout, "start minimization with Euler angles (%15.10f, %15.10f, %15.10f) (deg.)\n", \
		pattern->alpha*180/M_PI,pattern->beta*180/M_PI,pattern->gamma*180/M_PI);
#endif			/******************************************************************************/

	/* Iterate the minimizer.  This looop iterates the minimization method until it is good enough */
	iter = 0;
	do {
		iter++;
		status = nmsimplex_iterate(s);					/* do one iteration of the minimizer */
		if (status) { solver_error(status); break; }	/* status=0 is OK, check here for an error */

		size = nmsimplex_size(s);						/* get 'size' of minimizer */
		status = size < epsAbs ? NMSIMPLEX_SUCCESS : NMSIMPLEX_CONTINUE;	/* test size against epsAbs */

#if (DEBUG>1)		/******************************************************************************/
		fprintf(fout,"%5ld  f(", iter);
		for (i=0; i<np; i++) {
			fprintf(fout,"%.8f", s->x[i]);
			if (i<np-1) fprintf(fout,", ");
		}
		fprintf(fout,") = %15.8g size = %.5g\n", s->fval, size);
#endif			/******************************************************************************/
	} while (status == NMSIMPLEX_CONTINUE && iter < (size_t)maxIter); /* stop after maxIter regardless of size */

#if (DEBUG)		/******************************************************************************/
	if (status == NMSIMPLEX_SUCCESS) fprintf(fout,"converged to minimum successfully\n");
	else fprintf(fout,"did not converge successfully, status = %d\n",status);
	fprintf(fout, "  optimization changed Euler angles by (%15.10f, %15.10f, %15.10f) (deg.)\n", \
		(s->x[0]-pattern->alpha)*180/M_PI, (s->x[1]-pattern->beta)*180/M_PI, \
		(s->x[2]-pattern->gamma)*180/M_PI);
#endif			/******************************************************************************/

	pattern->alpha = s->x[0];							/* set to final values, resultant (alpha,beta,gamma) returned */
	pattern->beta = s->x[1];
	pattern->gamma = s->x[2];

#if (DEBUG)		/******************************************************************************/
	fprintf(fout, "after minimization, Euler angles are (%15.10f, %15.10f, %15.10f) (deg.)\n", \
		pattern->alpha*180/M_PI,pattern->beta*180/M_PI,pattern->gamma*180/M_PI);
#endif			/******************************************************************************/

	/* recalculate the err[] list in pattern */
	MatrixCopy33(recip,pattern->xtal.recip);		/* copy unrotated reciprocal lattice into recip */
	EulerMatrix(pattern->alpha,pattern->beta,pattern->gamma,M_Euler);/* rotation matrix from Euler angles */
	MatrixMultiply33(M_Euler,recip,recip);			/* rotate recip by Euler angles */
	for (i=0;i<(size_t)(pattern->Ni);i++) {
		VECTOR_COPY3(hkl,pattern->hkls[i]);			/* convert integer hkl to float for following MatrixMultiply31 */
		MatrixMultiply31(recip,hkl,G);
		normalize3(G);								/* normalized gvec of hkl[i] rotated by Euler angles */
		dot = dot3(G,pattern->Ghat[i]);				/* note, Ghat here is the measured, not calculated G */
		(pattern->err)[i] = acos(MIN(dot,1.0));
	}

	nmsimplex_free(s);									/* free the minimizer */
	return status;
}







