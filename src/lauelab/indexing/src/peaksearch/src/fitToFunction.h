/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
/**********************************************************

	Data Structure and Routines for peak fiting
       

/**********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "lmsder.h"

#include <errno.h>

#include "grid.h"
#include "grid_operations.h"
#include "minmax.h"
#include "point.h"

#ifndef _FITTOFUNCTION_H_
#define _FITTOFUNCTION_H_

typedef  struct {
  size_t n;
  double * y;
} ObservedValues ;

int fitToFunctionLorentz(Grid *image, double *fitx, double *fity,
		   double *background, double *intens,
		   double *widthx, double *widthy, double *tilt,double *chisq);

int fitToFunctionGauss(Grid *image, double *fitx, double *fity,
		   double *background, double *intens,
		   double *widthx, double *widthy, double *tilt,double *chisq);


#endif
