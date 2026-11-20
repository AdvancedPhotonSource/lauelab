/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
/**********************************************************

	Data Structure and Routines for peak fiting
       

/**********************************************************/

#include "fitToFunction.h"

/*******************************************************************/

static int expb_f_1D (const double *x, size_t p, void *params, double *f);
static int expb_df_1D (const double *x, size_t p, void *params, double *J);
static int fitLorentz_1D(double *a, void *params, double *results);
static int expb_f_2D (const double *x, size_t p, void *params, double *f);
static int expb_df_2D (const double *x, size_t p, void *params, double *J);
static int fitLorentz_2D(double *a, void *params, double *results);
static int Lorentz2DFit(double *a, Grid *image, double *a_fit);

/*******************************************************************/
/*
input: 
 a, the list of paramters of a function model (Lorentzian) to be decide after fitting
 params, the observed data valus to be fitted

output:
 f, the fitting function Yi-y[i], where Yi=a0*EZ + a3, y[i] is observed data
    (The funciton model here is for Lorentzian fit) 
 */
static int expb_f_1D (const double *a, size_t p, void *params,
	    double *f){
  size_t n = ((ObservedValues *)params)->n;
  double *y = ((ObservedValues *)params)->y;

  if(p != 4) return LMSDER_EBADLEN;

  double a0 = a[0];
  double a1 = a[1];
  double a2 = a[2];
  double a3= a[3];
  double Z, EZ, Yi;


  a2= (a2==0.)?0.0001:a2;

  size_t i;

  for (i = 0; i < n; i++)
    {
      /* Model Yi  */
      double x = i;
      Z =(x-a1)/a2;
      EZ =1/((Z*Z)+1.);

      Yi = a0*EZ + a3; //only consider p=4
      
      f[i] = (Yi - y[i]);
    }

  return LMSDER_SUCCESS;

}
/*******************************************************************/
/*******************************************************************
input: 
 a, the list of paramters of a function model (Lorentzian) to be decide after fitting
 params, the observed data valus to be fitted

output:
 J, the Jacobian matrix J is the derivative of the function f with respect to the
    four parameters in a  
 *******************************************************************/
static int expb_df_1D (const double *a, size_t p, void *params,
	     double *J){

  size_t n = ((ObservedValues *)params)->n;

  if(p != 4) return LMSDER_EBADLEN;

  double a0 = a[0];
  double a1 = a[1];
  double a2 = a[2];
//  double a3= a[3];
//  double Yi;
  double Z, EZ;


  a2= (a2==0.)?0.0001:a2;

  size_t i;

  for (i = 0; i < n; i++)
    {
      
      double x = i;
      Z =(x-a1)/a2;
      EZ =1/((Z*Z)+1.);

      J[i + 0*n] = EZ;     //df/da0
      J[i + 1*n] = a0*2*Z*(EZ*EZ)/a2;    //df/da1
      J[i + 2*n] = Z*J[i + 1*n];    //df/da2
      J[i + 3*n] = 1;    //df/da3
    
    }

  return LMSDER_SUCCESS;
}

/*******************************************************************/
/*******************************************************************
input: 
 initA, initial paramter list (total 4 elements) 
 params, the observed data to be fitted
output:
 a, the list with fitted parameters 
 *******************************************************************/
static int fitLorentz_1D(double *initA, void *params, double *results){
  
  
  lmsder_solver *s;

  int status;
//  size_t i;
  size_t iter = 0;

  const size_t n = ((ObservedValues *)params)->n;
  const size_t p = 4;

  if (n < p) return LMSDER_EBADLEN;


  
  
  lmsder_function f;

  
  f.f = &expb_f_1D;
  f.df = &expb_df_1D;
  f.n = n;
  f.p = p;
  f.params = params;


  s = lmsder_alloc (n, p);
  if (!s) return LMSDER_ENOMEM;
  status = lmsder_set (s, &f, initA);
  if (status != LMSDER_SUCCESS) {
    lmsder_free(s);
    return status;
  }

  do
    {
      iter++;
      status = lmsder_iterate (s);


    
      if (status)
        break;

      status = lmsder_test_delta (s, 1e-6, 1e-6);//1e-3,1e-3
    }while (status == LMSDER_CONTINUE && iter < 100);//20);
  // printf("1D iter=%d\n",iter);

  

#define FIT(i) (s->x[i])
  
  results[0]=FIT(0);
  results[1]=FIT(1);
  results[2]=FIT(2);
  results[3]=FIT(3);
  
  lmsder_free (s);
  return LMSDER_SUCCESS;
}

/*******************************************************************/
static int expb_f_2D (const double *a, size_t p, void *image,
	       double *f){
  int nx = ((Grid*)image)->width;
  int ny = ((Grid*)image)->height;

  int n=nx*ny;

  double *y = ((Grid *)image)->values;

  if(p < 6 || p > 7) return LMSDER_EBADLEN;
  double xp,yp,u,s,c,Yi;
  
  double a0 = a[0];
  double a1 = a[1];
  double a2 = a[2];
  double a3= a[3];
  double a4= a[4];
  double a5= a[5];
  
  
  if(p==7){
    double a6=a[6];
    s = sin(a6);
    c = cos(a6);
  }
  else {
    s=0.;
    c=1.;
  }
  //  printf("\n**xp**:\n");
	int i;
  for(i=0;i<n;i++){
    //   if(i%nx==0) printf("\n");
    xp =(i%nx - a4)* c/a2 - (i/nx - a5) * s /a2;
    yp =(i%nx - a4) *s/a3 +  (i/nx - a5) * c/a3;
    u = 1./((xp*xp) + (yp*yp) + 1.);
    Yi = a0 + a1 * u;
    //    printf("%5.2f ",xp);
    
    f[i] = (Yi - y[i]);
    
  }


  return LMSDER_SUCCESS;

}

/*******************************************************************/

static int expb_df_2D (const double *a, size_t p, void *image,
		double *J){

  int nx = ((Grid*)image)->width;
  int ny = ((Grid*)image)->height;

  int n=nx*ny;

//  double *y = ((Grid *)image)->values;
  /*
  printf("Image values:\n");
  for(int i=0;i<n;i++){
    printf("%d ",(int)y[i]);
    if(i%nx==0)
      printf("\n");
  }
  */
  if(p < 6 || p > 7) return LMSDER_EBADLEN;
  double xp,yp,u,s,c;
  
//  double a0 = a[0];
  double a1 = a[1];
  double a2 = a[2];
  double a3= a[3];
  double a4= a[4];
  double a5= a[5];
  
  
  if(p==7){
    double a6=a[6];
    s = sin(a6);
    c = cos(a6);
  }
  else {
    s=0.;
    c=1.;
  }
	int i;
  for(i=0;i<n;i++){
    xp =(i%nx - a4)* c/a2 - (i/nx - a5) * s /a2;
    yp =(i%nx - a4) *s/a3 +  (i/nx - a5) * c/a3;
    u = 1./((xp*xp) + (yp*yp) + 1.);

    J[i + 0*n] = 1.;
    J[i + 1*n] = u;
    u = 2.*a1*(u*u);
    J[i + 2*n] = u*(xp*xp)/a2;
    J[i + 3*n] = u*(yp*yp)/a3;
    J[i + 4*n] = u*(c/a2*xp + s/a3*yp);
    J[i + 5*n] = u*(-s/a2*xp + c/a3*yp);
    if(p==7)
      J[i + 6*n] = u*xp*yp*(a3/a2-a2/a3);
      
  }


  return LMSDER_SUCCESS;

}




/*******************************************************************/
static int fitLorentz_2D(double *init_a, void *image, double *a_fit){
 
  
  lmsder_solver *s;

  int status;
//  size_t i;
  size_t iter = 0;

  const size_t n = (size_t)(((Grid *)image)->width*((Grid *)image)->height);
  const size_t p = 7;

  if (n < p) return LMSDER_EBADLEN;


  
  
  lmsder_function f;
  //printf("array size: %d %d\n",sizeof(init_a),sizeof(double));
  
  f.f = &expb_f_2D;
  f.df = &expb_df_2D;
  f.n = n;
  f.p = p;
  f.params = image;


  s = lmsder_alloc (n, p);
  if (!s) return LMSDER_ENOMEM;
  status = lmsder_set (s, &f, init_a);
  if (status != LMSDER_SUCCESS) {
    lmsder_free(s);
    return status;
  }

  do
    {
      iter++;
      status = lmsder_iterate (s);


    
      if (status)
        break;

      status = lmsder_test_delta (s, 1e-6, 1e-6);
    }while (status == LMSDER_CONTINUE && iter < 100);//50);
  // printf("2D iter=%d\n",iter);

  

#define FIT(i) (s->x[i])
	int j;
for(j=0;j<p;j++)
  a_fit[j]=FIT(j);
  
  
  lmsder_free (s);
  return LMSDER_SUCCESS;
}


/*******************************************************************/
/*******************************************************************
input:
 image,  ROI of a peak to be fitted 
 a, initial list of paramters to be fit, 7*sizeof(double)
 a_fit,an empty arry for holding the fitted results of a

output:
 a_fit, the parameter list after 2D Lorentz fit
 *******************************************************************/  


static int Lorentz2DFit(double *a, Grid *image, double *a_fit){
  int nx=image->width;
  int ny=image->height;
  int status = LMSDER_ENOMEM;
  int i;
  Grid *image_roi = NULL;
  Point *center = NULL;
  double ax[4], ay[4], ax_fit[4], ay_fit[4];
  double *x = NULL, *y = NULL;
  ObservedValues paramsX, paramsY;
  int ix, iy;

  image_roi = grid_new_copy_region(image,0,0,nx-1,ny-1);
  if (!image_roi || grid_smooth_boxcar(image_roi,1)) goto cleanup;
  center = centroid_2(image_roi,0,0);
  if (!center) goto cleanup;
  ix=(int)(center->x);
  iy=(int)(center->y);

  ax[0]=grid_get_value(image,ix,iy)-a[0];
  ax[1]=center->x;
  ax[2]=a[2];
  ax[3]=a[0];
  ay[0]=grid_get_value(image,ix,iy)-a[0];
  ay[1]=center->y;
  ay[2]=a[3];
  ay[3]=a[0];

  x=malloc((size_t)nx*sizeof(double));
  y=malloc((size_t)ny*sizeof(double));
  if (!x || !y) goto cleanup;
  for(i=0;i<nx;i++) x[i]=grid_get_value(image,i,iy);
  for(i=0;i<ny;i++) y[i]=grid_get_value(image,ix,i);

  paramsX.n = nx; paramsX.y = x;
  paramsY.n = ny; paramsY.y = y;
  status = fitLorentz_1D(ax,&paramsX,ax_fit);
  if (status != LMSDER_SUCCESS) goto cleanup;
  status = fitLorentz_1D(ay,&paramsY,ay_fit);
  if (status != LMSDER_SUCCESS) goto cleanup;

  a[0]=(ax_fit[3] + ay_fit[3])/2.;
  a[1]=sqrt(fabs(ax_fit[0]*ay_fit[0]));
  a[2]=fabs(ax_fit[2]);
  a[3]=fabs(ay_fit[2]);
  a[4]=ax_fit[1];
  a[5]=ay_fit[1];

  status = fitLorentz_2D(a, (void *)image, a_fit);
  if (status == LMSDER_SUCCESS) a_fit[6]= mod(a_fit[6], M_PI);

cleanup:
  point_delete(center);
  grid_delete(image_roi);
  free(x);
  free(y);
  return status;
}
   
/*******************************************************************/
/*******************************************************************
This is the function to be called in peaksearch.c

input:
 image, the ROI image of a peak to be fitted 

all the other paramters are input & output, after function return they
are all reset with fitted values
changed Aug 2009 by JZT to reject data values of NAN.

 *******************************************************************/
int fitToFunctionLorentz(Grid *image, double *fitx, double *fity, double *background, double *intens,
		  double *widthx, double *widthy, double *tilt,double *chisq){
	double a[7], a_fit[7];
	int status;
	a[0]=*background;
	a[1]=*intens;
	a[2]=*widthx;
	a[3]=*widthy;
	a[4]=*fitx;
	a[5]=*fity;
	a[6]=0.;

	status = Lorentz2DFit(a,image,a_fit);
	if (status != LMSDER_SUCCESS) return status;

	int nx = image->width;
	int ny = image->height;
	int n=nx*ny;
	double s=sin(a_fit[6]), c=cos(a_fit[6]);
	double xp, yp,u,F,chi, datai, *data=image->values;
	double sumChi=0.,sumData=0.;

	int i;
	for(i=0;i<n;i++){
		datai = data[i];
		if (datai!=datai) continue;		/* skip NaNs in the data */
		xp =(i%nx - a_fit[4]) * c/a_fit[2] - (i/nx - a_fit[5]) * s/a_fit[2];
		yp =(i%nx - a_fit[4]) * s/a_fit[3] + (i/nx - a_fit[5]) * c/a_fit[3];
		u = 1./((xp*xp) + (yp*yp) + 1.);
		F = a_fit[0] + a_fit[1] * u;
		chi = (F-datai) * (F-datai);
		sumChi += chi;
		sumData += (datai*datai);
	}

  *fitx=a_fit[4];
  *fity=a_fit[5];
  *background=a_fit[0];
  *intens=a_fit[1];
  *widthx=(a_fit[2]);
  *widthy=(a_fit[3]);
  *tilt=a_fit[6]*180./M_PI;
  *chisq=sumChi/sumData;
  return LMSDER_SUCCESS;
}
     
  
  


  
  
  
  









