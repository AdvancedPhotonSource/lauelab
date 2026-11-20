/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#ifndef _MATH_UTIL_EULER_H_
#define	_MATH_UTIL_EULER_H_


/* space group numbers for structures known by this program */
/* these are from the International Tables, allowed values are [1-230] */
/*
 * #define FCC 225
 * #define BCC 229
 * #define DIA 227
 * #define SIMPLE_CUBIC 221
 * #define HEX 194
 * #define SAPPHIRE 167
 * #define TRICLINIC 1
 */

#ifndef CHECK_FREE
#define CHECK_FREE(A)   { if(A) free(A); (A)=NULL;}
#endif

#ifdef _MSC_VER						/* identifies this as a Microsoft compiler */
#define round(A) ceil(A-0.5)		/* added RX2011 for round() function */
#endif

#ifndef NAN							/* probably only needed for pcs */
#define NAN nan("")					/* probably only needed for cygwin on pc */
#endif

#ifndef isNAN						/* a good test for NAN  */
#define isNAN(A) ( (A) != (A) )
#endif


#define VECTOR_COPY3(A,B) {A[0]=B[0]; A[1]=B[1]; A[2]=B[2];}


void DeletePoints(size_t len, void *ptr, size_t pntLen, size_t numDel);

void EulerMatrix(double alpha,double beta,double gamma,double M_Euler[3][3]);
void rot2EulerAngles(double A[3][3], double *alpha, double *beta, double *gamma);
void MatrixRz(double Rz[3][3],double angle);
void MatrixRy(double Ry[3][3],double angle);

/* void lowestOrderHKL(int hkl[3]); */
/* void lowestAllowedHKL(int hkl[3], int structure); */
/* long allowedHKL(long h, long k, long l, int structure); */
int gcf(int n1, int n2, int n3);

double normalize3(double a[3]);
void cross(double a[3], double b[3], double c[3]);
void vector3cons(double a[3], double x);
double dot3(double a[3],double b[3]);
double determinant33(double a[3][3]);
void MatrixMultiply31(double a[3][3], double v[3], double c[3]);
void MatrixMultiply33(double a[3][3], double b[3][3], double c[3][3]);
void MatrixTranspose33(double A[3][3]);					/* transpose the 3x3 matrix A */
void MatrixCopy33(double dest[3][3], double source[3][3]);
double diff3(double a[3], double b[3]);
double matsDelta(double a[3][3], double b[3][3]);

char *num2sexigesmal(char str[40], double seconds, long places);


#endif
