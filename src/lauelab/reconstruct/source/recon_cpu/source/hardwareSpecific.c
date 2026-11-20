/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
/*
 *  hardwareSpecific.c
 *  reconstruct
 *
 *  Created by Jon Tischler on 1/19/09.
 *  Copyright 2009 ORNL. All rights reserved.
 *
 */


#include <math.h>
#include "WireScanDataTypesN.h"
#include "hardwareSpecific.h"

static size_t myEpoch(int year, int month, int date, int hour, int min, int sec);

fmatrix * distortion_map_i = NULL;
fmatrix * distortion_map_j = NULL;

int	positionerType = 0;				/*	0 means no correction (the default)		initialized here
										1 means PM500 correction from May 2006
										2 means Alio correction from Oct 2009 */


int positionerTypeFromFileTime(char *fileTime)
{
	int		year, month, date;
	int		hour, min, sec;
	size_t	t, t0, t1;
	int		i;

	//	file_time = 2009-07-18 22:34:17-0600
	i = sscanf(fileTime,"%04d-%02d-%02d %02d:%02d:%02d",&year,&month,&date,&hour,&min,&sec);
	if (i!=6) return 0;						/* could not read date time, default to no correction */

	t = myEpoch(year,month,date,hour,min,sec);
	t0 = myEpoch(2006,5,1,0,0,0);			/* May 1, 2006 */
	t1 = myEpoch(2009,10,1,0,0,0);			/* Oct 1, 2009 */

	if (t<t0) return 0;						/* PM500 correction is valid as of May 2006, this date is really early */
	if (t<t1) return 1;						/* between May 2006 and Oct 2009, use PM500 correction */
	return 2;								/* on or after Oct 2009, use Alio correction */
}


static size_t myEpoch(
int		year,
int		month,
int		date,
int		hour,
int		min,
int		sec)
{
	int		leap;
	size_t	t=0;							/* number of seconds since start of Jan 1, year=EPOCH, 00:00:00 */
	size_t	days;							/* number of days this year */
	int	mdays[] = {0,31,59,90,120,151,181,212,243,273,304,334};	/* number of days before each month */

	year -= EPOCH;							/* use an epoch of year==EPOCH, Jan 1, 00:00:00 */
	if (year<0 || year>136) return 0;		/* cannot deal with time before EPOCH or 136 year later */
	if (month<1 || month>12 || date<1 || date>31) 	/* check for valid inputs */
	if (hour<0 || hour>24 ||min<0 || min>60 || sec<0 || sec>60) return 0;

	leap = (year-1)/4;						/* number of leap years, one extra day for each leap year */
	leap += (year%4==0) && month>2;			/* and this year is a leap year and we are past February */
	days = (size_t)year*365L + leap;		/* number of days to start of this year */
	days += mdays[month-1];					/* number of days to start of this month */
	days += (date-1);						/* days to start of date */
	t = days * (24L*3600L);					/* convert days to seconds */
	t += 3600L*hour + 60L*min + sec;		/* add number of seconds for time of day */
	return t;
}



#ifdef USE_POSITIONER_CORRECTION

double X2corrected(		/* takes positioner X2 and returns the "real" X2, uses appropriate correction */
double X2)				/* input the PM500 X2 reading */
{
	switch (positionerType)
	{
		case 0:	return X2;
		case 1:	return X2correctedPM500(X2);
		case 2:	return X2;
	}
	return X2;
}

double Y2corrected(		/* takes positioner Y2 and returns the "real" Y2, uses appropriate correction */
double Y2)				/* input the PM500 Y2 reading */
{
	switch (positionerType)
	{
		case 0:	return Y2;
		case 1:	return Y2correctedPM500(Y2);
		case 2:	return Y2;
	}
	return Y2;
}

double Z2corrected(		/* takes positioner Z2 and returns the "real" Z2, uses appropriate correction */
double Z2)				/* input the PM500 Z2 reading */
{
	switch (positionerType)
	{
		case 0:	return Z2;
		case 1:	return Z2correctedPM500(Z2);
		case 2:	return Z2;
	}
	return Z2;
}






#ifdef USE_PM500_CORRECTION_MAY06
double X2correctedPM500(/* takes PM500 X2 and returns the "real" X2, uses data from Deming Shu's sensor */
double X2)				/* input the PM500 X2 reading */
{
	return(X2);
}


double Y2correctedPM500(/* takes PM500 Y2 and returns the "real" Y2, uses data from Deming Shu's sensor */
double Y2)				/* input the PM500 Y2 reading */
{
	double Y2correction[]={0.190672,0.122487,0.0709713,0.0194531,-0.0620639,-0.180244,-0.26843,-0.353281,-0.491464,-0.519646, \
						-0.467831,-0.242511,-0.0873608,0.117789,0.292936,0.413085,0.433239,0.388385,0.353539,0.298687,0.218837};
	double dY2;			/* Y value into Y2correction */
	long i;
	double delta;

	dY2 = fmod(Y2,20.);
	dY2 = dY2 < 0 ? dY2+20 : dY2;	/* dY2 now in range [0,20) */

	/* find interpolated value of Y2correction(dY2), we know that dY2 is in range [0,20) */
	i = (long)dY2;
	if (i > 19) i = 19;				/* negative values can round to the period */
	delta = fmod(dY2,1.);			/* fractional part from i to dY2 */

	Y2 += fmod(dY2,1.) * (Y2correction[i+1]-Y2correction[i]) + Y2correction[i];	/* add correction to original value */
	return(Y2);
}


double Z2correctedPM500(/* takes PM500 Z2 and returns the "real" Z2, uses data from Deming Shu's sensor */
double Z2)				/* input the PM500 Z2 reading */
{
	double Z2correction[]={0.989107,1.00074,0.823196,0.532596,0.160087,-0.205875,-0.556123,-0.794466,-0.938048,-0.903535, \
						-0.771879,-0.562128,-0.363329,-0.19691,-0.0523967,0.0983067,0.240439,0.405843,0.562625,0.701474, \
						0.786522,0.772175,0.632715,0.376184,0.0351449,-0.313989,-0.629154,-0.850827,-0.961081,-0.93489, \
						-0.764372,-0.544069,-0.331984,-0.118546,0.0546623,0.223944,0.393703,0.549969,0.710416,0.890823,0.989107};
	double dZ2;			/* Z value into Z2correction */
	long i;
	double delta;

	dZ2 = fmod(Z2,40.);
	dZ2 = dZ2 < 0 ? dZ2+40 : dZ2;	/* dZ2 now in range [0,40) */

	/* find interpolated value of Z2correction(dZ2), we know that dZ2 is in range [0,40) */
	i = (long)dZ2;
	if (i > 39) i = 39;				/* negative values can round to the period */
	delta = fmod(dZ2,1.);			/* fractional part from i to dZ2 */

	Z2 += fmod(dZ2,1.) * (Z2correction[i+1]-Z2correction[i]) + Z2correction[i];	/* add correction to original value */
	return(Z2);
}
#else
double X2correctedPM500(double X2)				/* dummy */
{ return(X2); }
double Y2correctedPM500(double Y2)				/* dummy */
{ return(Y2); }
double Z2correctedPM500(double Z2)				/* dummy */
{ return(Z2); }
#endif		// end of USE_PM500_CORRECTION_MAY06
#endif		// end of USE_POSITIONER_CORRECTION


#ifdef USE_DISTORTION_CORRECTION
#error "check this function, peakcorrection()"
/* takes a full chip un-distorted pixel position and returns the distortion corrected pixel position (all full chip un-binned pixels) */
/* if pixel is outside the range, it uses the correction from the nearest known pixel.  So this shold work for all inputs */
point_ccd peakcorrection(point_ccd pixel)
{
	long i,j;							/* input pixel trimmed to image size */
	point_ccd corrected;				/* the result, a distortion corrected pixel (full chip un-binned pixels) */
	if (!distortion_map_i || !distortion_map_j) return pixel;	/* no distortion map loadedd */

	i = MAX((long)round(pixel.i),0);	/* i is an integer in the range [0,2083] */
	i = MIN(i,2083);
	j = MAX((long)round(pixel.j),0);	/* j is an integer in the range [0,2083] */
	j = MIN(j,2083);
	#warning "change 2083 to a variablle, probably calibration.ccd_pixels_i-1 and calibration.ccd_pixels_j-1"

	corrected.i = pixel.i + fmatrix_get(distortion_map_i, i, j);	/* add the distortion correction */
	corrected.j = pixel.j + fmatrix_get(distortion_map_j, i, j);
	return corrected;
}
#endif


#ifdef USE_DISTORTION_CORRECTION
/* reads the x and y distortions of the detector (in pixels).  true pixel is itrue = i+distortion_map_i(i,j),   and jtrue = j+distortion_map_j(i,j) */
/* the distortion is a correction to the pixel */
void load_peak_correction_maps(char *filename)
{
	int		i, j;
	#warning "this routine only works for a 2084x2084 pixel detetor"
	if (verbose > 0) printf("\nloading distortion maps from: %s",filename);
	fflush(stdout);

	float * buffer;
	buffer = (float*)malloc (4 * 2084*2084 * 2);
	if (!buffer) {fprintf(stderr,"\nCould not allocate buffer %ld bytes in load_peak_correction_maps()\n",4*2084*2084*2); exit(1);}

	distortion_map_i = fmatrix_alloc (2084, 2084);
	distortion_map_j = fmatrix_alloc (2084, 2084);

	FILE *readfile;
	readfile = fopen(filename.c_str(), "r");
	if (!readfile) {
		error("\nCannot read peak correction file");
		exit(1);
	}
	fread(buffer, 4, 2084*2084 * 2, readfile);
		if (verbose > 1) { printf("\n   read the distortion maps"); fflush(stdout); }
	fclose(readfile);

	byteSwapArray(4, 2084*2084 * 2, buffer);
	#ifdef __BIG_ENDIAN__
		if (verbose > 2) { printf("\n   completed a little-endian -> big-endian byte swap (this computer should have a Motorola processor)"); fflush(stdout); }
	#endif 

	for (j = 0; j < 2084; j++) {
		for (i = 0; i < 2084; i++) {
			fmatrix_set(distortion_map_i, i, j, buffer[j*2084 + i]);
			fmatrix_set(distortion_map_j, i, j, buffer[2084*2084 + j*2084 + i]);
		}
	}
	if (verbose > 2) printf("\n   transfered distortion map to a matrix");
	free (buffer);
}
#endif
