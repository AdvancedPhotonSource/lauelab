/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#include "peaksearch.h"

#include <assert.h>
#include <math.h>

#define isNotNAN(A) ( (A) == (A) )

List*	get_blob_list(Grid* image, Grid* bitmap, int *status);
int	get_blob_points(Grid* image, Grid* bitmap, List* list, int x, int y);
bool	peakQualify(double fitX,double fitY,double centX,double centY,double widthx, double widthy, double chisq,double tilt, Genfileinf *ginf);
int	peakIntegral(Grid *image,double fitX,double fitY,Genfileinf *ginf,double *integr);
int	peakCorrection(double *fitX, double *fitY,Genfileinf *ginf);
double	biggestIntensity(List *peaks, double maxIntens, double *x, double *y);
void	removeSmallerNearbyPeaks(List *peaks, double px, double py, double pIntens, int minSeparation);
static void deletePointList(List *points);
static void deletePointValue(void *value);


#ifdef DEBUG
char	qualifyStr[2048];						/* stores reason peakQualify() rejected peak */
#endif



//	GridB* mask = gridB_new(imageRaw->width, imageRaw->height);		/* make the mask, this also sets all values to 0 */
//
List* boxsearch(
Grid*	imageRaw,			/* image to search on */
GridB*	mask,				/* mask for image */
int		boxsize,			/* use a box of size [2*boxsize+1][2*boxsize+1] */
// int		min_size,		/* minimum size in both x and y for valid blob */
// bool	maxima_search,		/* for big blobs do a bit of smoothing first */
long	NpeakMax,			/* maximum number of peaks to examine */
bool	smooth,				/* if true fit Lorentzian to smoothed image, otherwise use raw image */
Genfileinf *ginf)			/* general parameters */
{
	List * peaks = list_new();				/* for holding peak list */
	double	xoff=ginf->xoff, yoff=ginf->yoff;/* information from Genfileinf for doing peak fitting,e.g. fitToFunction */
	int		x, y;							/* generic pixel position */
	GridStats s;							/* stats structure for an image */
	int		width=imageRaw->width, height=imageRaw->height;
	long	Npts = width * height;
	int		x1, x2, y1, y2;					/* box for image_roi, used to process one blob */
	double	xcom,ycom;
	double	x0, y0;
	double	intens;
	long	ipeak, i;
	double	z0;								/* vertical offset of Lorentzian, background */
	double	A;								/* amplitude of Lorentzian */
	double	tilt;							/* angle of ellipsoidal Lorentzian peak */
	double	hwhmX, hwhmY;					/* hwhm of Lorentzian */
	double	chisq=0.;
	double	value;
	boxsize = min(boxsize,(min(width,height)-1)/2);		/* boxsize has to fit in the image */
	long	Nroi = (2*boxsize+1)*(2*boxsize+1);			/* number of pixels in a full ROI */
	#ifdef DEBUG
	double	testXY[NpeakMax][4];						/* store (x,y,intens,used) of tested peaks */
	#endif
	Grid *imageMedian = grid_new_copy(imageRaw);		/* create a new duplicate of raw image for smoothing */
	grid_smooth_median(imageMedian, 1);					/* median smooth, uses a 3x3 box to get rid of isolated noise spikes */

	for(ipeak=i=0; i<NpeakMax && gridB_get_total(mask)<Npts; i++) {
		s = grid_get_stats(imageMedian,mask);
		x = s.xMax;
		y = s.yMax;
		if ((x+y) != (x+y)) break;
		intens = s.valMax;
		#ifdef DEBUG
			// printf("testing pixel(%d,  %d) = %g\n",x,y,intens);
			testXY[i][0] = x;
			testXY[i][1] = y;
			testXY[i][2] = intens;
			testXY[i][3] = 0;
		#endif

		x1 = limit(x-boxsize,0,width-boxsize/2);		/* limit rectangular region to lie wholly within image */
		x2 = limit(x+boxsize,0,width-boxsize/2);
		y1 = limit(y-boxsize,0,height-boxsize/2);
		y2 = limit(y+boxsize,0,height-boxsize/2);

/*		if(intens<0.1 || x<0 || y>=width || y<0 || y>=height) continue;	// make sure the peak is in the image */

		GridB* mask_roi = gridB_new_copy_region(mask,x1,y1,x2,y2);
		gridB_set_region(mask, x1, y1, x2, y2, true);	/* set pixels of the roi to "used" in main mask */
		if (gridB_get_total(mask_roi)>(Nroi/2)) {		/* not enough pixels, skip this peak */
			#ifdef DEBUG
			printf("skip %ld \tonly %ld out of %ld pixels are available in this roi\n",i,Nroi-gridB_get_total(mask_roi),Nroi);
			#endif
			grid_delete((void *)mask_roi);				/* a bit of clean up */
			continue;
		}
		Grid* image_roi = grid_new_copy_region(imageMedian,x1,y1,x2,y2);

		/* set starting point of Lorentzian fit, initial guesses, not that width means hwhm */
		s = grid_get_stats(image_roi,mask_roi);
		x0 = s.xMax;									/* start center */
		y0 = s.yMax;
		xcom = s.xCOM;									/* use COM to compute hwhm */
		ycom = s.yCOM;
		hwhmX = limit(2*fabs(x0-xcom),6.0,boxsize)/2.0;
		hwhmY = limit(2*fabs(y0-ycom),6.0,boxsize)/2.0;
		z0 = s.average;									/* background */
		A = s.valMax - z0;								/* amplitude */
		tilt = 0.17;									/* about 10° (this works better than 1°) */
		/* printf("dx = %g,   dy = %g\n",xcom-x0,ycom-y0); */

		/* fit to the raw or smoothed image (not median smoothed) */
		Grid* image_roi_fit;
		if (smooth) image_roi_fit = grid_new_copy_region(imageMedian,x1,y1,x2,y2);
		else image_roi_fit = grid_new_copy_region(imageRaw,x1,y1,x2,y2);
		grid_set_masked_val(image_roi_fit,mask_roi,NAN);	/* 'disable' points NOT in ROI */

		if (ginf->peakShape == 1)		fitToFunctionGauss(image_roi_fit, &x0,&y0, &z0,&A, &hwhmX,&hwhmY, &tilt,&chisq);
		else if(ginf->peakShape == 0)	fitToFunctionLorentz(image_roi_fit, &x0,&y0, &z0,&A, &hwhmX,&hwhmY, &tilt,&chisq);
		else {
#ifdef LIBLAUE_BUILD
			grid_delete(imageMedian);
			return peaks;
#else
			fprintf(stderr,"ERROR -- in boxsearch(), ginf->peakShape = %d, it must be 0 or 1\n",ginf->peakShape);
			exit(1);
#endif
		}
//		value = grid_get_value(image_roi_fit,(int)round(x0),(int)round(y0));
		if (x0<0 || x0>=image_roi_fit->width || y0<0 || y0>=image_roi_fit->height || !(x0==x0) || !(y0==y0)) value = NAN;
		else value = grid_get_value(image_roi_fit,(int)round(x0),(int)round(y0));
		hwhmX = fabs(hwhmX);							/* +/- are the same thing */
		hwhmY = fabs(hwhmY);
		x0 += x1 + xoff + 1.;							/* translate from small roi to full image */
		y0 += y1 + yoff + 1.;							/* NOTE, these are 1 based pixels, remember to write as 0 based */
		xcom += x1 + xoff + 1.;
		ycom += y1 + yoff + 1.;


		if(peakQualify(x0,y0,xcom,ycom,hwhmX,hwhmY,chisq,tilt,ginf) && (value==value)) {
			double integr;
			if (peakIntegral(imageRaw,x0,y0,ginf,&integr)) integr = NAN;
			//printf("beforeCorrection : intens,x0,y0: %f %f %f \n",intens,x0,y0);
			peakCorrection(&x0,&y0,ginf);
			//printf("afterCorrection : intens,x0,y0,: %f %f %f \n",intens,x0,y0);
			list_append(peaks, peak_new_initialized(xcom,ycom,intens,x0,y0,
				A,z0,hwhmX,hwhmY,tilt,integr,boxsize,chisq));
			#ifdef DEBUG
				testXY[i][3] = 1;
			#endif
		}
		#ifdef DEBUG
		else printf("skip %ld \t%s",i,qualifyStr);
		#endif

//		roi_oneImage = mask_roi ? NaN :  roi_oneImage	// remove previously used pixels
//		roi_oneImageRaw[0,ix][0,iy] = imageRaw[p+x1][q+y1]
//		roi_oneImageRaw = mask_roi ? NaN :  roi_oneImageRaw

		grid_delete(image_roi);							/* clean up the small regions made in this loop */
		grid_delete(image_roi_fit);
		grid_delete((void *)mask_roi);
	}

	#ifdef DEBUG
		long imax=i;
		printf("\n  i\t\t testX\t\t\ttestY\t\t\tIntens\t\tused\n");
		for(i=0; i<imax; i+=1) {
			printf("% 3ld\t\t%7.2f\t\t%7.2f\t\t%5.0f",i,testXY[i][0],testXY[i][1],testXY[i][2]);
			if (testXY[i][3] > 0.) printf("\t\tX");
			printf("\n");
		}
	#endif
	return peaks;
}







/*
input:
	for a list of peaks, make sure that they are at least 2*boxSize distance between peaks
	list of peaks from processBlobs()
output:
	return the list of Peaks after removing ones that are too close together
	remove in order of descending intensity (so that the strongest peaks are kept)
 */
List * removeNearbyPeaks(
List	*peaks,											/* list of peaks (position & intensities */
int		minSeparation)									/* min distance between two peaks */
{
	double	x=1,y=1;									/* position of intensity, initial values are ignored */
	double intensity=INFINITY;							/* current peak intensity */

	if (minSeparation<1) return peaks;
	while(intensity > -INFINITY) {
		intensity = biggestIntensity(peaks,intensity,&x,&y);
		removeSmallerNearbyPeaks(peaks,x,y,intensity,minSeparation);	/* remove smaller peaks that are close to (px,py) */
	}
	return peaks;
}


/* find peak with biggest intensity less than maxIntens, retruns the intensity and sets (x,y) */
double biggestIntensity(
List	*peaks,											/* list of peaks (position & intensities */
double	maxIntens,										/* only consider intensities < maxIntens */
double	*x,												/* position of the max, set when exits */
double	*y)
{
	double	intens;										/* result, max intenstity less than maxIntens */
	double	peaki;
	intens = -INFINITY;									/* start with small value */
	*x = *y = NAN;										/* init to bad values */

	ListNode* peak=peaks->head;
	while(peak != EMPTY_NODE) {
		peaki = ((Peak*)peak->value)->intens;
		if (peaki>intens && peaki<maxIntens) {
			intens = peaki;
			*x = ((Peak*)peak->value)->fitX;
			*y = ((Peak*)peak->value)->fitY;
		}
		peak = peak->next;
	}
	return intens;
}


/*
input:
	for a list of peaks, make sure that they are at least 2*boxSize distance between peaks
	list of peaks from processBlobs()
	minSeparation, min separation between peaks
	px,py	position of peak that we are keeping (the test peak)
	pIntens, intensity of peak at (px,py)
output:
	return the list of Peaks after removing ones that are too close together

	remove all peaks less then intens that are within minSeparation of (px,py)
 */
void removeSmallerNearbyPeaks(
List	*peaks,					/* list of peaks (position & intensities */
double	px,						/* position of test peak */
double	py,
double	pIntens,				/* only consider peaks <= pIntens */
int		minSeparation)			/* min distance between two peaks is 2*boxsize */
{
	double	x,y;										/* postion of a peak */
	double	intens;										/* intensity of a peak */
	ListNode *peak;
	ListNode *nextPeak;

	peak = peaks->head;
	while(peak != EMPTY_NODE) {
		nextPeak = peak->next;							/* save next peak before I change anything */
		intens = ((Peak*)peak->value)->intens;
		x = ((Peak*)peak->value)->fitX;
		y = ((Peak*)peak->value)->fitY;

		if (fabs(px-x)<minSeparation && fabs(py-y)<minSeparation && intens<pIntens) {
			peak_delete((Peak*)(peak->value));			/* free contents of peak we do not want */
			peak->value = NULL;
			list_remove(peaks,peak);					/* remove (now empty) peak from list */
		}
		peak = nextPeak;
	}
}



/*
input:
	blobs: the result list after blobsearch,i.e.,a list of Points
	wimage: the original image data
	boxsize: user input for doing fitting
output:
	return the list of Peaks after been processed/fitted
 */
List * processBlobs(
List *blobs,					/* list of points where peaks are to be found */
WinViewImage *wimage,			/* input image */
Genfileinf *ginf,				/* general parameters */
int		NpeakMax,				/* maximum allowed number of peaks */
double	background,				/* image-wide fit background */
int		*status)
{
	List * peaks = list_new();						/* for holding peak list */
	double	xoff=ginf->xoff, yoff=ginf->yoff;
	if (status) *status = LMSDER_SUCCESS;
	if (!peaks) { if (status) *status = LMSDER_ENOMEM; return NULL; }		/* information from Genfileinf for doing peak fitting,e.g. fitToFunction */
	int		boxsize = ginf->boxsize;				/* local copy of boxsize */
	int		x1, x2, y1, y2;							/* box for image_roi, used to process one blob */
	int		width, height;							/* size of image (pixels) */
	double	x,y, intens;							/* center and intensity of one blob */

	Grid* image=wimage->data;						/* actual image values from wimage */
	width = image->width;
	height = image->height;
	NpeakMax = NpeakMax<=0 ? INT_MAX : NpeakMax;	/* for negative NpeakMax, allow no limit on number of spots */

	/* process each point in the blob list */
	ListNode* blob=blobs->head;
	while(blob !=EMPTY_NODE && (peaks->size < NpeakMax)) {
		x = ((Point*)blob->value)->x;
		y = ((Point*)blob->value)->y;
		intens = ((Point*)blob->value)->value;

		/* make sure the peak is in the image */
		if(intens>0.1 && x>0. && x<width && y>0. && y<height) {
			x1=round(round(x)-boxsize);				/* range of rectangular region */
			x2=round(round(x)+boxsize);
			y1=round(round(y)-boxsize);
			y2=round(round(y)+boxsize);

			x1=max(x1,0);							/* limit rectangular region to lay wholly within image */
/*			x1=min(x1,width-boxsize/2); */
			x1=min(x1,width-1);
			x2=max(x2,0);
/*			x2=min(x2,width-boxsize/2); */
			x2=min(x2,width-1);
			y1=max(y1,0);
/*			y1=min(y1,height-boxsize/2); */
			y1=min(y1,height-1);
			y2=max(y2,0);
/*			y2=min(y2,height-boxsize/2); */
			y2=min(y2,height-1);

			Grid* image_roi = grid_new_copy_region(image,x1,y1,x2,y2);
			if (!image_roi) goto allocation_error;
			if(image_roi->width >= boxsize/2 && image_roi->height >= boxsize/2) {
				Point *cent=centroid(image_roi,x1,y1);
				double centX, centY;
				int fit_status;
				if (!cent) { grid_delete(image_roi); goto allocation_error; }
				centX = cent->x + xoff + 1.;
				centY = cent->y + yoff + 1.;

				/* set starting point of fit, initial guesses, width is hwhm */
				double widthx=ginf->widthx, widthy=ginf->widthy, tilt=ginf->tilt;
				double fitX, fitY,fitIntens,fitBackground=background,chisq=0.;
				fitX=round((x2-x1)/2.);
				fitY=round((y2-y1)/2.);
				fitIntens=intens;
				if (ginf->peakShape == 1)		fit_status = fitToFunctionGauss(image_roi,&fitX,&fitY,&fitBackground, &fitIntens,&widthx, &widthy,&tilt,&chisq);
				else if(ginf->peakShape == 0)	fit_status = fitToFunctionLorentz(image_roi,&fitX,&fitY,&fitBackground, &fitIntens,&widthx, &widthy,&tilt,&chisq);
				else fit_status = 1;
				point_delete(cent);
				if (fit_status) {
					grid_delete(image_roi);
					if (status) *status = fit_status;
					return peaks;
				}
				fitX += x1 + xoff + 1.;				/* translate from small roi to full image */
				fitY += y1 + yoff +1.;				/* NOTE, these are 1 based pixels, remember to write as 0 based */

				if (peakCorrection(&fitX,&fitY,ginf)) { grid_delete(image_roi); goto allocation_error; }

				if(peakQualify(fitX,fitY,centX,centY,widthx,widthy,chisq,tilt,ginf)) {
					double integr;
					Peak *peak;
					if (peakIntegral(image,fitX,fitY,ginf,&integr)) { grid_delete(image_roi); goto allocation_error; }
					peak = peak_new_initialized(centX,centY,intens,fitX,fitY,
						fitIntens,fitBackground,widthx,widthy,tilt,integr,boxsize,chisq);
					if (!peak || list_append(peaks, peak)) {
						peak_delete(peak);
						grid_delete(image_roi);
						goto allocation_error;
					}
				} /* end if(peakQualify...) */
				#ifdef DEBUG
				// else printf("skip %ld \t%s",i,qualifyStr);
				else printf("skip \t%s",qualifyStr);
				#endif
			} /* end if(image_roi...) */
			grid_delete(image_roi);
		} /* end if(intens...) */

		blob = blob->next;
	} /* end while(blob...) */
	return peaks;

allocation_error:
	if (status) *status = LMSDER_ENOMEM;
	return peaks;
}


void	savePeaks(
List	*peaks,					/* list of peak positions the output */
char	*filename,				/* name of output file */
WinViewHeader* header,			/* header values from image file */
char	*inFileName,			/* name of file with input image */
Genfileinf *ginf,				/* general parameters */
double	threshold,				/* the threshold that was used to find peaks */
double	seconds,				/* execution time (sec) */
int		minSeparation,			/* minimum separation between any two peaks (default is 2*boxsize) */
bool	smooth,					/* if true fit Lorentzian to smoothed image, otherwise use raw image */
struct ExtraOutput_Header *exH,	/* some extra output that JZT added */
char	*pgm)					/* name of this program */
{
	ListNode* peak = peaks->head;
	int numPeaks = peaks->size;		/* number of fitted peaks */
	int i;
	char peakShape[1024];
	FILE *output;

	if ( !(output=fopen(filename,"w")) ) {
		fprintf(stderr,"Error: Can not open file %s to write\n",filename);
		exit(1);
	}
	if (ginf->peakShape == 0)		strcpy(peakShape,"Lorentzian");
	else if (ginf->peakShape == 1)	strcpy(peakShape,"Gaussian");
	else							sprintf(peakShape,"Unknown=%d\n",ginf->peakShape);

	/* write the header part of the file */
	fprintf(output,"$filetype		PixelPeakList\n");
	fprintf(output,"$inputImage		%s\n",inFileName);
	fprintf(output,"$xdim			%lu		// number of binned pixels along X\n",header->xdim);
	fprintf(output,"$ydim			%lu		// number of binned pixels along Y\n",header->ydim);
	fprintf(output,"$xDimDet		%lu		// total number of un-binned pixels in detector along X\n",header->xDimDet);
	fprintf(output,"$yDimDet		%lu		// total number of un-binned pixels in detector along Y\n",header->yDimDet);
	fprintf(output,"$startx			%lu			// starting X of ROI (un-binned pixels)\n",header->startx-1);
	fprintf(output,"$endx			%lu		// last X of ROI (un-binned pixels)\n",header->endx-1);
	fprintf(output,"$groupx			%lu			// binning along X for the ROI (un-binned pixels)\n",header->groupx);
	fprintf(output,"$starty			%lu			// starting Y of ROI (un-binned pixels)\n",header->starty-1);
	fprintf(output,"$endy			%lu		// last Y of ROI (un-binned pixels)\n",header->endy-1);
	fprintf(output,"$groupy			%lu			// binning along Y for the ROI (un-binned pixels)\n",header->groupy);
	fprintf(output,"$exposure		%g			// exposure time (sec)\n",header->exposure);
	fprintf(output,"$CCDshutterIN	%d			// CCD shutter, 1=IN, 0=OUT\n",exH->CCDshutterIN);
	if (isNotNAN(header->xSample)) fprintf(output,"$Xsample		%g		// sample position (micron)\n",header->xSample);
	if (isNotNAN(header->ySample)) fprintf(output,"$Ysample		%g\n",header->ySample);
	if (isNotNAN(header->zSample)) fprintf(output,"$Zsample		%g\n",header->zSample);
	if (isNotNAN(header->CCDy)) fprintf(output,"$CCDy			%g		// CCD positioner height (mm)\n",(header->CCDy)/1000.);
	if (isNotNAN(exH->depth)) fprintf(output,"$depth			%g			// depth for depth resolved images (micron)\n",exH->depth);
	if (exH->scanNum>=0) fprintf(output,"$scanNum		%ld		// scan number\n",exH->scanNum);
	if (exH->beamBad>=0) fprintf(output,"$beamBad		%d			// beam bad flag (TRUE==bad)\n",exH->beamBad);
	if (exH->lightOn>=0) fprintf(output,"$lightOn		%d			// flag, TRUE=illuminator ON\n",exH->lightOn);
	if (isNotNAN(exH->energy)) fprintf(output,"$energy			%g		// monochromator energy (keV)\n",exH->energy);
	if (isNotNAN(exH->hutchTemperature)) fprintf(output,"$hutchTemperature	%g	// hutch temperature (C)\n",exH->hutchTemperature);
	if (isNotNAN(exH->sampleDistance)) fprintf(output,"$sampleDistance	%g	// Keyence measure of sample posiiton (micron)\n",exH->sampleDistance);
	if (strlen(exH->monoMode)) fprintf(output,"$monoMode		%s	// monochromator mode or position\n",exH->monoMode);
	if (strlen(exH->dateExposed)) fprintf(output,"$dateExposed	%s\n",exH->dateExposed);
	if (strlen(exH->userName)) fprintf(output,"$userName		%s\n",exH->userName);
	if (strlen(exH->title)) fprintf(output,"$title			%s\n",exH->title);
	if (strlen(exH->sampleName)) fprintf(output,"$sampleName		%s\n",exH->sampleName);
	if (strlen(exH->beamline)) fprintf(output,"$beamline		%s\n",exH->beamline);
	if (strlen(exH->detector_ID)) fprintf(output,"$detector_ID	%s\n",exH->detector_ID);

	fprintf(output,"//\n");
	fprintf(output,"$boxsize		%d			// box size used for peak fitting\n",ginf->boxsize);
	fprintf(output,"$minwidth		%g		// min allowed width of a peak\n",ginf->minwidth);
	fprintf(output,"$maxwidth		%g			// max allowed width of a peak\n",ginf->maxwidth);
	fprintf(output,"$maxCentToFit	%g			// max diff between initial & fitted peak position\n",ginf->maxCentToFit);
	fprintf(output,"$maxRfactor		%g			// max allowed R-factor\n",ginf->maxRfactor);
	if (strlen(ginf->CCDFilename)) fprintf(output,"$CCD_DistortionFile	%s\n",ginf->CCDFilename);
	if ((exH->maskFile)[0]) fprintf(output,"$maskFile		%s\n",exH->maskFile);
	fprintf(output,"$threshold		%g		// threshold for blob searching\n",threshold);
	fprintf(output,"$minSeparation	%d			// minimum separation between any two peaks\n",minSeparation);
	fprintf(output,"$smooth			%d			// fit to smoothed image\n",smooth);
	fprintf(output,"$peakShape		%s	// shape for peak fit\n",peakShape);
	fprintf(output,"$totalSum		%g		// sum of all pixels in image\n",exH->sum);
	fprintf(output,"$sumAboveThreshold	%g	// sum of all pixels in image above threshold\n",exH->sumAboveThreshold);
	fprintf(output,"$numAboveThreshold	%lu	// number of pixels above threshold\n",exH->numAboveThreshold);
	if (exH->NpeakMax > 0) fprintf(output,"$NpeakMax		%d			// limit on number of peaks to search for\n",exH->NpeakMax);
	if (strlen(pgm)) fprintf(output,"$programName	%s\n",pgm);
	fprintf(output,"$executionTime	%.2f		// execution time (sec)\n",seconds);
	fprintf(output,"//\n");
	fprintf(output,"// fitted peak positions relative to the start of the ROI (not detector origin)\n");
	fprintf(output,"//    peak positions are in zero based binned pixels\n");
	fprintf(output,"$Npeaks		%d				// number of fitted peaks in following table\n",numPeaks);
//	fprintf(output,"$peakList	5 %d			// fitX fitY intens integral\n",numPeaks);
//	fprintf(output,"$peakList	4 %d			// fitX fitY intens integral\n",numPeaks);
	fprintf(output,"$peakList	8 %d			// fitX fitY intens integral hwhmX hwhmY tilt chisq\n",numPeaks);

	/* write the list of fitted peak positions */
	for(i=0;i<numPeaks;i++) {
		fprintf(output,"%13.3f%13.3f%16.4f%16.5f%11.3f%11.3f%11.4f   %.5g\n", ((Peak*)peak->value)->fitX-1,((Peak*)peak->value)->fitY-1,
			((Peak*)peak->value)->intens,((Peak*)peak->value)->integrIntens,((Peak*)peak->value)->fitPeakWidthX,
			((Peak*)peak->value)->fitPeakWidthY,((Peak*)peak->value)->fitTilt,((Peak*)peak->value)->chisq);
		peak = peak->next;
	}

//	fprintf(output,"$peakList	5 %d			// fitX fitY intens integral boxSize \n",numPeaks);
//	for(i=0;i<numPeaks;i++) {
//		fprintf(output,"%13.3f%13.3f%16.4f%16.5f%8d\n", ((Peak*)peak->value)->fitX-1,((Peak*)peak->value)->fitY-1,
//			((Peak*)peak->value)->intens,((Peak*)peak->value)->integrIntens,((Peak*)peak->value)->boxsize);
//		peak = peak->next;
//	}

	fclose(output);
}

void savePeaksIDL(List *peaks, char *filename) {
	FILE *output = fopen(filename,"w");
	if(output == NULL) {
		fprintf(stderr,"Error: Can not open file %s to write\n",filename);
		exit(1);
	}

	int numPeaks = peaks->size;
	printf("numPeaks=%d\n",numPeaks);
	fprintf(output,"%8d\n",numPeaks);

	ListNode* peak = peaks->head;

	int i;
	for(i=0;i<numPeaks;i++) {
	fprintf(output,"%13.3f%13.3f%16.4f%16.5f%8d\n", ((Peak*)peak->value)->fitX,((Peak*)peak->value)->fitY,
		((Peak*)peak->value)->intens,((Peak*)peak->value)->integrIntens,((Peak*)peak->value)->boxsize);
	peak = peak->next;
	}
	fclose(output);
}





/* returns the net peak integral for a region around a peak */
int peakIntegral(
Grid	*image,			/* image to use */
double	fitX,			/* peak posiiton */
double	fitY,
Genfileinf *ginf,
double	*integr) {		/* general parameters about the image */
	*integr=0.;

	int boxsize = ginf->boxsize;
	int xoff = (int)ginf->xoff;
	int yoff = (int)ginf->yoff;
	int xdim = image->width;
	int ydim = image->height;

	int xpeak = round(fitX);
	int ypeak = round(fitY);
	int xlow = xpeak - boxsize;
	int xmax = xpeak + boxsize;
	int ylow = ypeak - boxsize;
	int ymax = ypeak + boxsize;

	xlow = max(xlow,1+xoff);
	xlow = min(xlow, xdim + xoff);
	xmax = max(xmax,1+xoff);
	xmax = min(xmax, xdim + xoff);

	ylow = max(ylow,1+yoff);
	ylow = min(ylow, ydim+yoff);
	ymax = max(ymax,1+yoff);
	ymax = min(ymax, ydim+yoff);

	if(xlow < xmax-2 && ylow < ymax-2) {
		Grid * image_roi = grid_new_copy_region(image,round(xlow-1-xoff),round(ylow-1-yoff),round(xmax-1-xoff),round(ymax-1-yoff));
		Grid * image_roi_b = grid_new_copy_region(image,round(xlow-xoff),round(ylow-yoff),round(xmax-2-xoff),round(ymax-2-yoff));
		if (!image_roi || !image_roi_b) { grid_delete(image_roi); grid_delete(image_roi_b); return 1; }

		double roi_total = grid_get_total(image_roi);
		int xxdim = image_roi->width;
		int yydim = image_roi->height;
		int n_roi = xxdim*yydim;
		//double roi_b_total = grid_get_total(image_roi_b);
		int n_roi_b = image_roi_b->width * image_roi_b->height;

		double *edgebackground = malloc((n_roi - n_roi_b)* sizeof(double));
		if (!edgebackground) { grid_delete(image_roi); grid_delete(image_roi_b); return 1; }

		/* first row of image_roi */
		int i;
		for(i=0; i<xxdim;i++)
			edgebackground[i]=grid_get_value(image_roi,i,0);

		/* last row of image_roi */
		for(i=xxdim; i<2*xxdim;i++)
			edgebackground[i]=grid_get_value(image_roi,i-xxdim,yydim-1);

		/* first column of image_roi except first and last row */
		for(i=2*xxdim; i<2*xxdim+yydim-2;i++)
			edgebackground[i]=grid_get_value(image_roi,0,i-2*xxdim +1);

		for(i=2*xxdim+yydim-2; i<n_roi-n_roi_b;i++)
			edgebackground[i]=grid_get_value(image_roi,xxdim-1,i-(2*xxdim+yydim-2)+1);


		double background = median(edgebackground, n_roi-n_roi_b);

		*integr = (roi_total - background * n_roi)/1000.;

		free(edgebackground);
		grid_delete(image_roi);
		grid_delete(image_roi_b);
	}
	return 0;
}
/*****************************************************************************/


int peakCorrection(double *fitx,double *fity,Genfileinf *ginf) {
	if (strlen(ginf->CCDFilename) < 1) return 0;	/* no distortion file, so nothing to do */

	Calibparam *cp=default_calibparam(); /* need to be free */
	if (!cp) return 1;
	float dpsx = cp->dpsx;
	// float dpsy = cp->dpsy;
	int xdim = cp->xdim;
	// int ydim = cp->ydim;
	int binFactor = round (dpsx *1000.0/xdim/24.);

	delete_calibparam(cp);

	CCDTable *ct = loadCCDTable(ginf->CCDFilename); /* need to be free */
	if (!ct) return 1;

	float *cornerxy=malloc(4*sizeof(float));
	if (!cornerxy) { ccdTable_delete(ct); return 1; }

	switch (ct->cornerx0) {
		case 1:cornerxy[0]=ccdTable_getValue(ct,0,0,0);
			break;
		case 2:cornerxy[0]=ccdTable_getValue(ct,ct->nx-1,0,0);
			break;
		case 3:cornerxy[0]=ccdTable_getValue(ct,0,ct->ny-1,0);
			break;
		case 4:cornerxy[0]=ccdTable_getValue(ct,ct->nx-1,ct->ny-1,0);
	}

	switch(ct->cornery0) {
		case 1:cornerxy[1]=ccdTable_getValue(ct,0,0,1);
			break;
		case 2:cornerxy[1]=ccdTable_getValue(ct,ct->nx-1,0,1);
			break;
		case 3:cornerxy[1]=ccdTable_getValue(ct,0,ct->ny-1,1);
			break;
		case 4:cornerxy[1]=ccdTable_getValue(ct,ct->nx-1,ct->ny-1,1);
	}

	switch(ct->cornerx1) {
		case 1:cornerxy[2]=ccdTable_getValue(ct,0,0,0);
			break;
		case 2:cornerxy[2]=ccdTable_getValue(ct,ct->nx-1,0,0);
			break;
		case 3:cornerxy[2]=ccdTable_getValue(ct,0,ct->ny-1,0);
			break;
		case 4:cornerxy[2]=ccdTable_getValue(ct,ct->nx-1,ct->ny-1,0);
	}

	switch(ct->cornery1) {
		case 1:cornerxy[3]=ccdTable_getValue(ct,0,0,1);
			break;
		case 2:cornerxy[3]=ccdTable_getValue(ct,ct->nx-1,0,1);
			break;
		case 3:cornerxy[3]=ccdTable_getValue(ct,0,ct->ny-1,1);
			break;
		case 4:cornerxy[3]=ccdTable_getValue(ct,ct->nx-1,ct->ny-1,1);
	}

	float x0=ccdTable_getValue(ct,0,0,0);
	float y0=ccdTable_getValue(ct,0,0,1);
	float xxstep = (ccdTable_getValue(ct,ct->nx-1,0,0) - x0)/(ct->nx-1);
	float xystep = (ccdTable_getValue(ct,ct->nx-1,0,1) - y0)/(ct->nx-1);
	float yxstep = (ccdTable_getValue(ct,0,ct->ny-1,0) - x0)/(ct->ny-1);
	float yystep = (ccdTable_getValue(ct,0,ct->ny-1,1) - y0)/(ct->ny-1);

	float xoff = ginf->xoff;
	float yoff = ginf->yoff;
	float fitx0 = (*fitx - xoff)*(float)(binFactor) + xoff;
	float fity0 = (*fity - yoff)*(float)(binFactor) + yoff;
	int xcent = (round) (((fitx0 - x0)*yystep - (fity0 - y0)*yxstep)/(xxstep*yystep - xystep*yxstep));
	int ycent = (round) (((fitx0 - x0)*xystep - (fity0 - y0)*xxstep)/(xystep*yxstep - yystep*xxstep));

	xcent = (xcent>0)?xcent:0;
	xcent = (xcent<(ct->nx-1))?xcent:(ct->nx-1);

	ycent = (ycent>0)?ycent:0;
	ycent = (ycent<(ct->ny-1))?ycent:(ct->ny-1);

	float deltax, deltay;
	float cent0 = ccdTable_getValue(ct,xcent,ycent,0);
	float cent1 = ccdTable_getValue(ct,xcent,ycent,1);
	float cent2 = ccdTable_getValue(ct,xcent,ycent,2);
	float cent3 = ccdTable_getValue(ct,xcent,ycent,3);

	if(fitx0 <= cornerxy[0] || fitx0 >= cornerxy[2] || fity0 < cornerxy[1] || fity0 >= cornerxy[3]) {
		deltax = cent2/(float)(binFactor);
		deltay = cent2/(float)(binFactor);
	}
	else {
		float xcent2=0., ycent2=0.;
		if(fitx0 >= cent0 && fity0 >= cent1) {
			xcent2 = xcent+1;
			ycent2 = ycent+1;
		}
		if(fitx0 >= cent0 && fity0 < cent1) {
			xcent2 = xcent+1;
			ycent2 = ycent-1;
		}
		if(fitx0 < cent0 && fity0 >= cent1) {
			xcent2 = xcent-1;
			ycent2 = ycent+1;
		}
		if(fitx0 < cent0 && fity0 < cent1) {
			xcent2 = xcent-1;
			ycent2 = ycent-1;
		}

		xcent2 = (xcent2>0)?xcent2:0;
		xcent2 = (xcent2<(ct->nx-1))?xcent2:(ct->nx-1);

		ycent2 = (ycent2>0)?ycent2:0;
		ycent2 = (ycent2<(ct->ny-1))?ycent2:(ct->ny-1);

		float r1 = sqrt(pow (fitx0 - cent0,2) + pow (fity0 - cent1,2));
		float r2 = sqrt(pow (fitx0 - ccdTable_getValue(ct,xcent,ycent2,0),2) + pow (fity0 - ccdTable_getValue(ct,xcent,ycent2,1),2));
		float r3 = sqrt(pow (fitx0 - ccdTable_getValue(ct,xcent2,ycent,0),2) + pow (fity0 - ccdTable_getValue(ct,xcent2,ycent,1),2));
		float r4 = sqrt(pow (fitx0 - ccdTable_getValue(ct,xcent2,ycent2,0),2) + pow (fity0 - ccdTable_getValue(ct,xcent2,ycent2,1),2));
		deltax = (cent2/r1 + cent2/r2 + cent2/r3 + cent2/r4)/(1/r1 + 1/r2 +1/r3 +1/r4)/(float)(binFactor);
		deltay = (cent3/r1 + cent3/r2 + cent3/r3 + cent3/r4)/(1/r1 + 1/r2 +1/r3 +1/r4)/(float)(binFactor);
	}

	*fitx+=deltax;
	*fity+=deltay;
	free(cornerxy);
	ccdTable_delete(ct);
	return 0;
}


bool peakQualify(
double	fitx,			/* fitted peak position */
double	fity,
double	centx,			/* center of mass used as starting point for fit */
double	centy,
double	widthx,			/* fitted half-width of peak */
double	widthy,
double	chisq,			/* chi square of the fit */
double	tilt,			/* tilt (degree) */
Genfileinf *ginf)		/* general parameters, contains tolerances */
{
	double distance = sqrt((fitx-centx)*(fitx-centx) + (fity-centy)*(fity-centy));

	#ifdef DEBUG
	qualifyStr[0] = '\0';
	#endif

	if (!(fitx==fitx && fity==fity)) {
		#ifdef DEBUG
		sprintf(qualifyStr,"peak at (%.2f, %.2fg) bad result fit x/y = (%g, %g), width x/y = (%g, %g), tilt = %g, chisq=%g\n",centx,centy,fitx,fity,widthx,widthy,tilt,chisq);
		#endif
		return false;
	}
//	else if(widthx < ginf->minwidth || widthy < ginf->minwidth || widthx > ginf->maxwidth || widthy > ginf->maxwidth) {
	else if(widthx < ginf->minwidth/3.0 || widthy < ginf->minwidth/3.0 || widthx > ginf->maxwidth || widthy > ginf->maxwidth) {
		#ifdef DEBUG
			char	str[1024];
			if(widthx < ginf->minwidth/3.0) sprintf(str," widthx=%g < minwidth=%g\n",widthx,ginf->minwidth/3.0);
			else if(widthy < ginf->minwidth/3.0) sprintf(str," widthy=%g < minwidth=%g\n",widthy,ginf->minwidth/3.0);
			else if(widthx > ginf->maxwidth) sprintf(str," widthx=%g > maxwidth=%g\n",widthx,ginf->maxwidth);
			else if(widthy > ginf->maxwidth) sprintf(str," widthy=%g > maxwidth=%g\n",widthy,ginf->maxwidth);
			else sprintf(str," peak width does not pass criteria, don't include, widthx=%g, widthy=%g, allowed range is [%g, %g]\n",widthx,widthy,ginf->minwidth/2.5,ginf->maxwidth);
			sprintf(qualifyStr,"peak at (%.2f, %.2f) width does not pass criteria, don't include, %s",fitx,fity,str);
		#endif
		return false;
	}
	else if (distance > ginf->maxCentToFit) {
		#ifdef DEBUG
		sprintf(qualifyStr,"peak COM=(%.2f, %.2f) is too far from fitted position (%.2f, %.2f),  distance = %.2f\n",centx,centy,fitx,fity,distance);
		#endif
		return false;
	}
	else if(!(ginf->maxRfactor >= fabs(chisq))) {
		#ifdef DEBUG
		sprintf(qualifyStr,"peak's chisq = %.3f (bigger than %.3f), not added to list.\n",chisq,ginf->maxRfactor);
		#endif
		return false;
	}
	else if(!(fabs(tilt)<361.0)) {
		#ifdef DEBUG
		sprintf(qualifyStr,"peak's tilt = %.3f should be in range ±360 degree , not added to list.\n",tilt);
		#endif
		return false;
	}
	return true;
}


List* blobsearch(
Grid*	image,				/* image to search on */
double	threshold,			/* threshold used to identify a blob */
int		min_size,			/* minimum size in both x and y for valid blob */
bool	maxima_search,		/* for big blobs do a bit of smoothing first */
int		*status)
{

	Grid* bitmap_image = grid_new(image->width, image->height);
	List* all_maximas = NULL;
	List* blob_maximas;
	List* blobs = NULL;
	if (status) *status = 0;
	if (!bitmap_image) goto allocation_error;

	/* Build the bitmap directly in row-major order without copying the image first. */
	int x, y;
	for (y = 0; y < bitmap_image->height; y++) {
		for (x = 0; x < bitmap_image->width; x++) {
			int location = y * bitmap_image->width + x;
			bitmap_image->values[location] = image->values[location] < threshold ? 0.0 : 1.0;
		}
	}

	all_maximas = list_new();
	if (!all_maximas) goto allocation_error;

	blobs = get_blob_list(image, bitmap_image, status);
	if (!blobs || (status && *status)) goto allocation_error;
	// printf("number of blobs: %d\n",blobs->size);


	/*
	 * now we're going to loop over every point in every blob
	 * then we're going to operate on the blob's image section
	 */

	/* we want to calculate the min/max x/y and weighted average x/y for each blob */
	int xmin, xmax, ymin, ymax;
	double xtotal = 0.0, ytotal = 0.0;

	/* pointers for list of lists of points */
	ListNode* blob_node = blobs->head;

	List* point_list;
	/* pointers for list of points */
	ListNode* point_node;
	Point* point;

	/* for each blob in bloblist */
	while (blob_node != EMPTY_NODE) {

		xmin = INT_MAX; xmax = 0; ymin = INT_MAX; ymax = 0;

		/* value in this blob/node should be a list of points in the current blob */
		/* we loop over every point in this blob to calculate some data */
		point_list = blob_node->value;
		point_node = point_list->head;

		/* for each point in pointlist */
		while (point_node != EMPTY_NODE) {

			/* value in this point/node should be a point */
			point = point_node->value;

			/* compare these x/y values to the min/max values stored */
			xmin = min(xmin, point->x);
			ymin = min(ymin, point->y);
			xmax = max(xmax, point->x);
			ymax = max(ymax, point->y);

			xtotal += point->value * point->x;	/* ? This is for computing the center of mass, which is not being done here */
			ytotal += point->value * point->y;	/* ? */
			//	printf("%f %f %f\n",point->x, point->y,point->value);
			point_node = point_node->next;
		}


		/* if big enough spot */
		if ((xmax - xmin >= min_size) && (ymax - ymin >= min_size)) {

			/* grab a copy of the section of the image that contains the blob in question */
			Grid* image_roi = grid_new_copy_region(image, xmin, ymin, xmax, ymax);
			Point *center;
			if (!image_roi) goto allocation_error;

			int npix = 4;		/* 2*npix+1 is size of local area in which there can be only one maxima */

			/* find_maximas also blanks out 2*npix pixels on the borders (npix pixels on each side */
			/* so if the size of this region is less than 2*npix+1, we can't use our find_maximas function on it */
			if ( (xmax - xmin > 2*npix+1) && (ymax-ymin > 2*npix+1) && maxima_search) {
				if (grid_smooth_median(image_roi, 1) || grid_smooth_boxcar(image_roi, 1)) {
					grid_delete(image_roi);
					goto allocation_error;
				}
				center = centroid_2(image_roi, xmin, ymin);
			}
			else {
				/* too small an area for the find_maximas function */
				center = centroid(image_roi, xmin, ymin);
			}
			blob_maximas = list_new();
			if (!center || !blob_maximas || list_append(blob_maximas, center)) {
				point_delete(center);
				list_delete(blob_maximas);
				grid_delete(image_roi);
				goto allocation_error;
			}

			/* append the maximas in this list to the list of all maximas */
			list_concat( all_maximas, blob_maximas );
			list_delete( blob_maximas );
			grid_delete(image_roi);
		} /* if big enough spot */

		blob_node = blob_node->next;
	} /* Looping over all lists of blobs */

	blob_node = blobs->head;
	while (blob_node != EMPTY_NODE) {
		deletePointList(blob_node->value);
		blob_node = blob_node->next;
	}
	list_delete_nodes(blobs);
	grid_delete(bitmap_image);
	return all_maximas;

allocation_error:
	if (status) *status = 1;
	if (blobs) {
		blob_node = blobs->head;
		while (blob_node != EMPTY_NODE) {
			deletePointList(blob_node->value);
			blob_node = blob_node->next;
		}
		list_delete_nodes(blobs);
	}
	if (all_maximas) {
		ListNode *node = all_maximas->head;
		while (node != EMPTY_NODE) { point_delete(node->value); node = node->next; }
		list_delete_nodes(all_maximas);
	}
	grid_delete(bitmap_image);
	return NULL;
}

static void deletePointList(List *points)
{
	list_delete_with_values(points, deletePointValue);
}

static void deletePointValue(void *value)
{
	point_delete((Point *)value);
}

/*****************************************************************************/
List* get_blob_list(Grid* image, Grid* bitmap, int *status) {

	List* blob_list = list_new();
	List* blob_point_list;

	int x, y;
	if (!blob_list) { if (status) *status = 1; return NULL; }
	for (y = 0; y < image->height; y++) {
		for (x = 0; x < image->width; x++) {

			/* This is part of a blob */
			if( grid_get_value(bitmap, x, y) == 1) {

				blob_point_list = list_new();
				if (!blob_point_list || get_blob_points(image, bitmap, blob_point_list, x, y) ||
					list_append(blob_list, blob_point_list)) {
					deletePointList(blob_point_list);
					if (status) *status = 1;
					while (blob_list->head != EMPTY_NODE) {
						deletePointList(blob_list->head->value);
						list_remove(blob_list, blob_list->head);
					}
					list_delete(blob_list);
					return NULL;
				}
			}
		}
	}
	return blob_list;
}

/* given a starting point inside a blob, find all points in the blob */
int get_blob_points(Grid* image, Grid* bitmap, List* list, int x, int y) {
	const size_t capacity = (size_t)image->width * (size_t)image->height;
	size_t *stack = malloc(capacity * sizeof(*stack));
	size_t size = 0;
	int status = 0;

	if (!stack) return 1;
	stack[size++] = (size_t)y * (size_t)image->width + (size_t)x;
	grid_set_value(bitmap, x, y, 0);

	while (size > 0) {
		size_t index = stack[--size];
		int point_x = (int)(index % (size_t)image->width);
		int point_y = (int)(index / (size_t)image->width);
		Point *point = point_new_initialized(
			point_x, point_y, grid_get_value(image, point_x, point_y));
		int dx, dy;

		if (!point || list_append(list, point)) {
			point_delete(point);
			status = 1;
			break;
		}

		for (dx = -1; dx <= 1; dx++) {
			for (dy = -1; dy <= 1; dy++) {
				int neighbor_x = point_x + dx;
				int neighbor_y = point_y + dy;
				if (neighbor_x >= 0 && neighbor_x < image->width &&
					neighbor_y >= 0 && neighbor_y < image->height &&
					(dx != 0 || dy != 0) &&
					grid_get_value(bitmap, neighbor_x, neighbor_y) == 1) {
					stack[size++] = (size_t)neighbor_y * (size_t)image->width +
						(size_t)neighbor_x;
					grid_set_value(bitmap, neighbor_x, neighbor_y, 0);
				}
			}
		}
	}

	free(stack);
	return status;
}


/*****************************************************************************/

int compare_doubleReverse(double *a, double *b);		/* only called by sorListPoints */


/*
	When the threshold is too low, almost all of the time is spent in processBlobs() which processes each blob in blobs

	each blob is a Point structure that contains 3 numbers:
		x = ((Point*)blob->value)->x;				// the (x,y) location
		y = ((Point*)blob->value)->y;
		intens = ((Point*)blob->value)->value;		// the value at that (x,y) location

	So in an attempy to make processBlobs() more efficient, I first sort blobs so that the most intens peaks are first.
	Thus NpeakMax is more likely to processBlobs() without having to process all of the blobs, which is very slow.
*/

/* sorts a list of Points, so that the largest Point->value are first */
int sorListPoints(
List	*blobs)						/* a list of Points which gets resorted */
{
	size_t	sort_len = blobs->size;
	double *points;					/* holds (intens,x,y) to sort */
	if (!sort_len) return 0;
	points = (double*)calloc(3*sort_len,sizeof(double));			/* there are 3 values for each blob in blobs */
	if (!points) return 1;

	ListNode* blob=blobs->head;		/* fill array points */
	size_t	i=0;
	while(blob !=EMPTY_NODE) {		/* assign list values to the array */
		points[i++] = ((Point*)blob->value)->value;
		points[i++] = ((Point*)blob->value)->x;
		points[i++] = ((Point*)blob->value)->y;
		blob = blob->next;
	}

	qsort(points,sort_len, 3*sizeof(double), (void *)compare_doubleReverse);

	i = 0;
	blob=blobs->head;
	while(blob !=EMPTY_NODE) {		/* re-assign sorted values to list */
		((Point*)blob->value)->value = points[i++];
		((Point*)blob->value)->x = points[i++];
		((Point*)blob->value)->y = points[i++];
		blob = blob->next;
	}

	free(points);					/* done with points, free it */
	return 0;
}

/*
// this would sort in Increasing order
int compare_double(double *a, double *b);
int compare_double(
double	*a,
double	*b)
{
	if (*a == *b) return 0;
	else if (*a < *b) return -1;
	else return 1;
}
*/

/* only called by sorListPoints, causes the sort to be in Decreasing order */
int compare_doubleReverse(
double	*a,
double	*b)
{
	if (a[0] > b[0]) return -1;
	if (a[0] < b[0]) return 1;
	/* Preserve the historical x-major discovery order when intensities tie. */
	if (a[1] < b[1]) return -1;
	if (a[1] > b[1]) return 1;
	if (a[2] < b[2]) return -1;
	if (a[2] > b[2]) return 1;
	return 0;
}
