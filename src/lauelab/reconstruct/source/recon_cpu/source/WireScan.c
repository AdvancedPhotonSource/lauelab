/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
//#include <math.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include "mathUtil.h"
#include "microHDF5.h"
#include "WireScanDataTypesN.h"
#include "WireScan.h"
#include "readGeoN.h"
#include "misc.h"
#include "cosmicFilter.h"
#include <omp.h>
#include <stdbool.h>

#define TYPICAL_mA		102.		/* average current, used with normalization */
#define TYPICAL_cnt3	88100.		/* average value of cnt3, used with normalization */


#define FIX_ME_SLOW
#ifdef MULTI_IMAGE_FILE
#define MULTI_IMAGE_SKIP 1			/* number of images in multi image file to skip, to use first image set to 0 */
#define MULTI_IMAGE_SKIPV 1			/* number of points in vector to skip, to use first vector set to 0 */
#else
#define MULTI_IMAGE_SKIP 0			/* for a single image in each file, set to 0 */
#define MULTI_IMAGE_SKIPV 0
#endif

#ifndef ELAPSED_SEC
#define ELAPSED_SEC ( ((double)(clock() - tstart)) / ((double)CLOCKS_PER_SEC) )
#endif

const long MiB = (1<<20);			/* 2^20, one mega byte */
/* const int REVERSE_X_AXIS = 0;			//set to non-zero value to flip x axis - double-check */
/* const double PI = 3.14159265358979323;	// not used anymore */

#define CHECK_FREE(A)   { if(A) free(A); (A)=NULL; }

/* control functions */
int main (int argc, const char **argv);
int start(char* infile, char* outfile, char* geofile, double depth_start, double depth_end, double resolution, int first_image, int last_image, int out_pixel_type, int wireEdge, char* normalization);
void processAll( int file_num_start, int file_num_end, char* fn_base, char* fn_out_base, char* normalization);
//void readSingleImage(char* fn, int imageNum, int bottom_image, int ilow, int ihi, int jlow, int jhi, char* normalization);
//void readSingleImage(char* filename, int imageIndex, int ilow, int ihi, int jlow, int jhi, char* normalization);
//void readSingleImage(char *filename, int imageIndex, int file_num_start, int ilow, int ihi, int jlow, int jhi);
void readSingleImage(char *filename, int imageIndex, int slice, int ilow, int ihi, int jlow, int jhi);
int find_first_valid_i(int i1, int i2, int jlo, int jhi, point_xyz wire, BOOLEAN use_leading_wire_edge);
int find_last_valid_i(int i1, int i2, int jlo, int jhi, point_xyz wire, BOOLEAN use_leading_wire_edge);
point_xyz wirePosition2beamLine(point_xyz wire_pos);

/* File I/O */
void getImageInfo(char* fn_base, int file_num_start, int file_num_end, char	*normalization, Dvector *normalVector);
void get_intensity_map(char* filename_base, int file_num_start);
//void readImageSet(char* fn_base, int ilow, int ihi, int jlow, int jhi, int file_num_start, int file_num_end, Dvector *normalVector);
void readImageSet(char* fn_base, int ilow, int ihi, int jlow, int jhi, int file_num_start, int Nimages, Dvector *normalVector);
void writeAllHeaders(char* fn_in_first, char* fn_out_base, int file_num_start, int file_num_end);
int makeTemplateFile(const char* filenameIn,const char* filenameTemplate,char* buf);
//void write1Header(char* finalTemplate, char* fn_base, int file_num);
void write1Header(char* fn_base, int file_num);
void write_depth_data(size_t start_i, size_t end_i, char* fn_base);
void write_depth_datai(int file_num, size_t start_i, size_t end_i, char* fileName);

/* image memory and image manipulation */
void setup_depth_images(int numImages);
void clear_depth_images(vvector *is); //(ws_image_set *is);
void delete_images(void);
void add_pixel_intensity_at_depth(point_ccd pixel, double intensity, double depth);
//inline void add_pixel_intensity_at_index(point_ccd pixel, double intensity, long index);
//inline void add_pixel_intensity_at_index(size_t i, size_t j, double intensity, long index);
int calc_intensity_norm(double *intensity_sorted);
void p_buffer_swap();

/* actual calculations */
//inline double index_to_beam_depth(long index);
//inline double get_trapezoid_height(double partial_start, double partial_end, double full_start, double full_end, double depth);
double index_to_beam_depth(long index);
double get_trapezoid_height(double partial_start, double partial_end, double full_start, double full_end, double depth);
point_xyz pixel_to_point_xyz(point_ccd pixel);
double pixel_xyz_to_depth(point_xyz point_on_ccd_xyz, point_xyz wire_position, BOOLEAN use_leading_wire_edge);
double rotated_pixel_xyz_to_depth(point_xyz pixelPos, point_xyz wire_position, BOOLEAN use_leading_wire_edge);
void depth_resolve(int i_start, int i_stop, int nthreads);
void depth_resolve_pixel(double pixel_intensity, size_t i, size_t j, double partial_end, double partial_start, double full_start, double full_end, double *pixel_depths);
void print_imaging_parameters(ws_imaging_parameters ip);
long pixel_type2scaling(int itype);
void print_help_text(void);

omp_lock_t hdf_io;					// a lock is needed between threads such that only one R/W is performed on HDF5 files at a time.

/* Global variable definitions (declared as extern in WireScan.h) */
ws_calibration calibration;
ws_imaging_parameters imaging_parameters;
ws_image_set image_set;
ws_user_preferences user_preferences;
dmatrix * intensity_map;
dmatrix * intensity_norm;
vvector p_read_buffer[2];
vvector p_write_buffer[2];
int p_ibuff;
int rows_default;
int NUM_THREADS;
struct HDF5_Header in_header;
struct HDF5_Header output_header;
struct geoStructure geoIn;
int verbose;
float percent;
int cutoff;
int AVAILABLE_RAM_MiB;
int cosmic;
int detNum;
char distortionPath[FILENAME_MAX];
float norm_exponent;
float norm_threshold;
float norm_rescale;
/* Note: positionerType is defined in hardwareSpecific.c */

#ifdef DEBUG_ALL					/* temp debug variable for JZT */
int slowWay=0;						/* true if found reading stripes the slow way */
int verbosePixel=0;
	//	#define pixelTESTi 49
	//	#define pixelTESTj 60
//#define pixelTESTi 1092
//#define pixelTESTj 881
#if defined(pixelTESTi) && defined(pixelTESTj)
#define DEBUG_1_PIXEL
#endif
//	•abc()  from Igor
//	Intensity of Si_wire_1[881, 1092] = 60100
//	pixel[881, 1092] --> {13.85, 510.97, -2.23}mm
//	depth = 55.62 µm
void testing_depth(void);
void printPieceOfArrayInt(int ilo, int ihi, int jhlo, int jhi, int Nx, int Ny, unsigned short int buf[Nx][Ny],int itype);
void printPieceOfArrayDouble(int ilo, int ihi, int jhlo, int jhi, int Nx, int Ny, double buf[Nx][Ny]);
void printPieceOf_dmatrix(int ilo, int ihi, int jlo, int jhi, dmatrix *mat);
#endif



int main (int argc, const char *argv[]) {
	int		c;
	char	infile[FILENAME_MAX];
	char	outfile[FILENAME_MAX];
	char	geofile[FILENAME_MAX];
	char	paramfile[FILENAME_MAX];
	char	normalization[FILENAME_MAX];	/* if empty, then do not normalize */

	infile[0] = outfile[0] = geofile[0] = paramfile[0] = normalization[0] = '\0';
	double	depth_start = 0.;
	double	depth_end = 0.;
	double	resolution = 1;
	int		first_image = 0;
	int		last_image = 0;					/* defaults to last image in multi-image file */
	int		out_pixel_type = -1;			/* -1 flags that output image should have same type as input image */
	int		wireEdge = 1;					/* 1=leading edge of wire, 0=trailing edge of wire, -1=both edges */
	unsigned long required=0, requiredFlags=((1<<5)-1);	/* (1<<5)-1 == (2^5 - 1) requiredFlags are the arguments that must be set */
	long	lvalue;
	#ifdef DEBUG_ALL
	char	ApplicationsPath[FILENAME_MAX];		/* path to applications, used for h5repack */
	#endif

	/* initialize some globals */
	geoIn.wire.axis[0]=1; geoIn.wire.axis[0]=geoIn.wire.axis[0]=0;	/* default wire.axis is {1,0,0} */
	geoIn.wire.R[0] = geoIn.wire.R[1] = geoIn.wire.R[2] = 0;		/* default PM500 rotation of wire is 0 */
	distortionPath[0] = '\0';				/* start with it empty */
	verbose = 0;
	percent = 100;
	cutoff = 0;
	AVAILABLE_RAM_MiB = 128;
	cosmic = 0;								/* remove cosmic ray events when true */
	NUM_THREADS = 1;
	rows_default = 256;						// default max # of rows per slice
	detNum = 0;								/* detector number */
	norm_exponent = 0.0;					/* default to 0.0, means do not use */
	norm_threshold = -1.0;					/* default to auto (if norm_exponent is valid) */
	norm_rescale = 1.0;						/* default to 1, only used with valid norm_exponent */
	intensity_norm = NULL;					/* only fill this when norm_exponent is valid */
#ifdef DEBUG_ALL
	getParentPath(ApplicationsPath);
	printf("ApplicationsPath = '%s'\n",ApplicationsPath);
#endif

	int option_index = 0;				/* getopt_long stores the option index here. */
	static struct option long_options[] =
	{
		{"infile",			required_argument,	0,	'i'},
		{"outfile",			required_argument,	0,	'o'},
		{"geofile",			required_argument,	0,	'g'},
		{"depth_start",		optional_argument,	0,	's'},
		{"depth_end",		required_argument,	0,	'e'},
		{"resolution",		optional_argument,	0,	'r'},
		{"verbose",			optional_argument,	0,	'v'},
		{"first_image",		optional_argument,	0,	'f'},
		{"last_image",		required_argument,	0,	'l'},
		{"normalization",	optional_argument,	0,	'n'},
		{"percent",			optional_argument,	0,	'p'},
		{"wire_edge",		optional_argument,	0,	'w'},
		{"memory",			required_argument,	0,	'm'},
		{"type_output",		optional_argument,	0,	't'},
		{"CosmicFilter",	no_argument,		0,	'C'},
		{"distortion_map",	optional_argument,	0,	'd'},
		{"detector_number",	optional_argument,	0,	'D'},
		{"Norm_exponent",	optional_argument,	0,	'E'},
		{"Norm_threshold",	optional_argument,	0,	'T'},
		{"Parameters_File",	optional_argument,	0,	'F'},
		{"Number_of_threads",	optional_argument,	0,	'N'},
		{"Rows_per_stripe",	optional_argument,	0,	'R'},
		{"ignore",			optional_argument,	0,	'@'},
		{"help",			no_argument,		0,	'h'},
		{0, 0, 0, 0}
	};

	char optstring[] = "i:o:g:s:e:r:v:f:l:n:p:w:m:t:Cd:D:E:T:F:N:R:@:h:";	/* follow by ":" means argument is required, see man for getopt(3) */
	while ( (c = getopt_long (argc, (char * const *)argv, optstring, long_options, &option_index) ) > 0) {	/* c is -1 when all options read */
		switch (c) {
			case 0:
				break;

			case 'i':
				strncpy(infile,optarg,FILENAME_MAX-2);
				infile[FILENAME_MAX-1] = '\0';				/* strncpy may not terminate */
				required = required | (1<<0);
				break;

			case 'o':
				strncpy(outfile,optarg,FILENAME_MAX-2);
				outfile[FILENAME_MAX-1] = '\0';				/* strncpy may not terminate */
				required = required | (1<<1);
				break;

			case 'g':
				strncpy(geofile,optarg,FILENAME_MAX-2);
				geofile[FILENAME_MAX-1] = '\0';				/* strncpy may not terminate */
				required = required | (1<<2);
				break;

			case 's':
				depth_start = atof(optarg);
				break;

			case 'e':
				depth_end = atof(optarg);
				required = required | (1<<3);
				break;

			case 'r':
				resolution = atof(optarg);
				break;

			case 'v':
				verbose = (int)atol(optarg);
				if (verbose < 0) verbose = 0;
				break;

			case 'f':
				first_image = (int)atol(optarg);
				break;

			case 'l':
				last_image = (int)atol(optarg);
				break;

			case 'n':
				strncpy(normalization,optarg,FILENAME_MAX-2);
				normalization[FILENAME_MAX-1] = '\0';		/* strncpy may not terminate */
				break;

			case 'p':
				percent = (float)atof(optarg);
				percent = MAX(0,percent);
				percent = MIN(100,percent);
				break;

			case 'm':
				AVAILABLE_RAM_MiB = (int)atol(optarg);
				AVAILABLE_RAM_MiB = MAX(AVAILABLE_RAM_MiB,1);
				break;

			case 't':
				lvalue = atol(optarg);
				if (lvalue<0 ||lvalue>7 || lvalue==4) {
					error("-t switch needs to be followed by 0, 1, 2, 3, 5, 6, or 7\n");
					fprintf(stderr,"0=float(4 byte),   1=long(4 byte),  2=int(2 byte),  3=uint (2 byte)\n");
					fprintf(stderr,"5=double(8 byte),  6=int8 (1 byte), 7=uint8(1 type)\n");
					exit(1);
				}
				out_pixel_type = (int)lvalue;				/* type of output pixel uses old WinView values */
				break;

			case 'C':										/* do Cosmic ray removal */
				cosmic = 1;
				break;

			case 'w':
				if ('l'==optarg[0]) wireEdge = 1;			/* use only leading edge of wire */
				else if ('t'==optarg[0]) wireEdge = 0;		/* use only trailing edge of wire */
				else if ('b'==optarg[0]) {
					wireEdge = -1;							/* use both leading and trailing edges of wire */
					/* use type long for the output image (need + and - values) it not previously specified */
					out_pixel_type = out_pixel_type<0 ? 1 : out_pixel_type;
				}
				else {
					error("-w switch needs to be followed by l, t, or b\n");
					exit(1);
				}
				break;

			case 'd':
				strncpy(distortionPath,optarg,1022);
				distortionPath[1023-1] = '\0';				/* strncpy may not terminate */
				break;

			case 'D':
				detNum = (int)atol(optarg);
				required = required | (1<<4);
				if (detNum < 0 || detNum>2) {
					error("-D detector number must be 0, 1, or 2\n");
					exit(1);
				}
				break;

			case 'E':
				norm_exponent = atof(optarg);
				if (!(norm_exponent>0.0 && norm_exponent<=5.0)) {
					error("-E Normalization exponent must be in range (0,5], probably should be 0.5\n");
					exit(1);
				}
				break;

			case 'N':
				NUM_THREADS = atoi(optarg);
				// if input <=0, automatically get number of processors and subtract 1,
				//	which leaves 1 processor free for the OS
				// if input is not given, however, program will execute in a single thread all through.
				if(NUM_THREADS < 1) NUM_THREADS = MAX(omp_get_num_procs()-1,1);  
				break;

			case 'R':
				rows_default = atoi(optarg);
				if(rows_default < 1) rows_default = 256;  // default is 256 per slice
				break;

			case 'T':
				norm_threshold = atof(optarg);
				if (norm_threshold <= 0.0) norm_threshold = NAN;	/* flags need to auto determine norm_threshold */
				break;

			case '@':				/* a do nothing, just skip */
				break;

			case 'F':				/* read all the parameters from a key=value file */
				strncpy(paramfile,optarg,FILENAME_MAX-2);
				paramfile[FILENAME_MAX-1] = '\0';			/* strncpy may not terminate */
				geofile[0] = '\0';
				required = readAllParameters(paramfile,infile,outfile,normalization,&first_image,&last_image,\
					&depth_start,&depth_end,&resolution,&out_pixel_type,&wireEdge,&detNum,distortionPath) ? 0 : 0xFFFFFFFF;
				break;

			default:
				printf ("Unknown command line argument(s)");

			case 'h':
			case '?':
				print_help_text();
				exit(0);
		}
	}

	if (requiredFlags ^ (required & requiredFlags)) {		/* means all required arguments have been set */
		error("some required -D detector number must be 0, 1, or 2\n");
		print_help_text();
		exit(1);
	}

	if (verbose > 0) {
		time_t systime;
		systime = time(NULL);

		printf("\nStarting execution at %s\n",ctime(&systime));
		printf("infile = '%s'",infile);
		printf("\noutfile = '%s'",outfile);
		printf("\ngeofile = '%s'",geofile);
		printf("\ndistortion map = '%s'",distortionPath);
		if (paramfile[0]) printf("\nparamFile = '%s'",paramfile);
		printf("\ndepth range = [%g, %g]micron with resolution of %g micron",depth_start,depth_end,resolution);
		printf("\nimage index range = [%d, %d]  using %g%% of pixels",first_image,last_image,percent);
		if (normalization[0]) printf("\nnormalizing by value in tag:  '%s'",normalization);
		else printf("\nnot normalizing");

		if (wireEdge<0) printf("\nusing both leading and trailing edges of wire");
		else if (wireEdge) printf("\nusing only leading edge of wire (the usual)");
		else printf("\nusing oly TRAILING edge of wire");
		if (out_pixel_type >= 0) printf("\nwriting output images as type long");
		printf("\nusing %dMiB of RAM, and verbose = %d",AVAILABLE_RAM_MiB,verbose);
		printf("\nRunning with %d threads in parallel\n",NUM_THREADS);
		printf("\n\n");
	}
	fflush(stdout);

	if (!(norm_exponent>0)) {										/* norm_exponent is invalid */
		norm_exponent = 0;
		norm_threshold = 0;
		norm_rescale = 1;
	}
	start(infile, outfile, geofile, depth_start, depth_end, resolution, first_image, last_image, out_pixel_type, wireEdge, normalization);

	if (verbose) {
		time_t systime;
		systime = time(NULL);
		printf("\nExecution ended at %s\n",ctime(&systime));
	}
	return 0;
}




int start(
char *infile,						/* base name of input image files */
char *outfile,						/* base name of output image files */
char *geofile,						/* full path to geometry file */
double depth_start,					/* first depth in reconstruction range (micron) */
double depth_end,					/* last depth in reconstruction range (micron) */
double resolution,					/* depth resolution (micron) */
int first_image,					/* index to first input image file */
int last_image,						/* index to last input image file */
int out_pixel_type,					/* type to use for the output pixel */
int wireEdge,						/* 1=leading edge of wire, 0=trailing edge of wire, -1=both edges */
char *normalization)				/* optional tag for normalization */
{
	double	seconds;				/* seconds of CPU time used */
	time_t	executionTime;			/* number of seconds since program started */
	clock_t	tstart = clock();		/* clock() provides cpu usage, not total elapsed time */
	time_t	sec0 = time(NULL);		/* time (since EPOCH) when program starts */
	int		err=0;
	struct stat info;				/* status of outfile path */
	char	outFolder[FILENAME_MAX];/* holds pathname part of outfile */
	char	*p;
	char	errStr[FILENAME_MAX+256];

	if (strlen(geofile)<1) { }								/* skip if no geo file specified, could have been entered via -F command line flag */
	else if (!(err=(int)readGeoFromFile(geofile, & geoIn))) {	/* readGeoFromFile returns 1=error */
		geo2calibration(&geoIn, detNum);					/* take values from geoIn and put them into calibration */
	} else err = 1;
	if (err) {
		error("Could not load geometry from a file");
		exit(1);
	}
	int i;
	for (i=0;i<MAX_Ndetectors;i+=1) geoIn.d[i].used = 0;	/* mark all detectors in geo as unused */
	geoIn.d[detNum].used = 1;								/* mark as used the one being used */
	if (verbose > 0) {
		printCalibration(verbose);
		printf("\n");
		fflush(stdout);
	}

	if (verbose > 0) printf("running this program as user %d\n",getuid());

	/* create the output directory if it does not exist */
	strncpy(outFolder, outfile, FILENAME_MAX-2); 
	outFolder[FILENAME_MAX-1] = '\0';						/* ensure termination */
	p = strrchr(outFolder, '/');
	if (p || p==outFolder) {								/* make sure I actually have a path part */
		*p = '\0';											/* trim off file part from outFolder */
		info.st_mode = 0;
		stat(outFolder,&info);
		if (info.st_mode == 0) {							/* output folder does not exist, try to create it */
			if (verbose > 0) printf("output folder '%s' does not exist, create it\n",outFolder);
			if (mkdir(outFolder, S_IRWXU | S_IRWXG | S_IRWXO)) {	/* create with complete access to everyone, 777 */
				sprintf(errStr,"could not create output folder '%s'\n",outFolder);
				error(errStr);
				exit(1);
			}
		}
		else if (!(info.st_mode & S_IFDIR)) {					/* outFolder exists, but it is not a directory, cannot write here */
			sprintf(errStr,"output folder '%s' exists, but it is not a directory!",outFolder);
			error(errStr);
			exit(1);
		}
	}
/*
	struct stat info; 
	stat("/Users/tischler/dev/reconstructC_Big/testing/build", &info);
	printf("File mode: 0x%lX\n",info.st_mode);
	if (info.st_mode & S_IFDIR) printf("is a dir\n");
	else printf("NOT a dir\n");
		//	printf("owner mask = %o\n",(info.st_mode & S_IRWXU)>>6);
		//	printf("group mask = %o\n",(info.st_mode & S_IRWXG)>>3);
		//	printf("other mask = %o\n",(info.st_mode & S_IRWXO));
	printf("owner write = %o\n",!!(info.st_mode & S_IWUSR));
	printf("group write = %o\n",!!(info.st_mode & S_IWGRP));
	printf("other write = %o\n",!!(info.st_mode & S_IWOTH));

	uid_t uid;
	uid = getuid();
	printf("user: 0x%lX\n",info.st_uid);
	printf("I am: 0x%lX  =  %d\n",uid,uid);

	printf("for non-existant file");
	info.st_mode = 0;
	stat("/Users/tischler/dev/reconstructC_Big/testing/buildXXX", &info);
	printf("File mode: 0x%lX\n",info.st_mode);
*/

	/* write first part of summary, then close it and write last part after computing */
	FILE *f=NULL;
	char summaryFile[FILENAME_MAX];
	sprintf(summaryFile,"%ssummary.txt",outfile);
	if (!(f=fopen(summaryFile, "w"))) { printf("\nERROR -- start(), failed to open file '%s'\n\n",summaryFile); exit(1); }
	writeSummaryHead(f, infile, outfile, geofile, depth_start, depth_end, resolution, first_image, last_image, out_pixel_type, wireEdge, normalization);
	fclose(f);

	/* initialize image_set.*, contains partial input images & wire positions and partial output images & total intensity */
//	image_set.wire_scanned.v = NULL;
//	image_set.wire_scanned.alloc = image_set.wire_scanned.size = 0;
	p_read_buffer[0].v = p_read_buffer[1].v = NULL;
	p_read_buffer[0].alloc = p_read_buffer[1].alloc = 0;
	p_read_buffer[0].size = p_read_buffer[1].size = 0;
//	image_set.depth_resolved.v = NULL;
//	image_set.depth_resolved.alloc = image_set.depth_resolved.size = 0;
	p_write_buffer[0].v = p_write_buffer[1].v = NULL;
	p_write_buffer[0].alloc = p_write_buffer[1].alloc = 0;
	p_write_buffer[0].size = p_write_buffer[1].size = 0;
	image_set.wire_positions.v = NULL;
	image_set.wire_positions.alloc = image_set.wire_positions.size = 0;
	image_set.depth_intensity.v = NULL;
	image_set.depth_intensity.alloc = image_set.depth_intensity.size = 0;

	user_preferences.depth_resolution = resolution;				/* depth resolution and range of the reconstruction (micron) */
	depth_start = round(depth_start/resolution)*resolution;		/* depth range should have same resolution as step size */
	depth_end = round(depth_end/resolution)*resolution;
	user_preferences.depth_start = depth_start;
	user_preferences.depth_end = depth_end;
	user_preferences.NoutputDepths = (int) round((depth_end - depth_start) / resolution + 1.0);
	user_preferences.out_pixel_type = out_pixel_type;
	user_preferences.wireEdge = wireEdge;
	if (user_preferences.NoutputDepths < 1) {
		error("no output images to process");
		exit(1);
	}

	#ifdef USE_DISTORTION_CORRECTION
	load_peak_correction_maps(distortionPath);
	/*	load_peak_correction_maps("/Users/tischler/dev/reconstructXcode_Mar07/dXYdistortion"); */
	/*	load_peak_correction_maps("/home/nathaniel/Desktop/Reconstruction/WireScan/dXYdistortion"); */
	#endif

	/* *********************** this does everything *********************** */
	processAll(first_image, last_image, infile, outfile, normalization);

	delete_images();
	/* TODO: clear the depth-resolved images from memory*/
	seconds = ELAPSED_SEC;
	executionTime = time(NULL) - sec0;	/* number of seconds since program started */

	/* write remainder of summary file with the total intensity vs depth, for the user to check and see if the depth range is correct */
	if (!(f=fopen(summaryFile, "a"))) printf("\nERROR -- start(), failed to re-open file '%s'\n\n",summaryFile);
	else {														/* re-open file, this section added Apr 1, 2008  JZT */
		writeSummaryTail(f, (double)executionTime);
		fclose(f);
	}
	if (verbose) printf("total execution time for this process took %ld sec, for a CPU time of %.1f seconds\n",executionTime,seconds);

	/* de-allocate and zero out image_set.depth_intensity */
	CHECK_FREE(image_set.depth_intensity.v)
	image_set.depth_intensity.alloc = image_set.depth_intensity.size = 0;
#ifdef DEBUG_ALL					/* temp debug variable for JZT */
	if (slowWay) printf("\n\n********************************\n	reading the slow way\n********************************\n\n");
#endif

	return 0;
}




void processAll(
int		file_num_start,				/* index to first input image file */
int		file_num_end,				/* index to last input image file */
char	*infile,					/* base name of input image files */
char	*fn_out_base,				/* base name of output image files */
char	*normalization)				/* optional tag for normalization */
{
	Dvector normalVector;										/* normalization vector made using 'normalization' */

	if (verbose > 0) printf("loading image information...\n");
	fflush(stdout);

	init_Dvector(&normalVector);
	initHDF5structure(&in_header);								/* initialize in_header */
	initHDF5structure(&output_header);							/* initialize output_header */
	getImageInfo(infile, file_num_start, file_num_end, normalization, &normalVector);	/* sets many of the values in the structure imaging_parameters which is a global */

	#ifdef DEBUG_1_PIXEL
	testing_depth();
	#endif

	clock_t tstart = clock();

	// all shared variables defined before parallel region 
	int cur_start_i,cur_stop_i,nthreads,num_stripes;
	int	start_i, end_i;												/* first and last rows of the image to process, may be less than whole image depending upon depth range and wire range */
																	/*		actually for HDF5 files, you probably have to do the whole range */
	size_t	rows;													/* number of rows (i's) that can be processed at once, limited by memory.  (1<<20) = 2^20 = 1MiB */
	size_t	max_rows;												/* maximum number of rows that can be processed with this memory allocation */
	// record time
	double time1,time0;
	time0 = omp_get_wtime();
	// allow nested parallelism
	omp_set_nested(1);
	omp_init_lock(&hdf_io);
	nthreads = MIN(3,NUM_THREADS);
 #pragma omp parallel sections shared(start_i,end_i,max_rows,rows,cur_start_i,cur_stop_i,num_stripes) num_threads(nthreads)
 {	// start parallel session with 2 sections
  #pragma omp section
  { // parallel section1: get intensity map, create output files & write headers
	omp_set_lock(&hdf_io);  // get the lock so this thread will do some hdf5 io first
	get_intensity_map(infile, file_num_start);					/* finds cutoff, and saves the first image of the wire scan for later comparison */
	if (verbose>1) printf(" ... %g seconds spent in get_intensity_map(), done in thread #%d.\n",omp_get_wtime()-time0,omp_get_thread_num());

	/* set values in the output header */
	int	output_pixel_type;										/* WinView number type of output pixels */
	int	pixel_size;												/* for output image, number of bytes/pixel */
	output_pixel_type = (user_preferences.out_pixel_type < 0) ? imaging_parameters.in_pixel_type : user_preferences.out_pixel_type;
	pixel_size = (user_preferences.out_pixel_type < 0) ? imaging_parameters.in_pixel_bytes : WinView_itype2len(user_preferences.out_pixel_type);
	copyHDF5structure(&output_header, &in_header);				/* duplicate in_header into output_header */
	output_header.isize = pixel_size;							/* change size of pixels for output files */
	output_header.itype = output_pixel_type;
	if (norm_exponent > 0.0) norm_rescale = (double)pixel_type2scaling(output_pixel_type);
#ifdef MULTI_IMAGE_FILE
	init_Dvector(&output_header.xWire);							/* no wire positions in output file */
	init_Dvector(&output_header.yWire);
	init_Dvector(&output_header.zWire);
#else
	output_header.xWire = output_header.yWire = output_header.zWire = NAN;	/* no wire positions in output file */
#endif

	/* create all of the output files, and write the headers, with a dummy image filled with 0 */
	char fn_in_first[FILENAME_MAX];								/* name of first input file */
#ifdef MULTI_IMAGE_FILE
	strncpy(fn_in_first,infile,FILENAME_MAX-1);
#else
	sprintf(fn_in_first,"%s%d.h5",fn_base,file_num_start);
#endif

	#ifdef DEBUG_ALL
	if (verbose > 0) { fprintf(stderr,"allocating disk space for results..."); fflush(stdout); }
	#endif
	writeAllHeaders(fn_in_first,fn_out_base, 0, user_preferences.NoutputDepths - 1);
	#ifdef DEBUG_ALL
	if (verbose > 0) { fprintf(stderr,"     took %.2f sec",ELAPSED_SEC; fflush(stdout); }
	#endif

	if (verbose>1) printf(" ... %g seconds spent in creating output files & headers, done in thread #%d.\n",omp_get_wtime()-time0,omp_get_thread_num());
  } // end of parallel section 1
  #pragma omp section
  { // parallel section 2: determine # of rows per slice & allocate memory spaces
	/* [file_num_start, file_num_end] is the total range of files to read */
	rows = AVAILABLE_RAM_MiB * MiB;									/* total number of bytes available */
	rows -= (imaging_parameters.nROI_i * imaging_parameters.nROI_j * sizeof(double) * 3);	/* subtract space for intensity and distortion maps */
	rows /= (imaging_parameters.nROI_j * sizeof(double));									/* divide by number of bytes per line */
	rows /= (imaging_parameters.NinputImages + user_preferences.NoutputDepths) * 2 ;		/* divide by number of images to store */
	rows = MAX(rows,1);												/* always at least one row */
	max_rows = rows;												/* save maxium value for later */
	if (verbose > 0) printf("From the amount of RAM, can process %lu rows at once\n",rows);
	fflush(stdout);

	/* get starting row and stopping row positions in images (range of i) */
	if (verbose > 0) { printf("setup depth-resolved images in memory\n"); fflush(stdout); }
	start_i = 0;													/* start with whole image, then trim down depending upon wire range and depth range */
	end_i = (int)in_header.xdim - 1;
	if (verbose > 0) printf("process rows %d thru %d\n",start_i,end_i);

if (0) {
	if (user_preferences.wireEdge>=0) {								/* using only one edge of the wire */
		start_i = find_first_valid_i(start_i,end_i,0,imaging_parameters.nROI_j-1,imaging_parameters.wire_first_xyz,(BOOLEAN)(user_preferences.wireEdge));
		end_i = find_last_valid_i(start_i,end_i,0,imaging_parameters.nROI_j-1,imaging_parameters.wire_last_xyz,(BOOLEAN)(user_preferences.wireEdge));
	}
	else {															/* using both edges of the wire */
		int i1,i2;
		i1 = find_first_valid_i(start_i,end_i,0,imaging_parameters.nROI_j-1,imaging_parameters.wire_first_xyz,0);
		i2 = find_first_valid_i(start_i,end_i,0,imaging_parameters.nROI_j-1,imaging_parameters.wire_first_xyz,1);
		start_i = MIN(i1,i2);
		i1 = find_last_valid_i(start_i,end_i,0,imaging_parameters.nROI_j-1,imaging_parameters.wire_last_xyz,0);
		i2 = find_last_valid_i(start_i,end_i,0,imaging_parameters.nROI_j-1,imaging_parameters.wire_last_xyz,1);
		end_i = MAX(i1,i2);
	}
}
	if (start_i<0 || end_i<0) {
		char errStr[1024];
		sprintf(errStr,"Could not find valid starting or stopping rows, got [%d, %d]",start_i,end_i);
		error(errStr);
		exit(1);
	}
	rows = MIN(rows,(size_t)(end_i-start_i+1));						/* re-set in case [start_i,end_i] is smaller, only have a few left */
	rows = MIN(rows,(size_t) rows_default);							// rows cannot be bigger than rows_default
	imaging_parameters.rows_at_one_time = rows;						/* number of rows that can be processed at one time due to memory limitations */
	if (verbose > 0) printf("need to process rows %d thru %d, can do %lu rows at a time\n",start_i,end_i,rows);

	/* current row indicies */
	cur_start_i = start_i;											/* start and stop row for one band of image that fits into memory */
	cur_stop_i = (int)(start_i + rows - 1);
	cur_stop_i = MIN(end_i,cur_stop_i);

	/* in input and output images need space for (imaging_parameters.rows_at_one_time = rows) rows */
	/* allocate space for wire_scanned images of length (rows = imaging_parameters.rows_at_one_time) */
//	setup_depth_images(file_num_end-file_num_start+1);				/* allocate space and initialize the structure image_set, which contains the output */
	setup_depth_images(imaging_parameters.NinputImages);			/* allocate space and initialize the structure image_set, which contains the output */
	if (verbose > 0) print_imaging_parameters(imaging_parameters);

	if (verbose>1) 
	{	time1 = omp_get_wtime();
		printf(" ... %g seconds spent in setting up depth-image memory space, done in thread #%d.\n",time1-time0,omp_get_thread_num());
	}

	// number of stripes to process
	num_stripes = (end_i-start_i+1)%(int)rows ? (end_i-start_i+1)/(int)rows+1 : (end_i-start_i+1)/(int)rows;

	// read in first stripe, this is often slow unless the file is already cached by the system somewhere
	//	thus this is executed while the other thread is copying template file into individual output files.
	omp_set_lock(&hdf_io);  			// wait until the other thread is done with hdf5 IO
	if (verbose > 1) printf("> thread #%d: start reading first stripe ...\n",omp_get_thread_num());
	if (verbose > 0) printf("Reading first stripe: rows %d thru %d  (%d of %d)...\n",cur_start_i,cur_stop_i,cur_stop_i-cur_start_i+1,end_i-start_i+1);
	p_ibuff = 0; // use buffer #0 for reading in first stripe
//	readImageSet(infile, cur_start_i, cur_stop_i, 0, imaging_parameters.nROI_j - 1, file_num_start, file_num_end, normalization);
	readImageSet(infile, cur_start_i, cur_stop_i, 0, imaging_parameters.nROI_j - 1, file_num_start, imaging_parameters.NinputImages, &normalVector);
	omp_unset_lock(&hdf_io);			// release the lock to the other thread
	if (verbose>1) 
	{	printf(" ... %g seconds spent in reading first stripe, done in thread #%d.\n",omp_get_wtime()-time1,omp_get_thread_num());
	}
  }// end of parallel section2
 }// end of parallel execution of sections
	omp_destroy_lock(&hdf_io);	// lock no longer needed
 
	time1 = omp_get_wtime();
	if (verbose>1) printf(" %g seconds spent in first parallel part.\n",time1 - time0);
	time0 = time1;

	/* ** Loop over stripes ** */
	int istripe,prev_start_i=-1,prev_stop_i=-1;
	int n_threads_depth_resolve = MAX(NUM_THREADS - 1,1); // num of threads for depth-resovling
	double time11;
	if(verbose>1) printf("*** Enters loop over stripes ***\n");
	for(istripe=0;istripe<=num_stripes;istripe++) // total steps = num_stripes + 1 because need one extra cycle to write data 
	{	if (verbose>1) printf("** Stripe No. %d:\n",istripe);
	
		/* Switch between buffers for reading/writing data */
		//   index "p_ibuff" points to the buffer that R/W is carried out with,
		//   while depth-resolving calculations are performed on the other buffer, i.e., !p_ibuff
		p_ibuff = !p_ibuff; 

		time1 = omp_get_wtime();
		
		image_set.wire_scanned = p_read_buffer[!p_ibuff]; 
		image_set.depth_resolved = p_write_buffer[!p_ibuff];
		if (verbose>2) printf(" Now use buffers #%d for read-write\n",p_ibuff);

		// start/finish row numbers
		cur_start_i = istripe * (int)rows;
		cur_stop_i = MIN((istripe+1) * (int)rows - 1, end_i);
		imaging_parameters.current_selection_start = cur_start_i;
		imaging_parameters.current_selection_end = cur_stop_i;

		if (verbose>2) printf(" cur_start_i = %d, cur_stop_i = %d, rows = %d, istripe=%d\n",cur_start_i,cur_stop_i,rows,istripe);
		
		// start parallel execution
		omp_set_num_threads(2); // 2 threads for the sections, section 2 will spawn more threads for depth-resolving
		#pragma omp parallel sections private(time11)
		{
			#pragma omp section
			{	// section1: write previous stripe to disk and read-in next stripe to buffer
				if(verbose>0) printf("> thread #%d: Start reading/writing data buffer ...\n",omp_get_thread_num());

				// part1: write last-processed stripe to disk
				if(prev_start_i>=0) 
				{	if(verbose>2) printf(" Start writing data of the previous stripe, prev_start_i = %d, prev_stop_i = %d.\n",prev_start_i,prev_stop_i);
					write_depth_data((size_t)prev_start_i, (size_t)prev_stop_i, fn_out_base);
					time11 = omp_get_wtime();
					// clear the buffer that has been written out, does not de-allocate
					clear_depth_images(&p_write_buffer[p_ibuff]);
					if(verbose>1) printf("> thread #%d: ...%g seconds spent in writing previous stripe to output files;\n",omp_get_thread_num(),omp_get_wtime()-time1);
				}
				else
					time11 = omp_get_wtime();

				// start-end row numbers for the next stripe
				int next_start_i = cur_stop_i +1;
				int next_stop_i = MIN((istripe+2) * (int)rows - 1, end_i);
				if(verbose>2) printf(" To read: next start_i = %d, next stop_i = %d\n",next_start_i, next_stop_i);
				if (next_start_i <= end_i)
				{
					p_read_buffer[p_ibuff].size = 0; // need to reset this number as it will be incremented during readImageSet().
//					readImageSet(infile, next_start_i, next_stop_i, 0, imaging_parameters.nROI_j - 1, file_num_start, file_num_end, normalization);
					readImageSet(infile, next_start_i, next_stop_i, 0, imaging_parameters.nROI_j - 1, file_num_start, imaging_parameters.NinputImages, &normalVector);
					if(verbose>1) printf("> thread #%d: ... %g seconds spent in reading data of the next stripe.\n",omp_get_thread_num(),omp_get_wtime()-time11);
				}
				else
					if(verbose>1) printf("> thread #%d: no more stripes to read-in, going idle.\n",omp_get_thread_num());
			}// end of parallel section 1
			#pragma omp section
			{ // section2: depth-resolving with multiple threads
				if(cur_start_i<=cur_stop_i) depth_resolve(cur_start_i, cur_stop_i, n_threads_depth_resolve); 
				if(verbose>1) printf(" ... %g seconds spent in depth-resolving, done with %d threads.\n",omp_get_wtime()-time1,n_threads_depth_resolve);
			} 
		}// end of parallel processing	

		prev_start_i = cur_start_i; // store these indices for the writing thread on the next cycle
		prev_stop_i = cur_stop_i;
		
	} // end of loop over stripes
	
	
	/* loop through ram-managable stripes of the image and process them */
//	while (cur_start_i <= end_i ) {
//		imaging_parameters.current_selection_start = cur_start_i;
//		imaging_parameters.current_selection_end = cur_stop_i;

//		clear_depth_images(&image_set);				/* sets all images in image_set.depth_resolved and image_set.wire_scanned to zero, does not de-allocate the space they use, or change .size or .alloc */
													// /* NOTE, do NOT clear image_set.depth_intensity or image_set.wire_positions */
//		if (verbose > 1) printf("\n");
//		if (verbose > 0) printf("\nprocessing rows %d thru %d  (%d of %d), %.2f sec so far...",cur_start_i,cur_stop_i,cur_stop_i-cur_start_i+1,end_i-start_i+1,ELAPSED_SEC);
//		fflush(stdout);

//		/* read stripes from the input image files */
//		readImageSet(infile, cur_start_i, cur_stop_i, 0, imaging_parameters.nROI_j - 1, file_num_start, imaging_parameters.NinputImages, &normalVector);

//		if (verbose > 1) printf("\n\tdepth resolving...");
//		if (verbose == 2) printf("       ");
//		fflush(stdout);

//		/* depth resolve the set of stripes just read */
//		depth_resolve(cur_start_i, cur_stop_i);

//		if (verbose > 1) printf("\n\twriting out data");
//		if (verbose == 2) printf("      ");
//		fflush(stdout);

//		/* write the depth resolved stripes to the output image files */
//		write_depth_data((size_t)cur_start_i, (size_t)cur_stop_i, fn_out_base);

//		cur_start_i = cur_stop_i + 1;					/* increase row limits for next stripe */
//		cur_stop_i = MIN(cur_stop_i+(int)rows,end_i);	/* make sure loop doesn't go outside of the assigned area. */
//	}
	imaging_parameters.rows_at_one_time = max_rows;		/* save this for output to summary file */

	if (verbose > 1) printf("\n\nfinishing after %.2f sec...\n", ELAPSED_SEC);
	fflush(stdout);
}





/* depth sort out the intensity for for the pixels in one stripe */
void depth_resolve(
	int i_start,			/* starting row of this stripe */
	int i_stop,				/* final row of this stripe*/
	int nthreads)			// number of threads for parallel run
{
	point_ccd pixel_edge;					/* pixel indicies for an edge of a pixel (e.g. [117,90.5]) */
	point_xyz front_edge;					/* xyz coords of the front edge of a pixel */
	point_xyz back_edge;					/* xyz coords of the back edge of a pixel */
	double	diff_value;						/* intensity difference between two wire steps for a pixel */
//	dvector pixel_values;					/* vector to hold one pixel's values at all depths */
	dvector *pixel_values_thread;			// vector in each thread to hold one pixel's values at depths
	double *depth_back_thread;
	double *depth_front_thread;
	double *pixel_depths_thread;
	double *totals_thread;
	size_t n_depths;
	size_t n_wire_positions;
	size_t	step;							/* index over the input images */
	size_t	idep;							/* index into depths */
	size_t	i,j;							/* loop indicies */
	double time1,time0;
	
	time0 = omp_get_wtime();

	//pixel_values.size = pixel_values.alloc = imaging_parameters.NinputImages - 1 - 1;
	//pixel_values.v = calloc(pixel_values.alloc,sizeof(double));				/* allocate space for array of doubles in the vector */
	//if (!(pixel_values.v)) { fprintf(stderr,"\ncannot allocate space for pixel_values, %ld points\n",pixel_values.alloc); exit(1); }
	pixel_values_thread = calloc(nthreads,sizeof(dvector));
	n_depths = image_set.depth_resolved.size;
	n_wire_positions = image_set.wire_positions.size;
	depth_back_thread = malloc((size_t)nthreads * 2 * n_wire_positions * sizeof(double));
	depth_front_thread = malloc((size_t)nthreads * 2 * n_wire_positions * sizeof(double));
	pixel_depths_thread = calloc((size_t)nthreads * n_depths, sizeof(double));
	totals_thread = calloc((size_t)nthreads * n_depths, sizeof(double));
	if (!(pixel_values_thread) || !(depth_back_thread) || !(depth_front_thread) || !(pixel_depths_thread) || !(totals_thread)) { fprintf(stderr,"\ncannot allocate thread-local reconstruction buffers\n"); exit(1); }
	for (i=0;i<nthreads;i++)
	{	pixel_values_thread[i].size = pixel_values_thread[i].alloc = imaging_parameters.NinputImages - 1;
		pixel_values_thread[i].v = calloc(pixel_values_thread[i].alloc,sizeof(double));		/* allocate space for array of doubles in the vector */
		if (!(pixel_values_thread[i].v)) { fprintf(stderr,"\ncannot allocate space for pixel_values_thread[%d], %ld points\n",i,pixel_values_thread[i].alloc); exit(1); }
	}
	
	#ifdef DEBUG_1_PIXEL
	if (i_start<=pixelTESTi && pixelTESTi<=i_stop) { printf("\n\n  ****** start story of one pixel, [%g, %g]\n",(double)pixelTESTi,(double)pixelTESTj); verbosePixel = 1; }
	#endif


//	get_difference_images();												/* sequential subtraction on all of the input images. */


	#ifdef DEBUG_1_PIXEL
	verbosePixel = 0;
	#endif

	/* start parallel execution */
  #pragma omp parallel private(pixel_edge,front_edge,back_edge,diff_value,step,idep,j) num_threads(nthreads)
  {	
    #pragma omp single
	if(verbose>1) printf(" depth_resolving rows from %d to %d with %d threads\n",i_start,i_stop,omp_get_num_threads());

    #pragma omp for // run this for loop in parallel
	for (i = i_start; i <= (size_t)i_stop; i++) {							/* loop over selected part of i */
		int ith = omp_get_thread_num();
		double *depth_back = depth_back_thread + (size_t)ith * 2 * n_wire_positions;
		double *depth_front = depth_front_thread + (size_t)ith * 2 * n_wire_positions;
		double *pixel_depths = pixel_depths_thread + (size_t)ith * n_depths;
		double *depth_totals = totals_thread + (size_t)ith * n_depths;
		pixel_edge.i = (double)i;
		for (j=0; j < (size_t)imaging_parameters.nROI_j; j++) {				/* loop over all of j, wire travels in the j direction for the orange detector */
			if (j==0) {														/* only need to recompute this for first j */
				pixel_edge.j = 0.5 + ((double)j - 1.0);						/* upstream edge of pixel for the first pixel, (j-1) is the 'previous' j */
				back_edge = pixel_to_point_xyz(pixel_edge);					/* this is the back edge of the first pixel */
			}
			else {
				back_edge = front_edge;										/* reuse last front edge as the current back edge */
			}
#ifdef DEBUG_1_PIXEL
			verbosePixel = (i==pixelTESTi) && (j==pixelTESTj);
#endif
			pixel_edge.j = 0.5 + (double)j;
			front_edge = pixel_to_point_xyz(pixel_edge);					/* the front edge of this pixel */
			if ( dmatrix_get(intensity_map, i, j)  < cutoff) continue;	/* not enough intensity, skip this pixel */
			for (idep=0;idep<pixel_values_thread[ith].alloc;idep++) pixel_values_thread[ith].v[idep]=0.;	/* clear the pixel vector along depth, set all to zero */
			memset(pixel_depths, 0, n_depths * sizeof(double));
			point_xyz rotated_back_edge = MatrixMultiply31(calibration.wire.rho,back_edge);
			point_xyz rotated_front_edge = MatrixMultiply31(calibration.wire.rho,front_edge);
			for (step=0; step<n_wire_positions; step++) {
				BOOLEAN edge = user_preferences.wireEdge < 0 ? 1 : user_preferences.wireEdge;
				depth_back[step] = rotated_pixel_xyz_to_depth(rotated_back_edge, image_set.wire_positions.v[step], edge);
				depth_front[step] = rotated_pixel_xyz_to_depth(rotated_front_edge, image_set.wire_positions.v[step], edge);
				if (user_preferences.wireEdge < 0) {
					depth_back[n_wire_positions + step] = rotated_pixel_xyz_to_depth(rotated_back_edge, image_set.wire_positions.v[step], 0);
					depth_front[n_wire_positions + step] = rotated_pixel_xyz_to_depth(rotated_front_edge, image_set.wire_positions.v[step], 0);
				}
			}
#ifdef DEBUG_1_PIXEL
			if (verbosePixel)
				printf("\nback_edge = {%g, %g, %g},  front_edge = {%g, %g, %g} for pixel[%lu, %lu]",back_edge.x,back_edge.y,back_edge.z,front_edge.x,front_edge.y,front_edge.z,i,j);
#endif

			/* load the pixel vector full of values for this pixel
			 *  - 1 - 1 because images have already been differenced within the matricies
			 * meaning that the last image hasn't been differenced against anything and so we ignore it.
			 */
			for (step=0; step < pixel_values_thread[ith].size; step++) {
				/* pixel locations are real coordinates on detector, but image is stripe of image from middle of image - correct for this. */
				pixel_values_thread[ith].v[step] = dmatrix_get(image_set.wire_scanned.v[step], i - imaging_parameters.current_selection_start, j);

			}

			if (cosmic) {
				//	***************************************************************
				//		PUT IN CODE HERE TO REMOVE COSMICS FROM pixel_values.v[]
				//	***************************************************************
				cosmic_filter(pixel_values_thread[ith].v, pixel_values_thread[ith].size);
			}

//			for (step=0; step < pixel_values.size - 1; step++) {
//				pixel_values.v[step] -= pixel_values.v[step+1];				/* Change pixel intensities to the sequential subtraction of intensities. */
//			}


			#pragma message "TODO: put any curve-fitting stuff here before we go through the pixel in a line"
			#pragma message "are the limits of this loop correct?, should it be one longer?"
			for (step=0; step < pixel_values_thread[ith].size - 1-1; step++) {			/* loop over all of the differenced intensities of this pixel */
				diff_value = pixel_values_thread[ith].v[step] - pixel_values_thread[ith].v[step+1];
				pixel_values_thread[ith].v[step] = diff_value;
				#ifdef DEBUG_1_PIXEL
				if (verbosePixel) printf("\n∆ pixel[%ld] values = %g",step,diff_value);
				#endif
				if (diff_value==0) continue;								/* only process for non-zero intensity */
				else if (user_preferences.wireEdge<0) {						/* using both leading and trailing edges of the wire */
					depth_resolve_pixel(diff_value, i, j, depth_back[step+1], depth_front[step], depth_back[step], depth_front[step+1], pixel_depths);
					depth_resolve_pixel(-diff_value, i, j, depth_back[n_wire_positions+step+1], depth_front[n_wire_positions+step], depth_back[n_wire_positions+step], depth_front[n_wire_positions+step+1], pixel_depths);
				}
/*				else if (user_preferences.wireEdge && diff_value>0 || !(user_preferences.wireEdge) && diff_value<0) {*/
				else {														/* change, do both positives and negatives, when only a single edge */
					depth_resolve_pixel(user_preferences.wireEdge ? diff_value : -diff_value, i, j, depth_back[step+1], depth_front[step], depth_back[step], depth_front[step+1], pixel_depths);
				}
				#ifdef DEBUG_1_PIXEL
				if (verbosePixel) printf("\n∆ pixel[%ld] values = %g",step,diff_value);
				#endif
			}
			for (idep=0; idep<n_depths; idep++) {
				if (pixel_depths[idep] != 0) {
					* dmatrix_ptr(image_set.depth_resolved.v[idep], i - imaging_parameters.current_selection_start, j) += pixel_depths[idep];
					depth_totals[idep] += pixel_depths[idep];
				}
			}
			#ifdef DEBUG_1_PIXEL
			if (verbosePixel) { printf("\n  ****** done with story of one pixel[%lu, %lu]\n",i,j); verbosePixel = 0; }
			#endif
		}
	} // end of parallel for loop over rows
  } // end of omp parallel
	for (i=0;i<(size_t)nthreads;i++) {
		for (idep=0;idep<n_depths;idep++) {
			image_set.depth_intensity.v[idep] += totals_thread[i * n_depths + idep];
		}
		free(pixel_values_thread[i].v);
	}
	free(totals_thread);
	free(pixel_depths_thread);
	free(depth_front_thread);
	free(depth_back_thread);
	free(pixel_values_thread);
	if(verbose > 1)
	{	time1 = omp_get_wtime();
		printf("\t\t depth-resolving done in %g seconds\n",time1-time0);
	}
	#ifdef DEBUG_1_PIXEL
	verbosePixel = 0;
	#endif
	return;
}


/* Given the difference intensity at one pixel for two wire positions, distribute the difference intensity into the depth histogram */
/* This routine only tests for zero pixel_intensity, it does not avoid negative intensities,  this routine can accumulate negative intensities. */
/* This routine assumes that the wire is moving "forward" */
void depth_resolve_pixel(
double pixel_intensity,				/* difference of the intensity at the two wire positions */
size_t	i,							/* indicies to the the pixel being processed, relative to the full stored image, range is (xdim,ydim) */
size_t	j,
double partial_end,					/* depth from the back pixel edge at the second wire position */
double partial_start,				/* depth from the front pixel edge at the first wire position */
double full_start,					/* depth from the back pixel edge at the first wire position */
double full_end,					/* depth from the front pixel edge at the second wire position */
double *pixel_depths)				/* thread-local accumulation for this pixel */
{
	double	area;							/* area of trapezoid */
	double	maxDepth;						/* depth of deepest reconstructed image (micron) */
	double	dDepth;							/* local version of user_preferences.depth_resolution */
	long	m;								/* index to depth */

	if (pixel_intensity==0) return;											/* do not process pixels without intensity */

	dDepth = user_preferences.depth_resolution;								/* just a local copy */
	maxDepth = dDepth*(image_set.depth_resolved.size- 1) + user_preferences.depth_start;	/* max reconstructed depth (mciron) */

	if (partial_end < user_preferences.depth_start || partial_start > maxDepth) return;		/* trapezoid does not overlap depth-resolved region, do not process */

	if (full_end < full_start) {			/* in case mid points are backwards, ensure proper order by swapping */
		double swap;
		swap = full_end;
		full_end = full_start;
		full_start = swap;
	}
	area = (full_end + partial_end - full_end - partial_start) / 2;			/* area of trapezoid assuming a height of 1, used for normalizing */
	if (area < 0 || isnan(area)) return;									/* do not process if trapezoid has no area (or is NAN) */

	long imax = (long)image_set.depth_resolved.size- 1;						/* imax is maximum allowed value of index */
	long start_index, end_index;											/* range of output images for this trapezoid */
	start_index = (long)floor((partial_start - user_preferences.depth_start) / dDepth);
	start_index = MAX((long)0,start_index);									/* start_index lie in range [0,imax] */
	start_index = MIN(imax,start_index);
	end_index = (long)ceil((partial_end - user_preferences.depth_start) / dDepth);
	end_index = MAX(start_index,end_index);									/* end_index must lie in range [start_index, imax] */
	end_index = MIN(imax,end_index);

	#ifdef DEBUG_1_PIXEL
	if (verbosePixel) printf("\n\ttrapezoid over range (% .3f, % .3f) micron == image index[%ld, %ld],  area=%g",partial_start,partial_end,start_index,end_index,area);
	#endif

	double area_in_range = 0;
	double depth_1, depth_2, height_1, height_2;							/* one part of the trapezoid that overlaps the current bin */
	double depth_i, depth_i1;												/* depth range of depth bin i */
	for (m = start_index; m <= end_index; m++) {							/* loop over possible depth indicies (m is index to depth-resolved image) */
		area_in_range = 0;
		depth_i = index_to_beam_depth(m) - (dDepth*0.5);					/* ends of current depth bin */
		depth_i1 = depth_i + dDepth;

		if (full_start > depth_i && partial_start < depth_i1) {				/* this depth bin overlaps first part of trapezoid (sloping up from zero) */
			depth_1 = MAX(depth_i,partial_start);
			depth_2 = MIN(depth_i1,full_start);
			height_1 = get_trapezoid_height(partial_start, partial_end, full_start, full_end, depth_1);
			height_2 = get_trapezoid_height(partial_start, partial_end, full_start, full_end, depth_2);
			area_in_range += ((height_1 + height_2) / 2 * (depth_2 - depth_1));
		}

		if (full_end > depth_i && full_start < depth_i1) {					/* this depth bin overlaps second part of trapezoid (the flat top) */
			depth_1 = MAX(depth_i,full_start);
			depth_2 = MIN(depth_i1,full_end);
			area_in_range += (depth_2 - depth_1);							/* the height of both points is 1, so area is just the width */
		}

		if (partial_end > depth_i && full_end < depth_i1) {					/* this depth bin overlaps third part of trapezoid (sloping down to zero) */
			depth_1 = MAX(depth_i,full_end);
			depth_2 = MIN(depth_i1,partial_end);
			height_1 = get_trapezoid_height(partial_start, partial_end, full_start, full_end, depth_1);
			height_2 = get_trapezoid_height(partial_start, partial_end, full_start, full_end, depth_2);
			area_in_range += ((height_1 + height_2) / 2 * (depth_2 - depth_1));
		}

		if (area_in_range>0) {
			double intensity = pixel_intensity * (area_in_range / area);
			#ifdef DEBUG_1_PIXEL
			if (verbosePixel && i==pixelTESTi && j==pixelTESTj) printf("\n\t\t adding %g to pixel [%lu, %lu] at depth index %ld",intensity,i,j,m);
			#endif
			pixel_depths[m] += intensity;
		}
	}
}



/* for a trapezoid of max height 1, find the actual height at x=depth, y=0 outside of [partial_start,partial_end] & y=1 in [full_start,full_end] */
/* the order of the conditionals was chosen by their likelihood, the most likely test should come first, the least likely last. */
// inline double get_trapezoid_height(
double get_trapezoid_height(
double	partial_start,				/* first depth where trapezoid becomes non-zero */
double	partial_end,				/* last depth where trapezoid is non-zero */
double	full_start,					/* first depth of the flat top */
double	full_end,					/* last depth of the flat top */
double	depth)						/* depth we want the value for */
{
	if ( depth <= partial_start || depth >= partial_end )	return 0;								/* depth is outside trapezoid */
	else if( depth < full_start )	return (depth - partial_start) / (full_start - partial_start);	/* depth in first sloping up part */
	else if( depth > full_end )		return (partial_end - depth) / (partial_end - full_end);		/* depth in sloping down part */
	return 1;																						/* depth in flat middle */
}



/*inline double pixel_to_depth(point_ccd pixel, point_xyz wire_position, BOOLEAN use_leading_wire_edge);
 *inline double pixel_to_depth(point_ccd pixel, point_xyz wire_position, BOOLEAN use_leading_wire_edge)
 *{
 *	double depth;
 *	point_xyz point_on_ccd_xyz;
 *	point_on_ccd_xyz = pixel_to_point_xyz(pixel);
 *	depth = pixel_xyz_to_depth(point_on_ccd_xyz, wire_position, use_leading_wire_edge);
 *	return depth;
 *}
 */




/* Returns depth (starting point of ray with one end point at point_on_ccd_xyz that is tangent */
/* to leading (or trailing) edge of the wire and intersects the incident beam.  The returned depth is relative to the Si position (origin) */
/* depth is measured along the incident beam from the origin, not just the z value. */
double pixel_xyz_to_depth(
point_xyz point_on_ccd_xyz,			/* end point of ray, an xyz location on the detector */
point_xyz wire_position,			/* wire center, used to find the tangent point, has been PM500 corrected, origin subtracted, rotated by rho */
BOOLEAN use_leading_wire_edge)		/* which edge of wire are using here, TRUE for leading edge */
{
	return rotated_pixel_xyz_to_depth(MatrixMultiply31(calibration.wire.rho,point_on_ccd_xyz), wire_position, use_leading_wire_edge);
}


double rotated_pixel_xyz_to_depth(
point_xyz pixelPos,					/* pixel position already rotated by calibration.wire.rho */
point_xyz wire_position,			/* wire center, used to find the tangent point, has been PM500 corrected, origin subtracted, rotated by rho */
BOOLEAN use_leading_wire_edge)		/* which edge of wire are using here, TRUE for leading edge */
{
	point_xyz	ki;										/* incident beam direction */
	point_xyz	S;										/* point where rays intersects incident beam */
	double		pixel_to_wireCenter_y;					/* vector from pixel to wire center, y,z coordinates */
	double		pixel_to_wireCenter_z;
//	double		pixel_to_wireCenter_len;				/* length of vector pixel_to_wireCenter (only y & z components) */
	double		pixel_to_wireCenter_lensq;				/* square of length of vector pixel_to_wireCenter (only y & z components) */
	double		wire_radius;							/* wire radius */
	double		phi0,tanphi0;									/* angle from yhat to wireCenter (measured at the pixel) */
	double		dphi,tandphi;									/* angle between line from detector to centre of wire and to tangent of wire */
	double		tanphi;									/* phi is angle from yhat to tangent point on wire */
	double		b_reflected;
	double		depth;									/* the result */
	double		numerator,denominator;					// temp variables

	/* change coordinate system so that wire axis lies along {1,0,0}, a rotated system */

	ki.x = calibration.wire.ki.x;						/* ki = rho x {0,0,1} */
	ki.y = calibration.wire.ki.y;
	ki.z = calibration.wire.ki.z;

	pixel_to_wireCenter_y = wire_position.y - pixelPos.y; /* vector from point on detector to wire centre. */
	pixel_to_wireCenter_z = wire_position.z - pixelPos.z;
//	pixel_to_wireCenter_len = sqrt(pixel_to_wireCenter_y*pixel_to_wireCenter_y + pixel_to_wireCenter_z*pixel_to_wireCenter_z);/* length of vector pixel_to_wireCenter */
	pixel_to_wireCenter_lensq = pixel_to_wireCenter_y*pixel_to_wireCenter_y + pixel_to_wireCenter_z*pixel_to_wireCenter_z;/* square of length of vector pixel_to_wireCenter */

	wire_radius = calibration.wire.diameter / 2;		/* wire radius */
//	phi0 = atan2(pixel_to_wireCenter_z , pixel_to_wireCenter_y);	/* angle from yhat to wireCenter (measured at the pixel) */
	tanphi0 = pixel_to_wireCenter_z / pixel_to_wireCenter_y;
//	dphi = asin(wire_radius / pixel_to_wireCenter_len);	/* angle between line from detector to centre of wire and line to tangent of wire */
	tandphi = wire_radius / sqrt(pixel_to_wireCenter_lensq - wire_radius*wire_radius);
//	tanphi = tan(phi0+(use_leading_wire_edge ? -dphi : dphi));	/* phi is angle from yhat to V (measured at the pixel) */
	numerator = use_leading_wire_edge ? (tanphi0-tandphi) : (tanphi0+tandphi);
	denominator = use_leading_wire_edge ? (1+tanphi0*tandphi) : (1-tanphi0*tandphi);
	tanphi = numerator / denominator;
	

	b_reflected = pixelPos.z - pixelPos.y * tanphi;		/* line from pixel to tangent point is:   z = y*tan(phio±dphi) + b */
	/* line of incident beam is:   y = kiy/kiz * z		Thiis line goes through origin, so intercept is 0 */
	/* find intersection of this line and line from pixel to tangent point */
	S.z = b_reflected / (1-tanphi * ki.y / ki.z);		/* intersection of two lines at this z value */
	S.y = ki.y / ki.z * S.z;							/* corresponding y of point on incident beam */
	S.x = ki.x / ki.z * S.z;							/* corresponding z of point on incident beam */
	depth = DOT3(ki,S);

/*	if (verbosePixel) {
 *		printf("\n    -- pixel on detector = {%.3f, %.3f, %.3f}",point_on_ccd_xyz.x,point_on_ccd_xyz.y,point_on_ccd_xyz.z);
 *		printf("\n       wire center = {%.3f, %.3f, %.3f} relative to Si (micron)",wire_position.x,wire_position.y,wire_position.z);
 *		printf("\n       pixel_to_wireCenter = {%.9lf, %.9lf}µm,  |v|=%.9f",pixel_to_wireCenter_y,pixel_to_wireCenter_z,pixel_to_wireCenter_len);
 *		printf("\n       phi0 = %g (rad),   dphi = %g (rad),   tanphi = %g,   depth = %.2f (micron)\n",phi0,dphi,tanphi,DOT3(ki,S));
 *	}
 */
	return depth;										/* depth measured along incident beam (remember that ki is normalized) */
}





/* Take the indicies to a detector pixel and returns an 3vector point in beam-line coordinates of the pixel centre
 * Here is the only place where the corrections for a ROI (binning & sub-region of detector) has been used.  Hopefully it is the only place needed.
 * All pixel values (binned & un-binned) are zero based.
 * This routine uses the same conventions a used in Igor
 */
point_xyz pixel_to_point_xyz(
point_ccd pixel)					/* input, binned ROI (zero-based) pixel value on detector, can be non-integer, and can lie outside range (e.g. -05 is acceptable) */
{
	point_xyz coordinates;								/* point with coordinates in R3 to return */
	point_ccd corrected_pixel;							/* pixel data to be filled by the peak_correction method */
	double	x,y,z;										/* 3d coordinates */

	#pragma message "here is the only place where the pixel is swapped for the transpose in an HDF5 file"
	corrected_pixel.i = pixel.j;						/* the transpose swap needed with the HDF5 files */
	corrected_pixel.j = pixel.i;

	/* convert pixel from binned ROI value to full frame un-binned pixels, both binned and un-binned are zero based. */
	corrected_pixel.i = corrected_pixel.i * imaging_parameters.bini + imaging_parameters.starti;		/* convert from binned ROI to full chip un-binned pixel */
	corrected_pixel.j = corrected_pixel.j * imaging_parameters.binj + imaging_parameters.startj;
	corrected_pixel.i += (imaging_parameters.bini-1)/2.;	/* move from leading edge of pixel to the pixel center(e) */
	corrected_pixel.j += (imaging_parameters.binj-1)/2.;	/*	this is needed because the center of a pixel changes with binning */

	#ifdef DEBUG_1_PIXEL
	if (verbosePixel) printf("\nin pixel_to_point_xyz(), pixel = [%g, %g] (binned ROI, on input),   size is (%g, %g) (micron)",pixel.i,pixel.j,calibration.pixel_size_i,calibration.pixel_size_j);
	if (verbosePixel) printf("\n   corrected_pixel = [%g, %g] (un-binned full chip pixels)",corrected_pixel.i,corrected_pixel.j);
	#endif
	corrected_pixel = PEAKCORRECTION(corrected_pixel);		/* do the distortion correction */

	#if defined(DEBUG_ALL) && defined(USE_DISTORTION_CORRECTION)
	if (verbosePixel) printf("\n   distortion corrected_pixel = [%g, %g] (un-binned full chip pixels)",corrected_pixel.i,corrected_pixel.j);
	#endif

	/* get 3D coordinates in detector frame of the pixel */
	x = (corrected_pixel.i - 0.5*(calibration.ccd_pixels_i - 1)) * calibration.pixel_size_i;	/* (x', y', z') position of pixel (detector frame) */
	y = (corrected_pixel.j - 0.5*(calibration.ccd_pixels_j - 1)) * calibration.pixel_size_j;	/* note, in detector frame all points on detector have z'=0 */
	/*if (REVERSE_X_AXIS) x = -x; */

	x += calibration.P.x;									/* translate by P (P is from geoN.detector.P) */
	y += calibration.P.y;
	z  = calibration.P.z;

	/* finally, rotate (x,y,z) by rotation vector geo.detector.R using precomputed matrix calibration.detector_rotation[3][3] */
	coordinates.x = calibration.detector_rotation[0][0] * x + calibration.detector_rotation[0][1] * y + calibration.detector_rotation[0][2] * z;
	coordinates.y = calibration.detector_rotation[1][0] * x + calibration.detector_rotation[1][1] * y + calibration.detector_rotation[1][2] * z;
	coordinates.z = calibration.detector_rotation[2][0] * x + calibration.detector_rotation[2][1] * y + calibration.detector_rotation[2][2] * z;
	#ifdef DEBUG_1_PIXEL
	if (verbosePixel) printf("\n   pixel xyz coordinates = (%g, %g, %g)\n",coordinates.x,coordinates.y,coordinates.z);
	#endif
	return coordinates;									/* return point_xyz coordinates */
}








/* allocate space and initialize the structure image_set, which contains the output */
void setup_depth_images(
int numImages)						/* number of input images, needed for .wire_scanned and .wire_positions */
{
	long	Ndepths;				/* number of depth points */
	long	i;
	size_t	j;						/* index into vectors, offset by MULTI_IMAGE_SKIPV */
	double	wireXdefault;			/* default value for wireX, wireY, wireZ, in case wire vector not present, really only for wireX */
	double	wireYdefault;
	double	wireZdefault;
	
	Ndepths = user_preferences.NoutputDepths;
	if (Ndepths<1 || numImages<1) {											/* nothing to do */
		image_set.depth_intensity.v =NULL;
		image_set.depth_resolved.v = NULL;
		image_set.depth_intensity.alloc = image_set.depth_intensity.size = 0;
		//image_set.depth_resolved.alloc = image_set.depth_resolved.size = 0;
		// set up the write-out buffers
		p_write_buffer[0].alloc = p_write_buffer[0].size = 0; 
		p_write_buffer[1].alloc = p_write_buffer[1].size = 0;
		//image_set.wire_scanned.alloc = image_set.wire_scanned.size = 0;
		// set up the read-in buffers
		p_read_buffer[0].alloc = p_read_buffer[0].size = 0; 
		p_read_buffer[1].alloc = p_read_buffer[1].size = 0;
		image_set.wire_positions.alloc = image_set.wire_positions.size = 0;
		return;
	}
	if (image_set.depth_intensity.v || image_set.depth_resolved.v || image_set.wire_positions.v || image_set.wire_scanned.v) {
		error("ERROR -- setup_depth_images(), one of 'image_set.*.v' is not NULL\n");
		exit(1);
	}
	if (image_set.depth_intensity.alloc || image_set.depth_resolved.alloc || image_set.wire_positions.alloc || image_set.wire_scanned.alloc) {
		error("ERROR -- setup_depth_images(), one of 'image_set.*.alloc' is not NULL\n");
		exit(1);
	}

	image_set.depth_intensity.v = calloc((size_t)Ndepths,sizeof(double));/* allocate space for array of doubles in the vector */
	if (!(image_set.depth_intensity.v)) { fprintf(stderr,"\ncannot allocate space for image_set.depth_intensity, %ld points\n",Ndepths); exit(1); }
	image_set.depth_intensity.alloc = image_set.depth_intensity.size = Ndepths;
	for (i=0; i<Ndepths; i++) image_set.depth_intensity.v[i] = 0.;		/* init to all zeros */

	/* allocate space for write-buffers as array of pointers to dmatrices */
//	image_set.depth_resolved.v = calloc((size_t)Ndepths,sizeof(dmatrix *));	/* allocate space for array of pointers to dmatrices (these are image) in the vector */
	p_write_buffer[0].v = calloc((size_t)Ndepths,sizeof(dmatrix *));
	p_write_buffer[1].v = calloc((size_t)Ndepths,sizeof(dmatrix *));
//	if (!(image_set.depth_resolved.v)) { fprintf(stderr,"\ncannot allocate space for image_set.depth_resolved, %ld points\n",Ndepths); exit(1); }
	if (!(p_write_buffer[0].v) || !(p_write_buffer[1].v)) { fprintf(stderr,"\ncannot allocate space for image_set.depth_resolved, %ld points\n",Ndepths); exit(1); }
//	image_set.depth_resolved.alloc = image_set.depth_resolved.size = Ndepths;
	p_write_buffer[0].alloc = p_write_buffer[1].alloc = Ndepths;
	p_write_buffer[0].size = p_write_buffer[1].size = Ndepths;
	for (i=0; i<Ndepths; i++) {
		/* pointers to dmatrix containing space for the image, initialized to zero */
//		image_set.depth_resolved.v[i] = dmatrix_calloc((size_t)(imaging_parameters.rows_at_one_time), (size_t)(imaging_parameters.nROI_j));	/* pointers to dmatrix containing space for the image, initialized to zero */
		p_write_buffer[0].v[i] = dmatrix_calloc((size_t)(imaging_parameters.rows_at_one_time), (size_t)(imaging_parameters.nROI_j));
		p_write_buffer[1].v[i] = dmatrix_calloc((size_t)(imaging_parameters.rows_at_one_time), (size_t)(imaging_parameters.nROI_j));
	}

#pragma message "Using a default value of 0 for wire X position"
	wireXdefault = isnan(in_header.wirebaseX) ? 0.0 : in_header.wirebaseX;	/* use wirebase as default if wire vector position not present */
	wireYdefault = isnan(in_header.wirebaseY) ? 0.0 : in_header.wirebaseY;
	wireZdefault = isnan(in_header.wirebaseZ) ? 0.0 : in_header.wirebaseZ;

/* *************** */
	/* allocate for .wire_scanned and .wire_positions for numImages input images */
	point_xyz wire_pos;
	image_set.wire_positions.v = calloc((size_t)numImages+1,sizeof(point_xyz));/* allocate space for array of doubles in the vector */
	if (!(image_set.wire_positions.v)) { fprintf(stderr,"\ncannot allocate space for image_set.wire_positions, %d points\n",numImages); exit(1); }
	image_set.wire_positions.alloc = numImages+1;						/* room allocated */
	image_set.wire_positions.size = numImages+1;						/* and set length used also */
	for (i=0; i<=numImages; i++) {
		j = i + MULTI_IMAGE_SKIPV;										/* used vector positions starting with point MULTI_IMAGE_SKIPV, not 0 */
		wire_pos.x = (in_header.xWire.N>j) ? in_header.xWire.v[j] : wireXdefault;	/* default value for wire X,Y,Z */
		wire_pos.y = (in_header.yWire.N>j) ? in_header.yWire.v[j] : wireYdefault;
		wire_pos.z = (in_header.zWire.N>j) ? in_header.zWire.v[j] : wireZdefault;
		image_set.wire_positions.v[i] = wirePosition2beamLine(wire_pos);/* correct raw wire position: PM500 distortion, origin, PM500 rotation, wire axis rotation */
		/* #pragma message "the wire position is corrected here when it is read in for: PM500, origin, rotation (by rho)" */
	}

	/* allocate space for read-buffers as array of pointers to dmatrices */
//	image_set.wire_scanned.v = calloc((size_t)numImages,sizeof(dmatrix *));	/* allocate space for array of pointers to dmatrices these are pieces of the input images */
	p_read_buffer[0].v = calloc((size_t)numImages,sizeof(dmatrix *));
	p_read_buffer[1].v = calloc((size_t)numImages,sizeof(dmatrix *));
//	if (!(image_set.wire_scanned.v)) { fprintf(stderr,"\ncannot allocate space for image_set.wire_scanned, %d points\n",numImages); exit(1); }
	if (!(p_read_buffer[0].v) || !(p_read_buffer[1].v)) { fprintf(stderr,"\ncannot allocate space for p_read_buffer, %d points\n",numImages); exit(1); }
//	image_set.wire_scanned.alloc = numImages;							/* room allocated */
	p_read_buffer[0].alloc = p_read_buffer[1].alloc = numImages;
//	image_set.wire_scanned.size = 0;									/* but nothing set */
	p_read_buffer[0].size = p_read_buffer[1].size = 0;
	for (i=0; i<numImages; i++) {
		/* pointers to dmatrix containing space for the image */
//		image_set.wire_scanned.v[i] = dmatrix_calloc((size_t)(imaging_parameters.rows_at_one_time), (size_t)(imaging_parameters.nROI_j));	/* pointers to dmatrix containing space for the image */
		p_read_buffer[0].v[i] = dmatrix_calloc((size_t)(imaging_parameters.rows_at_one_time), (size_t)(imaging_parameters.nROI_j));
		p_read_buffer[1].v[i] = dmatrix_calloc((size_t)(imaging_parameters.rows_at_one_time), (size_t)(imaging_parameters.nROI_j));
	}
//	image_set.wire_scanned = p_read_buffer[0]; // now image_set.wire_scanned is the same as p_read_buffer[0], this may not be necessary here
}
/*	for (i = (long)(user_preferences.depth_start / user_preferences.depth_resolution); i <= (long)(user_preferences.depth_end / user_preferences.depth_resolution); i ++ ) {
 *		image = dmatrix_calloc(imaging_parameters.nROI_i, imaging_parameters.rows_at_one_time);
 *		image_set.depth_resolved.push_back(image);
 *		image_set.depth_intensity.push_back(0);
 *	}
 */


/* this just sets the images in image_set to zero, it does NOT de-allocate the space they use */
/* for .depth_resolved & .wire_scanned, set all the elements to zero, assumes space already allocated */
// Change: input of this function is changed from ws_image_set to vvector for more flexible handling of the read/write buffers
void clear_depth_images(
	vvector *is) //	ws_image_set *is)
{
	size_t i;
//	for (i=0; i < is->depth_resolved.alloc; i++) dmatrix_set_zero(is->depth_resolved.v[i]);
//	for (i=0; i < is->wire_scanned.alloc; i++) dmatrix_set_zero(is->wire_scanned.v[i]);
	for (i=0; i < is->alloc; i++) dmatrix_set_zero(is->v[i]);
}

/* delete the images stored in image_set, and deallocate everything too, do: .wire_scanned, .depth_resolved, and .wire_positions, but NOT .depth_intensity */
// Change: also deallocates the read/write buffers
void delete_images(void)
{
	dmatrix * image;

	/* de-allocate and zero out read-buffers*/
//	while (image_set.wire_scanned.alloc) {
	while (p_read_buffer[0].alloc) {
		//image = image_set.wire_scanned.v[--(image_set.wire_scanned.alloc)];
		image = p_read_buffer[0].v[--(p_read_buffer[0].alloc)];
		if (image != 0) dmatrix_free(image);
		image = p_read_buffer[1].v[--(p_read_buffer[1].alloc)];
		if (image != 0) dmatrix_free(image);
	}
	//CHECK_FREE(image_set.wire_scanned.v)
	CHECK_FREE(p_read_buffer[0].v);
	CHECK_FREE(p_read_buffer[1].v);
	image_set.wire_scanned.alloc = image_set.wire_scanned.size = 0;
	p_read_buffer[0].alloc = p_read_buffer[0].size = 0;	
	p_read_buffer[1].alloc = p_read_buffer[1].size = 0;	

	/* de-allocate and zero out write-buffers*/
//	while (image_set.depth_resolved.alloc) {
	while (p_write_buffer[0].alloc) {
		//image = image_set.depth_resolved.v[--(image_set.depth_resolved.alloc)];
		image = p_write_buffer[0].v[--(p_write_buffer[0].alloc)];
		if (image != 0) dmatrix_free(image);
		image = p_write_buffer[1].v[--(p_write_buffer[1].alloc)];
		if (image != 0) dmatrix_free(image);
	}
	//CHECK_FREE(image_set.depth_resolved.v)
	CHECK_FREE(p_write_buffer[0].v);
	CHECK_FREE(p_write_buffer[1].v);
	image_set.depth_resolved.alloc = image_set.depth_resolved.size = 0;
	p_write_buffer[0].alloc = p_write_buffer[0].size = 0;	
	p_write_buffer[1].alloc = p_write_buffer[1].size = 0;	

	/* de-allocate and zero out .wire_positions */
	CHECK_FREE(image_set.wire_positions.v)
	image_set.wire_positions.alloc = image_set.wire_positions.size = 0;
}
/*
 *	void delete_images()
 *	{
 *		dmatrix * image;
 *
 *		while (image_set.wire_scanned.size())
 *		{
 *			image = image_set.wire_scanned.back();
 *			if (image != 0) dmatrix_free ( image );
 *			image_set.wire_scanned.pop_back();
 *		}
 *		//image_set.wire_scanned.clear();
 *		image_set.wire_positions.clear();
 *	}
*/

/* subtract from each image from its following image */
void get_difference_images(void)
{
	size_t m;
	for (m=0; m < (image_set.wire_scanned.size)-1; m++) {
		#ifdef DEBUG_1_PIXEL
		if (verbosePixel) printf("pixel[%d,%d] raw image[% 3d] = %g\n",pixelTESTi,pixelTESTj,(int)m,dmatrix_get(image_set.wire_scanned.v[m], pixelTESTi -  imaging_parameters.current_selection_start, pixelTESTj));
		#endif
		dmatrix_sub(image_set.wire_scanned.v[m], image_set.wire_scanned.v[m+1]);	/* dmatrix_sub(a,b) -->  a -= b */
	}
}

/*  FILE IO  */



/* sets many of the values in the gobal structure imaging_parameters */
/* get header information from first and last input images, this is called at start of program */
void getImageInfo(
char	*infile,						/* base file input name */
int		file_num_start,					/* index to first input file */
int		file_num_end,					/* index to last input file */
char	*normalization,					/* optional tag for normalization */
Dvector *normalVector)					/* normalization vector made using 'normalization' */
{
	point_xyz wire_pos;
	char	filename[FILENAME_MAX];						/* full filename */

	#ifndef PRINT_HDF5_MESSAGES
	H5Eset_auto2(H5E_DEFAULT,NULL,NULL);				/* turn off printing of HDF5 errors */
	#endif

#ifdef MULTI_IMAGE_FILE
	strncpy(filename,infile,FILENAME_MAX-1);
#else
	sprintf(filename,"%s%d.h5",fn_base,file_num_start);
#endif
	if (readHDF5header(filename, &in_header)) goto error_path;
	if (verbose > 0) {
		printf("\n");
		printHeader(&in_header);
		printf("\n");
	}
	imaging_parameters.nROI_i = (int)in_header.xdim;			/* number of binned pixels along the x direction of one image */
	imaging_parameters.nROI_j = (int)in_header.ydim;			/* number of binned pixels in one full stored image along detector y */
	imaging_parameters.in_pixel_type = in_header.itype;	/* type (e.g. float, int, ...) of a pixel value, uses the WinView pixel types */
	imaging_parameters.in_pixel_bytes = in_header.isize;	/* number of bytes used to specify one pixel strength (bytes) */
	imaging_parameters.starti = (int)in_header.startx;		/* definition of the ROI for a full image all pixel coordinates are zero based */
	imaging_parameters.endi   = (int)in_header.endx;
	imaging_parameters.startj = (int)in_header.starty;
	imaging_parameters.endj   = (int)in_header.endy;
	imaging_parameters.bini   = (int)in_header.groupx;
	imaging_parameters.binj   = (int)in_header.groupy;

	positionerType = positionerTypeFromFileTime(in_header.fileTime);		/* sets global value position type, needed for wirePosition2beamLine() */
	imaging_parameters.NinputImages = file_num_end - file_num_start + 1;	/* number of input image files to process */
/*	imaging_parameters.NinputImages *= in_header.Nimages;					// total number of input images (number of files * imags per file) */
	imaging_parameters.NinputImages *= (in_header.Nimages-MULTI_IMAGE_SKIP);/* total number of input images (number of files * imags per file) */

#ifdef MULTI_IMAGE_FILE
	wire_pos.x = (in_header.xWire.N>0) ? in_header.xWire.v[0] : NAN;		/* first wire position */
	wire_pos.y = (in_header.yWire.N>0) ? in_header.yWire.v[0] : NAN;
	wire_pos.z = (in_header.zWire.N>0) ? in_header.zWire.v[0] : NAN;
	imaging_parameters.wire_first_xyz = wirePosition2beamLine(wire_pos);	/* correct raw wire position: PM500 distortion, origin, PM500 rotation, wire axis rotation */

	/* wire position of the last image in the wire scan */
	wire_pos.x = (in_header.xWire.N>0) ? in_header.xWire.v[in_header.xWire.N-1] : NAN;
	wire_pos.y = (in_header.yWire.N>0) ? in_header.yWire.v[in_header.yWire.N-1] : NAN;
	wire_pos.z = (in_header.zWire.N>0) ? in_header.zWire.v[in_header.zWire.N-1] : NAN;
//	wire_pos.x = in_header.xWire.v[in_header.xWire.N-1];
//	wire_pos.y = in_header.yWire.v[in_header.yWire.N-1];
//	wire_pos.z = in_header.zWire.v[in_header.zWire.N-1];
	imaging_parameters.wire_last_xyz = wirePosition2beamLine(wire_pos);		/* correct raw wire position: PM500 distortion, origin, PM500 rotation, wire axis rotation */
#endif

	empty_Dvector(normalVector);
	if (normalization[0]) (*normalVector).N = readHDF5oneHeaderVector(infile, normalization, normalVector);

	size_t	i;
	#ifdef TYPICAL_mA
	if (!strcmp(normalization,"mA")) {							/* for beam current, divide by 104 */
		for (i=0;i<normalVector->N;i++) normalVector->v[i] /= TYPICAL_mA;
	}
	#endif
	#ifdef TYPICAL_cnt3
	if (!strcmp(normalization,"cnt3")) {
		for (i=0;i<normalVector->N;i++) normalVector->v[i] /= TYPICAL_cnt3;
	}
	#endif

	return;

	error_path:
	printf("\t%s;%s\n",filename,infile);
	error("getImageInfo(), could not read first of last header information, imaging_parameters.* not set\n");
	exit(1);
}


/* loads first image into intensity_map, and finds cutoff, the intensity that decides which pixels to use (uses percent to find cutoff) */
/* set the global 'cutoff' */
void get_intensity_map(
char	*infile,					/* base name of image file */
int		file_num_start)				/* index of image file */
{
	size_t	dimi = imaging_parameters.nROI_i;	/* full size of image */
	size_t	dimj = imaging_parameters.nROI_j;
	char	filename[FILENAME_MAX];		/* full filename */

	intensity_map = dmatrix_calloc(dimi, dimj);	/* get memory for one whole image */

#ifdef MULTI_IMAGE_FILE
	strncpy(filename,infile,FILENAME_MAX-1);
	file_num_start=file_num_start;
#else
	sprintf(filename,"%s%d.h5",filename_base,file_num_start);
#endif

	/* read the first image (actually MULTI_IMAGE_SKIP), reads data (of any kind) into a double array */
	if (HDF5ReadROIdoubleSlice(filename, "entry1/data/data", &(intensity_map->data), 0,(dimi-1), 0,(dimj-1), &in_header, MULTI_IMAGE_SKIP)) {
		error("\nFailed to open intensity map file");
		exit(1);
	}



#ifdef DEBUG_1_PIXEL
printf(" ***in get_intensity_map()...\n");
printf("pixelTESTi=%d, pixelTESTj=%d    intensity_map->data[%u*%lu+%u] = %lg\n",pixelTESTi,pixelTESTj,pixelTESTi,intensity_map->tda,pixelTESTj,intensity_map->data[pixelTESTi*intensity_map->tda+pixelTESTj]);
printf("pixelTESTi=%d, pixelTESTj+1=%d    intensity_map->data[%u*%lu+%u] = %lg\n",pixelTESTi,pixelTESTj+1,pixelTESTi,intensity_map->tda,pixelTESTj+1,intensity_map->data[pixelTESTi*intensity_map->tda+pixelTESTj+1]);
printPieceOf_dmatrix(pixelTESTi-2, pixelTESTi+2, pixelTESTj-2, pixelTESTj+2, intensity_map);
printf(" ***done with get_intensity_map()\n");
#endif


	/* remove image noise below certain value */
	size_t	sort_len = dimj*dimi;
	double *intensity_sorted;		/* std::vector<double> intensity_sorted;		This line is in WireScan.h, move it to here */
	intensity_sorted = (double*)calloc(sort_len,sizeof(double));
	if (!intensity_sorted) { fprintf(stderr,"\nCould not allocate intensity_sorted %ld bytes in get_intensity_map()\n",sort_len); exit(1); }

	memcpy(intensity_sorted,intensity_map->data,dimi*dimj*sizeof(double));
	qsort(intensity_sorted,(dimj*dimi), sizeof(double), (void *)compare_double);

	if (norm_exponent > 0.0) calc_intensity_norm(intensity_sorted);

/*
#if DEBUG
	if (intensity_norm) {
		FILE *f;
		f = fopen("intensity_norm_image.rawBin","w");
		fwrite((void*)(intensity_norm->data), 8, 2048*2048, f);
		fclose(f);
	}
#endif
*/

	cutoff = (int)intensity_sorted[ (size_t)floor((double)sort_len * MIN((100.0 - percent)/100.0,1.)) ];
	cutoff = MAX(cutoff,1);
	CHECK_FREE(intensity_sorted);

	if (verbose > 0) printf("\nignoring pixels with a value less than %d",cutoff);
	fflush(stdout);
	return;
}



int calc_intensity_norm(		/* calc normalization image, image0 ^ norm_exponent, sets intensity_norm */
double *intensity_sorted)		/* all the intensities, sorted  */
{
	size_t	dimi = imaging_parameters.nROI_i;				/* full size of image */
	size_t	dimj = imaging_parameters.nROI_j;
	size_t	i,j;

	if (!(norm_exponent > 0.0)) {							/* norm_exponent is invalid, do nothing */
		intensity_norm = NULL;
		return 1;
	}
	else if (!(norm_threshold>0.0)) {						/* norm_threshold is invalid, use average + 5 standard deviations */
		size_t	Nhalf;
		double N;
		N = floor( 0.5 * (double)(dimj*dimi) );				/* use lowest half of the pixels from intensity_map to describe bkg */
		Nhalf = (size_t) N;
		if (Nhalf < 10) {									/* too few pixels */
			if (verbose > 0) printf("\nHalf of the image is only %ld pixels, that is too few of auto-calc of normalization threshold",Nhalf);
			return 1;
		}
		double Xi2=0.0, Xi=0.0, avg, sdev, valuei;			/* used to calc average and standard deviation */
		for (i=0;i<Nhalf;i++) {
			valuei = intensity_sorted[i];
			Xi  += valuei;									/* Sum{ x[i] } */
			Xi2 += valuei*valuei;							/* Sum{ x[i]^2 } */
		}
		avg = Xi / N;										/* Sum{ x[i] } / N */
		sdev = sqrt( (Xi2 - N*avg*avg) / (N-1) );			/* sqrt[ 1/(N-1) * Sum{ (x[i] - avg)^2 } ] */
		norm_threshold = avg + 5*sdev;
	}
	if (!(norm_threshold>0.0)) {							/* norm_threshold is invalid, do nothing */
		if (verbose > 0) printf("\nCannot make normalization image with norm_threshold = %g",norm_threshold);
		return 1;
	}
	double	raw, scaled;
	double	baseValue=pow(norm_threshold,-norm_exponent);	/* values for pixels < norm_threshold, you multiply by baseValue, not divide */
	intensity_norm = dmatrix_calloc(dimi, dimj);			/* allocate memory for normalization image */
	for (j=0;j<dimj;j++) {
		for (i=0;i<dimi;i++) {
			raw = dmatrix_get(intensity_map, i, j);
			if (raw < norm_threshold) scaled = baseValue;	/* pixel raw < norm_threshold, use base value */
			else scaled = pow(raw,-norm_exponent);			/* multiply by this scaled value, not divide */
			dmatrix_set(intensity_norm, i, j, scaled);	/* set the normalization value */
		}
	}
	if (verbose > 0) printf("\nMade normalization image using threshold=%g, exponent=%g\n",norm_threshold,norm_exponent);
	return 0;
}



#pragma message "This routine may be WRONG, is now fixed?"
void readImageSet(
char	*fn_base,					/* base name of input image files */
int		ilow,						/* range of ROI to read from file */
int		ihi,
int		jlow,						/* for best speed, jhi-jlow+1 == ydim */
int		jhi,
int		file_num_start,				/* index of first input image */
int		Nimages,					/* number of images to read */
Dvector *normalVector)				/* normalization vector made using 'normalization' */
{
	int		f;
	char	filename[FILENAME_MAX];			/* full filename */

	size_t dimi = ihi - ilow + 1;
	size_t dimj = jhi - jlow + 1;
	if (verbose > 1) {
		printf("\n\tload %d new images of x=[%d,%d] nx=%lu,  y=[%d,%d] ny=%lu,  (%lu pixels)...",Nimages,ilow,ihi,dimi,jlow,jhi,dimj,dimi*dimj);
		fflush(stdout);
	}
	file_num_start=file_num_start;	/* do this to shut up compiler message */


#ifdef MULTI_IMAGE_FILE
	strncpy(filename,fn_base,FILENAME_MAX-1);
#endif
	for (f=0; f<Nimages; f++) {
		#ifdef DEBUG_1_PIXEL
		if (f==0) verbosePixel=1;
		#endif

#ifndef MULTI_IMAGE_FILE
		sprintf(filename,"%s%d.h5",fn_base,f+file_num_start);
		readSingleImage(filename, f,0, ilow,ihi, jlow,jhi);		/* load a single image from disk, slice=0 */
#else
		readSingleImage(filename, f,f+MULTI_IMAGE_SKIP, ilow,ihi, jlow,jhi);	/* load a single image from disk */
#endif

		#ifdef DEBUG_1_PIXEL
		verbosePixel=0;
		#endif
	}
	if (verbose > 2) printf("\n\tdone loading images into memory");

	if (normalVector->N >= (size_t)Nimages) {
		if (verbose > 2) printf("\n\tnormalize input images by ion chamber or beam current");
		for (f=0; f<Nimages; f++) dmatrix_scale(p_read_buffer[p_ibuff].v[f],normalVector->v[f]);
		/* dmatrix_scale(a, x):  a(i,j) <--  x * a(i,j) is stored in a. */
	}

	if (norm_exponent>0) {
		if (verbose > 2) printf("\n\tnormalize input images by (image1)^%g, with norm_threshold = %g  and  norm_rescale = %g",norm_exponent,norm_threshold,norm_rescale);
		/* first make a new matrix from intensity_norm that covers this ROI */
		//dmatrix *image = image_set.wire_scanned.v[0];
		dmatrix *image = p_read_buffer[p_ibuff].v[0];
		dmatrix *roi_scaling;
		roi_scaling = dmatrix_calloc(image->size1, image->size2);		/* allocate normalization roi, must match size of image_set images (NOT dimi,dimj)  */
		double	value;
		size_t	i,j;
		for (j = 0; j < dimj; j++) {
			for (i = 0; i < dimi; i++) {
				value = dmatrix_get(intensity_norm, i+ilow, j+jlow);
				dmatrix_set(roi_scaling, i, j, value);					/* set roi_scaling to intensity_norm values */
			}
		}

		/* apply roi_scaling, multiply each of the new roi's in image_set by roi_scaling, element by element */
		for (f=0; f<Nimages; f++) {
			//dmatrix_mul_elements (image_set.wire_scanned.v[f], roi_scaling);
			dmatrix_mul_elements (p_read_buffer[p_ibuff].v[f], roi_scaling);
			/* dmatrix_mul_elements(a,b):  a(i,j) <-- a(i,j) * b(i,j) is stored in a, b remains unchanged */
		}
		dmatrix_free(roi_scaling);					/* done with roi, free it */
	}

	if (verbose > 2) printf("\nimages ready for depth resolving\n");
	fflush(stdout);
}

#pragma message "This routine may be WRONG for multiimage files, now at least LOOKS right (and may be too)"
void readSingleImage(
char	*filename,							/* fully qualified file name */
int		imageIndex,							/* index to images, image number that appears in the full file name - first image number */
int		slice,								/* index to particular image in file for 3D files, for 2D images use slice=0 */
int		ilow,								/* range of ROI to read from file */
int		ihi,								/* these are in terms of the image stored in the file, not raw un-binned pixels of the detector */
int		jlow,
int		jhi)
{
	int		i,j;

	/* matrix to store image */
	dmatrix *image;
	int dimi = ihi - ilow + 1;
	int dimj = jhi - jlow + 1;				/* for best performance jlow-jhi+1 == ydim */
	double *buf = NULL;

	/* set image_set.wire_scanned.size to be big enough (probably just increment .size by 1) */
	//if ( (unsigned int)imageIndex >= image_set.wire_scanned.alloc) {/* do not have enough dmatrix arrays allocated */
	if ( (unsigned int)imageIndex >= p_read_buffer[p_ibuff].alloc) {/* do not have enough dmatrix arrays allocated */
		//fprintf(stderr,"\nERROR -- readSingleImage(), need room for image #%d, but only have .alloc = %ld",imageIndex,image_set.wire_scanned.alloc);
		fprintf(stderr,"\nERROR -- readSingleImage(), need room for image #%d, but only have .alloc = %ld",imageIndex,p_read_buffer[p_ibuff].alloc);
		exit(2);
	}
	//image_set.wire_scanned.size = imageIndex+1;		/* number of input images read so far */
	p_read_buffer[p_ibuff].size = imageIndex+1;
	//image = image_set.wire_scanned.v[imageIndex];
	image = p_read_buffer[p_ibuff].v[imageIndex];

	/* read data (of any kind) into a double array */
	#ifdef DEBUG_ALL
	slowWay = ((size_t)(jhi-jlow+1)<(in_header.ydim)) || slowWay;	/* check stripe orientation */
	#endif

	if (HDF5ReadROIdoubleSlice(filename, "entry1/data/data", &buf, (long)ilow, (long)ihi, (long)jlow, (long)jhi, &in_header,(size_t)slice)) {
		error("Error reading image from file");
		exit(1);
	}

	/* transfer into a dmatrix, cannot do memcpy because last stripe is narrower & so there could be a mismatch */
 	int tda = image->tda;
	for (i = 0; i < dimi; i++) {
		for (j = 0; j < dimj; j++) {
			//dmatrix_set(image, (size_t)i, (size_t)j, buf[i*dimj + j]);
			image->data[i * tda + j] = buf[i*dimj + j];
		}
	}

#ifdef DEBUG_1_PIXEL
if (verbosePixel && ilow<=pixelTESTi && pixelTESTi<=ihi) {
	printf("\n ++++++++++ in readSingleImage(), finished reading i=[%d, %d], j=[%d, %d]",ilow,ihi,jlow,jhi);
	printf("\n ++++++++++ pixel[%d,%d] = %g,     ROI: i=[%d,%d], j=[%d, %d],  Nj=%d",pixelTESTi,pixelTESTj, buf[dimj*(pixelTESTi-ilow) + pixelTESTj],ilow,ihi,jlow,jhi,dimj);
	printf("\n ++++++++++ matrix pixel[%d-%d,%d] = %g",pixelTESTi,ilow,pixelTESTj,dmatrix_get(image, pixelTESTi-ilow, pixelTESTj));
	printf("\n            dmatrix.size1 = %lu,  dmatrix.size2 = %lu,  dmatrix.tda = %lu",image->size1,image->size2,image->tda);
	fflush(stdout);
}
#endif

	CHECK_FREE(buf);
}



/* write correct headers and blank (all zero) images for every output HDF5 file */
void writeAllHeaders(
char	*fn_in_first,				/* full path (including the .h5) to the first input file */
char	*fn_out_base,				/* full path up to index of the reconstructed output files */
int		file_num_start,				/* first output file number */
int		file_num_end)				/* last output file number */
{
	size_t	pixel_size = (user_preferences.out_pixel_type < 0) ? imaging_parameters.in_pixel_bytes : WinView_itype2len(user_preferences.out_pixel_type);
	size_t	NpixelsImage = imaging_parameters.nROI_i * imaging_parameters.nROI_j;	/* number of pixels in one whole image */
	int		i;
	char	filenameTemp[L_tmpnam];			/* a temp file, delete at end of this routine */
	char	finalTemplate[L_tmpnam];		/* final template file */
	char	*buf=NULL;
	int		dims[2] = {(int)output_header.xdim, (int)output_header.ydim};
	hid_t	file_id=0;
	char	fout_name[FILENAME_MAX];

	/* buf is the same size of one of the output images and is initialized by calloc() to all zeros */
	buf = (char*)calloc(NpixelsImage , pixel_size);
	if (!buf) {fprintf(stderr,"\nCould not allocate buf %lu bytes in writeAllHeaders()\n", NpixelsImage*imaging_parameters.in_pixel_bytes); exit(1);}

	/* create the first output file using fn_in_first as a template */
	strcpy(filenameTemp,tmpnam(NULL));					/* get unique filenames */
	strcpy(finalTemplate,tmpnam(NULL));

/*	
 *	/tmp/ccz301rF.o: In function 'writeAllHeaders':
 *	WireScan.c:(.text+0xbfe): warning: the use of 'tmpnam' is dangerous, better use 'mkstemp'
 */	
	
#ifdef FIX_ME_SLOW
	//time_t	sec0 = time(NULL);
	//copyFile(fn_in_first,filenameTemp,1);
	//printf("  copy took %ld sec\n", time(NULL) - sec0);
#endif

	// make new template file without selelcted objects, adds "depth", and write new "data" to it
	makeTemplateFile(fn_in_first,filenameTemp,buf);
	// release the lock so the other thread can start reading in the first stripe
	if(!omp_test_lock(&hdf_io))		omp_unset_lock(&hdf_io); 

	if(verbose>1) printf("> thread #%d: Done with creating template files, start copying them into output files;\n",omp_get_thread_num());

#ifdef FIX_ME_SLOW
	// /* delete the main data in file, and delete the wire positions from the template otuput file */
	// if ((file_id=H5Fopen(filenameTemp,H5F_ACC_RDWR,H5P_DEFAULT))<=0) { fprintf(stderr,"error after file open, file_id = %d\n",file_id); goto error_path; }
	// if (deleteDataFromFile(file_id,"entry1/data","data")) { fprintf(stderr,"error trying to delete \"/entry1/data/data\", file_id = %d\n",file_id); goto error_path; }		/* delete the data */
	// if (deleteDataFromFile(file_id,"entry1/wire","wireX")) { fprintf(stderr,"error trying to delete \"/entry1/wire/wireX\", file_id = %d\n",file_id); goto error_path; }	/* delete the wire positions */
	// if (deleteDataFromFile(file_id,"entry1/wire","wireY")) { fprintf(stderr,"error trying to delete \"/entry1/wire/wireY\", file_id = %d\n",file_id); goto error_path; }
	// if (deleteDataFromFile(file_id,"entry1/wire","wireZ")) { fprintf(stderr,"error trying to delete \"/entry1/wire/wireZ\", file_id = %d\n",file_id); goto error_path; }
	// if (deleteDataFromFile(file_id,"entry1/wire","wirescan")) { fprintf(stderr,"error trying to delete \"/entry1/wire/wirescan\", file_id = %d\n",file_id); goto error_path; }
	// if (deleteDataFromFile(file_id,"entry1/wire","H_downCts")) { fprintf(stderr,"error trying to delete \"/entry1/wire/H_downCts\", file_id = %d\n",file_id); goto error_path; }
	// if (deleteDataFromFile(file_id,"entry1/wire","H_upCts")) { fprintf(stderr,"error trying to delete \"/entry1/wire/H_upCts\", file_id = %d\n",file_id); goto error_path; }
	// if (deleteDataFromFile(file_id,"entry1/wire","wirebaseX")) { fprintf(stderr,"error trying to delete \"/entry1/wire/wirebaseX\", file_id = %d\n",file_id); goto error_path; }
	// if (deleteDataFromFile(file_id,"entry1/wire","wirebaseY")) { fprintf(stderr,"error trying to delete \"/entry1/wire/wirebaseY\", file_id = %d\n",file_id); goto error_path; }
	// if (deleteDataFromFile(file_id,"entry1/wire","wirebaseZ")) { fprintf(stderr,"error trying to delete \"/entry1/wire/wirebaseZ\", file_id = %d\n",file_id); goto error_path; }
	// if (deleteDataFromFile(file_id,"entry1","wire")) { fprintf(stderr,"error trying to delete \"/entry1/wire\", file_id = %d\n",file_id); goto error_path; }
	// /* write the depth, reconstruction program name, and norm_* values */
	// if (writeDepthInFile(file_id,0.0)) { fprintf(stderr,"error writing the depth in the output template file, file_id = %d\n",file_id); goto error_path; }

#warning "FIX writeReconPgmInFile()"
//	if (writeReconPgmInFile(file_id)) { fprintf(stderr,"error writing the reconstruction program name in output template file, file_id = %d\n",file_id); goto error_path; }
//	if (norm_exponent > 0.0 )  {
//		if (writeImageNormalizationsInFile(file_id, norm_exponent, norm_threshold, norm_rescale)) { fprintf(stderr,"error writing the norm_* values in the output template file, file_id = %d\n",file_id); goto error_path; }
//	}
//	if (cosmic)  {
//		if (writeImageCosmicInFile(file_id,cosmic)) { fprintf(stderr,"error writing the cosmic in the output template file, file_id = %d\n",file_id); goto error_path; }
//	}
//	if (H5Fclose(file_id)) { fprintf(stderr,"file close error\n"); goto error_path; } else file_id = 0;

	// /* make a re-packed version of file */
	// if (repackFile(filenameTemp,finalTemplate)) { fprintf(stderr,"error after calling repackFile()\n"); goto error_path; }
	
	// /* re-create the /entry1/data/data, same full size, but with appropriate data type */
	// if(createNewData(finalTemplate,"entry1/data/data",2,dims,getHDFtype(output_header.itype))) fprintf(stderr,"error after calling createNewData()\n");

	/* write entire image back into file, but using different data type, using HDF5WriteROI() */
	/*	for (i=0;i<(1024*1024);i++) wholeImage[i] = wholeImage[i] & 0x7FFFFFFF;	// trim off high order bit */
	//if ((HDF5WriteROI(finalTemplate,"entry1/data/data",buf,0,(output_header.xdim)-1,0,(output_header.ydim)-1,getHDFtype(output_header.itype),&output_header)))
	//	{ fprintf(stderr,"error from HDF5WriteROI()\n"); goto error_path; }
#endif

	/* create each of the output files with the correct depth in it */
	//for (i = file_num_start; i <= file_num_end; i++) write1Header(finalTemplate,fn_out_base, i);
	for (i = file_num_start; i <= file_num_end; i++)	// duplicate template file with correct name, this is SLOW
	{	sprintf(fout_name,"%s%d.h5",fn_out_base,i);		//  therefore can have anothre thread reading h5 file at the same time
		copyFile(filenameTemp,fout_name,1);
	}
	omp_set_lock(&hdf_io);							// wait until the other thread is done with reading h5 file
	if(verbose>1) printf("\n> thread #%d: start writing depth into output files\n",omp_get_thread_num());
	for (i = file_num_start; i <= file_num_end; i++) write1Header(fn_out_base, i);
	omp_unset_lock(&hdf_io);

	/* delete both unused files */

#ifdef FIX_ME_SLOW
	deleteFile(filenameTemp);
//	deleteFile(finalTemplate);
#endif
	CHECK_FREE(buf);
	return;

	error_path:
	CHECK_FREE(buf);
	exit(1);
}

/* Make a template file for reconstructed images from a raw file */
int makeTemplateFile(
	const char* filenameIn, 		// raw file name
	const char* filenameTemplate, 	// template file name
	char* buf)						// data buffer
{
	hid_t fid_src, fid_dst;
	int	dims[2] = {(int)output_header.xdim, (int)output_header.ydim};

	// open file
	fid_src = H5Fopen(filenameIn,H5F_ACC_RDONLY,H5P_DEFAULT);

	// create new file
	fid_dst = H5Fcreate(filenameTemplate,H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	// traverse & copy desired objects by calling cp_h5obj
	#if H5_VERSION_GE(1,12,0)
	H5Ovisit3(fid_src, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, cp_h5obj, &fid_dst, H5O_INFO_ALL);
	#else
	H5Ovisit(fid_src, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, cp_h5obj, &fid_dst);
	#endif

	// re-create "data" in new file
	if (createNewData(fid_dst,"entry1/data/data",2,dims,getHDFtype(output_header.itype)))
	{
		fprintf(stderr, "ERROR in makeTemplateFile(): createNewData() failed.\n");
		exit(1);
	}
	
	// write data from buffer
	HDF5WriteROI(fid_dst,"entry1/data/data",buf,0,(output_header.xdim)-1,0,(output_header.ydim)-1,getHDFtype(output_header.itype),&output_header);

	// write depth etc into new file
	writeDepthInFile(fid_dst,0.0);
	
	// norm-by-exponent
	if (norm_exponent > 0.0 )  {
		if (writeImageNormalizationsInFile(fid_dst, norm_exponent, norm_threshold, norm_rescale)) { fprintf(stderr,"error writing the norm_* values in the output template file, file_id = %d\n",fid_dst); goto error_path; }
	}
	// cosmic
	if (cosmic)  {
		if (writeImageCosmicInFile(fid_dst,cosmic)) { fprintf(stderr,"error writing the cosmic in the output template file, file_id = %d\n",fid_dst); goto error_path; }
	}
	
	// close files
	if(H5Fclose(fid_src)<0) {	printf("Error closing raw file %s\n",filenameIn); exit(1);	}
	if(H5Fclose(fid_dst)<0) {	printf("Error closing template file %s\n",filenameTemplate); exit(1);}
	
	return 0;
	error_path:
		1;
}


/* write the correct header and a single image of all zeros for an output HDF5 file */
void write1Header(
//char	*finalTemplate,				/* template output file, a cleaned up version of the input file */
char	*fn_base,					/* full path up to index of the reconstructed output files */
int		file_num)					/* output file number to write */
{
	double	depth;						/* depth from origin, measured from this Si standard (micron) */
	char	fout_name[FILENAME_MAX];	/* full name of file to write */
	hid_t	file_id;
	int		err;

	depth = index_to_beam_depth(file_num);
	if (fabs(depth)>1e7) {
		fprintf(stderr,"\nwrite1Header(), depthSi=%g out of range +-10e6\n",depth);
		exit(1);
	}

//	/* duplicate template file with correct name */
	sprintf(fout_name,"%s%d.h5",fn_base,file_num);
//	copyFile(finalTemplate,fout_name,1);

	/* open file, write the depth, & close the file */
	if ((file_id=H5Fopen(fout_name,H5F_ACC_RDWR,H5P_DEFAULT))<=0) { fprintf(stderr,"ERROR -- H5Fopen(), cannot open the file '%s'\n",fout_name); exit(1); }
	if ((err=writeDepthInFile(file_id,depth))) { fprintf(stderr,"ERROR -- writeDepthInFile(), cannot write the depth to the file '%s'\n",fout_name); exit(1); }
	if ((err = H5Fclose(file_id))) { fprintf(stderr,"ERROR -- write1Header(), file close error = %d\n",err); exit(1); }
}



/* write out one stripe of the reconstructed image, and the correct depth */
/* multiple image version */
void write_depth_data(
size_t	start_i,					/* start i of this stripe */
size_t	end_i,						/* end i of this stripe */
char	*fn_base)					/* base name of file, just add index and .h5 */
{
	int file_num_end = user_preferences.NoutputDepths - 1;
	int m;
	char fileName[FILENAME_MAX];

	if (verbose>2) {
		size_t pixels = (size_t)(end_i-start_i+1) * (size_t) imaging_parameters.nROI_j;
		printf("\nWrite out the final processed ROI's, i=[%lu, %lu],  j=[%d, %d],  %lu pixels",start_i,end_i,0,(imaging_parameters.nROI_j - 1),pixels);
		fflush(stdout);
	}

	/*	if (verbose == 2) printf("     "); */
	for (m=0; m <= file_num_end; m++) {									/* output file numbers are in the range [0, file_num_end] */
		sprintf(fileName,"%s%d.h5",fn_base,m);
		write_depth_datai(m, start_i, end_i, fileName);
	}
}

/* single image version, write both the depth and the data */
void write_depth_datai(
int		file_num,					/* the file number to write also the index into the number of output images, zero based */
size_t	start_i,					/* start and end i of this stripe */
size_t	end_i,
char	*fileName)					/* fully qualified name of file */
{
	int		output_pixel_type, pixel_size;
	//dmatrix *outMatrix = image_set.depth_resolved.v[file_num];		/* pointer to the dmatrix with data to write */
	dmatrix *outMatrix = p_write_buffer[p_ibuff].v[file_num];		/* pointer to the dmatrix with data to write */
	hid_t file_id;
	herr_t err;

	if (outMatrix->size2 != outMatrix->tda) {							/* I will be assuming that this is true, so check here */
		error("write_depth_datai(), outMatrix.size1 != outMatrix.tda");	/* this is needed so I can just write the outMatrix->data directly */
		exit(1);
	}

#ifdef DEBUG_1_PIXEL
	if (start_i<=pixelTESTi && pixelTESTi<=end_i)
		printf("\t%%%%\t about to write stripe[%lu, %lu] of output image % 3d, pixel[%d,%d] = %g\t\tmax pixel = %g\n", \
			start_i,end_i,file_num,pixelTESTi,pixelTESTj,dmatrix_get(outMatrix, pixelTESTi -  start_i, pixelTESTj),dmatrix_max(outMatrix));
#endif

	output_pixel_type = (user_preferences.out_pixel_type < 0) ? imaging_parameters.in_pixel_type : user_preferences.out_pixel_type;
	pixel_size = (user_preferences.out_pixel_type < 0) ? imaging_parameters.in_pixel_bytes : WinView_itype2len(user_preferences.out_pixel_type);
	if (dmatrix_max(outMatrix)==0 && dmatrix_min(outMatrix)==0) return;	/* do not write if this is all zeros (file is preset to zero) */

	/*	WinViewWriteROI(readfile, (char*)cbuf, output_pixel_type, imaging_parameters.nROI_i, 0, imaging_parameters.nROI_i - 1, start_i, end_i); */
	struct HDF5_Header header;
	header.xdim = in_header.xdim;						/* copy only the needed values into header */
	header.ydim = in_header.ydim;
	header.isize = pixel_size;
	header.itype = output_pixel_type;

	if (norm_rescale > 1.0) {							/* rescale the output to fill the integer range */
		dmatrix_scale(outMatrix, norm_rescale);		/* multiply all elements of outMatrix by norm_rescale */
		/* dmatrix_scale(a, x):  a(i,j) <--  x * a(i,j) is stored in a. */
	}
	
	if ((file_id=H5Fopen(fileName,H5F_ACC_RDWR,H5P_DEFAULT))<=0) { fprintf(stderr,"ERROR -- H5Fopen(), cannot open the file '%s'\n",fileName); exit(1); }
	err = HDF5WriteROI(file_id,"entry1/data/data",(void*)(outMatrix->data), start_i, end_i, 0, (size_t)(imaging_parameters.nROI_j - 1), H5T_IEEE_F64LE, &header);
	if (err) { fprintf(stderr,"ERROR -- HDF5WriteROI(), cannot write ROI to the file '%s'\n",fileName); exit(1); }
	if ((err = H5Fclose(file_id))) { fprintf(stderr,"ERROR -- write_depth_datai(), file close error = %d\n",err); exit(1); }
}


long pixel_type2scaling(			/* return max val of a pixel type, used to set norm_rescale */
int		itype)						/* WinView itype, pixel types */
{
	switch (itype) {
		case 1:						/* int32 (4 byte), signed, max value = 2,147,483,647 */
			return (1<<15);			/* (1<<15) = 32766 */
		case 2:						/* int16 (2 byte), max value = 32767 */
			return (1<<7);			/* 2^7 = 128 */
		case 3:						/* unsigned int16 (2 byte), max value = 65535 */
			return (1<<8) - 1;		/* 2^8 - 1 = 255 */
		case 6:						/* signed int8 (1 byte), max value = 127 */
			return (1<<3);			/* 2^3 - 1 = 15 */
		case 4:						/* string/char (1 byte), max value = 255 */
		case 7:						/* unsigned int8 (1 byte) */
			return (1<<4) - 1;		/* 2^8 - 1 = 255 */
		case 0:						/* float (4 byte) */
		case 5:						/* double (8 byte) */
			return 1;
		default:
			return 1;
	}
}


/* convert index of a depth resolved image to its depth along the beam (micron) */
/* this is the depth of the center of the bin */
//inline double index_to_beam_depth(
double index_to_beam_depth(
long	index)						/* index to depth resolved images */
{
	double absolute_depth = index * user_preferences.depth_resolution;
	absolute_depth += user_preferences.depth_start;
	return absolute_depth;
}


#pragma message "find_first_valid_i() and find_last_valid_i() my be unusable with this detector"
/* find the lowest possible row (i) in the image given the depth range and image size.  Only the first wire position of the scan is needed. */
int find_first_valid_i(
int ilo,							/* check i in the range [ilo,ihi], thses are usually the ends of the image */
int ihi,
int j1,								/* check pixels (i,j1) and(i,j2) */
int j2,
point_xyz wire,						/* first wire position (xyz) of wire scan (in beam line coords relative to the Si) */
BOOLEAN use_leading_wire_edge)		/* true=used leading endge of wire, false=use trailing edge of wire */
{
	int i;							/* index to a row in the image */
	double	d;						/* computed depth from either j1 or j2 */
	point_ccd pixel_edge;			/* pixel indicies for the edge of a pixel (e.g. [117,90.5]) */
	point_xyz back_edge;			/* xyz postition of the trailing edge of the pixel (in beam line coords relative to the Si) */
	double depth_end = user_preferences.depth_end;							/* max depth of a reconstructed image */

	for (i=ilo; i<=ihi; i++) {
		pixel_edge.i = 0.5 + (double)i;
		pixel_edge.j = (double)j1;
		back_edge = pixel_to_point_xyz(pixel_edge);							/* the back (low) edge of this pixel, using j1 */
		d = pixel_xyz_to_depth(back_edge, wire, use_leading_wire_edge);
		if (user_preferences.depth_start <= d && d <= depth_end) return i;	/* this d lies in our depth range, so i is OK */
		if (i==ilo && user_preferences.depth_start > d && d <= depth_end) return i;	/* limiting i is negative */

		pixel_edge.j = (double)j2;
		back_edge = pixel_to_point_xyz(pixel_edge);							/* the back (low) edge of this pixel, using j2 */
		d = pixel_xyz_to_depth(back_edge, wire, use_leading_wire_edge);
		if (user_preferences.depth_start <= d && d <= depth_end) return i;	/* this d lies in our depth range, so i is OK */
		if (i==ilo && user_preferences.depth_start > d && d <= depth_end) return i;	/* limiting i is negative */
	}
	return -1;																/* none of the i were acceptable */
}


/* find the highest possible row (i) in the image given the depth range and image size.  Only the last wire position of the scan is needed. */
int find_last_valid_i(
int ilo,							/* check i in the range [ilo,ihi] */
int ihi,
int j1,								/* check pixels (i,j1) and(i,j2) ilo & ihi are usually edges of image */
int j2,
point_xyz wire,						/* last wire position (xyz) of wire scan (in beam line coords relative to the Si) */
BOOLEAN use_leading_wire_edge)		/* true=used leading endge of wire, false=use trailing edge of wire */
{
	int i;							/* index to a row in the image */
	double	d;						/* computed depth from either j1 or j2 */
	point_ccd pixel_edge;			/* pixel indicies for the edge of a pixel (e.g. [117,90.5]) */
	point_xyz front_edge;			/* xyz postition of the leading edge of the pixel (in beam line coords relative to the Si) */
	double depth_end = user_preferences.depth_end;							/* max depth of a reconstructed image */

	for (i=ihi; i>=ilo; i--) {
		pixel_edge.i = -0.5 + (double)i;
		pixel_edge.j = (double)j1;
		front_edge = pixel_to_point_xyz(pixel_edge);						/* the front (low) edge of this pixel, using j1 */
		d = pixel_xyz_to_depth(front_edge, wire, use_leading_wire_edge);
		if (user_preferences.depth_start <= d && d <= depth_end) return i;	/* this d lies in our depth range, so i is OK */
		if (i==ihi && d > depth_end) return i;								/* limiting i is past end of detector */

		pixel_edge.j = (double)j2;
		front_edge = pixel_to_point_xyz(pixel_edge);						/* the front (low) edge of this pixel, using j2 */
		d = pixel_xyz_to_depth(front_edge, wire, use_leading_wire_edge);
		if (user_preferences.depth_start <= d && d <= depth_end) return i;	/* this d lies in our depth range, so i is OK */
		if (i==ihi && d > depth_end) return i;								/* limiting i is past end of detector */
	}
	return -1;																/* none of the i were acceptable */
}



/* convert PM500 {x,y,z} to beam line {x,y,z} */
point_xyz wirePosition2beamLine(
point_xyz wire_pos)								/* PM500 {x,y,z} values */
{
	double x,y,z;

	x = X2corrected(wire_pos.x);				/* do PM500 distortion correction for wire */
	y = Y2corrected(wire_pos.y);
	z = Z2corrected(wire_pos.z);
	x -= calibration.wire.centre_at_si_xyz.x;	/* offset wire to origin (the Si position) */
	y -= calibration.wire.centre_at_si_xyz.y;
	z -= calibration.wire.centre_at_si_xyz.z;

	/* rotate by the orientation of the positioner, this does not make the wire axis parallel to beam-line x-axis */
	wire_pos.x = calibration.wire.rotation[0][0]*x + calibration.wire.rotation[0][1]*y + calibration.wire.rotation[0][2]*z;	/* {X2,Y2,Z2} = w.Rij x {x,y,z},   rotate by R (a small rotation) */
	wire_pos.y = calibration.wire.rotation[1][0]*x + calibration.wire.rotation[1][1]*y + calibration.wire.rotation[1][2]*z;
	wire_pos.z = calibration.wire.rotation[2][0]*x + calibration.wire.rotation[2][1]*y + calibration.wire.rotation[2][2]*z;

/* #pragma message "what should I do about the rotation to put wire axis parallel to beam line axis" */
	wire_pos = MatrixMultiply31(calibration.wire.rho,wire_pos);	/* wire_centre = rho x wire_centre, rotate wire position so wire axis lies along {1,0,0} */

	return wire_pos;
}






#ifdef DEBUG_ALL
void printPieceOf_dmatrix(				/* print some of the dmatrix */
int		ilo,								/* i range to print is [ihi, ilo] */
int		ihi,
int		jlo,								/* j range to print is [jhi, jlo] */
int		jhi,
dmatrix *mat)
{
	int		i,j;
	printf("\nfor dmatrix_get(mat,i,j)  (j is the fast index)\nj\t i=");
	for (i=ilo;i<=ihi;i++) printf("\t% 5d",i); printf("\n");
	for (j=jlo;j<=jhi;j++) {
		printf("%d\t",j);
		for (i=ilo;i<=ihi;i++) printf("\t %g",dmatrix_get(mat,(size_t)i,(size_t)j));
		printf("\n");
	}
}


void printPieceOfArrayDouble(				/* print some of the double array */
int		ilo,								/* i range to print is [ihi, ilo] */
int		ihi,
int		jlo,								/* j range to print is [jhi, jlo] */
int		jhi,
int		Nx,									/* size of array,  buf[Nx][Ny] */
int		Ny,
double buf[Nx][Ny])
{
	int		i,j;
	printf("\nfor double array[j][i]\nj\t i=");
	for (i=ilo;i<=ihi;i++) printf("\t% 5d",i); printf("\n");
	for (j=jlo;j<=jhi;j++) {
		printf("%d\t",j);
		for (i=ilo;i<=ihi;i++) printf("\t %g",buf[j][i]);
		printf("\n");
	}
}

void printPieceOfArrayInt(					/* print some of the array */
int		ilo,								/* i range to print is [ihi, ilo] */
int		ihi,
int		jlo,								/* j range to print is [jhi, jlo] */
int		jhi,
int		Nx,									/* size of array,  buf[Nx][Ny] */
int		Ny,
unsigned short int buf[Nx][Ny],
int		itype)
{
	int		i,j;
	printf("\nfor int array[j][i]\nj\t i=");
	for (i=ilo;i<=ihi;i++) printf("\t% 5d",i); printf("\n");
	for (j=jlo;j<=jhi;j++) {
		printf("%d\t",j);

		if (itype==3)
		{
			for (i=ilo;i<=ihi;i++) printf("\t %u",buf[j][i]);
		}
		else if (itype==2) {
			for (i=ilo;i<=ihi;i++) printf("\t %hd",buf[j][i]);
		}
		printf("\n");
	}
}
#endif




#if (0)
/*	THIS ROUTINE IS NOT USED, BUT KEEP IT AROUND ANYHOW */
double slitWidth(point_xyz a, point_xyz b, point_xyz sa, point_xyz sb);
/* calculate width of wire step perpendicular to the ray from source to wire, the width of the virtual slit */
double slitWidth(					/* distance between wire points a and b perp to ray = { (a+b)/2 - (sa+sb)/2 } */
point_xyz a,							/* first wire position */
point_xyz b,							/* second wire position */
point_xyz sa,						/* first source point */
point_xyz sb)						/* second source point */
{
	double dx,dy,dz;					/* vector from b to a (between two wire positions) */
	double dw2;							/* square of distance between two wire positions, |b-a|^2 */
	double dirx,diry,dirz;				/* direction vector from average source to average wire (not normalized) */
	double dir2;						/* square of length of dir */
	double dot;							/* dir .dot. dw */
	double wid2;						/* square of the answer */

	dirx = ((b.x+a.x) - (sb.x+sa.x))/2.;/* (a+b)/2 - (sa+sb)/2, ray from pixel to wire */
	diry = ((b.y+a.y) - (sb.y+sa.y))/2.;
	dirz = ((b.z+a.z) - (sb.z+sa.z))/2.;
	dir2 = dirx*dirx + diry*diry + dirz*dirz;
	if (dir2 <=0) return 0;				/* slit width is zero */

	dx = b.x - a.x;						/* vector between two wire positions */
	dy = b.y - a.y;
	dz = b.z - a.z;
	dw2 = sqrt(dx*dx + dy*dy + dz*dz);

	dot = dirx*dx + diry*dy + dirz*dz;

	wid2 = dw2 - (dot*dot)/dir2;		/* this line is just pythagoras */
	wid2 = MAX(wid2,0);
	return sqrt(wid2);
}
#endif



void print_imaging_parameters(
ws_imaging_parameters ip)
{
	printf("\n\n*************  value of ws_imaging_parameters structure  *************\n");
	printf("\t\t[nROI_i, nROI_j] = [%d, %d]\n", ip.nROI_i,ip.nROI_j);
	printf("\t\tstartx=%d,  endx=%d,  binx=%d,  no. of points = %d\n",ip.starti,ip.endi,ip.bini,ip.endi-ip.starti+1);
	printf("\t\tstarty=%d,  endy=%d,  biny=%d,  no. of points = %d\n",ip.startj,ip.endj,ip.binj,ip.endj-ip.startj+1);
	printf("\t\tthere are %d input images\n",ip.NinputImages);
	printf("\t\tinput pixel is of type %d,   (length = %d bytes) // 0=float, 1=long, 2=short, 3=ushort, 4=char, 5=double, 6=signed char, 7=uchar\n",ip.in_pixel_type, ip.in_pixel_bytes);
	printf("\t\tfirst wire position is  {%g, %g, %g}\n",ip.wire_first_xyz.x,ip.wire_first_xyz.y,ip.wire_first_xyz.z);
	printf("\t\t last wire position is  {%g, %g, %g}\n",ip.wire_last_xyz.x,ip.wire_last_xyz.y,ip.wire_last_xyz.z);
	printf("\t\tcan measure rows_at_one_time = %lu\n",ip.rows_at_one_time);
	printf("\t\tcurrent row range is [%d, %d]\n", ip.current_selection_start,ip.current_selection_end);
	printf("*************  end of ws_imaging_parameters structure  *************\n");
	printf("\n");
}


void print_help_text(void)
{
	printf("\nUsage: WireScan -i <file> -o <file> -g <file> [-s <#>] -e <#> [-r <#>] [-v <#>] [-f <#>] -l <#> [-p <#>]  [-t <#>]  [-m <#>] [-?] \n\n");
	printf("\n-i <file>,\t --infile=<file>\t\tlocation and leading section of file names to process");
	printf("\n-o <file>,\t --outfile=<file>\t\tlocation and leading section of file names to create");
	printf("\n-g <file>,\t --geofile=<file>\t\tlocation of file containing parameters from the wirescan");
	printf("\n-d <file>,\t --distortion map=<file>\tlocation of file with the distortion map, dXYdistortion");
	printf("\n-s <#>,\t\t --depth-start=<#>\t\tdepth to begin recording values at - inclusive");
	printf("\n-e <#>,\t\t --depth-end=<#>\t\tdepth to stop recording values at - inclusive");
	printf("\n-r <#>,\t\t --resolution=<#>\t\tum depth covered by a single depth-resolved image");
	printf("\n-v <#>,\t\t --verbose=<#>\t\t\toutput detailed output of varying degrees (0, 1, 2, 3)");
	printf("\n-f <#>,\t\t --first-image=<#>\t\tnumber of first image to process - inclusive");
	printf("\n-l <#>,\t\t --last-image=<#>\t\tnumber of last image to process - inclusive");
	printf("\n-n <tag>,\t --normalization=<tag>\t\ttag of variable in header to use for normalizing incident intensity, optional");
	printf("\n-p <#>,\t\t --percent-to-process=<#>\tonly process the p%% brightest pixels in image");
	printf("\n-w <l,t,b>,\t --wire-edges\t\t\tuse leading, trailing, or both edges of wire, (for both, output images will then be longs)");
	printf("\n-t <#>,\t\t --type-output-pixel=<#>\ttype of output pixel (uses old WinView numbers), optional");
	printf("\n-E <#>,\t\t --type-output-pixel=<#>\texponent for image intensity scaling, optional (0.5 is typical value)");
	printf("\n-T <#>,\t\t --type-output-pixel=<#>\tthreshold for image intensity scaling, optional (only used if -E provided)");
	printf("\n-C,\t\t --CosmicFilter\t\t\tfilter out the hot pixels from cosmics");
	printf("\n-m <#>,\t\t --memory=<#>\t\t\tdefine the amount of memory in MiB that the programme is allowed to use");
	printf("\n-N <file>,\t --num-threads=<\x23>\t\tnumber of threads to run in parallel");
	printf("\n-R <file>,\t --num-threads=<\x23>\t\tnumber of rows to process per batch");
	printf("\n-?,\t\t --help\t\t\t\tdisplay this help");
	printf("\n\n");
	printf("Example: WireScan -i /images/image_ -o /result/image_ -g /geo/file -s 0 -e 100 -r 1 -v 1 -f 1 -l 401 -p 1\n\n");
}


#ifdef DEBUG_1_PIXEL
/*
	head->xDimDet	= get1HDF5data_int(file_id,"/entry1/detector/Nx",&ivalue) ? XDIMDET : ivalue;
	head->yDimDet	= get1HDF5data_int(file_id,"/entry1/detector/Ny",&ivalue) ? YDIMDET : ivalue;
	head->startx	= get1HDF5data_int(file_id,"/entry1/detector/startx",&ivalue) ? STARTX : ivalue;
	head->endx		= get1HDF5data_int(file_id,"/entry1/detector/endx",&ivalue) ? ENDX : ivalue;
	head->groupx	= get1HDF5data_int(file_id,"/entry1/detector/groupx",&ivalue) ? GROUPX : ivalue;
	head->starty	= get1HDF5data_int(file_id,"/entry1/detector/starty",&ivalue) ? STARTY : ivalue;
	head->endy		= get1HDF5data_int(file_id,"/entry1/detector/endy",&ivalue) ? ENDY : ivalue;
	head->groupy	= get1HDF5data_int(file_id,"/entry1/detector/groupy",&ivalue) ? GROUPY : ivalue;


ROI: un-binned
Xstart=820, Xsize=120		along beam
Ystart=1045, Ysize=100		perpendicular to beam

  depth = -167.396710542 µm, using leading ede
  depth = -167.396710541619
  depth = -219.436661679 µm, using trailing edge
  depth = -219.436661679071

value of [50,60] = 9660 for file 932
value of [60,50] = 11852 for h5 file
*/
void testing_depth(void)
{
	point_ccd	pixel;
	point_xyz	xyzPixel;
	point_xyz	xyzWire;
//	double		depth;
//	BOOLEAN		use_leading_wire_edge=1;		// 1==989.266,   0==989.265
//	double		leadingIgor = -167.396710541619;
//	double		trailingIgor = -219.436661679071;


	pixel.i = pixelTESTi;
	pixel.j = pixelTESTj;
	xyzWire.x = -0.05;
	xyzWire.y = -5815.28;
	xyzWire.z = -1572.63;

	double H,F, root2=1./sqrt(2.);
	H =  root2 * xyzWire.y + root2 * xyzWire.z;
	F = -root2 * xyzWire.y + root2 * xyzWire.z;

	printf("\n\n\n start of testing_depth()\n");
	/*
	printf("\n\ndetector P = {%.6f, %.6f, %.6f}\n",calibration.P.x,calibration.P.y,calibration.P.z);
	printf("calibration.detector_rotation = \n");
	printf("%13.9f	%13.9f	%13.9f	\n",calibration.detector_rotation[0][0],calibration.detector_rotation[0][1],calibration.detector_rotation[0][2]);
	printf("%13.9f	%13.9f	%13.9f	\n",calibration.detector_rotation[1][0],calibration.detector_rotation[1][1],calibration.detector_rotation[1][2]);
	printf("%13.9f	%13.9f	%13.9f	\n",calibration.detector_rotation[2][0],calibration.detector_rotation[2][1],calibration.detector_rotation[2][2]);
	printf("***pixel X = %.9lf\n",calibration.detector_rotation[0][0] * 209773.0 + calibration.detector_rotation[0][1] * 14587.0 + calibration.detector_rotation[0][2] * 510991.0);
	*/

	xyzPixel = pixel_to_point_xyz(pixel);						/* input, binned ROI (zero-based) pixel value on detector, can be non-integer, and can lie outside range (e.g. -05 is acceptable) */
	printf("pixel = [%g, %g] --> {%.12lf, %.12lf, %.12lf}mm\n",pixel.i,pixel.j,xyzPixel.x/1000.,xyzPixel.y/1000.,xyzPixel.z/1000.);
	printf("wire starts at {%.6f, %.6f, %.6f}µm   H=%g, F=%g\n",xyzWire.x,xyzWire.y,xyzWire.z,H,F);

	xyzWire = wirePosition2beamLine(xyzWire);
	printf("corrected wire at {%.6f, %.6f, %.6f}µm\n",xyzWire.x,xyzWire.y,xyzWire.z);


//	depth = pixel_xyz_to_depth(xyzPixel,xyzWire,use_leading_wire_edge);
//	printf("depth (leading) = %.9f µm,  Igor got %.9f,  ∆=%.9f\n",depth,leadingIgor,depth-leadingIgor);
//
//	depth = pixel_xyz_to_depth(xyzPixel,xyzWire,0);
//	printf("depth (trailing) = %.9f µm,  Igor got %.9f,  ∆=%.9f\n",depth,trailingIgor,depth-trailingIgor);

	printf(" done with testing_depth()\n\n");
	return;
}
#endif
