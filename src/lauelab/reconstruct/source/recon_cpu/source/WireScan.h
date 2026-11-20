/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#ifndef WireScanHeader
#define WireScanHeader

#include "WireScanDataTypesN.h"
#include "microHDF5.h"
#include "hardwareSpecific.h"


#ifndef MAX
#define MAX(X,Y) ( ((X)<(Y)) ? (Y) : (X) )
#endif
#ifndef MIN
#define MIN(X,Y) ( ((X)>(Y)) ? (Y) : (X) )
#endif


typedef short unsigned int BOOLEAN;

/* data structures containing information for the wire scan */
extern ws_calibration calibration;
extern ws_imaging_parameters imaging_parameters;
extern ws_image_set image_set;
extern ws_user_preferences user_preferences;

extern dmatrix * intensity_map;
extern dmatrix * intensity_norm;

/* extra data-holders defined for parallel version */
extern vvector p_read_buffer[2]; 	// buffer to hold input data for parallel execution
extern vvector p_write_buffer[2]; 	// buffer to hold output data for parallel execution
extern int p_ibuff;				// index to which buffer to be worked on
extern int rows_default;			// limit on # of rows per stripe, default 64
extern int	NUM_THREADS;			// default to 1/

extern struct HDF5_Header in_header;
//extern struct HDF5_Header first_header;
extern struct HDF5_Header output_header;
extern struct geoStructure geoIn;

extern int		verbose;							/* default to 0 */
extern float	percent;							/* default to 100 */
extern int		cutoff;								/* default to 0 */
extern int		AVAILABLE_RAM_MiB;					/* default to 128 */
extern int		cosmic;								/* default to 0 */
extern int		detNum;								/* detector number, default to 0 */
extern char	distortionPath[FILENAME_MAX];		/* full path to the distortion map */
extern float	norm_exponent;						/* exponent to use on normalization image */
extern float	norm_threshold;						/* threshold to use on normalization image */
extern float	norm_rescale;						/* if using norm_exponent, multiply result by this */

#endif
