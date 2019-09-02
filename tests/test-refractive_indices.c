/* Copyright (c) 2018, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "xraylib.h"
#include "xraylib-error-private.h"
#ifdef NDEBUG
  #undef NDEBUG
#endif
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv) {
	xrl_error *error = NULL;
	double re, im;
	xrlComplex cplx;

	/* The refractive index funtions accept both chemical formulas and NIST catalog entries.
	 * If the latter is used, it is possible to use the NIST density, if the density that gets passed is 0 or less
	 */

	re = Refractive_Index_Re("H2O", 1.0, 1.0, &error);
	assert(error == NULL);
	assert(fabs(re - 0.999763450676632) < 1E-9);

	im = Refractive_Index_Im("H2O", 1.0, 1.0, &error);
	assert(error == NULL);
	assert(fabs(im - 4.021660592312145e-05) < 1E-9);

	cplx = Refractive_Index("H2O", 1.0, 1.0, &error);
	assert(error == NULL);
	assert(fabs(cplx.re - 0.999763450676632) < 1E-9);
	assert(fabs(cplx.im - 4.021660592312145e-05) < 1E-9);

	re = Refractive_Index_Re("Air, Dry (near sea level)", 1.0, 1.0, &error);
	assert(error == NULL);
	assert(fabs(re - 0.999782559048) < 1E-9);

	im = Refractive_Index_Im("Air, Dry (near sea level)", 1.0, 1.0, &error);
	assert(error == NULL);
	assert(fabs(im - 0.000035578193) < 1E-9);

	cplx = Refractive_Index("Air, Dry (near sea level)", 1.0, 1.0, &error);
	assert(error == NULL);
	assert(fabs(cplx.re - 0.999782559048) < 1E-9);
	assert(fabs(cplx.im - 0.000035578193) < 1E-9);

	re = Refractive_Index_Re("Air, Dry (near sea level)", 1.0, 0.0, &error);
	assert(error == NULL);
	assert(fabs(re - 0.999999737984) < 1E-12);

	im = Refractive_Index_Im("Air, Dry (near sea level)", 1.0, 0.0, &error);
	assert(error == NULL);
	assert(fabs(im - 0.000000042872) < 1E-12);

	cplx = Refractive_Index("Air, Dry (near sea level)", 1.0, 0.0, &error);
	assert(error == NULL);
	assert(fabs(cplx.re - 0.999999737984) < 1E-12);
	assert(fabs(cplx.im - 0.000000042872) < 1E-12);

	cplx = Refractive_Index("Air, Dry (near sea level)", 1.0, -1.0, &error);
	assert(error == NULL);
	assert(fabs(cplx.re - re) < 1E-12);
	assert(fabs(cplx.im - im) < 1E-12);

	/* bad input */
	re = Refractive_Index_Re(NULL, 1.0, 1.0, &error);
	assert(error != NULL);
	assert(re == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, UNKNOWN_COMPOUND) == 0);
	xrl_clear_error(&error);
	
	im = Refractive_Index_Im(NULL, 1.0, 1.0, &error);
	assert(error != NULL);
	assert(im == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, UNKNOWN_COMPOUND) == 0);
	xrl_clear_error(&error);

	cplx = Refractive_Index(NULL, 1.0, 1.0, &error);
	assert(error != NULL);
	assert(cplx.re == 0.0 && cplx.im == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, UNKNOWN_COMPOUND) == 0);
	xrl_clear_error(&error);

	re = Refractive_Index_Re("", 1.0, 1.0, &error);
	assert(error != NULL);
	assert(re == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, UNKNOWN_COMPOUND) == 0);
	xrl_clear_error(&error);
	
	im = Refractive_Index_Im("", 1.0, 1.0, &error);
	assert(error != NULL);
	assert(im == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, UNKNOWN_COMPOUND) == 0);
	xrl_clear_error(&error);

	cplx = Refractive_Index("", 1.0, 1.0, &error);
	assert(error != NULL);
	assert(cplx.re == 0.0 && cplx.im == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, UNKNOWN_COMPOUND) == 0);
	xrl_clear_error(&error);

	re = Refractive_Index_Re("H2O", 0.0, 1.0, &error);
	assert(error != NULL);
	assert(re == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);
	
	im = Refractive_Index_Im("H2O", 0.0, 1.0, &error);
	assert(error != NULL);
	assert(im == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	cplx = Refractive_Index("H2O", 0.0, 1.0, &error);
	assert(error != NULL);
	assert(cplx.re == 0.0 && cplx.im == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	re = Refractive_Index_Re("H2O", 1.0, 0.0, &error);
	assert(error != NULL);
	assert(re == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_DENSITY) == 0);
	xrl_clear_error(&error);
	
	im = Refractive_Index_Im("H2O", 1.0, 0.0, &error);
	assert(error != NULL);
	assert(im == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_DENSITY) == 0);
	xrl_clear_error(&error);

	cplx = Refractive_Index("H2O", 1.0, 0.0, &error);
	assert(error != NULL);
	assert(cplx.re == 0.0 && cplx.im == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_DENSITY) == 0);
	xrl_clear_error(&error);


	
	return 0;
}
