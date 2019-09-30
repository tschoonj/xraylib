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
#include <math.h>



int main(int argc, char **argv) {
	xrl_error *error = NULL;
	double cs_photo, cs_compt, cs_rayl, cs_total, cs_energy;
	double (* const cs[])(int, double, xrl_error **) = {CS_Photo, CS_Compt, CS_Rayl, CS_Total, CS_Energy};
	double data_max[] = {1001, 801, 801, 801, 20001};
	double data_min[] = {0.09, 0.09, 0.09, 0.09, 0.9};

	int i;

	/* CS_Photo */
	cs_photo = CS_Photo(10, 10.0, &error);
	assert(error == NULL);
	assert(fabs(cs_photo - 11.451033638148562) < 1E-4);

	/* CS_Compt */
	cs_compt = CS_Compt(10, 10.0, &error);
	assert(error == NULL);
	assert(fabs(cs_compt - 0.11785269096475783) < 1E-4);

	/* CS_Rayl */
	cs_rayl = CS_Rayl(10, 10.0, &error);
	assert(error == NULL);
	assert(fabs(cs_rayl - 0.39841164641058013) < 1E-4);

	/* CS_Total */
	cs_total = CS_Total(10, 10.0, &error);
	assert(error == NULL);
	/* internal consistency check */
	assert(fabs(cs_total - cs_photo - cs_compt - cs_rayl) < 1E-3);

	/* CS_Energy */
	cs_energy = CS_Energy(10, 10.0, &error);
	assert(error == NULL);
	assert(fabs(cs_energy - 11.420221747941419) < 1E-4);

	/* bad input */
	for (i = 0 ; i < sizeof(cs)/sizeof(cs[0]) ; i++) {
		double v;

		/* bad Z */
		v = cs[i](-1, 10.0, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
		assert(v == 0.0);
		xrl_clear_error(&error);

		v = cs[i](ZMAX, 10.0, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
		assert(v == 0.0);
		xrl_clear_error(&error);

		/* bad energy */
		v = cs[i](26, 0.0, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
		assert(v == 0.0);
		xrl_clear_error(&error);

		v = cs[i](26, -1.0, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
		assert(v == 0.0);
		xrl_clear_error(&error);

		v = cs[i](26, data_max[i], &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, SPLINT_X_TOO_HIGH) == 0);
		assert(v == 0.0);
		xrl_clear_error(&error);

		v = cs[i](26, data_min[i], &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, SPLINT_X_TOO_LOW) == 0);
		assert(v == 0.0);
		xrl_clear_error(&error);
	}
		

	return 0;	
}
