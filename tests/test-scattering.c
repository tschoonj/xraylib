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
#include <stdio.h>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

int main(int argc, char **argv) {
	xrl_error *error = NULL;
	double rv;

	/* FF_Rayl */
	rv = FF_Rayl(26, 0.0, &error);
	assert(error == NULL);
	assert(rv == 26.0);

	rv = FF_Rayl(92, 10.0, &error);
	assert(error == NULL);
	assert(fabs(rv - 2.1746) < 1E-6);

	rv = FF_Rayl(0, 10.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	rv = FF_Rayl(99, 10.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	rv = FF_Rayl(98, 10.0, &error);
	assert(error == NULL);
	assert(fabs(rv - 2.4621) < 1E-6);

	rv = FF_Rayl(9, -10.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_Q) == 0);
	xrl_clear_error(&error);

	rv = FF_Rayl(9, 1E9 - 1, &error);
	assert(error == NULL);

	rv = FF_Rayl(9, 1E9 + 1, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, SPLINT_X_TOO_HIGH) == 0);
	xrl_clear_error(&error);

	/* SF_Compt */
	rv = SF_Compt(26, 0.1, &error);
	assert(error == NULL);
	assert(fabs(rv - 2.891) < 1E-6);

	rv = SF_Compt(92, 10.0, &error);
	assert(error == NULL);
	assert(fabs(rv - 89.097) < 1E-6);

	rv = SF_Compt(0, 10.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	rv = SF_Compt(99, 10.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	rv = SF_Compt(98, 10.0, &error);
	assert(error == NULL);
	assert(fabs(rv - 94.631) < 1E-6);

	rv = SF_Compt(9, 0.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_Q) == 0);
	xrl_clear_error(&error);

	rv = SF_Compt(9, 1E9 - 1, &error);
	assert(error == NULL);

	rv = SF_Compt(9, 1E9 + 1, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, SPLINT_X_TOO_HIGH) == 0);
	xrl_clear_error(&error);

	/* DCS_Thoms */
	rv = DCS_Thoms(M_PI/4, &error);
	assert(error == NULL);
	assert(fabs(rv - 0.05955590775) < 1E-6);

	/* DCS_KN */
	rv = DCS_KN(10.0, M_PI/4, &error);
	assert(error == NULL);
	assert(fabs(rv - 0.058880292827846535) < 1E-6);

	rv = DCS_KN(0.0, 0.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	/* DCS_Rayl */
	rv = DCS_Rayl(26, 10.0, M_PI/4.0, &error);
	assert(fabs(rv - 0.17394690792051704) < 1E-6);
	assert(error == NULL);

	rv = DCS_Rayl(26, 10.0, 0.0, &error);
	assert(fabs(rv - 0.5788126901827545) < 1E-6);
	assert(error == NULL);

	rv = DCS_Rayl(0, 10.0, M_PI/4.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	rv = DCS_Rayl(99, 10.0, M_PI/4.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	rv = DCS_Rayl(98, 10.0, M_PI/4.0, &error);
	assert(error == NULL);
	assert(fabs(rv - 0.7582632962268532) < 1E-6);

	rv = DCS_Rayl(26, 0.0, M_PI/4.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	/* DCS_Compt*/
	rv = DCS_Compt(26, 10.0, M_PI/4.0, &error);
	assert(fabs(rv - 0.005489497545806117) < 1E-6);
	assert(error == NULL);

	rv = DCS_Compt(92, 1.0, M_PI/3, &error);
	assert(fabs(rv - 0.0002205553556181471) < 1E-6);
	assert(error == NULL);

	rv = DCS_Compt(0, 10.0, M_PI/4.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	rv = DCS_Compt(99, 10.0, M_PI/4.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	rv = DCS_Compt(98, 10.0, M_PI/4.0, &error);
	assert(error == NULL);
	assert(fabs(rv - 0.0026360563424557386) < 1E-6);

	rv = DCS_Compt(26, 0.0, M_PI/4.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);
	
	rv = DCS_Compt(26, 10.0, 0.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_Q) == 0);
	xrl_clear_error(&error);

	/* MomentTransf */
	rv = MomentTransf(10.0, M_PI, &error);
	assert(fabs(rv - 0.8065544290795198) < 1E-6);
	assert(error == NULL);

	rv = MomentTransf(0.0, M_PI, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	/* CS_KN */
	rv = CS_KN(10.0, &error);
	assert(fabs(rv - 0.6404703229290962) < 1E-6);
	assert(error == NULL);

	rv = CS_KN(0.0, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	/* ComptonEnergy */
	rv = ComptonEnergy(10.0, M_PI/4, &error);
	assert(fabs(rv - 9.943008884806082) < 1E-6);
	assert(error == NULL);

	rv = ComptonEnergy(10.0, 0.0, &error);
	assert(fabs(rv - 10.0) < 1E-6);
	assert(error == NULL);

	rv = ComptonEnergy(0.0, M_PI/4, &error);
	assert(error != NULL);
	assert(rv == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	return 0;
}
