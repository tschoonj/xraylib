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
	double cs;

	cs = DCSP_Rayl(26, 10.0, M_PI/4, M_PI/4, &error);
	assert(error == NULL);
	assert(fabs(cs - 0.17394690792051704) < 1E-6);
	
	cs = DCSP_Rayl(0, 10.0, M_PI/4, M_PI/4, &error);
	assert(cs == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = DCSP_Rayl(ZMAX + 1, 10.0, M_PI/4, M_PI/4, &error);
	assert(cs == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = DCSP_Rayl(26, 0.0, M_PI/4, M_PI/4, &error);
	assert(cs == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	cs = DCSP_Rayl(26, 10.0, 0.0, M_PI/4, &error);
	assert(error == NULL);
	assert(fabs(cs - 0.5788126901827545) < 1E-6);

	cs = DCSP_Compt(26, 10.0, M_PI/4, M_PI/4, &error);
	assert(error == NULL);
	assert(fabs(cs - 0.005489497545806118) < 1E-6);
	
	cs = DCSP_Compt(0, 10.0, M_PI/4, M_PI/4, &error);
	assert(cs == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = DCSP_Compt(ZMAX + 1, 10.0, M_PI/4, M_PI/4, &error);
	assert(cs == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = DCSP_Compt(26, 0.0, M_PI/4, M_PI/4, &error);
	assert(cs == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	cs = DCSP_Compt(26, 10.0, 0.0, M_PI/4, &error);
	assert(cs == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_Q) == 0);
	xrl_clear_error(&error);

	cs = DCSP_KN(10.0, M_PI/4, M_PI/4, &error);
	assert(error == NULL);
	assert(fabs(cs - 0.05888029282784654) < 1E-6);
	
	cs = DCSP_KN(0.0, 0.0, M_PI/4, &error);
	assert(cs == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	cs = DCSP_Thoms(M_PI/4, M_PI/4, &error);
	assert(error == NULL);
	assert(fabs(cs - 0.05955590775) < 1E-6);
	
	return 0;
}
