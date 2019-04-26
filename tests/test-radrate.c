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

int main(int argc, char **argv) {
	xrl_error *error = NULL;
	double rr = 0.0;

	/* some simple IUPAC tests */
	rr = RadRate(26, KL3_LINE, &error);
	assert(fabs(rr - 0.58357) < 1E-6);
	assert(error == NULL);

	rr = RadRate(92, KN2_LINE, &error);
	assert(fabs(rr - 0.01452) < 1E-6);
	assert(error == NULL);

	rr = RadRate(56, L3M1_LINE, &error);
	assert(fabs(rr - 0.031965) < 1E-6);
	assert(error == NULL);

	rr = RadRate(82, M5N7_LINE, &error);
	assert(fabs(rr - 0.86638) < 1E-6);
	assert(error == NULL);

	/* bad input */
	rr = RadRate(0, KL3_LINE, &error);
	assert(rr == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	rr = RadRate(-1, KL3_LINE, &error);
	assert(rr == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	rr = RadRate(ZMAX + 1, KL3_LINE, &error);
	assert(rr == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	rr = RadRate(ZMAX, KL3_LINE, &error);
	assert(rr == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	rr = RadRate(110, KL3_LINE, &error);
	assert(rr == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	rr = RadRate(109, KL3_LINE, &error);
	assert(error == NULL);
	assert(rr > 0);

	rr = RadRate(26, 1000, &error);
	assert(rr == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, UNKNOWN_LINE) == 0);
	xrl_clear_error(&error);

	rr = RadRate(26, M5N7_LINE, &error);
	assert(rr == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	/* Siegbahn macros */
	rr = RadRate(26, KA_LINE, &error);
	assert(error == NULL);
	assert(fabs(rr - 0.88156) < 1E-6);
	assert(fabs(rr - (RadRate(26, KL1_LINE, NULL) + RadRate(26, KL2_LINE, NULL) + RadRate(26, KL3_LINE, NULL))) < 1E-6);

	rr = RadRate(26, KB_LINE, &error);
	assert(error == NULL);
	assert(fabs(rr - (1.0 - RadRate(26, KA_LINE, NULL))) < 1E-6);

	rr = RadRate(10, KA_LINE, &error);
	assert(error == NULL);
	assert(rr == 1.0);

	rr = RadRate(10, KB_LINE, &error);
	assert(rr == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	rr = RadRate(56, LA_LINE, &error);
	assert(error == NULL);
	assert(fabs(rr - 0.828176) < 1E-6);
	assert(fabs(rr - (RadRate(56, L3M4_LINE, NULL) + RadRate(56, L3M5_LINE, NULL))) < 1E-6);

	/* Lb is never allowed! */
	rr = RadRate(56, LB_LINE, &error);
	assert(rr == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	return 0;
}
