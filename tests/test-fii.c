/* Copyright (c) 2018, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modifiication, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specifiic prior written permission.

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
	double fii;

	fii = Fii(26, 1.0, &error);
	assert(error == NULL);
	assert(fabs(fii + 12.11016272606279) < 1E-6);

	fii = Fii(92, 10.0, &error);
	assert(error == NULL);
	assert(fabs(fii + 9.812520269504303) < 1E-6);

	fii = Fii(56, 100.0, &error);
	assert(error == NULL);
	assert(fabs(fii + 0.6440539604345072) < 1E-6);

	/* bad input */
	fii = Fii(0, 10.0, &error);
	assert(fii == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	fii = Fii(101, 10.0, &error);
	assert(fii == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	fii = Fii(100, 10.0, &error);
	assert(fii != 0.0);
	assert(error == NULL);

	fii = Fii(59, 0.0, &error);
	assert(fii == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	fii = Fii(59, -10.0, &error);
	assert(fii == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	fii = Fii(59, 0.0011, &error);
	assert(error == NULL);

	fii = Fii(59, 0.0009, &error);
	assert(fii == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, SPLINT_X_TOO_LOW) == 0);
	xrl_clear_error(&error);

	fii = Fii(59, 9999, &error);
	assert(error == NULL);

	fii = Fii(59, 10001, &error);
	assert(fii == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, SPLINT_X_TOO_HIGH) == 0);
	xrl_clear_error(&error);

	return 0;
}
