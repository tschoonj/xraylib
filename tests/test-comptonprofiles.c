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
	double profile, profile1, profile2;

	/* pz == 0.0 */
	profile = ComptonProfile(26, 0.0, &error);
	assert(fabs(profile - 7.060) < 1E-6);
	assert(error == NULL);

	profile = ComptonProfile_Partial(26, N1_SHELL, 0.0, &error);
	assert(fabs(profile - 1.550) < 1E-6);
	assert(error == NULL);

	profile1 = ComptonProfile_Partial(26, L2_SHELL, 0.0, &error);
	profile2 = ComptonProfile_Partial(26, L3_SHELL, 0.0, &error);
	assert(fabs(profile1 - profile2) < 1E-6);
	assert(fabs(profile1 - 0.065) < 1E-6);
	assert(error == NULL);

	/* pz == 100.0 */
	profile = ComptonProfile(26, 100.0, &error);
	assert(fabs(profile - 1.800E-05) < 1E-8);
	assert(error == NULL);

	profile = ComptonProfile_Partial(26, N1_SHELL, 100.0, &error);
	assert(fabs(profile - 5.100E-09) < 1E-12);
	assert(error == NULL);

	profile1 = ComptonProfile_Partial(26, L2_SHELL, 100.0, &error);
	profile2 = ComptonProfile_Partial(26, L3_SHELL, 100.0, &error);
	assert(fabs(profile1 - profile2) < 1E-10);
	assert(fabs(profile1 - 1.100E-08) < 1E-10);
	assert(error == NULL);

	/* pz == 50.0 -> interpolated! */
	profile = ComptonProfile(26, 50.0, &error);
	assert(fabs(profile - 0.0006843950273082384) < 1E-8);
	assert(error == NULL);

	profile = ComptonProfile_Partial(26, N1_SHELL, 50.0, &error);
	assert(fabs(profile - 2.4322755767709126e-07) < 1E-10);
	assert(error == NULL);

	profile1 = ComptonProfile_Partial(26, L2_SHELL, 50.0, &error);
	profile2 = ComptonProfile_Partial(26, L3_SHELL, 50.0, &error);
	assert(fabs(profile1 - profile2) < 1E-10);
	assert(fabs(profile1 - 2.026953933016568e-06) < 1E-10);
	assert(error == NULL);

	/* bad input */
	profile = ComptonProfile(0, 0.0, &error);
	assert(profile == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	profile = ComptonProfile(102, 0.0, &error);
	assert(error == NULL);

	profile = ComptonProfile(103, 0.0, &error);
	assert(profile == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	profile = ComptonProfile(26, -1.0, &error);
	assert(profile == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_PZ) == 0);
	xrl_clear_error(&error);

	profile = ComptonProfile(26, 101, &error);
	assert(profile == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, SPLINT_X_TOO_HIGH) == 0);
	xrl_clear_error(&error);

	profile = ComptonProfile_Partial(0, K_SHELL, 0.0, &error);
	assert(profile == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	profile = ComptonProfile_Partial(102, K_SHELL, 0.0, &error);
	assert(error == NULL);

	profile = ComptonProfile_Partial(103, K_SHELL, 0.0, &error);
	assert(profile == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	profile = ComptonProfile_Partial(26, K_SHELL, -1.0, &error);
	assert(profile == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_PZ) == 0);
	xrl_clear_error(&error);

	profile = ComptonProfile_Partial(26, -1, 0.0, &error);
	assert(profile == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_SHELL) == 0);
	xrl_clear_error(&error);

	profile = ComptonProfile_Partial(26, N2_SHELL, 0.0, &error);
	assert(profile == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_SHELL) == 0);
	xrl_clear_error(&error);

	profile = ComptonProfile_Partial(26, K_SHELL, 101, &error);
	assert(profile == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, SPLINT_X_TOO_HIGH) == 0);
	xrl_clear_error(&error);



	return 0;
}
