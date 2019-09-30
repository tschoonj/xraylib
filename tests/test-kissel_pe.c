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
	double cs, cs2, ec;

	/* some simpler ones */
	cs = CS_FluorLine_Kissel(29, L3M5_LINE, 10.0, &error);
	assert(error == NULL);
	assert(fabs(1.677692151560 - cs) < 1E-6);

	cs = CS_FluorLine_Kissel(29, L1M5_LINE, 10.0, &error);
	assert(error == NULL);
	assert(fabs(0.000126120244 - cs) < 1E-8);

	cs = CS_FluorLine_Kissel(29, L1M2_LINE, 10.0, &error);
	assert(error == NULL);
	assert(fabs(0.029951600106 - cs) < 1E-6);

	cs = CS_FluorLine_Kissel(29, KL3_LINE, 10.0, &error);
	assert(error == NULL);
	assert(fabs(49.51768761506201 - cs) < 1E-6);

	cs = CS_FluorLine_Kissel(82, M5N7_LINE, 30.0, &error);
	assert(error == NULL);
	assert(fabs(0.538227139546 - cs) < 1E-6);

	cs = CS_FluorLine_Kissel(82, M5N7_LINE, 100.0, &error);
	assert(error == NULL);
	assert(fabs(0.102639909656483 - cs) < 1E-6);

	/* lets try some bad input */
	cs = CS_FluorLine_Kissel(0, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine_Kissel(ZMAX, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_SHELL) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine_Kissel(ZMAX + 1, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine_Kissel(1, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_SHELL) == 0);
	xrl_clear_error(&error);

	/* N-lines are not supported at all due to missing RadRate/CosKronTransProb data */
	cs = CS_FluorLine_Kissel(92, N1O3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine_Kissel(26, KL3_LINE, 0.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine_Kissel(26, KL3_LINE, 301, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, SPLINT_X_TOO_HIGH) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine_Kissel(26, KL3_LINE, 10.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(26, KL2_LINE, 10.0, &error);
	assert(error == NULL);
	assert(fabs(cs - CS_FluorLine_Kissel(26, KA_LINE, 10.0, NULL)) < 1E-6);

	cs = CS_FluorLine_Kissel(92, L3M5_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, TOO_LOW_EXCITATION_ENERGY) == 0);
	xrl_clear_error(&error);
	cs = CS_FluorLine_Kissel(92, L3M5_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, L3M4_LINE, 30.0, &error);
	assert(error == NULL);
	assert(fabs(cs - CS_FluorLine_Kissel(92, LA_LINE, 30.0, NULL)) < 1E-6);

	cs = 0.0;
	cs += CS_FluorLine_Kissel(92, LB1_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, LB2_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, LB3_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, LB4_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, LB5_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, LB6_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, LB7_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, LB9_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, LB10_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, LB15_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, LB17_LINE, 30.0, &error);
	assert(error == NULL);
	/* The following 2 lines have no corresponding RadRates... L3O45 (LB5) does though
	cs += CS_FluorLine_Kissel(92, L3O4_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, L3O5_LINE, 30.0, &error);
	assert(error == NULL);*/
	cs += CS_FluorLine_Kissel(92, L3N6_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine_Kissel(92, L3N7_LINE, 30.0, &error);
	assert(error == NULL);
	cs2 = CS_FluorLine_Kissel(92, LB_LINE, 30.0, NULL);
	assert(fabs(cs2 - cs) < 1E-6);

	/* CS_Photo_Partial tests */
	cs = CS_Photo_Partial(26, K_SHELL, 20.0, &error);
	assert(error == NULL);
	assert(fabs(cs - 22.40452459077649) < 1E-6);

	cs = CS_Photo_Partial(26, K_SHELL, 6.0, &error);
	assert(error != NULL);
	assert(cs == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, TOO_LOW_EXCITATION_ENERGY) == 0);
	xrl_clear_error(&error);
	
	cs = CS_Photo_Partial(26, N5_SHELL, 16.0, &error);
	assert(error != NULL);
	assert(cs == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_SHELL) == 0);
	xrl_clear_error(&error);
	
	cs = CS_Photo_Partial(26, SHELLNUM_K, 16.0, &error);
	assert(error != NULL);
	assert(cs == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, UNKNOWN_SHELL) == 0);
	xrl_clear_error(&error);
	
	cs = CS_Photo_Partial(26, K_SHELL, 0.0, &error);
	assert(error != NULL);
	assert(cs == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);
	
	cs = CS_Photo_Partial(26, K_SHELL, 301, &error);
	assert(error != NULL);
	assert(cs == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, SPLINT_X_TOO_HIGH) == 0);
	xrl_clear_error(&error);
	
	cs = CS_Photo_Partial(0, K_SHELL, 0.0, &error);
	assert(error != NULL);
	assert(cs == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	ec = ElectronConfig(26, M5_SHELL, &error);
	assert(error == NULL);
	assert(ec == 3.6);

	ec = ElectronConfig(26, N7_SHELL, &error);
	assert(error != NULL);
	assert(ec == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_SHELL) == 0);
	xrl_clear_error(&error);

	ec = ElectronConfig(26, SHELLNUM_K, &error);
	assert(error != NULL);
	assert(ec == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, UNKNOWN_SHELL) == 0);
	xrl_clear_error(&error);

	ec = ElectronConfig(0, K_SHELL, &error);
	assert(error != NULL);
	assert(ec == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	return 0;
}
