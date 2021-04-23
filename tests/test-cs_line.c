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

static struct {int line_lower; int line_upper; int shell;} line_mappings[] = {
  {KP5_LINE, KL1_LINE, K_SHELL},
  {L1P5_LINE, L1L2_LINE, L1_SHELL},
  {L2Q1_LINE, L2L3_LINE, L2_SHELL},
  {L3Q1_LINE, L3M1_LINE, L3_SHELL},
};

int main(int argc, char **argv) {
	xrl_error *error = NULL;
	double cs, cs2, rr;
	int i, j;

	cs = CS_FluorLine(29, L3M5_LINE, 10.0, &error);
	assert(error == NULL);
	assert(fabs(0.13198670698075143 - cs) < 1E-6);

	cs = CS_FluorLine(29, L1M5_LINE, 10.0, &error);
	assert(error == NULL);
	assert(fabs(7.723944209880828e-06 - cs) < 1E-8);

	cs = CS_FluorLine(29, L1M2_LINE, 10.0, &error);
	assert(error == NULL);
	assert(fabs(0.0018343168459245755 - cs) < 1E-6);

	cs = CS_FluorLine(29, KL3_LINE, 10.0, &error);
	assert(error == NULL);
	assert(fabs(49.21901698835919 - cs) < 1E-6);

	/* lack of FL23_TRANS CosKronTransProb values for Z < Cu will cause UNAVAILABLE_CK errors for L3 lines */
	cs = CS_FluorLine(26, L3M5_LINE, 10.0, &error);
	assert(cs == 0.0);
	assert(error != NULL);
	assert(strcmp(error->message, UNAVAILABLE_CK) == 0);
	xrl_clear_error(&error);
	cs = CS_FluorLine(26, L2M4_LINE, 10.0, &error);
	assert(fabs(0.0200667 - cs) < 1E-6);
	assert(error == NULL);
	cs = CS_FluorLine(26, L1M2_LINE, 10.0, &error);
	assert(fabs(0.000830915 - cs) < 1E-6);
	assert(error == NULL);

	/* lets try some bad input */
	cs = CS_FluorLine(0, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine(ZMAX, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine(ZMAX + 1, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine(1, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	/* M-lines are not supported at all due to missing jump factors for the M-subshells */
	cs = CS_FluorLine(92, M5N7_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine(26, KL3_LINE, 0.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine(26, KL3_LINE, 1001, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, SPLINT_X_TOO_HIGH) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine(26, KL3_LINE, 10.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(26, KL2_LINE, 10.0, &error);
	assert(error == NULL);
	assert(fabs(cs - CS_FluorLine(26, KA_LINE, 10.0, NULL)) < 1E-6);

	cs = CS_FluorLine(92, L3M5_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, TOO_LOW_EXCITATION_ENERGY) == 0);
	xrl_clear_error(&error);
	cs = CS_FluorLine(92, L3M5_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, L3M4_LINE, 30.0, &error);
	assert(error == NULL);
	assert(fabs(cs - CS_FluorLine(92, LA_LINE, 30.0, NULL)) < 1E-6);

	cs = 0.0;
	cs += CS_FluorLine(92, LB1_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, LB2_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, LB3_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, LB4_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, LB5_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, LB6_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, LB7_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, LB9_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, LB10_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, LB15_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, LB17_LINE, 30.0, &error);
	assert(error == NULL);
	/* The following 2 lines have no corresponding RadRates... L3O45 (LB5) does though
	cs += CS_FluorLine(92, L3O4_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, L3O5_LINE, 30.0, &error);
	assert(error == NULL);*/
	cs += CS_FluorLine(92, L3N6_LINE, 30.0, &error);
	assert(error == NULL);
	cs += CS_FluorLine(92, L3N7_LINE, 30.0, &error);
	assert(error == NULL);
	cs2 = CS_FluorLine(92, LB_LINE, 30.0, NULL);
	assert(fabs(cs2 - cs) < 1E-6);

	/* FluorShell tests */
	cs = CS_FluorShell(0, K_SHELL, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorShell(ZMAX, K_SHELL, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_SHELL) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorShell(ZMAX + 1, K_SHELL, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorShell(1, K_SHELL, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_SHELL) == 0);
	xrl_clear_error(&error);

	/* M-lines are not supported at all due to missing jump factors for the M-subshells */
	cs = CS_FluorShell(92, M1_SHELL, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_SHELL) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorShell(92, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_SHELL) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorShell(26, K_SHELL, 0.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorShell(26, K_SHELL, 1001, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, SPLINT_X_TOO_HIGH) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorShell(26, K_SHELL, 5, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, TOO_LOW_EXCITATION_ENERGY) == 0);
	xrl_clear_error(&error);


    for (i = 0 ; i < sizeof(line_mappings)/sizeof(line_mappings[0]) ; i++) {
		cs = CS_FluorShell(92, line_mappings[i].shell, 120.0, &error);
		assert(error == NULL);
		cs2 = 0;
		rr = 0;
		for (j = line_mappings[i].line_lower ; j <= line_mappings[i].line_upper ; j++) {
			rr += RadRate(92, j, NULL);
			cs2 += CS_FluorLine(92, j, 120.0, NULL);
		}
		assert(fabs(cs2 - rr*cs) < 1E-6);
	}

	return 0;
}
