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
	double line_energy = 0.0;

	/* some simple IUPAC tests */
	line_energy = LineEnergy(26, KL3_LINE, &error);
	assert(fabs(line_energy - 6.4039) < 1E-6);
	assert(error == NULL);

	line_energy = LineEnergy(92, KL1_LINE, &error);
	assert(fabs(line_energy - 93.844) < 1E-6);
	assert(error == NULL);

	line_energy = LineEnergy(56, L3M1_LINE, &error);
	assert(fabs(line_energy - 3.9542) < 1E-6);
	assert(error == NULL);

	line_energy = LineEnergy(82, M5N7_LINE, &error);
	assert(fabs(line_energy - 2.3477) < 1E-6);
	assert(error == NULL);

	/* bad input */
	line_energy = LineEnergy(0, KL3_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(-1, KL3_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(ZMAX + 1, KL3_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(ZMAX, KL3_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(105, KL3_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(104, KL3_LINE, &error);
	assert(error == NULL);
	assert(line_energy > 0);

	line_energy = LineEnergy(26, 1000, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, UNKNOWN_LINE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(26, M5N7_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	/* Siegbahn macros */
	line_energy = LineEnergy(26, KA_LINE, &error);
	assert(error == NULL);
	assert(fabs(line_energy - 6.399505664957576) < 1E-6);

	line_energy = LineEnergy(26, KB_LINE, &error);
	assert(error == NULL);
	assert(fabs(line_energy - 7.058) < 1E-6);

	line_energy = LineEnergy(1, KA_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(26, LA_LINE, &error);
	assert(error == NULL);
	assert(fabs(line_energy - 0.7045) < 1E-6);

	line_energy = LineEnergy(26, LB_LINE, &error);
	assert(error == NULL);
	assert(fabs(line_energy - 0.724378) < 1E-6);

	line_energy = LineEnergy(0, KA_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(92, L1N67_LINE, &error);
	assert(error == NULL);
	assert(fabs(line_energy - (LineEnergy(92, L1N6_LINE, NULL) + LineEnergy(92, L1N7_LINE, NULL)) / 2.0) < 1E-6);

	/* test that LB_LINE starts at Z=13 */
	line_energy = LineEnergy(12, LB_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(13, LB_LINE, &error);
	assert(fabs(line_energy - 0.112131) < 1E-6);
	assert(error == NULL);

	/* test that LA_LINE starts at Z=21 */
	line_energy = LineEnergy(20, LA_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(21, LA_LINE, &error);
	assert(fabs(line_energy - 0.3956) < 1E-6);
	assert(error == NULL);

	/* test KO_LINE and KP_LINE */
	/* KO_LINE starts at 48 */
	line_energy = LineEnergy(47, KO_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(48, KO_LINE, &error);
	assert(fabs(line_energy - 26.709) < 1E-6);
	assert(fabs(line_energy - LineEnergy(48, KO1_LINE, NULL)) < 1E-6);
	assert(error == NULL);

	/* KP_LINE starts at 82 */
	line_energy = LineEnergy(81, KP_LINE, &error);
	assert(line_energy == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	line_energy = LineEnergy(82, KP_LINE, &error);
	assert(fabs(line_energy - 88.0014) < 1E-6);
	assert(fabs(line_energy - LineEnergy(82, KP1_LINE, NULL)) < 1E-6);
	assert(error == NULL);

	return 0;
}
