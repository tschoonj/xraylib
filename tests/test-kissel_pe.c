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
  {M1P5_LINE, M1N1_LINE, M1_SHELL},
  {M2P5_LINE, M2N1_LINE, M2_SHELL},
  {M3Q1_LINE, M3N1_LINE, M3_SHELL},
  {M4P5_LINE, M4N1_LINE, M4_SHELL},
  {M5P5_LINE, M5N1_LINE, M5_SHELL},
};

static int test_lines[] ={
	KL3_LINE,
	L1M3_LINE,
	L2M4_LINE,
	L3M5_LINE,
	M1N3_LINE,
	M2N4_LINE,
	M3N5_LINE,
	M4N6_LINE,
	M5N7_LINE,
};

static struct {
	double (*shell)(int, int, double, xrl_error **);
	double(*line)(int, int, double, xrl_error **);
	double line_vals_expected[9];
	} function_mappings[] = {
	{CS_FluorShell_Kissel, CS_FluorLine_Kissel, {1.488296, 0.021101, 0.431313, 0.701276, 0.000200, 0.004753, 0.004467, 0.099232, 0.134301}},
	{CSb_FluorShell_Kissel, CSb_FluorLine_Kissel, {588.359681, 8.341700, 170.508401, 277.231590, 0.078971, 1.878953, 1.766078, 39.228801, 53.092598}},
	{CS_FluorShell_Kissel_Cascade, CS_FluorLine_Kissel_Cascade, {1.488296, 0.021101, 0.431313, 0.701276, 0.000200, 0.004753, 0.004467, 0.099232, 0.134301}},
	{CS_FluorShell_Kissel_Radiative_Cascade, CS_FluorLine_Kissel_Radiative_Cascade, {1.488296, 0.017568, 0.413908, 0.671135, 0.000092, 0.001906, 0.001758, 0.043009, 0.055921}},
	{CS_FluorShell_Kissel_Nonradiative_Cascade, CS_FluorLine_Kissel_Nonradiative_Cascade, {1.488296, 0.021101, 0.100474, 0.169412, 0.000104, 0.001204, 0.001106, 0.018358, 0.025685}},
	{CS_FluorShell_Kissel_no_Cascade, CS_FluorLine_Kissel_no_Cascade, {1.488296, 0.017568, 0.083069, 0.139271, 0.000053, 0.000417, 0.000327, 0.003360, 0.004457}},
	{CSb_FluorShell_Kissel_Cascade, CSb_FluorLine_Kissel_Cascade, {588.359681, 8.341700, 170.508401, 277.231590, 0.078971, 1.878953, 1.766078, 39.228801, 53.092598}},
	{CSb_FluorShell_Kissel_Radiative_Cascade, CSb_FluorLine_Kissel_Radiative_Cascade, {588.359681, 6.945250, 163.627802, 265.316234, 0.036434, 0.753313, 0.695153, 17.002549, 22.106758}},
	{CSb_FluorShell_Kissel_Nonradiative_Cascade, CSb_FluorLine_Kissel_Nonradiative_Cascade, {588.359681, 8.341700, 39.719983, 66.972714, 0.041157, 0.475948, 0.437288, 7.257269, 10.153862}},
	{CSb_FluorShell_Kissel_no_Cascade, CSb_FluorLine_Kissel_no_Cascade, {588.359681, 6.945250, 32.839384, 55.057358, 0.020941, 0.164658, 0.129253, 1.328221, 1.762099}},
};

int main(int argc, char **argv) {
	xrl_error *error = NULL;
	double cs, cs2, ec;
	int i, j, k;

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

	/* see https://github.com/tschoonj/xraylib/issues/187 */
	cs = CS_FluorLine_Kissel(47, L2M4_LINE, 3.5282, &error);
	assert(error == NULL);

	/* lets try some bad input */
	cs = CS_FluorLine_Kissel(0, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine_Kissel(ZMAX, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine_Kissel(ZMAX + 1, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	cs = CS_FluorLine_Kissel(1, KL3_LINE, 10.0, &error);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_LINE) == 0);
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

	/* see https://github.com/tschoonj/xraylib/issues/187 */
	cs = CSb_Photo_Partial(47, L2_SHELL, 3.5282, &error);
	assert(error == NULL);
	assert(fabs(cs - 1.569549E+04)/cs < 1E-6);

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

	/* FluorShell tests */
    for (i = 0 ; i < sizeof(function_mappings)/sizeof(function_mappings[0]) ; i++) {

		for (j = 0 ; j < 9 ; j++) {
			cs = function_mappings[i].line(92, test_lines[j], 120.0, &error);
			assert(error == NULL);
			assert(fabs(cs - function_mappings[i].line_vals_expected[j]) < 1E-6);
		}

		cs = function_mappings[i].shell(0, K_SHELL, 10.0, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
		xrl_clear_error(&error);

		cs = function_mappings[i].shell(ZMAX, K_SHELL, 10.0, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, INVALID_SHELL) == 0);
		xrl_clear_error(&error);

		cs = function_mappings[i].shell(ZMAX + 1, K_SHELL, 10.0, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
		xrl_clear_error(&error);

		cs = function_mappings[i].shell(1, K_SHELL, 10.0, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, INVALID_SHELL) == 0);
		xrl_clear_error(&error);

		cs = function_mappings[i].shell(92, KL3_LINE, 10.0, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, INVALID_SHELL) == 0);
		xrl_clear_error(&error);

		cs = function_mappings[i].shell(26, K_SHELL, 0.0, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
		xrl_clear_error(&error);

		cs = function_mappings[i].shell(26, K_SHELL, 1001, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, SPLINT_X_TOO_HIGH) == 0);
		xrl_clear_error(&error);

		cs = function_mappings[i].shell(26, K_SHELL, 5, &error);
		assert(error != NULL);
		assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
		assert(strcmp(error->message, TOO_LOW_EXCITATION_ENERGY) == 0);
		xrl_clear_error(&error);

    	for (j = 0 ; j < sizeof(line_mappings)/sizeof(line_mappings[0]) ; j++) {
			double cs = function_mappings[i].shell(92, line_mappings[j].shell, 120.0, &error);
			assert(error == NULL);
			double cs2 = 0;
			double rr = 0;
			for (k = line_mappings[j].line_lower ; k <= line_mappings[j].line_upper ; k++) {
				rr += RadRate(92, k, NULL);
				cs2 += function_mappings[i].line(92, k, 120.0, NULL);
			}
			assert(fabs(cs2 - rr*cs) < 1E-6);
		}
	}
	return 0;
}
