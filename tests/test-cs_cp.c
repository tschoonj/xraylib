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

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#define COMPOUND "Ca5(PO4)3F" /* Fluorapatite */
#define NIST_COMPOUND "Ferrous Sulfate Dosimeter Solution"

#define test_cp_f(fun) \
	sum = 0.0; \
	for (i = 0 ; i < cd->nElements ; i++) { \
		sum += cd->massFractions[i] * fun(cd->Elements[i], 10.0, &error); \
		assert(error == NULL); \
	} \
	assert(fabs(sum - fun ## _CP(COMPOUND, 10.0, &error)) < 1E-6); \
	assert(error == NULL); \
	sum = 0.0; \
	for (i = 0 ; i < cdn->nElements ; i++) { \
		sum += cdn->massFractions[i] * fun(cdn->Elements[i], 10.0, &error); \
		assert(error == NULL); \
	} \
	assert(fabs(sum - fun ## _CP(NIST_COMPOUND, 10.0, &error)) < 1E-6); \
	assert(error == NULL); \
	rv = fun ## _CP("ajajajajaja", 10.0, &error); \
	assert(rv == 0.0); \
	assert(error != NULL); \
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT); \
	assert(strcmp(error->message, UNKNOWN_COMPOUND) == 0); \
	xrl_clear_error(&error); \
	rv = fun ## _CP(COMPOUND, -1.0, &error); \
	assert(rv == 0.0); \
	assert(error != NULL); \
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT); \
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0); \
	xrl_clear_error(&error);

#define test_cp_ff(fun) \
	sum = 0.0; \
	for (i = 0 ; i < cd->nElements ; i++) { \
		sum += cd->massFractions[i] * fun(cd->Elements[i], 10.0, M_PI/4, &error); \
		assert(error == NULL); \
	} \
	assert(fabs(sum - fun ## _CP(COMPOUND, 10.0, M_PI/4, &error)) < 1E-6); \
	assert(error == NULL); \
	sum = 0.0; \
	for (i = 0 ; i < cdn->nElements ; i++) { \
		sum += cdn->massFractions[i] * fun(cdn->Elements[i], 10.0, M_PI/4, &error); \
		assert(error == NULL); \
	} \
	assert(fabs(sum - fun ## _CP(NIST_COMPOUND, 10.0, M_PI/4, &error)) < 1E-6); \
	assert(error == NULL); \
	rv = fun ## _CP("ajajajajaja", 10.0, M_PI/4, &error); \
	assert(rv == 0.0); \
	assert(error != NULL); \
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT); \
	assert(strcmp(error->message, UNKNOWN_COMPOUND) == 0); \
	xrl_clear_error(&error); \
	rv = fun ## _CP(COMPOUND, -1.0, M_PI/4, &error); \
	assert(rv == 0.0); \
	assert(error != NULL); \
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT); \
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0); \
	xrl_clear_error(&error);

#define test_cp_fff(fun) \
	sum = 0.0; \
	for (i = 0 ; i < cd->nElements ; i++) { \
		sum += cd->massFractions[i] * fun(cd->Elements[i], 10.0, M_PI/4, M_PI/4, &error); \
		assert(error == NULL); \
	} \
	assert(fabs(sum - fun ## _CP(COMPOUND, 10.0, M_PI/4, M_PI/4, &error)) < 1E-6); \
	assert(error == NULL); \
	sum = 0.0; \
	for (i = 0 ; i < cdn->nElements ; i++) { \
		sum += cdn->massFractions[i] * fun(cdn->Elements[i], 10.0, M_PI/4, M_PI/4, &error); \
		assert(error == NULL); \
	} \
	assert(fabs(sum - fun ## _CP(NIST_COMPOUND, 10.0, M_PI/4, M_PI/4, &error)) < 1E-6); \
	assert(error == NULL); \
	rv = fun ## _CP("ajajajajaja", 10.0, M_PI/4, M_PI/4, &error); \
	assert(rv == 0.0); \
	assert(error != NULL); \
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT); \
	assert(strcmp(error->message, UNKNOWN_COMPOUND) == 0); \
	xrl_clear_error(&error); \
	rv = fun ## _CP(COMPOUND, -1.0, M_PI/4, M_PI/4, &error); \
	assert(rv == 0.0); \
	assert(error != NULL); \
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT); \
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0); \
	xrl_clear_error(&error);

int main(int argc, char **argv) {
	xrl_error *error = NULL;
	struct compoundData *cd = NULL;
	struct compoundDataNIST *cdn = NULL;
	double sum, rv;
	int i;


	/* CS_CP routines will first try and parse as a compound. If this fails, NIST compounds will be tried instead */
	cd = CompoundParser(COMPOUND, &error);
	assert(cd != NULL);
	assert(error == NULL);

	cdn = GetCompoundDataNISTByName(NIST_COMPOUND, &error);
	assert(cdn != NULL);
	assert(error == NULL);

	test_cp_f(CS_Total)
	test_cp_f(CS_Photo)
	test_cp_f(CS_Compt)
	test_cp_f(CS_Rayl)
	test_cp_f(CS_Total_Kissel)
	test_cp_f(CS_Photo_Total)
	test_cp_f(CSb_Total)
	test_cp_f(CSb_Photo)
	test_cp_f(CSb_Compt)
	test_cp_f(CSb_Rayl)
	test_cp_f(CSb_Total_Kissel)
	test_cp_f(CSb_Photo_Total)
	test_cp_f(CS_Energy)
	test_cp_ff(DCS_Rayl)
	test_cp_ff(DCS_Compt)
	test_cp_ff(DCSb_Rayl)
	test_cp_ff(DCSb_Compt)
	test_cp_fff(DCSP_Rayl)
	test_cp_fff(DCSP_Compt)
	test_cp_fff(DCSPb_Rayl)
	test_cp_fff(DCSPb_Compt)

	FreeCompoundData(cd);
	FreeCompoundDataNIST(cdn);

	return 0;
}
