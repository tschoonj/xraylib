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
#include <stddef.h>
#include <string.h>
#include "xraylib-radionuclides-internal.h"

int main(int argc, char *argv[]) {
	xrl_error *error = NULL;
	char **nuclides= NULL;
	int nNuclides = 0;
	int i;
	struct radioNuclideData *rnd = NULL;

	nuclides = GetRadioNuclideDataList(&nNuclides, &error);
	assert(error == NULL);
	assert(nuclides != NULL);
	assert(nNuclides == nNuclideDataList);
	nNuclides = 0;
	for (i = 0 ; nuclides[i] != NULL ; i++) {
		nNuclides++;
		xrlFree(nuclides[i]);
	}
	xrlFree(nuclides);
	assert(nNuclides == nNuclideDataList);
	
	nuclides = GetRadioNuclideDataList(&nNuclides, NULL);
	assert(nuclides != NULL);

	for (i = 0 ; nuclides[i] != NULL ; i++) {
		rnd = GetRadioNuclideDataByName(nuclides[i], &error);
		assert(rnd != NULL);
		assert(error == NULL);
		assert(strcmp(rnd->name, nuclides[i]) == 0);
		assert(strcmp(rnd->name, nuclideDataList[i].name) == 0);
		assert(rnd->Z == nuclideDataList[i].Z);
		assert(rnd->A == nuclideDataList[i].A);
		assert(rnd->N == nuclideDataList[i].N);
		assert(rnd->Z_xray == nuclideDataList[i].Z_xray);
		assert(rnd->nXrays == nuclideDataList[i].nXrays);
		assert(memcmp(rnd->XrayLines, nuclideDataList[i].XrayLines, sizeof(int) * rnd->nXrays) == 0);
		assert(memcmp(rnd->XrayIntensities, nuclideDataList[i].XrayIntensities, sizeof(double) * rnd->nXrays) == 0);
		assert(rnd->nGammas == nuclideDataList[i].nGammas);
		assert(memcmp(rnd->GammaEnergies, nuclideDataList[i].GammaEnergies, sizeof(double) * rnd->nGammas) == 0);
		assert(memcmp(rnd->GammaIntensities, nuclideDataList[i].GammaIntensities, sizeof(double) * rnd->nGammas) == 0);
		FreeRadioNuclideData(rnd);
		xrlFree(nuclides[i]);
	}
	xrlFree(nuclides);

	for (i = 0 ; i < nNuclideDataList; i++) {
		rnd = GetRadioNuclideDataByIndex(i, &error);
		assert(rnd != NULL);
		assert(error == NULL);
		assert(strcmp(rnd->name, nuclideDataList[i].name) == 0);
		assert(rnd->Z == nuclideDataList[i].Z);
		assert(rnd->A == nuclideDataList[i].A);
		assert(rnd->N == nuclideDataList[i].N);
		assert(rnd->Z_xray == nuclideDataList[i].Z_xray);
		assert(rnd->nXrays == nuclideDataList[i].nXrays);
		assert(memcmp(rnd->XrayLines, nuclideDataList[i].XrayLines, sizeof(int) * rnd->nXrays) == 0);
		assert(memcmp(rnd->XrayIntensities, nuclideDataList[i].XrayIntensities, sizeof(double) * rnd->nXrays) == 0);
		assert(rnd->nGammas == nuclideDataList[i].nGammas);
		assert(memcmp(rnd->GammaEnergies, nuclideDataList[i].GammaEnergies, sizeof(double) * rnd->nGammas) == 0);
		assert(memcmp(rnd->GammaIntensities, nuclideDataList[i].GammaIntensities, sizeof(double) * rnd->nGammas) == 0);
		FreeRadioNuclideData(rnd);
	}

	/* bad input */
	rnd = GetRadioNuclideDataByIndex(-1, &error);
	assert(rnd == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, "-1 is out of the range of indices covered by the radionuclide database") == 0);
	xrl_clear_error(&error);

	rnd = GetRadioNuclideDataByIndex(nNuclideDataList, &error);
	assert(rnd == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, "10 is out of the range of indices covered by the radionuclide database") == 0);
	xrl_clear_error(&error);

	rnd = GetRadioNuclideDataByName(NULL, &error);
	assert(rnd == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, "radioNuclideString cannot be NULL") == 0);
	xrl_clear_error(&error);

	rnd = GetRadioNuclideDataByName("", &error);
	assert(rnd == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, " was not found in the radionuclide database") == 0);
	xrl_clear_error(&error);

	return 0;
}

