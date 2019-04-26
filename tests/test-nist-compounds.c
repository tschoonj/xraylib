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
#include "xraylib-nist-compounds-internal.h"

int main(int argc, char *argv[]) {
	xrl_error *error = NULL;
	char **compounds = NULL;
	int nCompounds = 0;
	int i;
	struct compoundDataNIST *cdn = NULL;

	compounds = GetCompoundDataNISTList(&nCompounds, &error);
	assert(error == NULL);
	assert(compounds != NULL);
	assert(nCompounds == nCompoundDataNISTList);
	nCompounds = 0;
	for (i = 0 ; compounds[i] != NULL ; i++) {
		nCompounds++;
		xrlFree(compounds[i]);
	}
	xrlFree(compounds);
	assert(nCompounds == nCompoundDataNISTList);
	
	compounds = GetCompoundDataNISTList(&nCompounds, NULL);
	assert(compounds != NULL);

	for (i = 0 ; compounds[i] != NULL ; i++) {
		cdn = GetCompoundDataNISTByName(compounds[i], &error);
		assert(cdn != NULL);
		assert(error == NULL);
		assert(strcmp(cdn->name, compounds[i]) == 0);
		assert(strcmp(cdn->name, compoundDataNISTList[i].name) == 0);
		assert(cdn->nElements == compoundDataNISTList[i].nElements);
		assert(memcmp(cdn->Elements, compoundDataNISTList[i].Elements, sizeof(int) * cdn->nElements) == 0);
		assert(memcmp(cdn->massFractions, compoundDataNISTList[i].massFractions, sizeof(double) * cdn->nElements) == 0);
		assert(cdn->density == compoundDataNISTList[i].density);
		FreeCompoundDataNIST(cdn);
		xrlFree(compounds[i]);
	}
	xrlFree(compounds);

	for (i = 0 ; i < nCompoundDataNISTList ; i++) {
		cdn = GetCompoundDataNISTByIndex(i, &error);
		assert(cdn != NULL);
		assert(error == NULL);
		assert(strcmp(cdn->name, compoundDataNISTList[i].name) == 0);
		assert(cdn->nElements == compoundDataNISTList[i].nElements);
		assert(memcmp(cdn->Elements, compoundDataNISTList[i].Elements, sizeof(int) * cdn->nElements) == 0);
		assert(memcmp(cdn->massFractions, compoundDataNISTList[i].massFractions, sizeof(double) * cdn->nElements) == 0);
		assert(cdn->density == compoundDataNISTList[i].density);
		FreeCompoundDataNIST(cdn);
	}

	/* bad input */
	cdn = GetCompoundDataNISTByIndex(-1, &error);
	assert(cdn == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, "-1 is out of the range of indices covered by the NIST compound database") == 0);
	xrl_clear_error(&error);

	cdn = GetCompoundDataNISTByIndex(nCompoundDataNISTList, &error);
	assert(cdn == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, "180 is out of the range of indices covered by the NIST compound database") == 0);
	xrl_clear_error(&error);

	cdn = GetCompoundDataNISTByName(NULL, &error);
	assert(cdn == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, "compoundString cannot be NULL") == 0);
	xrl_clear_error(&error);

	cdn = GetCompoundDataNISTByName("", &error);
	assert(cdn == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, " was not found in the NIST compound database") == 0);
	xrl_clear_error(&error);

	return 0;
}
