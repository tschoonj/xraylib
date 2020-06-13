/*
Copyright (c) 2014-2018, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "config.h"
#include "xraylib-aux.h"
#include "xrayvars.h"
#include "xraylib.h"
#include "xraylib-error-private.h"
#include "xraylib-nist-compounds-internal.h" 
#include <string.h>
#include <search.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>


static int CompareCompoundDataNIST(const void *a, const void *b) {
	struct compoundDataNIST *ac = (struct compoundDataNIST *) a;
	struct compoundDataNIST *bc = (struct compoundDataNIST *) b;
	return strcmp(ac->name, bc->name);
}

struct compoundDataNIST *GetCompoundDataNISTByName(const char compoundString[], xrl_error **error) {

	struct compoundDataNIST *key = malloc(sizeof(struct compoundDataNIST));
	struct compoundDataNIST *rv;
#ifndef _WIN32
	size_t nelp;
#else
	unsigned int nelp;
#endif
	if (key == NULL) {
		xrl_set_error(error, XRL_ERROR_MEMORY, MALLOC_ERROR, strerror(errno));
		return NULL;
	}
	if (compoundString == NULL) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "compoundString cannot be NULL");
		return NULL;
	}
	key->name = xrl_strdup(compoundString);
	

#ifndef _WIN32
	nelp = nCompoundDataNISTList;

	rv = lfind(key, compoundDataNISTList, &nelp, sizeof(struct compoundDataNIST), CompareCompoundDataNIST);
#else
	nelp = nCompoundDataNISTList;

	rv = _lfind(key, compoundDataNISTList, &nelp, sizeof(struct compoundDataNIST), CompareCompoundDataNIST);
#endif

	free(key->name);

	if (rv != NULL) {
		key->name = xrl_strdup(rv->name);
		key->nElements = rv->nElements; 
		key->Elements = malloc(sizeof(int)*rv->nElements);
		memcpy(key->Elements, rv->Elements, sizeof(int)*rv->nElements);
		key->massFractions = malloc(sizeof(double)*rv->nElements);
		memcpy(key->massFractions, rv->massFractions, sizeof(double)*rv->nElements);
		key->density = rv->density;
	}
	else {
		free(key);
		xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "%s was not found in the NIST compound database", compoundString);
		key = NULL;
	}
	return key;
}

struct compoundDataNIST *GetCompoundDataNISTByIndex(int compoundIndex, xrl_error **error) {
	struct compoundDataNIST *key;

	if (compoundIndex < 0 || compoundIndex >= nCompoundDataNISTList) {
		xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "%d is out of the range of indices covered by the NIST compound database", compoundIndex);
		return NULL;
	}

	key = malloc(sizeof(struct compoundDataNIST));
	if (key == NULL) {
		xrl_set_error(error, XRL_ERROR_MEMORY, MALLOC_ERROR, strerror(errno));
		return NULL;
	}
	key->name = xrl_strdup(compoundDataNISTList[compoundIndex].name);
	key->nElements = compoundDataNISTList[compoundIndex].nElements; 
	key->Elements = malloc(sizeof(int)*compoundDataNISTList[compoundIndex].nElements);
	memcpy(key->Elements, compoundDataNISTList[compoundIndex].Elements, sizeof(int)*compoundDataNISTList[compoundIndex].nElements);
	key->massFractions = malloc(sizeof(double)*compoundDataNISTList[compoundIndex].nElements);
	memcpy(key->massFractions, compoundDataNISTList[compoundIndex].massFractions, sizeof(double)*compoundDataNISTList[compoundIndex].nElements);
	key->density = compoundDataNISTList[compoundIndex].density;
	
	return key;
}

char **GetCompoundDataNISTList(int *nCompounds, xrl_error **error) {
	int i;
	char **rv;

	if (nCompounds != NULL)
		*nCompounds = nCompoundDataNISTList;

	rv = malloc(sizeof(char *)*(nCompoundDataNISTList+1));
	if (rv == NULL) {
		xrl_set_error(error, XRL_ERROR_MEMORY, MALLOC_ERROR, strerror(errno));
		return NULL;
	}

	for (i = 0 ; i < nCompoundDataNISTList; i++)
		rv[i] = xrl_strdup(compoundDataNISTList[i].name);

	rv[nCompoundDataNISTList] = NULL;

	return rv;
}

void FreeCompoundDataNIST(struct compoundDataNIST *compoundData) {
	free(compoundData->name);
	free(compoundData->Elements);
	free(compoundData->massFractions);
	free(compoundData);
}
