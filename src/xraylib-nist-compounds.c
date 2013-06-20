#include "xrayvars.h"
#include "xraylib-nist-compounds-internal.h" 
#include <string.h>
#include <search.h>
#include <xraylib-aux.h>
#include <stdlib.h>
#include <stdio.h>

static int CompareCompoundDataNIST(const void *a, const void *b) {
	struct compoundDataNIST *ac = (struct compoundDataNIST *) a;
	struct compoundDataNIST *bc = (struct compoundDataNIST *) b;
	return strcmp(ac->name, bc->name);
}

struct compoundDataNIST *GetCompoundDataNISTByName(const char compoundString[]) {

	struct compoundDataNIST *key = malloc(sizeof(struct compoundDataNIST));
	struct compoundDataNIST *rv;
	char *buffer;
#ifndef _WIN32
	size_t nelp;
#else
	unsigned int nelp;
#endif
	key->name = strdup(compoundString);
	

#ifndef _WIN32
	nelp = nCompoundDataNISTList;

	rv = lfind(key, compoundDataNISTList, &nelp, sizeof(struct compoundDataNIST), CompareCompoundDataNIST);
#else
	nelp = nCompoundDataNISTList;

	rv = _lfind(key, compoundDataNISTList, &nelp, sizeof(struct compoundDataNIST), CompareCompoundDataNIST);
#endif

	free(key->name);

	if (rv != NULL) {
		key->name = strdup(rv->name);
		key->nElements = rv->nElements; 
		key->Elements = malloc(sizeof(int)*rv->nElements);
		memcpy(key->Elements, rv->Elements, sizeof(int)*rv->nElements);
		key->massFractions = malloc(sizeof(double)*rv->nElements);
		memcpy(key->massFractions, rv->massFractions, sizeof(double)*rv->nElements);
		key->density = rv->density;
	}
	else {
		free(key);
		buffer = malloc(sizeof(char)*(strlen("xraylib-nist-compounds: no match found for ")+strlen(compoundString)+1));
		sprintf(buffer,"xraylib-nist-compounds: no match found for %s", compoundString);
		ErrorExit(buffer);
		free(buffer);
		key = NULL;
	}
	return key;
}

struct compoundDataNIST *GetCompoundDataNISTByIndex(int compoundIndex) {
	struct compoundDataNIST *key;

	if (compoundIndex < 0 || compoundIndex >= nCompoundDataNISTList) {
		char buffer[1000];
		sprintf(buffer,"xraylib-nist-compounds: no match found for index %i", compoundIndex);
		ErrorExit(buffer);
		/* compoundIndex out of range */
		return NULL;
	}

	key = malloc(sizeof(struct compoundDataNIST));
	key->name = strdup(compoundDataNISTList[compoundIndex].name);
	key->nElements = compoundDataNISTList[compoundIndex].nElements; 
	key->Elements = malloc(sizeof(int)*compoundDataNISTList[compoundIndex].nElements);
	memcpy(key->Elements, compoundDataNISTList[compoundIndex].Elements, sizeof(int)*compoundDataNISTList[compoundIndex].nElements);
	key->massFractions = malloc(sizeof(double)*compoundDataNISTList[compoundIndex].nElements);
	memcpy(key->massFractions, compoundDataNISTList[compoundIndex].massFractions, sizeof(double)*compoundDataNISTList[compoundIndex].nElements);
	key->density = compoundDataNISTList[compoundIndex].density;
	
	return key;
}

char **GetCompoundDataNISTList(int *nCompounds) {
	int i;
	char **rv;

	if (nCompounds != NULL)
		*nCompounds = nCompoundDataNISTList;

	rv = malloc(sizeof(char *)*(nCompoundDataNISTList+1));

	for (i = 0 ; i < nCompoundDataNISTList; i++)
		rv[i] = strdup(compoundDataNISTList[i].name);

	rv[nCompoundDataNISTList] = NULL;

	return rv;
}

void FreeCompoundDataNIST(struct compoundDataNIST *compoundData) {
	free(compoundData->name);
	free(compoundData->Elements);
	free(compoundData->massFractions);
	free(compoundData);
}
