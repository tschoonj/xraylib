#include "xraylib-nist-compounds-internal.h" 
#include <string.h>
#include <search.h>
#include <xraylib-aux.h>
#include <stdlib.h>

static int CompareCompoundDataNIST(const void *a, const void *b) {
	struct compoundDataNIST *ac = (struct compoundDataNIST *) a;
	struct compoundDataNIST *bc = (struct compoundDataNIST *) b;
	return strcmp(ac->name, bc->name);
}

struct compoundDataNIST *GetCompoundDataNISTByName(const char compoundString[]) {

	struct compoundDataNIST *key = malloc(sizeof(struct compoundDataNIST));
	key->name = strdup(compoundString);
	
	struct compoundDataNIST *rv;

#ifndef _WIN32
	size_t nelp = nCompoundDataNISTList;

	rv = lfind(key, compoundDataNISTList, &nelp, sizeof(struct compoundDataNIST), CompareCompoundDataNIST);
#else
	unsigned int nelp = nCompoundDataNISTList;

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
		key = NULL;
	}
	return key;
}

struct compoundDataNIST *GetCompoundDataNISTByIndex(int compoundIndex) {
	struct compoundDataNIST *key;

	if (compoundIndex < 0 || compoundIndex >= nCompoundDataNISTList) {
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

	if (nCompounds != NULL)
		*nCompounds = nCompoundDataNISTList;

	char **rv = malloc(sizeof(char *)*(nCompoundDataNISTList+1));

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
