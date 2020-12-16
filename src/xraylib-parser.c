/*Copyright (c) 2010, 2011, 2013, Tom Schoonjans
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
#include "xraylib.h"
#include "xraylib-error-private.h"
#include "xrayvars.h"
#include "xrayglob.h"
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <locale.h>


struct compoundAtom {
	int Element;
	double nAtoms;
};

struct compoundAtoms {
	int nElements;
	struct compoundAtom *singleElements;
};

static int compareCompoundAtoms(const void *i1, const void *i2) {
	struct compoundAtom *ca1 = (struct compoundAtom *) i1;
	struct compoundAtom *ca2 = (struct compoundAtom *) i2;

	return (ca1->Element - ca2->Element);
}

static int compareInt(const void *A, const void *B);

static int CompoundParserSimple(char compoundString[], struct compoundAtoms *ca, xrl_error **error) {

	int nbrackets=0;
	int nuppers=0;
	int i,j;
	char **upper_locs = NULL;
	char **brackets_begin_locs=NULL;
	char **brackets_end_locs=NULL;
	int nbracket_pairs=0;
	char *tempElement;
       	char *tempSubstring;
       	double tempnAtoms;
       	struct MendelElement *res;
       	struct compoundAtom *res2, key2;
    	struct compoundAtoms *tempBracketAtoms;
        char *tempBracketString;
	int ndots;
	char *endPtr;


	if (islower(compoundString[0]) || isdigit(compoundString[0])) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: Found a lowercase character or digit where not allowed");
		return 0;
	}

	for (i = 0 ; compoundString[i] != '\0' ; i++) {
		if (compoundString[i] == '(') {
			nbrackets++;
			if (nbrackets == 1) {
				brackets_begin_locs = realloc(brackets_begin_locs, sizeof(char *) * ++nbracket_pairs);
				brackets_begin_locs[nbracket_pairs-1] = compoundString+i;
			}
		}
		else if (compoundString[i] == ')') {
			nbrackets--;
			if (nbrackets == 0) {
				brackets_end_locs = realloc(brackets_end_locs, sizeof(char *) * nbracket_pairs);
				brackets_end_locs[nbracket_pairs-1] = compoundString+i;
			}
		}
		else if (nbrackets > 0) {
			/* this is ok... */
		}
		else if (nbrackets == 0 && isupper(compoundString[i])) {
			upper_locs = realloc(upper_locs, sizeof(char *) * ++nuppers);
			upper_locs[nuppers-1] = compoundString+i;
		}
		else if (compoundString[i] == ' '){
			xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: Spaces are not allowed in compound formula");
			return 0;
		}
		else if (i > 0 && islower(compoundString[i]) && isdigit(compoundString[i-1])) {
			xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: Found a lowercase character where not allowed");
			return 0;
		}
		else if (islower(compoundString[i]) || isdigit(compoundString[i]) || compoundString[i] == '.') {
			/* this is ok... */
		}
		else {
			xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: Invalid character %c detected", compoundString[i]);
			return 0;
		}

		if (nbrackets < 0) {
			xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: Brackets not matching");
			return 0;
		}
	}

	if (nuppers == 0 && nbracket_pairs == 0) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: No elements found");
		return 0;
	}
	if (nbrackets > 0) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: Brackets not matching");
		return 0;
	}

	/*parse locally*/
	for (i = 0 ; i < nuppers ; i++) {
		if (islower(upper_locs[i][1]) && !islower(upper_locs[i][2])) {
			/*second letter is lowercase and third one isn't -> valid */
			tempElement = xrl_strndup(upper_locs[i],2);
			/*get corresponding atomic number */
			res = bsearch(tempElement, MendelArraySorted, MENDEL_MAX, sizeof(struct MendelElement), matchMendelElement);
			if (res == NULL) {
				xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: unknown symbol %s detected", tempElement);
				return 0;
			}
			/*determine element subscript */
			j = 2;
			ndots = 0;
			while (isdigit(upper_locs[i][j]) || upper_locs[i][j] == '.') {
				j++;
				if (upper_locs[i][j] == '.')
					ndots++;
			}
			if (ndots > 1) {
				xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: only one dot allowed in subscripts of the chemical formula");
				return 0;
			}
			if (j == 2) {
				tempnAtoms = 1.0;
			}
			else {
				tempSubstring = xrl_strndup(upper_locs[i] + 2, j - 2);
				tempnAtoms =  strtod(tempSubstring, &endPtr);
				if (endPtr != tempSubstring+strlen(tempSubstring)) {
					xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: could not convert subscript %s to a real number", tempSubstring);
					return 0;
				}

				/*zero subscript is not allowed */
				if (tempnAtoms == 0.0) {
					xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: zero subscript detected");
					return 0;
				}
				free(tempSubstring);
			}
			free(tempElement);
		}
		else if (!islower(upper_locs[i][1])) {
			/*second letter is not lowercase -> valid */
			tempElement = xrl_strndup(upper_locs[i], 1);
			/*get corresponding atomic number */
			res = bsearch(tempElement, MendelArraySorted, MENDEL_MAX, sizeof(struct MendelElement), matchMendelElement);
			if (res == NULL) {
				xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: unknown symbol %s detected", tempElement);
				return 0;
			}
			/*determine element subscript */
			j = 1;
			ndots = 0;
			while (isdigit(upper_locs[i][j]) || upper_locs[i][j] == '.') {
				j++;
				if (upper_locs[i][j] == '.')
					ndots++;
			}
			if (ndots > 1) {
				xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: only one dot allowed in subscripts of the chemical formula");
				return 0;
			}
			if (j == 1) {
				tempnAtoms = 1.0;
			}
			else {
				tempSubstring = xrl_strndup(upper_locs[i] + 1, j - 1);
				tempnAtoms =  strtod(tempSubstring, &endPtr);
				if (endPtr != tempSubstring + strlen(tempSubstring)) {
					xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: could not convert subscript %s to a real number", tempSubstring);
					return 0;
				}

				/*zero subscript is not allowed */
				if (tempnAtoms == 0.0) {
					xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: zero subscript detected");
					return 0;
				}
				free(tempSubstring);
			}
			free(tempElement);
		}
		else {
			xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula");
			return 0;
		}
		/*atomic number identification ok -> add it to the array if necessary */
		if (ca->nElements == 0) {
			/*array is empty */
			ca->singleElements = malloc(sizeof(struct compoundAtom));
			ca->singleElements[0].Element = res->Zatom;
			ca->singleElements[0].nAtoms = tempnAtoms;
			ca->nElements++;
		}
		else {
			/*array is not empty */
			/*check if current element is already present in the array */
			key2.Element = res->Zatom;
			res2 = bsearch(&key2, ca->singleElements, ca->nElements, sizeof(struct compoundAtom), compareCompoundAtoms);
			if (res2 == NULL) {
				/*element not in array -> add it */
				ca->singleElements = (struct compoundAtom *) realloc((struct compoundAtom *) ca->singleElements,(++ca->nElements)*sizeof(struct compoundAtom));
				ca->singleElements[ca->nElements-1].Element = res->Zatom;
				ca->singleElements[ca->nElements-1].nAtoms = tempnAtoms;
				/*sort array */
				qsort(ca->singleElements,ca->nElements,sizeof(struct compoundAtom), compareCompoundAtoms);
			}
			else {
				/*element is in array -> update it */
				res2->nAtoms += tempnAtoms;
			}
		}
	}
	if (nuppers > 0)
		free(upper_locs);

	/*handle the brackets... */
	for (i = 0 ; i < nbracket_pairs ; i++) {
		tempBracketAtoms = malloc(sizeof(struct compoundAtoms));
		tempBracketString = xrl_strndup(brackets_begin_locs[i]+1,(size_t) (brackets_end_locs[i]-brackets_begin_locs[i]-1));
		tempBracketAtoms->nElements = 0;
		tempBracketAtoms->singleElements = NULL;
		/*recursive call */
		if (CompoundParserSimple(tempBracketString, tempBracketAtoms, error) == 0) {
			return 0;
		}
		free(tempBracketString);
		/*check if the brackets pair is followed by a subscript */
		j=1;
		ndots=0;
		while (isdigit(brackets_end_locs[i][j]) || brackets_end_locs[i][j] == '.') {
			j++;
			if (brackets_end_locs[i][j] == '.')
				ndots++;
		}
		if (ndots > 1) {
			xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: only one dot allowed in subscripts of the chemical formula");
			return 0;
		}
		if (j==1) {
			tempnAtoms = 1.0;
		}
		else {
			tempSubstring = xrl_strndup(brackets_end_locs[i]+1,j-1);
			tempnAtoms =  strtod(tempSubstring,&endPtr);
			if (endPtr != tempSubstring+strlen(tempSubstring)) {
				xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: could not convert subscript %s to a real number", tempSubstring);
				return 0;
			}
			/*zero subscript is not allowed */
			if (tempnAtoms == 0.0) {
				xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical formula: could not convert subscript %s to a real number", tempSubstring);
				return 0;
			}
			free(tempSubstring);
		}

		/*add them to the array... */
		if (ca->nElements == 0) {
			/*array is empty */
			ca->nElements = tempBracketAtoms->nElements;
			ca->singleElements = tempBracketAtoms->singleElements;
			for (j = 0 ; j < ca->nElements ; j++)
				ca->singleElements[j].nAtoms *= tempnAtoms;
		}
		else {
			for (j = 0 ; j < tempBracketAtoms->nElements ; j++) {
				key2.Element = tempBracketAtoms->singleElements[j].Element;
				res2 = bsearch(&key2,ca->singleElements, ca->nElements, sizeof(struct compoundAtom), compareCompoundAtoms);
				if (res2 == NULL) {
					/*element not in array -> add it */
					ca->singleElements = realloc(ca->singleElements,(++ca->nElements)*sizeof(struct compoundAtom));
					ca->singleElements[ca->nElements-1].Element = key2.Element;
					ca->singleElements[ca->nElements-1].nAtoms = tempBracketAtoms->singleElements[j].nAtoms*tempnAtoms;
					/*sort array */
					qsort(ca->singleElements,ca->nElements,sizeof(struct compoundAtom), compareCompoundAtoms);
				}
				else {
					/*element is in array -> update it */
					res2->nAtoms +=tempBracketAtoms->singleElements[j].nAtoms*tempnAtoms;
				}
			}
			free(tempBracketAtoms->singleElements);
			free(tempBracketAtoms);
		}
	}
	if (nbracket_pairs > 0) {
		free(brackets_begin_locs);
		free(brackets_end_locs);
	}



	return 1;
}




struct compoundData* CompoundParser(const char compoundString[], xrl_error **error) {
	struct compoundAtoms ca = {0.0, NULL};
	int rvCPS,i;
	double sum = 0.0;

	char *compoundStringCopy;
	char *backup_locale;

	if (compoundString == NULL) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Compound cannot be NULL");
		return NULL;
	}

	/* the locale is changed to default locale because we'll be using strtod later on */
	backup_locale = setlocale(LC_NUMERIC, "C");

	compoundStringCopy = xrl_strdup(compoundString);

	rvCPS = CompoundParserSimple(compoundStringCopy, &ca, error);

	setlocale(LC_NUMERIC, backup_locale);

	if (rvCPS) {
		struct compoundData *cd = malloc(sizeof(struct compoundData));
		cd->nElements = ca.nElements;
		cd->nAtomsAll = 0.0;
		cd->Elements = malloc(sizeof(int) * ca.nElements);
		cd->massFractions = malloc(sizeof(double) * ca.nElements);
		cd->nAtoms = malloc(sizeof(double) * ca.nElements);
		for (i = 0 ; i < ca.nElements ; i++) {
			sum += AtomicWeight(ca.singleElements[i].Element, NULL) * ca.singleElements[i].nAtoms;
			cd->nAtomsAll += ca.singleElements[i].nAtoms;
		}
		for (i = 0 ; i < ca.nElements ; i++) {
			cd->Elements[i] = ca.singleElements[i].Element;
			cd->massFractions[i] = AtomicWeight(ca.singleElements[i].Element, NULL) * ca.singleElements[i].nAtoms / sum;
			cd->nAtoms[i] = ca.singleElements[i].nAtoms;
		}
		cd->molarMass = sum;
		free(ca.singleElements);
		free(compoundStringCopy);

		return cd;
	}
	else {
		if (ca.singleElements)
			free(ca.singleElements);
		free(compoundStringCopy);
		return NULL;
	}
}

void FreeCompoundData(struct compoundData *cd) {
	free(cd->Elements);
	free(cd->massFractions);
	free(cd->nAtoms);
	free(cd);
}


struct compoundData * add_compound_data(struct compoundData A, double weightA, struct compoundData B, double weightB) {
	struct compoundData *rv, *longest, *shortest;
	int i,j,found=0;
	double *longestW, *shortestW;


	rv = malloc(sizeof(struct compoundData)) ;

	if (A.nElements >= B.nElements) {
		longest = &A;
		shortest = &B;
		longestW = &weightA;
		shortestW = &weightB;
	}
	else {
		longest = &B;
		shortest = &A;
		longestW = &weightB;
		shortestW = &weightA;
	}

	rv->Elements = malloc(sizeof(int) * longest->nElements);
	memcpy(rv->Elements,longest->Elements, sizeof(int)*longest->nElements);
	rv->nElements = longest->nElements;

	/*determine the unique Elements from A and B */
	for (i = 0 ; i < shortest->nElements ; i++) {
		found = 0;
		for (j = 0 ; j < longest->nElements ; j++) {
			if (shortest->Elements[i] == longest->Elements[j]) {
				found = 1;
				break;
			}
		}
		if (!found) {
			/*add to array */
			rv->Elements = realloc(rv->Elements, sizeof(int) * ++(rv->nElements));
			rv->Elements[rv->nElements-1] = shortest->Elements[i];
		}
	}

	/*sort array */
	qsort(rv->Elements, rv->nElements, sizeof(int),compareInt );

	/* the following lines are highly questionable... */
	rv->nAtomsAll = longest->nAtomsAll + shortest->nAtomsAll;
	rv->molarMass = longest->molarMass + shortest->molarMass;
	rv->nAtoms = (double *) calloc(rv->nElements,sizeof(double));

	rv->massFractions = (double *) calloc(rv->nElements,sizeof(double));

	for (i = 0 ; i < rv->nElements ; i++) {
		for (j = 0 ; j < longest->nElements ; j++) {
			if (rv->Elements[i] == longest->Elements[j])
				rv->massFractions[i] += longest->massFractions[j]**longestW;
		}
		for (j = 0 ; j < shortest->nElements ; j++) {
			if (rv->Elements[i] == shortest->Elements[j])
				rv->massFractions[i] += shortest->massFractions[j]**shortestW;
		}
	}

	return rv;
}

static int compareInt(const void *A, const void *B) {

	return (int)*((int*)A) - (int)*((int*)B);
}


char *AtomicNumberToSymbol(int Z, xrl_error **error) {
	if (Z < 1 || Z > MENDEL_MAX ) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, Z_OUT_OF_RANGE);
		return NULL;
	}

	return xrl_strdup(MendelArray[Z-1].name);
}

int SymbolToAtomicNumber(const char *symbol, xrl_error **error) {
	int i;

	if (symbol == NULL) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Symbol cannot be NULL");
		return 0;
	}

	for (i=0 ; i < MENDEL_MAX ; i++) {
		if (strcmp(symbol,MendelArray[i].name) == 0)
			return MendelArray[i].Zatom;
	}

	xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid chemical symbol");
	return 0;
}



void xrlFree(void *Ptr) {
	/*just a wrapper around free really... because we don't trust msvcrtXX.dll */
	free(Ptr);
}
