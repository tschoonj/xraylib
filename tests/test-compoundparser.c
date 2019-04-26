/* Copyright (c) 2017, Tom Schoonjans
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
#include <math.h>

int main(int argc, char *argv[]) {
	xrl_error *error = NULL;
	char *symbol = NULL;
	struct compoundData *cd = NULL;
	int Z;

	/* taken from https://github.com/KenanY/chemical-formula/blob/master/test/index.js */
	/* good formulas */
	assert(CompoundParser("C19H29COOH", NULL) != NULL);
	assert(CompoundParser("C12H10", NULL) != NULL);
	assert(CompoundParser("C12H6O2", NULL) != NULL);
	assert(CompoundParser("C6H5Br", NULL) != NULL);
	assert(CompoundParser("C3H4OH(COOH)3", NULL) != NULL);
	assert(CompoundParser("HOCH2CH2OH", NULL) != NULL);
	assert(CompoundParser("C5H11NO2", NULL) != NULL);
	assert(CompoundParser("CH3CH(CH3)CH3", NULL) != NULL);
	assert(CompoundParser("NH2CH(C4H5N2)COOH", NULL) != NULL);
	assert(CompoundParser("H2O", NULL) != NULL);
	assert(CompoundParser("Ca5(PO4)3F", NULL) != NULL);
	assert(CompoundParser("Ca5(PO4)3OH", NULL) != NULL);
	assert(CompoundParser("Ca5.522(PO4.48)3OH", NULL) != NULL);
	assert(CompoundParser("Ca5.522(PO.448)3OH", NULL) != NULL);

	/* bad formulas */
	assert(CompoundParser("CuI2ww", NULL) == NULL);
	assert(CompoundParser("0C", NULL) == NULL);
	assert(CompoundParser("2O", NULL) == NULL);
	assert(CompoundParser("13Li", NULL) == NULL);
	assert(CompoundParser("2(NO3)", NULL) == NULL);
	assert(CompoundParser("H(2)", NULL) == NULL);
	assert(CompoundParser("Ba(12)", NULL) == NULL);
	assert(CompoundParser("Cr(5)3", NULL) == NULL);
	assert(CompoundParser("Pb(13)2", NULL) == NULL);
	assert(CompoundParser("Au(22)11", NULL) == NULL);
	assert(CompoundParser("Au11(H3PO4)2)", NULL) == NULL);
	assert(CompoundParser("Au11(H3PO4))2", NULL) == NULL);
	assert(CompoundParser("Au(11(H3PO4))2", NULL) == NULL);
	assert(CompoundParser("Ca5.522(PO.44.8)3OH", NULL) == NULL);
	assert(CompoundParser("Ba[12]", NULL) == NULL);
	assert(CompoundParser("Auu1", NULL) == NULL);
	assert(CompoundParser("AuL1", NULL) == NULL);
	assert(CompoundParser(NULL, NULL) == NULL);
	assert(CompoundParser("  ", NULL) == NULL);
	assert(CompoundParser("\t", NULL) == NULL);
	assert(CompoundParser("\n", NULL) == NULL);
	assert(CompoundParser("Au L1", NULL) == NULL);
	assert(CompoundParser("Au\tFe", NULL) == NULL);

	cd = CompoundParser("H2SO4", NULL);
	assert(cd != NULL);
	assert(cd->nElements == 3);
	assert(fabs(cd->molarMass - 98.09) < 1E-6);
	assert(fabs(cd->nAtomsAll- 7.0) < 1E-6);
	assert(cd->Elements[0] == 1);
	assert(cd->Elements[1] == 8);
	assert(cd->Elements[2] == 16);
	assert(fabs(cd->massFractions[0] - 0.02059333265368539) < 1E-6);
	assert(fabs(cd->massFractions[1] - 0.6524620246712203) < 1E-6);
	assert(fabs(cd->massFractions[2] - 0.32694464267509427) < 1E-6);
	assert(fabs(cd->nAtoms[0] - 2.0) < 1E-6);
	assert(fabs(cd->nAtoms[1] - 4.0) < 1E-6);
	assert(fabs(cd->nAtoms[2] - 1.0) < 1E-6);
	FreeCompoundData(cd);

	assert(SymbolToAtomicNumber("Fe", &error) == 26);
	assert(error == NULL);

	assert(SymbolToAtomicNumber("Uu", &error) == 0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, "Invalid chemical symbol") == 0);
	xrl_clear_error(&error);

	assert(SymbolToAtomicNumber(NULL, &error) == 0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, "Symbol cannot be NULL") == 0);
	xrl_clear_error(&error);

	symbol = AtomicNumberToSymbol(26, &error);
	assert(strcmp(symbol, "Fe") == 0);
	assert(error == NULL);
	xrlFree(symbol);

	symbol = AtomicNumberToSymbol(-2, &error);
	assert(symbol == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);
	
	/* database currently goes up to Bh */
	symbol = AtomicNumberToSymbol(108, &error);
	assert(symbol == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	/* cross validation */
	for (Z = 1 ; Z <= 107 ; Z++) {
		symbol = AtomicNumberToSymbol(Z, NULL);
		assert(symbol != NULL);
		assert(SymbolToAtomicNumber(symbol, NULL) == Z);
		xrlFree(symbol);
	}

	return 0;
}
