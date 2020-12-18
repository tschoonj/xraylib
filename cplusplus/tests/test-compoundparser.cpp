/* Copyright (c) 2017, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifdef NDEBUG
  #undef NDEBUG
#endif
#include "xraylib++.h"
#include "xraylib-error-private.h"
#include <cmath>
#include <cassert>
#include <cstring>

int main(int argc, char *argv[]) {

	std::vector<std::string> good_compounds = {
        "C19H29COOH",
        "C12H10",
        "C12H6O2",
        "C6H5Br",
        "C3H4OH(COOH)3",
        "HOCH2CH2OH",
        "C5H11NO2",
        "CH3CH(CH3)CH3",
        "NH2CH(C4H5N2)COOH",
        "H2O",
        "Ca5(PO4)3F",
        "Ca5(PO4)3OH",
        "Ca5.522(PO4.48)3OH",
        "Ca5.522(PO.448)3OH",
	};

	std::vector<std::string> bad_compounds = {
       "CuI2ww",
       "0C",
       "2O",
       "13Li",
       "2(NO3)",
       "H(2)",
       "Ba(12)",
       "Cr(5)3",
       "Pb(13)2",
       "Au(22)11",
       "Au11(H3PO4)2)",
       "Au11(H3PO4))2",
       "Au(11(H3PO4))2",
       "Ca5.522(PO.44.8)3OH",
       "Ba[12]",
       "Auu1",
       "AuL1",
       "  ",
       "\t",
       "\n",
       "Au L1",
       "Au\tFe",
	};

	for (auto compound : good_compounds) {
		xrlpp::CompoundParser(compound);
	}

	for (auto compound : bad_compounds) {
		try {
			xrlpp::CompoundParser(compound);
			abort();
		}
		catch (std::invalid_argument &e) {
			continue;
		}
	}

	xrlpp::compoundData cd = xrlpp::CompoundParser("H2SO4");
	assert(cd.nElements == 3);
	assert(fabs(cd.molarMass - 98.09) < 1E-6);
	assert(fabs(cd.nAtomsAll- 7.0) < 1E-6);
	assert(cd.Elements[0] == 1);
	assert(cd.Elements[1] == 8);
	assert(cd.Elements[2] == 16);
	assert(fabs(cd.massFractions[0] - 0.02059333265368539) < 1E-6);
	assert(fabs(cd.massFractions[1] - 0.6524620246712203) < 1E-6);
	assert(fabs(cd.massFractions[2] - 0.32694464267509427) < 1E-6);
	assert(fabs(cd.nAtoms[0] - 2.0) < 1E-6);
	assert(fabs(cd.nAtoms[1] - 4.0) < 1E-6);
	assert(fabs(cd.nAtoms[2] - 1.0) < 1E-6);

	assert(xrlpp::SymbolToAtomicNumber("Fe") == 26);

	try {
		xrlpp::SymbolToAtomicNumber("Uu");
		abort();
	}
	catch (std::invalid_argument &e) {
		assert(strcmp(e.what(), "Invalid chemical symbol") == 0);
	}

	assert(xrlpp::AtomicNumberToSymbol(26) == "Fe");

	try {
		xrlpp::AtomicNumberToSymbol(-2);
		abort();
	}
	catch (std::invalid_argument &e) {
		assert(strcmp(e.what(), Z_OUT_OF_RANGE) == 0);
	}

	/* database currently goes up to Bh */
	try {
		xrlpp::AtomicNumberToSymbol(108);
		abort();
	}
	catch (std::invalid_argument &e) {
		assert(strcmp(e.what(), Z_OUT_OF_RANGE) == 0);
	}

	/* cross validation */
	for (int Z = 1 ; Z <= 107 ; Z++) {
		std::string symbol = xrlpp::AtomicNumberToSymbol(Z);
		assert(xrlpp::SymbolToAtomicNumber(symbol) == Z);
	}

	return 0;
}
