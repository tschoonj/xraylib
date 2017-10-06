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
#include <assert.h>
#include <stddef.h>

int main(int argc, char *argv[]) {
	// taken from https://github.com/KenanY/chemical-formula/blob/master/test/index.js
	// good formulas
	assert(CompoundParser("C19H29COOH") != NULL);
	assert(CompoundParser("C12H10") != NULL);
	assert(CompoundParser("C12H6O2") != NULL);
	assert(CompoundParser("C6H5Br") != NULL);
	assert(CompoundParser("C3H4OH(COOH)3") != NULL);
	assert(CompoundParser("HOCH2CH2OH") != NULL);
	assert(CompoundParser("C5H11NO2") != NULL);
	assert(CompoundParser("CH3CH(CH3)CH3") != NULL);
	assert(CompoundParser("NH2CH(C4H5N2)COOH") != NULL);
	assert(CompoundParser("H2O") != NULL);
	assert(CompoundParser("Ca5(PO4)3F") != NULL);
	assert(CompoundParser("Ca5(PO4)3OH") != NULL);
	assert(CompoundParser("Ca5.522(PO4.48)3OH") != NULL);
	assert(CompoundParser("Ca5.522(PO.448)3OH") != NULL);

	// bad formulas
	assert(CompoundParser("CuI2ww") == NULL);
	assert(CompoundParser("0C") == NULL);
	assert(CompoundParser("2O") == NULL);
	assert(CompoundParser("13Li") == NULL);
	assert(CompoundParser("2(NO3)") == NULL);
	assert(CompoundParser("H(2)") == NULL);
	assert(CompoundParser("Ba(12)") == NULL);
	assert(CompoundParser("Cr(5)3") == NULL);
	assert(CompoundParser("Pb(13)2") == NULL);
	assert(CompoundParser("Au(22)11") == NULL);
	assert(CompoundParser("Au11(H3PO4)2)") == NULL);
	assert(CompoundParser("Au11(H3PO4))2") == NULL);
	assert(CompoundParser("Au(11(H3PO4))2") == NULL);
	assert(CompoundParser("Ca5.522(PO.44.8)3OH") == NULL);
	assert(CompoundParser("Ba[12]") == NULL);
	assert(CompoundParser("Auu1") == NULL);
	assert(CompoundParser("AuL1") == NULL);

	return 0;
}
