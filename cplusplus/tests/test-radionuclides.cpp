/* Copyright (c) 2018, Tom Schoonjans
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

	std::vector<std::string> nuclides = xrlpp::GetRadioNuclideDataList();
	assert(nuclides.size() == 10);
	
	for (int i = 0 ; i < nuclides.size() ; i++) {
		auto nuclide = nuclides[i];
		auto rnd1 = xrlpp::GetRadioNuclideDataByName(nuclide);
		assert(rnd1.name == nuclide);
		auto rnd2 = xrlpp::GetRadioNuclideDataByIndex(i);
		assert(rnd2.name == nuclide);
	}

	auto rnd = xrlpp::GetRadioNuclideDataByIndex(3);
	assert(rnd.name == "125I");
	assert(rnd.A == 125);
	assert(rnd.N == 72);
	assert(rnd.Z == 53);
	assert(rnd.Z_xray == 52);
	assert(rnd.nGammas== 1);
	assert(fabs(rnd.GammaEnergies[0] - 35.4919) < 1E-4);
	assert(fabs(rnd.GammaIntensities[0] - 0.0668) < 1E-4);
	assert(rnd.nXrays== 20);
	assert(rnd.XrayLines[0] == -86);
	assert(fabs(rnd.XrayIntensities[0] - 0.0023) < 1E-4);

	try {
		xrlpp::GetRadioNuclideDataByIndex(-1);
		abort();
	}
	catch (std::invalid_argument &e) {

	}

	try {
		xrlpp::GetRadioNuclideDataByIndex(10);
		abort();
	}
	catch (std::invalid_argument &e) {

	}

	try {
		xrlpp::GetRadioNuclideDataByName("");
		abort();
	}
	catch (std::invalid_argument &e) {

	}

	try {
		xrlpp::GetRadioNuclideDataByName("jwefhjoehoeoehr");
		abort();
	}
	catch (std::invalid_argument &e) {

	}

	return 0;
}

