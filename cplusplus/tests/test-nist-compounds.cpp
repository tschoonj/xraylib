/* Copyright (c) 2020, Tom Schoonjans
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
	std::vector<std::string> compounds = xrlpp::GetCompoundDataNISTList();
	assert(compounds.size() == 180);
	
	for (int i = 0 ; i < compounds.size() ; i++) {
		auto compound = compounds[i];
		auto cdn1 = xrlpp::GetCompoundDataNISTByName(compound);
		assert(cdn1.name == compound);
		auto cdn2 = xrlpp::GetCompoundDataNISTByIndex(i);
		assert(cdn2.name == compound);
	}

	auto cdn1 = xrlpp::GetCompoundDataNISTByIndex(5);
	assert(cdn1.nElements == 4);
	assert(cdn1.Elements == std::vector<int>({6, 7, 8, 18}));
	assert(fabs(cdn1.massFractions[0] - 0.000124) < 1E-6);
	assert(fabs(cdn1.density - 0.001205) < 1E-6);
	assert(cdn1.name == "Air, Dry (near sea level)");

	auto cdn2 = xrlpp::GetCompoundDataNISTByName("Air, Dry (near sea level)");
	assert(cdn2.nElements == 4);
	assert(cdn2.Elements == std::vector<int>({6, 7, 8, 18}));
	assert(fabs(cdn2.massFractions[0] - 0.000124) < 1E-6);
	assert(fabs(cdn2.density - 0.001205) < 1E-6);
	assert(cdn2.name == "Air, Dry (near sea level)");


	/* bad input */
	try {
		xrlpp::GetCompoundDataNISTByIndex(-1);
		abort();
	}
	catch (std::invalid_argument &e) {

	}

	try {
		xrlpp::GetCompoundDataNISTByIndex(180);
		abort();
	}
	catch (std::invalid_argument &e) {

	}

	try {
		xrlpp::GetCompoundDataNISTByName("");
		abort();
	}
	catch (std::invalid_argument &e) {

	}

	return 0;
}
