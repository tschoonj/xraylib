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

int main(int argc, char **argv) {
	double width;

	width = xrlpp::AtomicLevelWidth(26, K_SHELL);
	assert(fabs(width - 1.19E-3) < 1E-6);
	
	width = xrlpp::AtomicLevelWidth(92, N7_SHELL);
	assert(fabs(width - 0.31E-3) < 1E-8);
	
	try {
		width = xrlpp::AtomicLevelWidth(185, K_SHELL);
		abort();
	}
	catch (std::invalid_argument &e) {
		assert(strcmp(e.what(), Z_OUT_OF_RANGE) == 0);
	}
	
	try {
		width = xrlpp::AtomicLevelWidth(26, -5);
		abort();
	}
	catch (std::invalid_argument &e) {
		assert(strcmp(e.what(), UNKNOWN_SHELL) == 0);
	}
	
	try {
		width = xrlpp::AtomicLevelWidth(26, N3_SHELL);
		abort();
	}
	catch (std::invalid_argument &e) {
		assert(strcmp(e.what(), INVALID_SHELL) == 0);
	}
	
	return 0;	
}
