/* Copyright (C) 2018 Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans 'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "xraylib.h"
#include "xraylib-error-private.h"
#ifdef NDEBUG
  #undef NDEBUG
#endif
#include <assert.h>
#include <string.h>

static void test_literal(void) {
	xrl_error *error = NULL;

	xrl_set_error_literal(&error, XRL_ERROR_MEMORY, "%s %d %x");

	assert(xrl_error_matches(error, XRL_ERROR_MEMORY));
	assert(strcmp(error->message, "%s %d %x") == 0);
	xrl_error_free(error);
}

static void test_copy(void) {
	xrl_error *error = NULL, *copy = NULL;

	xrl_set_error_literal(&error, XRL_ERROR_MEMORY, "%s %d %x");
	copy = xrl_error_copy(error);

	assert(xrl_error_matches(copy, XRL_ERROR_MEMORY));
	assert(strcmp(copy->message, "%s %d %x") == 0);
	xrl_error_free(error);
	xrl_error_free(copy);
}

int main(int argc, char *argv[]) {

	test_literal();
	test_copy();

	return 0;
}
