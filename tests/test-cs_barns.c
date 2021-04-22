/* Copyright (c) 2018, Tom Schoonjans
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
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

int main(int argc, char **argv) {
	xrl_error *error = NULL;
	double cs, aw;

#define IE(name, Z, E) \
	cs = CS_ ## name(Z, E, &error); \
	assert(error == NULL); \
	assert(cs > 0.0); \
	aw = AtomicWeight(Z, &error); \
	assert(aw > 0.0); \
	assert(error == NULL); \
	assert(CSb_ ## name(Z, E, &error) == cs * aw / AVOGNUM);

#define IE_BAD(name) \
	cs = CSb_ ## name(-1, 10.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);\
	xrl_clear_error(&error);\
	cs = CSb_ ## name(ZMAX, 10.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);\
	xrl_clear_error(&error);\
	cs = CSb_ ## name(26, 0.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);\
	xrl_clear_error(&error);

#define IIE(name, Z, line, E) \
	cs = CS_ ## name(Z, line, E, &error); \
	assert(error == NULL); \
	assert(cs > 0.0); \
	aw = AtomicWeight(Z, &error); \
	assert(aw > 0.0); \
	assert(error == NULL); \
	assert(CSb_ ## name(Z, line, E, &error) == cs * aw / AVOGNUM);

#define IIE_BAD(name) \
	cs = CSb_ ## name(-1, KL3_LINE, 10.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);\
	xrl_clear_error(&error);\
	cs = CSb_ ## name(ZMAX, KL3_LINE, 10.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, INVALID_LINE) == 0);\
	xrl_clear_error(&error);\
	cs = CSb_ ## name(26, -500, 10.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, INVALID_LINE) == 0);\
	xrl_clear_error(&error);\
	cs = CSb_ ## name(26, KL3_LINE, 0.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);\
	xrl_clear_error(&error);

#define IEE(name, Z, E, theta) \
	cs = DCS_ ## name(Z, E, theta, &error); \
	assert(error == NULL); \
	assert(cs > 0.0); \
	aw = AtomicWeight(Z, &error); \
	assert(aw > 0.0); \
	assert(error == NULL); \
	assert(DCSb_ ## name(Z, E, theta, &error) == cs * aw / AVOGNUM);

#define IEE_BAD(name) \
	cs = DCSb_ ## name(-1, 10.0, M_PI/4.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);\
	xrl_clear_error(&error);\
	cs = DCSb_ ## name(ZMAX, 10.0, M_PI/4.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);\
	xrl_clear_error(&error);\
	cs = DCSb_ ## name(26, 0.0, M_PI/4.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);\
	xrl_clear_error(&error);

#define IEEE(name, Z, E, theta, phi) \
	cs = DCSP_ ## name(Z, E, theta, phi, &error); \
	assert(error == NULL); \
	assert(cs > 0.0); \
	aw = AtomicWeight(Z, &error); \
	assert(aw > 0.0); \
	assert(error == NULL); \
	assert(DCSPb_ ## name(Z, E, theta, phi, &error) == cs * aw / AVOGNUM);

#define IEEE_BAD(name) \
	cs = DCSPb_ ## name(-1, 10.0, M_PI/4.0, M_PI/4.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);\
	xrl_clear_error(&error);\
	cs = DCSPb_ ## name(ZMAX, 10.0, M_PI/4.0, M_PI/4.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);\
	xrl_clear_error(&error);\
	cs = DCSPb_ ## name(26, 0.0, M_PI/4.0, M_PI/4.0, &error);\
	assert(cs == 0.0);\
	assert(error != NULL);\
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);\
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);\
	xrl_clear_error(&error);



	IE(Total, 26, 10.0)
	IE(Photo, 26, 10.0)
	IE(Rayl, 26, 10.0)
	IE(Compt, 26, 10.0)
	IIE(FluorLine, 26, KL3_LINE, 10.0)
	IIE(FluorShell, 26, K_SHELL, 10.0)
	IEE(Rayl, 26, 10.0, M_PI/4.0)
	IEE(Compt, 26, 10.0, M_PI/4.0)
	IEEE(Rayl, 26, 10.0, M_PI/4.0, M_PI/4.0)
	IEEE(Compt, 26, 10.0, M_PI/4.0, M_PI/4.0)

	IE_BAD(Total)
	IE_BAD(Photo)
	IE_BAD(Rayl)
	IE_BAD(Compt)
	IIE_BAD(FluorLine)
	IEE_BAD(Rayl)
	IEE_BAD(Compt)
	IEEE_BAD(Rayl)
	IEEE_BAD(Compt)

	return 0;
}
