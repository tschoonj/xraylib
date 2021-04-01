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

int main(int argc, char **argv) {
	xrl_error *error = NULL;
	double coskron, sum;

	/* good values */
	coskron = CosKronTransProb(92, FL13_TRANS, &error);
	assert(error == NULL);
	assert(fabs(coskron - 0.620) < 1E-6);

	coskron = CosKronTransProb(75, FL12_TRANS, &error);
	assert(error == NULL);
	assert(fabs(coskron - 1.03E-1) < 1E-6);

	coskron = CosKronTransProb(51, FL23_TRANS, &error);
	assert(error == NULL);
	assert(fabs(coskron - 1.24E-1) < 1E-6);

	coskron = CosKronTransProb(86, FM45_TRANS, &error);
	assert(error == NULL);
	assert(fabs(coskron - 6E-2) < 1E-6);


	/* bad values */
	/* no data for Z < 12 */
	coskron = CosKronTransProb(11, FL12_TRANS, &error);
	assert(error != NULL);
	assert(coskron == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT );
	assert(strcmp(error->message, INVALID_CK) == 0);
	xrl_clear_error(&error);

	/* apparently we have data for Z = 109... */
	coskron = CosKronTransProb(109, FM45_TRANS, &error);
	assert(error == NULL);
	assert(fabs(coskron - 1.02E-1) < 1E-6);

	/* ... but not for Z = 110+ */
	coskron = CosKronTransProb(110, FL12_TRANS, &error);
	assert(error != NULL);
	assert(coskron == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT );
	assert(strcmp(error->message, INVALID_CK) == 0);
	xrl_clear_error(&error);

	/* let's try going out of range with Z */
	coskron = CosKronTransProb(0, FL12_TRANS, &error);
	assert(error != NULL);
	assert(coskron == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT );
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	coskron = CosKronTransProb(ZMAX + 1, FL12_TRANS, &error);
	assert(error != NULL);
	assert(coskron == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT );
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	xrl_clear_error(&error);

	/* now some non-existent Coster-Kronig transitions */
	coskron = CosKronTransProb(26, 0, &error);
	assert(error != NULL);
	assert(coskron == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT );
	assert(strcmp(error->message, UNKNOWN_CK) == 0);
	xrl_clear_error(&error);

	coskron = CosKronTransProb(92, FM45_TRANS, &error);
	assert(error == NULL);
	assert(coskron > 0.0);
	coskron = CosKronTransProb(92, FM45_TRANS + 1, &error);
	assert(error != NULL);
	assert(coskron == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT );
	assert(strcmp(error->message, UNKNOWN_CK) == 0);
	xrl_clear_error(&error);
	
	/* now some internal consistency checks */
	sum = FluorYield(92, L1_SHELL, NULL) + AugerYield(92, L1_SHELL, NULL) + CosKronTransProb(92, FL12_TRANS, NULL) + CosKronTransProb(92, FL13_TRANS, NULL) + CosKronTransProb(92, FLP13_TRANS, NULL);
	assert(fabs(sum - 1.0) < 1E-6);
	sum = FluorYield(92, L2_SHELL, NULL) + AugerYield(92, L2_SHELL, NULL) + CosKronTransProb(92, FL23_TRANS, NULL);
	assert(fabs(sum - 1.0) < 1E-6);
	sum = FluorYield(92, M4_SHELL, NULL) + AugerYield(92, M4_SHELL, NULL) + CosKronTransProb(92, FM45_TRANS, NULL);
	assert(fabs(sum - 1.0) < 1E-6);
	sum = FluorYield(92, M2_SHELL, NULL) + AugerYield(92, M2_SHELL, NULL) + CosKronTransProb(92, FM23_TRANS, NULL) + CosKronTransProb(92, FM24_TRANS, NULL) + CosKronTransProb(92, FM25_TRANS, NULL);
	assert(fabs(sum - 1.0) < 1E-6);
	

	return 0;
}
