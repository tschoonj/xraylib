/*
Copyright (c) 2009-2018, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "config.h"
#include "xraylib.h"
#include "xrayvars.h"
#include <stdlib.h>
#include "xraylib-error-private.h"

#define CS_CP_BEGIN \
		struct compoundData *cd = NULL; \
		struct compoundDataNIST *cdn = NULL; \
		int i;\
		double rv = 0.0;\
		int nElements = 0; \
		int *Elements = NULL;\
		double *massFractions = NULL;\
		\
		if ((cd = CompoundParser(compound, NULL)) != NULL) { \
			nElements = cd->nElements; \
			Elements = cd->Elements; \
			massFractions = cd->massFractions; \
		} \
		else if ((cdn = GetCompoundDataNISTByName(compound, NULL)) != NULL) { \
			nElements = cdn->nElements; \
			Elements = cdn->Elements; \
			massFractions = cdn->massFractions; \
		} \
		else { \
			xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNKNOWN_COMPOUND); \
			return 0.0; \
		} \
		\
		for (i = 0 ; i < nElements ; i++) { \
			double tmp = 0.0;

#define CS_CP_END \
			if (tmp == 0.0) { \
				rv = 0.0; \
				break; \
			} \
			rv += tmp; \
		} \
		\
		if (cd) \
			FreeCompoundData(cd);\
		else if (cdn) \
			FreeCompoundDataNIST(cdn);\
		\
		return rv;

#define CS_CP_F(function) \
	double function ## _CP(const char compound[], double E, xrl_error **error) {\
		CS_CP_BEGIN \
			tmp = function(Elements[i], E, error) * massFractions[i]; \
		CS_CP_END \
	}


#define CS_CP_FF(function) \
	double function ## _CP(const char compound[], double E, double theta, xrl_error **error) {\
		CS_CP_BEGIN \
			tmp = function(Elements[i], E, theta, error) * massFractions[i]; \
		CS_CP_END \
	}


#define CS_CP_FFF(function) \
	double function ## _CP(const char compound[], double E, double theta, double phi, xrl_error **error) {\
		CS_CP_BEGIN \
			tmp = function(Elements[i], E, theta, phi, error) * massFractions[i]; \
		CS_CP_END \
	}

CS_CP_F(CS_Total)
CS_CP_F(CS_Photo)
CS_CP_F(CS_Rayl)
CS_CP_F(CS_Compt)
CS_CP_F(CSb_Total)
CS_CP_F(CSb_Photo)
CS_CP_F(CSb_Rayl)
CS_CP_F(CSb_Compt)
CS_CP_F(CS_Energy)
CS_CP_FF(DCS_Rayl)
CS_CP_FF(DCS_Compt)
CS_CP_FF(DCSb_Rayl)
CS_CP_FF(DCSb_Compt)
CS_CP_FFF(DCSP_Rayl)
CS_CP_FFF(DCSP_Compt)
CS_CP_FFF(DCSPb_Rayl)
CS_CP_FFF(DCSPb_Compt)
CS_CP_F(CS_Photo_Total)
CS_CP_F(CSb_Photo_Total)
CS_CP_F(CS_Total_Kissel)
CS_CP_F(CSb_Total_Kissel)
