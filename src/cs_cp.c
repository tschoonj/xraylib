/*
Copyright (c) 2009, 2010, 2013, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "xraylib.h"
#include "xrayvars.h"
#include <stdlib.h>

double CS_Total_CP(const char compound[], double E) {
	struct compoundData *cd;
	struct compoundDataNIST *cdn;
	int i;
	double rv = 0.0;

	if ((cd = CompoundParser(compound)) == NULL) {
		ErrorExit("CS_Total_CP: CompoundParser failed. Trying NIST compound database.");
		if ((cdn = GetCompoundDataNISTByName(compound)) == NULL) {
			ErrorExit("CS_Total_CP: Compound not found in NIST compound database.");
			return rv;
		}
		else {
			for (i = 0 ; i < cdn->nElements ; i++)
				rv += CS_Total(cdn->Elements[i], E)*cdn->massFractions[i];

			FreeCompoundDataNIST(cdn);
		}
	}
	else {
		for (i = 0 ; i < cd->nElements ; i++)
			rv += CS_Total(cd->Elements[i], E)*cd->massFractions[i];

		FreeCompoundData(cd);
	}

	return rv;
}

#define CS_CP_F(function) \
	double function ## _CP(const char compound[], double E) {\
		struct compoundData *cd;\
		struct compoundDataNIST *cdn;\
		int i;\
		double rv = 0.0;\
		\
		if ((cd = CompoundParser(compound)) == NULL) { \
			ErrorExit(#function ": CompoundParser failed. Trying NIST compound database."); \
			if ((cdn = GetCompoundDataNISTByName(compound)) == NULL) { \
				ErrorExit(#function ": Compound not found in NIST compound database."); \
				return rv; \
			} \
			else {\
				for (i = 0 ; i < cdn->nElements ; i++) \
					rv += function(cdn->Elements[i], E)*cdn->massFractions[i]; \
				\
				FreeCompoundDataNIST(cdn);\
			}\
		} \
		else {\
			for (i = 0 ; i < cd->nElements ; i++) \
				rv += function(cd->Elements[i], E)*cd->massFractions[i]; \
			\
			FreeCompoundData(cd);\
		}\
		\
		return rv;\
	}


#define CS_CP_FF(function) \
	double function ## _CP(const char compound[], double E, double theta) {\
		struct compoundData *cd;\
		struct compoundDataNIST *cdn;\
		int i;\
		double rv = 0.0;\
		\
		if ((cd = CompoundParser(compound)) == NULL) { \
			ErrorExit(#function ": CompoundParser failed. Trying NIST compound database."); \
			if ((cdn = GetCompoundDataNISTByName(compound)) == NULL) { \
				ErrorExit(#function ": Compound not found in NIST compound database."); \
				return rv; \
			} \
			else {\
				for (i = 0 ; i < cdn->nElements ; i++) \
					rv += function(cdn->Elements[i], E, theta)*cdn->massFractions[i]; \
				\
				FreeCompoundDataNIST(cdn);\
			}\
		} \
		else {\
			for (i = 0 ; i < cd->nElements ; i++) \
				rv += function(cd->Elements[i], E, theta)*cd->massFractions[i]; \
			\
			FreeCompoundData(cd);\
		}\
		\
		return rv;\
	}


#define CS_CP_FFF(function) \
	double function ## _CP(const char compound[], double E, double theta, double phi) {\
		struct compoundData *cd;\
		struct compoundDataNIST *cdn;\
		int i;\
		double rv = 0.0;\
		\
		if ((cd = CompoundParser(compound)) == NULL) { \
			ErrorExit(#function ": CompoundParser failed. Trying NIST compound database."); \
			if ((cdn = GetCompoundDataNISTByName(compound)) == NULL) { \
				ErrorExit(#function ": Compound not found in NIST compound database."); \
				return rv; \
			} \
			else {\
				for (i = 0 ; i < cdn->nElements ; i++) \
					rv += function(cdn->Elements[i], E, theta, phi)*cdn->massFractions[i]; \
				\
				FreeCompoundDataNIST(cdn);\
			}\
		} \
		else {\
			for (i = 0 ; i < cd->nElements ; i++) \
				rv += function(cd->Elements[i], E, theta, phi)*cd->massFractions[i]; \
			\
			FreeCompoundData(cd);\
		}\
		\
		return rv;\
	}


#define CS_CP_IF(function) \
	double function ## _CP(const char compound[], int line, double E) {\
		struct compoundData *cd;\
		struct compoundDataNIST *cdn;\
		int i;\
		double rv = 0.0;\
		\
		if ((cd = CompoundParser(compound)) == NULL) { \
			ErrorExit(#function ": CompoundParser failed. Trying NIST compound database."); \
			if ((cdn = GetCompoundDataNISTByName(compound)) == NULL) { \
				ErrorExit(#function ": Compound not found in NIST compound database."); \
				return rv; \
			} \
			else {\
				for (i = 0 ; i < cdn->nElements ; i++) \
					rv += function(cdn->Elements[i], line, E)*cdn->massFractions[i]; \
				\
				FreeCompoundDataNIST(cdn);\
			}\
		} \
		else {\
			for (i = 0 ; i < cd->nElements ; i++) \
				rv += function(cd->Elements[i], line, E)*cd->massFractions[i]; \
			\
			FreeCompoundData(cd);\
		}\
		\
		return rv;\
	}

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
