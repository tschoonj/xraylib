/*
Copyright (c) 2010-2018, Tom Schoonjans, Bruno Golosio
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans and Bruno Golosio''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "config.h"
#define KD 4.15179082788e-4
#include "xrayglob.h"
#include "xraylib.h"
#include <stdlib.h>
#include <math.h>
#include "xraylib-error-private.h"

#define REFR_BEGIN \
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
		if (density <= 0.0) { \
			density = cdn->density;\
		} \
	} \
	else { \
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNKNOWN_COMPOUND); \
		return rv; \
	} \
	if (density <= 0.0) { \
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, NEGATIVE_DENSITY); \
		return rv; \
	} \
	if (E <= 0.0) { \
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, NEGATIVE_ENERGY); \
		return rv; \
	}

#define REFR_END \
	if (cd) \
		FreeCompoundData(cd); \
	else if (cdn) \
		FreeCompoundDataNIST(cdn);

double Refractive_Index_Re(const char compound[], double E, double density, xrl_error **error) {
	struct compoundData *cd = NULL;
	struct compoundDataNIST *cdn = NULL;
	double rv = 0.0;
	int i;

	REFR_BEGIN

	/* Real part is 1-delta */
	for (i = 0 ; i < nElements ; i++) {
		double fi = 0.0;
		double atomic_weight = 0.0;
		fi = Fi(Elements[i], E, error);
		if (fi == 0.0)
			return 0.0;
		atomic_weight = AtomicWeight(Elements[i], error);
		if (atomic_weight == 0.0)
			return 0.0;
		rv += massFractions[i] * KD * (Elements[i] + fi) / atomic_weight / E / E;
	}

	REFR_END

	/* rv == delta! */
	return 1.0 - (rv * density);
}




double Refractive_Index_Im(const char compound[], double E, double density, xrl_error **error) {
	struct compoundData *cd = NULL;
	struct compoundDataNIST *cdn = NULL;
	int i;
	double rv = 0.0;

	REFR_BEGIN

	for (i = 0 ; i < nElements ; i++) {
		double cs = 0.0;
		cs = CS_Total(Elements[i], E, error);
		if (cs == 0.0)
			return 0.0;
		rv += cs * massFractions[i];
	}

	REFR_END

	/*9.8663479e-9 is calculated as planck's constant * speed of light / 4Pi */
	return rv * density * 9.8663479e-9 / E;
}

xrlComplex Refractive_Index(const char compound[], double E, double density, xrl_error **error) {
	struct compoundData *cd = NULL;
	struct compoundDataNIST *cdn = NULL;
	int i;
	xrlComplex rv = {0.0, 0.0};
	double delta = 0.0;
	double im = 0.0;

	REFR_BEGIN

	for (i = 0 ; i < nElements ; i++) {
		double fi = 0.0;
		double atomic_weight = 0.0;
		double cs = 0.0;
		fi = Fi(Elements[i], E, error);
		if (fi == 0.0)
			return rv;

		atomic_weight = AtomicWeight(Elements[i], error);
		if (atomic_weight == 0.0)
			return rv;

		cs = CS_Total(Elements[i], E, error);
		if (cs == 0.0)
			return rv;

		im += cs * massFractions[i];
		delta += massFractions[i] * KD * (Elements[i] + fi) / atomic_weight / E / E;
	}

	REFR_END

	rv.re = 1.0 - (delta * density);
	rv.im = im * density * 9.8663479e-9 / E;

	return rv;
}

XRL_EXTERN
void Refractive_Index2(const char compound[], double E, double density, xrlComplex* result, xrl_error **error);

void Refractive_Index2(const char compound[], double E, double density, xrlComplex* result, xrl_error **error) {
	xrlComplex z = Refractive_Index(compound, E, density, error);

	result->re = z.re;
	result->im = z.im;
}

