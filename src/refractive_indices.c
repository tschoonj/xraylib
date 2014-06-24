/*
Copyright (c) 2010-2013, Tom Schoonjans, Bruno Golosio
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans and Bruno Golosio''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define KD 4.15179082788e-4
#include "xrayglob.h"
#include "xraylib.h"
#include <stdlib.h>
#include <math.h>

double Refractive_Index_Re(const char compound[], double E, double density) {
	struct compoundData *cd;
	double delta = 0.0;
	int i;

	if ((cd = CompoundParser(compound)) == NULL) {
		ErrorExit("Refractive_Index_Re: CompoundParser error");
		return 0.0;
	} 
	else if (E <= 0.0) {
		ErrorExit("Refractive_Index_Re: energy must be greater than zero");
		return 0.0;
	}
	else if (density <= 0.0) {
		ErrorExit("Refractive_Index_Re: density must be greater than zero");
		return 0.0;
	}

	/* Real part is 1-delta */
	for (i=0 ; i < cd->nElements ; i++) {
		delta += cd->massFractions[i]*KD*(cd->Elements[i]+Fi(cd->Elements[i],E))/AtomicWeight(cd->Elements[i])/E/E;
	}


	FreeCompoundData(cd);

	return 1.0-(delta*density);
}




double Refractive_Index_Im(const char compound[], double E, double density) {
	struct compoundData *cd;
	int i;
	double rv = 0.0;

	if ((cd = CompoundParser(compound)) == NULL) {
		ErrorExit("Refractive_Index_Im: CompoundParser error");
		return 0.0;
	} 
	else if (E <= 0.0) {
		ErrorExit("Refractive_Index_Im: energy must be greater than zero");
		return 0.0;
	}
	else if (density <= 0.0) {
		ErrorExit("Refractive_Index_Im: density must be greater than zero");
		return 0.0;
	}

	for (i = 0 ; i < cd->nElements ; i++) 
		rv += CS_Total(cd->Elements[i], E)*cd->massFractions[i];

	FreeCompoundData(cd);
	/*9.8663479e-9 is calculated as planck's constant * speed of light / 4Pi */
	return rv*density*9.8663479e-9/E;	
}

xrlComplex Refractive_Index(const char compound[], double E, double density) {
	struct compoundData *cd;
	int i;
	xrlComplex rv;
	double delta = 0.0;
	double im = 0.0;

	rv.re = 0.0;
	rv.im = 0.0;

	if ((cd = CompoundParser(compound)) == NULL) {
		ErrorExit("Refractive_Index: CompoundParser error");
		return rv;
	} 
	else if (E <= 0.0) {
		ErrorExit("Refractive_Index: energy must be greater than zero");
		return rv;
	}
	else if (density <= 0.0) {
		ErrorExit("Refractive_Index: density must be greater than zero");
		return rv;
	}

	for (i=0 ; i < cd->nElements ; i++) {
		delta += cd->massFractions[i]*KD*(cd->Elements[i]+Fi(cd->Elements[i],E))/AtomicWeight(cd->Elements[i])/E/E;
		im += CS_Total(cd->Elements[i], E)*cd->massFractions[i];
	}

	rv.re = 1.0-(delta*density);
	rv.im = im*density*9.8663479e-9/E;

	return rv;
}

void Refractive_Index2(const char compound[], double E, double density, xrlComplex* result) {
	xrlComplex z = Refractive_Index(compound, E, density);

	result->re = z.re;
	result->im = z.im;
}

