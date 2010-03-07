/*
Copyright (c) 2010, Tom Schoonjans, Bruno Golosio
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

float Refractive_Index_Re(const char compound[], float E, float density) {
	struct compoundData cd;
	float delta = 0.0;
	int i;

	if (CompoundParser(compound,&cd) == 0) {
		ErrorExit("Refractive_Index_Re: CompoundParser error");
		return 0.0;
	} 

	//Real part is 1-delta
	for (i=0 ; i < cd.nElements ; i++) {
		delta += cd.massFractions[i]*KD*(cd.Elements[i]+Fi(cd.Elements[i],E))/AtomicWeight(cd.Elements[i])/E/E;
	}


	FREE_COMPOUND_DATA(cd)

	return 1.0-(delta*density);
}




float Refractive_Index_Im(const char compound[], float E, float density) {

	//9.8663479e-9 is calculated by planck's constant * speed of light / 4Pi
	return CS_Total_CP(compound,E)*density*9.8663479e-9/E;	

}
