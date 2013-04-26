/*
Copyright (c) 2009, 2010, 2011, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "xraylib.h"
#include "xrayvars.h"
#include "xrayglob.h"

static float AugerYield2(int Z, int shell);

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//                Fractional non Radiative Rate                     //
//                                                                  //
//          Z : atomic number                                       //
//          auger_trans: macro identifying initial                  //
//            ionized shell and two resulting                       //
//            ejected electrons                                     //
//                                                                  //
/////////////////////////////////////////////////////////////////// */

float AugerRate(int Z, int auger_trans) {
	float rv;
	float yield, yield2;

	rv = 0.0;

	if (Z > ZMAX || Z < 1) {
		ErrorExit("Invalid Z detected in AugerRate");
		return rv;
	}
	else if (auger_trans < K_L1L1_AUGER || auger_trans > M4_M5Q3_AUGER) {
		ErrorExit("Invalid Auger transition detected in AugerRate");
		return rv;
	}


	if (auger_trans >= K_L1L1_AUGER && auger_trans < L1_L2L2_AUGER  ) {
		yield = AugerYield(Z, K_SHELL);
		yield2 = AugerYield2(Z, K_SHELL);
		if (yield < 1E-8 || yield2 < 1E-8) 
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]*yield2/yield;
	}
	else if (auger_trans >= L1_L2L2_AUGER && auger_trans < L2_L3L3_AUGER) {
		yield = AugerYield(Z, L1_SHELL);
		yield2 = AugerYield2(Z, L1_SHELL);
		if (yield < 1E-8 || yield2 < 1E-8) 
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]*yield2/yield;
	}
	else if (auger_trans >= L2_L3L3_AUGER && auger_trans < L3_M1M1_AUGER) {
		yield = AugerYield(Z, L2_SHELL);
		yield2 = AugerYield2(Z, L2_SHELL);
		if (yield < 1E-8 || yield2 < 1E-8) 
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]*yield2/yield;
	}
	else if (auger_trans >= L3_M1M1_AUGER && auger_trans < M1_M2M2_AUGER) {
		yield = AugerYield(Z, L3_SHELL);
		yield2 = AugerYield2(Z, L3_SHELL);
		if (yield < 1E-8 || yield2 < 1E-8) 
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]*yield2/yield;
	}
	else if (auger_trans >= M1_M2M2_AUGER && auger_trans < M2_M3M3_AUGER) {
		yield = AugerYield(Z, M1_SHELL);
		yield2 = AugerYield2(Z, M1_SHELL);
		if (yield < 1E-8 || yield2 < 1E-8) 
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]*yield2/yield;
	}
	else if (auger_trans >= M2_M3M3_AUGER && auger_trans < M3_M4M4_AUGER) {
		yield = AugerYield(Z, M2_SHELL);
		yield2 = AugerYield2(Z, M2_SHELL);
		if (yield < 1E-8 || yield2 < 1E-8) 
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]*yield2/yield;
	}
	else if (auger_trans >= M3_M4M4_AUGER && auger_trans < M4_M5M5_AUGER) {
		yield = AugerYield(Z, M3_SHELL);
		yield2 = AugerYield2(Z, M3_SHELL);
		if (yield < 1E-8 || yield2 < 1E-8) 
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]*yield2/yield;
	}
	else if (auger_trans >= M4_M5M5_AUGER && auger_trans <= M4_M5Q3_AUGER) {
		yield = AugerYield(Z, M4_SHELL);
		yield2 = AugerYield2(Z, M4_SHELL);
		if (yield < 1E-8 || yield2 < 1E-8) 
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]*yield2/yield;
	}

	return rv;
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//                AugerYield                                        //
//                                                                  //
//          Z : atomic number                                       //
//          shell: macro identifying excited shell                  //
//                                                                  //
/////////////////////////////////////////////////////////////////// */

float AugerYield(int Z, int shell) {

	float rv;

	rv = 0.0;

	if (Z > ZMAX || Z < 1) {
		ErrorExit("Invalid Z detected in AugerYield");
		return rv;
	}
	else if (shell < K_SHELL || shell > M5_SHELL) {
		ErrorExit("Invalid Auger transition detected in AugerYield");
		return rv;
	}
	
	rv = 1.0 - FluorYield(Z, shell);
	if (shell == L1_SHELL) {
		rv -= CosKronTransProb(Z, FL12_TRANS);
		rv -= CosKronTransProb(Z, FL13_TRANS);
	}
	else if (shell == L2_SHELL) {
		rv -= CosKronTransProb(Z, FL23_TRANS);
	}
	else if (shell == M1_SHELL) {
		rv -= CosKronTransProb(Z, FM12_TRANS);
		rv -= CosKronTransProb(Z, FM13_TRANS);
		rv -= CosKronTransProb(Z, FM14_TRANS);
		rv -= CosKronTransProb(Z, FM15_TRANS);
	}
	else if (shell == M2_SHELL) {
		rv -= CosKronTransProb(Z, FM23_TRANS);
		rv -= CosKronTransProb(Z, FM24_TRANS);
		rv -= CosKronTransProb(Z, FM25_TRANS);
	}
	else if (shell == M3_SHELL) {
		rv -= CosKronTransProb(Z, FM34_TRANS);
		rv -= CosKronTransProb(Z, FM35_TRANS);
	}
	else if (shell == M4_SHELL) {
		rv -= CosKronTransProb(Z, FM45_TRANS);
	}

	return rv;
}
static float AugerYield2(int Z, int shell) {
	float rv;

	rv = 0.0;

	if (Z > ZMAX || Z < 1) {
		ErrorExit("Invalid Z detected in AugerYield2");
		return rv;
	}
	else if (shell < K_SHELL || shell > M5_SHELL) {
		ErrorExit("Invalid shell number detected in AugerYield2");
		return rv;
	}
	
	rv = Auger_Transition_Total[Z][shell];
	if (shell == L1_SHELL) {
		rv += CosKronTransProb(Z, FL12_TRANS);
		rv += CosKronTransProb(Z, FL13_TRANS);
	}
	else if (shell == L2_SHELL) {
		rv += CosKronTransProb(Z, FL23_TRANS);
	}
	else if (shell == M1_SHELL) {
		rv += CosKronTransProb(Z, FM12_TRANS);
		rv += CosKronTransProb(Z, FM13_TRANS);
		rv += CosKronTransProb(Z, FM14_TRANS);
		rv += CosKronTransProb(Z, FM15_TRANS);
	}
	else if (shell == M2_SHELL) {
		rv += CosKronTransProb(Z, FM23_TRANS);
		rv += CosKronTransProb(Z, FM24_TRANS);
		rv += CosKronTransProb(Z, FM25_TRANS);
	}
	else if (shell == M3_SHELL) {
		rv += CosKronTransProb(Z, FM34_TRANS);
		rv += CosKronTransProb(Z, FM35_TRANS);
	}
	else if (shell == M4_SHELL) {
		rv += CosKronTransProb(Z, FM45_TRANS);
	}

	return rv;

}

