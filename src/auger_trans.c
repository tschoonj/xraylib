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

double AugerRate(int Z, int auger_trans) {
	double rv;

	rv = 0.0;

	if (Z > ZMAX || Z < 1) {
		ErrorExit("Invalid Z detected in AugerRate");
		return rv;
	}
	else if (auger_trans < K_L1L1_AUGER || auger_trans > M4_M5Q3_AUGER) {
		ErrorExit("Invalid Auger transition detected in AugerRate");
		return rv;
	}

	rv = Auger_Rates[Z][auger_trans];
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

double AugerYield(int Z, int shell) {

	double rv;

	rv = 0.0;

	if (Z > ZMAX || Z < 1) {
		ErrorExit("Invalid Z detected in AugerYield");
		return rv;
	}
	else if (shell < K_SHELL || shell > M5_SHELL) {
		ErrorExit("Invalid Auger transition detected in AugerYield");
		return rv;
	}

	rv = Auger_Yields[Z][shell];

	return rv;
}
