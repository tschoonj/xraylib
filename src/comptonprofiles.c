/*
Copyright (c) 2010-2018, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "config.h"
#include "splint.h"
#include "xrayglob.h"
#include "xraylib.h"
#include "math.h"
#include "xraylib-error-private.h"

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//                Compton scattering profile                        //
//                                                                  //
//          Z : atomic number                                       //
//          pz : momentum                                           //
//                                                                  //
/////////////////////////////////////////////////////////////////// */


double ComptonProfile(int Z, double pz, xrl_error **error) {
	double q, ln_q;
	double ln_pz;
	int splint_rv;

	if (Z < 1 || Z > ZMAX || NShells_ComptonProfiles[Z] < 0) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, Z_OUT_OF_RANGE);
		return 0.0;
	}  

	if (pz < 0.0) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, NEGATIVE_PZ);
		return 0.0;
	}
	
	ln_pz = log(pz + 1.0);

	splint_rv = splint(pz_ComptonProfiles[Z]-1, Total_ComptonProfiles[Z]-1, Total_ComptonProfiles2[Z]-1,  Npz_ComptonProfiles[Z], ln_pz, &ln_q, error);

	if (!splint_rv)
		return 0.0;

	q = exp(ln_q); 

	return q;
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//              subshell Compton scattering profile                 //
//                                                                  //
//          Z : atomic number                                       //
//          shell : shell macro                                     //
//          pz : momentum                                           //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double ComptonProfile_Partial(int Z, int shell, double pz, xrl_error **error) {
	double q, ln_q;
	double ln_pz;
	int splint_rv;


	if (Z < 1 || Z > ZMAX || NShells_ComptonProfiles[Z] < 1) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, Z_OUT_OF_RANGE);
		return 0.0;
	}

	if (shell >= NShells_ComptonProfiles[Z] || shell < K_SHELL || UOCCUP_ComptonProfiles[Z][shell] == 0.0 ) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, INVALID_SHELL);
		return 0.0;
	}

	if (pz < 0.0) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, NEGATIVE_PZ);
		return 0.0;
	}
	
	ln_pz = log(pz + 1.0);

	splint_rv = splint(pz_ComptonProfiles[Z]-1, Partial_ComptonProfiles[Z][shell]-1,Partial_ComptonProfiles2[Z][shell]-1, Npz_ComptonProfiles[Z], ln_pz, &ln_q, error);

	if (!splint_rv)
		return 0.0;

	q = exp(ln_q); 

	return q;
}

XRL_EXTERN
double ElectronConfig_Biggs(int Z, int shell, xrl_error **error);

double ElectronConfig_Biggs(int Z, int shell, xrl_error **error) {
	if (Z < 1 || Z > ZMAX || NShells_ComptonProfiles[Z] < 0) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, Z_OUT_OF_RANGE);
		return 0.0;
	}

	if (shell >= NShells_ComptonProfiles[Z] || UOCCUP_ComptonProfiles[Z][shell] == 0.0 ) {
		xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, INVALID_SHELL);
		return 0.0;
	}

  	return UOCCUP_ComptonProfiles[Z][shell]; 
}

