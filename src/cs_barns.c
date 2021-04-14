/*
Copyright (c) 2009, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "config.h"
#include "xrayglob.h"
#include "xraylib.h"
#include "xraylib-error-private.h"

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//                  Cross sections in barn/atom                     //
//  (see cross_sections.c, scattering.c and polarized.c)            //
//                                                                  //
/////////////////////////////////////////////////////////////////// */

#define INIT \
	double aw; \
	aw = AtomicWeight(Z, error); \
	if (aw == 0.0) \
		return 0.0;


double CSb_Total(int Z, double E, xrl_error **error) {
	double cs = CS_Total(Z, E, error);
	if (cs == 0.0)
		return 0.0;
	INIT
	return cs * aw / AVOGNUM;
}

double CSb_Photo(int Z, double E, xrl_error **error) {
	double cs = CS_Photo(Z, E, error);
	if (cs == 0.0)
		return 0.0;
	INIT
	return cs * aw / AVOGNUM;
}

double CSb_Rayl(int Z, double E, xrl_error **error) {
	double cs = CS_Rayl(Z, E, error);
	if (cs == 0.0)
		return 0.0;
	INIT
	return cs * aw / AVOGNUM;
}

double CSb_Compt(int Z, double E, xrl_error **error) {
	double cs = CS_Compt(Z, E, error);
	if (cs == 0.0)
		return 0.0;
	INIT
	return cs * aw / AVOGNUM;
}

double CSb_FluorLine(int Z, int line, double E, xrl_error **error) {
	double cs = CS_FluorLine(Z, line, E, error);
	if (cs == 0.0)
		return 0.0;
	INIT
	return cs * aw / AVOGNUM;
}

double CSb_FluorShell(int Z, int shell, double E, xrl_error **error) {
	double cs = CS_FluorShell(Z, shell, E, error);
	if (cs == 0.0)
		return 0.0;
	INIT
	return cs * aw / AVOGNUM;
}

double DCSb_Rayl(int Z, double E, double theta, xrl_error **error) {
	double cs = DCS_Rayl(Z, E, theta, error);
	if (cs == 0.0)
		return 0.0;
	INIT
	return cs * aw / AVOGNUM;
}

double DCSb_Compt(int Z, double E, double theta, xrl_error **error) {
	double cs = DCS_Compt(Z, E, theta, error);
	if (cs == 0.0)
		return 0.0;
	INIT
	return cs * aw / AVOGNUM;
}

double DCSPb_Rayl(int Z, double E, double theta, double phi, xrl_error **error) {
	double cs = DCSP_Rayl(Z, E, theta, phi, error);
	if (cs == 0.0)
		return 0.0;
	INIT
	return cs * aw / AVOGNUM;
}

double DCSPb_Compt(int Z, double E, double theta, double phi, xrl_error **error) {
	double cs = DCSP_Compt(Z, E, theta, phi, error);
	if (cs == 0.0)
		return 0.0;
	INIT
	return cs * aw / AVOGNUM;
}

