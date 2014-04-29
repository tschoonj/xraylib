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

#include <math.h>
#include "splint.h"
#include "xrayglob.h"
#include "xraylib.h"


/*////////////////////////////////////////////////////////////////////
//                                                                  //
//                  Total cross section  (cm2/g)                    //
//               (Photoelectric + Compton + Rayleigh)               //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double CS_Total(int Z, double E)
{
  if (Z<1 || Z>ZMAX || NE_Photo[Z]<0 || NE_Rayl[Z]<0 || NE_Compt[Z]<0) {
    ErrorExit("Z out of range in function CS_Total");
    return 0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CS_Total");
    return 0;
  }

  return CS_Photo(Z, E) + CS_Rayl(Z, E) + CS_Compt(Z, E);
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//         Photoelectric absorption cross section  (cm2/g)          //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double CS_Photo(int Z, double E)
{
  double ln_E, ln_sigma, sigma;

  if (Z<1 || Z>ZMAX || NE_Photo[Z]<0) {
    ErrorExit("Z out of range in function CS_Photo");
    return 0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CS_Photo");
    return 0;
  }

  ln_E = log(E * 1000.0);

  splint(E_Photo_arr[Z]-1, CS_Photo_arr[Z]-1, CS_Photo_arr2[Z]-1,
	 NE_Photo[Z], ln_E, &ln_sigma);

  sigma = exp(ln_sigma);

  return sigma;
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//            Rayleigh scattering cross section  (cm2/g)            //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double CS_Rayl(int Z, double E)
{
  double ln_E, ln_sigma, sigma;

  if (Z<1 || Z>ZMAX || NE_Rayl[Z]<0) {
    ErrorExit("Z out of range in function CS_Rayl");
    return 0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CS_Rayl");
    return 0;
  }

  ln_E = log(E * 1000.0);

  splint(E_Rayl_arr[Z]-1, CS_Rayl_arr[Z]-1, CS_Rayl_arr2[Z]-1,
	 NE_Rayl[Z], ln_E, &ln_sigma);
  sigma = exp(ln_sigma);

  return sigma;
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//            Compton scattering cross section  (cm2/g)             //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double CS_Compt(int Z, double E) 
{
  double ln_E, ln_sigma, sigma;

  if (Z<1 || Z>ZMAX || NE_Compt[Z]<0) {
    ErrorExit("Z out of range in function CS_Compt");
    return 0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CS_Compt");
    return 0;
  }

  ln_E = log(E * 1000.0);

  splint(E_Compt_arr[Z]-1, CS_Compt_arr[Z]-1, CS_Compt_arr2[Z]-1,
	 NE_Compt[Z], ln_E, &ln_sigma);

  sigma = exp(ln_sigma);

  return sigma;
}


/*////////////////////////////////////////////////////////////////////
//                                                                  //
//            Mass energy-absorption coefficient (cm2/g)            //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double CS_Energy(int Z, double E)
{
	double ln_E, ln_sigma, sigma;
	if (Z < 1 || Z > 92 || NE_Energy[Z] < 0) {
		ErrorExit("Z out of range in function CS_Energy");
		return 0;
	}
	if (E <= 0.0) {
		ErrorExit("Z <= 0 in function CS_Energy");
		return 0;
	}
	ln_E = log(E);
	splint(E_Energy_arr[Z]-1, CS_Energy_arr[Z]-1, CS_Energy_arr2[Z]-1, NE_Energy[Z], ln_E, &ln_sigma);

	sigma = exp(ln_sigma);

	return sigma;
}


