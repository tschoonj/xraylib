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
#include <math.h>
#include "splint.h"
#include "xrayglob.h"
#include "xraylib.h"
#include "xraylib-error-private.h"
#include <stddef.h>


/*////////////////////////////////////////////////////////////////////
//                                                                  //
//         Differential Rayleigh scattering cross section           // 
//                for polarized beam (cm2/g/sterad)                 //
//                                                                  //
//          Z : atomic number                                       //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//          phi : scattering azimuthal angle (rad)                  //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double DCSP_Rayl(int Z, double E, double theta, double phi, xrl_error **error)
{
  double F, q;
  xrl_error *tmp_error = NULL;
                                                        
  if (Z < 1 || Z > ZMAX) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, Z_OUT_OF_RANGE);
    return 0.0;
  }

  if (E <= 0.0) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, NEGATIVE_ENERGY);
    return 0.0;
  }

  /* this will always return a valid value */
  q = MomentTransf(E, theta, NULL);

  F = FF_Rayl(Z, q, &tmp_error);
  if (tmp_error != NULL) {
    xrl_propagate_error(error, tmp_error);
    return 0.0;
  }

  return  AVOGNUM / AtomicWeight(Z, NULL) * F * F * DCSP_Thoms(theta, phi, NULL);
}                                                                              

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//           Differential Compton scattering cross section          //
//                for polarized beam (cm2/g/sterad)                 //
//          Z : atomic number                                       //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//          phi : scattering azimuthal angle (rad)                  //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double DCSP_Compt(int Z, double E, double theta, double phi, xrl_error **error)
{ 
  double S, q;
  xrl_error *tmp_error = NULL;
                                                        
  if (Z < 1 || Z > ZMAX) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, Z_OUT_OF_RANGE);
    return 0.0;
  }

  if (E <= 0.0) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, NEGATIVE_ENERGY);
    return 0.0;
  }

  /* this will always return a valid value */
  q = MomentTransf(E, theta, NULL);

  /* this may error out if beta/q is zero */
  S = SF_Compt(Z, q, &tmp_error);
  if (tmp_error != NULL) {
    xrl_propagate_error(error, tmp_error);
    return 0.0;
  }

  return AVOGNUM / AtomicWeight(Z, NULL) * S * DCSP_KN(E, theta, phi, NULL);
}


/*////////////////////////////////////////////////////////////////////
//                                                                  //
//      Klein-Nishina differential scattering cross section         // 
//                for polarized beam (barn)                         //
//                                                                  //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//          phi : scattering azimuthal angle (rad)                  //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double DCSP_KN(double E, double theta, double phi, xrl_error **error)
{ 
  double k0_k, k_k0, k_k0_2, cos_th, sin_th, cos_phi;
  
  if (E <= 0.0) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, NEGATIVE_ENERGY);
    return 0.0;
  }

  cos_th = cos(theta);
  sin_th = sin(theta);
  cos_phi = cos(phi);
  
  k0_k = 1.0 + (1.0 - cos_th) * E / MEC2 ;
  k_k0 = 1.0 / k0_k;
  k_k0_2 = k_k0 * k_k0;
  
  return (RE2/2.) * k_k0_2 * (k_k0 + k0_k - 2 * sin_th * sin_th 
			      * cos_phi * cos_phi);
} 


/*////////////////////////////////////////////////////////////////////
//                                                                  //
//       Thomson differential scattering cross section              //
//                for polarized beam (barn)                         //
//                                                                  //
//          theta : scattering polar angle (rad)                    //
//          phi : scattering azimuthal angle (rad)                  //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double DCSP_Thoms(double theta, double phi, xrl_error **error)
{ 
  double sin_th, cos_phi ;

  sin_th = sin(theta) ;
  cos_phi = cos(phi);
  return RE2 * (1.0 - sin_th * sin_th * cos_phi * cos_phi);
}
