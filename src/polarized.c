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
//         Differential Rayleigh scattering cross section           // 
//                for polarized beam (cm2/g/sterad)                 //
//                                                                  //
//          Z : atomic number                                       //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//          phi : scattering azimuthal angle (rad)                  //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double  DCSP_Rayl(int Z, double E, double theta, double phi)
{
  double F, q;                                                      
                                                        
  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function DCSP_Rayl");
    return 0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function DCSP_Rayl");
    return 0;
  }

  q = MomentTransf(E , theta);
  F = FF_Rayl(Z, q);
  return  AVOGNUM / AtomicWeight(Z) * F*F * DCSP_Thoms(theta, phi);
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
double  DCSP_Compt(int Z, double E, double theta, double phi)
{ 
  double S, q;                                                      
                                                        
  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function DCSP_Compt");
    return 0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function DCSP_Compt");
    return 0;
  }

  q = MomentTransf(E, theta);
  S = SF_Compt(Z, q);
  return  AVOGNUM / AtomicWeight(Z) * S * DCSP_KN(E, theta, phi);
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
double DCSP_KN(double E, double theta, double phi)
{ 
  double k0_k, k_k0, k_k0_2, cos_th, sin_th, cos_phi;
  
  if (E <= 0.) {
    ErrorExit("Energy <=0 in function DCSP_KN");
    return 0;
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
double  DCSP_Thoms(double theta, double phi)
{ 
  double sin_th, cos_phi ;

  sin_th = sin(theta) ;
  cos_phi = cos(phi);
  return RE2 * (1.0 - sin_th * sin_th * cos_phi * cos_phi);
}



