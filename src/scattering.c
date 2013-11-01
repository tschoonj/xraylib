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
//          Atomic form factor for Rayleigh scattering              //
//                                                                  //
//          Z : atomic number                                       //
//          q : momentum transfer                                   //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double FF_Rayl(int Z, double q)
{
  double FF;

  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function FF_Rayl");
    return 0;
  }

  if (q == 0) return Z;

  if (q < 0.) {
    ErrorExit("q < 0 in function FF_Rayl");
    return 0;
  }

  splint(q_Rayl_arr[Z]-1, FF_Rayl_arr[Z]-1, FF_Rayl_arr2[Z]-1,
	 Nq_Rayl[Z], q, &FF);

  return FF;
}


/*////////////////////////////////////////////////////////////////////
//                                                                  //
//       Incoherent scattering function for Compton scattering      //
//                                                                  //
//          Z : atomic number                                       //
//          q : momentum transfer                                   //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double SF_Compt(int Z, double q)
{
  double SF;

  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function SF_Compt");
    return 0;
  }

  if (q <= 0.) {
    ErrorExit("q <=0 in function SF_Compt");
    return 0;
  }

  splint(q_Compt_arr[Z]-1, SF_Compt_arr[Z]-1, SF_Compt_arr2[Z]-1,
	 Nq_Compt[Z], q, &SF);

  return SF;
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//       Thomson differential scattering cross section (barn)       //
//                                                                  //
//          theta : scattering polar angle (rad)                    //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double  DCS_Thoms(double theta)
{ 
  double cos_theta;

  cos_theta = cos(theta);

  return (RE2/2.0) * (1.0 + cos_theta*cos_theta);
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//    Klein-Nishina differential scattering cross section (barn)    //
//                                                                  //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double  DCS_KN(double E, double theta)
{ 
  double cos_theta, t1, t2;

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function DCS_KN");
    return 0;
  }

  cos_theta = cos(theta);
  t1 = (1.0 - cos_theta) * E / MEC2 ;
  t2 = 1.0 + t1;
  
  return (RE2/2.) * (1.0 + cos_theta*cos_theta + t1*t1/t2) /t2 /t2;
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//  Differential Rayleigh scattering cross section (cm2/g/sterad)   //
//                                                                  //
//          Z : atomic number                                       //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double  DCS_Rayl(int Z, double E, double theta)
{ 
  double F, q ;                                                      
                                                        
  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function DCS_Rayl");
    return 0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function DCS_Rayl");
    return 0;
  }

  q = MomentTransf(E, theta);
  F = FF_Rayl(Z, q);
  return  AVOGNUM / AtomicWeight_arr[Z] * F*F * DCS_Thoms(theta);
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//  Differential Compton scattering cross section (cm2/g/sterad)    //
//                                                                  //
//          Z : atomic number                                       //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double  DCS_Compt(int Z, double E, double theta)
{ 
  double S, q ;                                                      
                                                        
  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function DCS_Compt");
    return 0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function DCS_Compt");
    return 0;
  }

  q = MomentTransf(E, theta);
  S = SF_Compt(Z, q);
  return  AVOGNUM / AtomicWeight_arr[Z] * S * DCS_KN(E, theta);
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//    Momentum transfer for X-ray photon scattering (angstrom-1)    //
//                                                                  //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double  MomentTransf(double E, double theta)
{
  if (E <= 0.) {
    ErrorExit("Energy <=0 in function MomentTransf");
    return 0;
  }
  
  return E / KEV2ANGST * sin(theta / 2.0) ;
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//            Total klein-Nishina cross section (barn)              //
//                                                                  //
//          E : Energy (keV)                                        //   
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double CS_KN(double E)
{ 
  double a, a3, b, b2, lb;
  double sigma;

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CS_KN");
    return 0;
  }

  a = E / MEC2;
  a3 = a*a*a;
  b = 1 + 2*a;
  b2 = b*b;
  lb = log(b);

  sigma = 2*PI*RE2*( (1+a)/a3*(2*a*(1+a)/b-lb) + 0.5*lb/a - (1+3*a)/b2); 
  return sigma;
}


/*////////////////////////////////////////////////////////////////////
//                                                                  //
//              Photon energy after Compton scattering (keV)        //
//                                                                  //
//          E0 : Photon Energy before scattering (keV)              //   
//          theta : scattering polar angle (rad)                    //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
double ComptonEnergy(double E0, double theta)
{ 
  double cos_theta, alpha;

  if (E0 <= 0.) {
    ErrorExit("Energy <=0 in function ComptonEnergy");
    return 0;
  }

  cos_theta = cos(theta);
  alpha = E0/MEC2;

  return E0 / (1 + alpha*(1 - cos_theta));
}
