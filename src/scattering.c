#include <math.h>
#include "splint.h"
#include "xrayglob.h"
#include "xraylib.h"

//////////////////////////////////////////////////////////////////////
//                                                                  //
//          Atomic form factor for Rayleigh scattering              //
//                                                                  //
//          Z : atomic number                                       //
//          q : momentum transfer                                   //
//                                                                  //
//////////////////////////////////////////////////////////////////////
float FF_Rayl(int Z, float q)
{
  float FF;

  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function FF_Rayl");
    return 0;
  }

  if (q <= 0.) {
    ErrorExit("q <=0 in function FF_Rayl");
    return 0;
  }

  splint(q_Rayl_arr[Z]-1, FF_Rayl_arr[Z]-1, FF_Rayl_arr2[Z]-1,
	 Nq_Rayl[Z], q, &FF);

  return FF;
}


//////////////////////////////////////////////////////////////////////
//                                                                  //
//       Incoherent scattering function for Compton scattering      //
//                                                                  //
//          Z : atomic number                                       //
//          q : momentum transfer                                   //
//                                                                  //
//////////////////////////////////////////////////////////////////////
float SF_Compt(int Z, float q)
{
  float SF;

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

//////////////////////////////////////////////////////////////////////
//                                                                  //
//       Thomson differential scattering cross section (barn)       //
//                                                                  //
//          theta : scattering polar angle (rad)                    //
//                                                                  //
//////////////////////////////////////////////////// /////////////////
float  DCS_Thoms(float theta)
{ 
  float cos_theta;

  cos_theta = cos(theta);

  return (RE2/2.0) * (1.0 + cos_theta*cos_theta);
}

//////////////////////////////////////////////////////////////////////
//                                                                  //
//    Klein-Nishina differential scattering cross section (barn)   //
//                                                                  //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//                                                                  //
//////////////////////////////////////////////////// /////////////////
float  DCS_KN(float E, float theta)
{ 
  float cos_theta, t1, t2;

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function DCS_KN");
    return 0;
  }

  cos_theta = cos(theta);
  t1 = (1.0 - cos_theta) * E / MEC2 ;
  t2 = 1.0 + t1;
  
  return (RE2/2.) * (1.0 + cos_theta*cos_theta + t1*t1/t2) /t2 /t2;
}

//////////////////////////////////////////////////////////////////////
//                                                                  //
//  Differential Rayleigh scattering cross section (cm2/g/sterad)   //
//                                                                  //
//          Z : atomic number                                       //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//                                                                  //
//////////////////////////////////////////////////// /////////////////
float  DCS_Rayl(int Z, float E, float theta)
{ 
  float F, q ;                                                      
                                                        
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

//////////////////////////////////////////////////////////////////////
//                                                                  //
//  Differential Compton scattering cross section (cm2/g/sterad)    //
//                                                                  //
//          Z : atomic number                                       //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//                                                                  //
//////////////////////////////////////////////////// /////////////////
float  DCS_Compt(int Z, float E, float theta)
{ 
  float S, q ;                                                      
                                                        
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

//////////////////////////////////////////////////////////////////////
//                                                                  //
//    Momentum transfer for X-ray photon scattering (angstrom-1)    //
//                                                                  //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//                                                                  //
//////////////////////////////////////////////////// /////////////////
float  MomentTransf(float E, float theta)
{
  if (E <= 0.) {
    ErrorExit("Energy <=0 in function MomentTransf");
    return 0;
  }
  
  return E / KEV2ANGST * sin(theta / 2.0) ;
}

//////////////////////////////////////////////////////////////////////
//                                                                  //
//            Total klein-Nishina cross section (barn)              //
//                                                                  //
//          E : Energy (keV)                                        //   
//                                                                  //
//////////////////////////////////////////////////////////////////////
float CS_KN(float E)
{ 
  float a, a3, b, b2, lb;
  float sigma;

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


//////////////////////////////////////////////////////////////////////
//                                                                  //
//              Photon energy after Compton scattering (keV)        //
//                                                                  //
//          E0 : Photon Energy before scattering (keV)              //   
//          theta : scattering polar angle (rad)                    //
//                                                                  //
//////////////////////////////////////////////////////////////////////
float ComptonEnergy(float E0, float theta)
{ 
  float cos_theta, alpha;

  if (E0 <= 0.) {
    ErrorExit("Energy <=0 in function ComptonEnergy");
    return 0;
  }

  cos_theta = cos(theta);
  alpha = E0/MEC2;

  return E0 / (1 + alpha*(1 - cos_theta));
}

















