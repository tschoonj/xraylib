#include <math.h>
#include "splint.h"
#include "xrayglob.h"
#include "xraylib.h"


//////////////////////////////////////////////////////////////////////
//                                                                  //
//         Differential Rayleigh scattering cross section           // 
//                for polarized beam (cm2/g/sterad)                 //
//                                                                  //
//          Z : atomic number                                       //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//          phi : scattering azimuthal angle (rad)                  //
//                                                                  //
//////////////////////////////////////////////////// /////////////////
float  DCSP_Rayl(int Z, float E, float theta, float phi)
{
  float F, q;                                                      
                                                        
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

//////////////////////////////////////////////////////////////////////
//                                                                  //
//           Differential Compton scattering cross section          //
//                for polarized beam (cm2/g/sterad)                 //
//          Z : atomic number                                       //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//          phi : scattering azimuthal angle (rad)                  //
//                                                                  //
//////////////////////////////////////////////////// /////////////////
float  DCSP_Compt(int Z, float E, float theta, float phi)
{ 
  float S, q;                                                      
                                                        
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


//////////////////////////////////////////////////////////////////////
//                                                                  //
//      Klein-Nishina differential scattering cross section         // 
//                for polarized beam (barn)                         //
//                                                                  //
//          E : Energy (keV)                                        //
//          theta : scattering polar angle (rad)                    //
//          phi : scattering azimuthal angle (rad)                  //
//                                                                  //
//////////////////////////////////////////////////// /////////////////
float DCSP_KN(float E, float theta, float phi)
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


//////////////////////////////////////////////////////////////////////
//                                                                  //
//       Thomson differential scattering cross section              //
//                for polarized beam (barn)                         //
//                                                                  //
//          theta : scattering polar angle (rad)                    //
//          phi : scattering azimuthal angle (rad)                  //
//                                                                  //
//////////////////////////////////////////////////// /////////////////
float  DCSP_Thoms(float theta, float phi)
{ 
  float sin_th, cos_phi ;

  sin_th = sin(theta) ;
  cos_phi = cos(phi);
  return RE2 * (1.0 - sin_th * sin_th * cos_phi * cos_phi);
}



