#include <math.h>
#include "splint.h"
#include "xrayglob.h"
#include "xraylib.h"


//////////////////////////////////////////////////////////////////////
//                                                                  //
//                  Total cross section  (cm2/g)                    //
//               (Photoelectric + Compton + Rayleigh)               //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
//////////////////////////////////////////////////////////////////////
float CS_Total(int Z, float E)
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

//////////////////////////////////////////////////////////////////////
//                                                                  //
//         Photoelectric absorption cross section  (cm2/g)          //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
//////////////////////////////////////////////////////////////////////
float CS_Photo(int Z, float E)
{
  float ln_E, ln_sigma, sigma;

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

//////////////////////////////////////////////////////////////////////
//                                                                  //
//            Rayleigh scattering cross section  (cm2/g)            //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
//////////////////////////////////////////////////////////////////////
float CS_Rayl(int Z, float E)
{
  float ln_E, ln_sigma, sigma;

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

//////////////////////////////////////////////////////////////////////
//                                                                  //
//            Compton scattering cross section  (cm2/g)             //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
//////////////////////////////////////////////////////////////////////
float CS_Compt(int Z, float E) 
{
  float ln_E, ln_sigma, sigma;

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


