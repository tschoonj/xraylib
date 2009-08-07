#include "splint.h"
#include "xrayglob.h"
#include "xraylib.h"


//////////////////////////////////////////////////////////////////////
//                                                                  //
//                  Anomalous Scattering Factor Fii                 //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
//////////////////////////////////////////////////////////////////////
float Fii(int Z, float E)
{
  float fii;

  if (Z<1 || Z>ZMAX || NE_Fii[Z]<0) {
    ErrorExit("Z out of range in function Fii");
    return 0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function Fii");
    return 0;
  }

  splint(E_Fii_arr[Z]-1, Fii_arr[Z]-1, Fii_arr2[Z]-1,
         NE_Fii[Z], E, &fii);


  return fii;

}

