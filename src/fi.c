#include <math.h>
#include "splint.h"
#include "xrayglob.h"
#include "xraylib.h"


//////////////////////////////////////////////////////////////////////
//                                                                  //
//                  Anomalous Scattering Factor Fi                  //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
//////////////////////////////////////////////////////////////////////
float Fi(int Z, float E)
{
  float fi;

  if (Z<1 || Z>ZMAX || NE_Fi[Z]<0) {
    ErrorExit("Z out of range in function Fi");
    return 0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function fi");
    return 0;
  }

  splint(E_Fi_arr[Z]-1, Fi_arr[Z]-1, Fi_arr2[Z]-1,
         NE_Fi[Z], E, &fi);


  return fi;

}

