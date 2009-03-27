#include "xrayglob.h"
#include "xraylib.h"
#define KL2 -KL2_LINE-1
#define KL3 -KL3_LINE-1
#define KM2 -KM2_LINE-1
#define KM3 -KM3_LINE-1

//////////////////////////////////////////////////////////////////////
//                                                                  //
//                    Fractional Radiative Rate                     //
//                                                                  //
//          Z : atomic number                                       //
//          line :                                                  //
//            KA_LINE 0                                             //
//            KB_LINE 1                                             //
//            LA_LINE 2                                             //
//            LB_LINE 3                                             //
//                                                                  //
//////////////////////////////////////////////////////////////////////
      
float RadRate(int Z, int line)
{
  float rad_rate, rr1, rr2;

  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function RadRate");
    return 0;
  }

  if (line>=0 && line<2) {
    if (line == KA_LINE) {
      rr1 = RadRate_arr[Z][KL2];
      rr2 = RadRate_arr[Z][KL3];
    }
    else if (line == KB_LINE) {
      rr1 = RadRate_arr[Z][KM2];
      rr2 = RadRate_arr[Z][KM3];
    }
    if (rr1<0. || rr2<0.) {
      ErrorExit("Line not available in function RadRate");
      return 0;
    }
    return rr1 + rr2;
  }

  if (line == LA_LINE) {
    line = L3M5_LINE;
  }
  else if (line == LB_LINE) {
    line = L2M4_LINE;
  }

  line = -line - 1;
  if (line<0 || line>=LINENUM) {
    ErrorExit("Line not available in function RadRate");
    return 0;
  }

  rad_rate = RadRate_arr[Z][line];
  if (rad_rate < 0.) {
    ErrorExit("Line not available in function RadRate");
    return 0;
  }

  return rad_rate;
}

                          
                          
