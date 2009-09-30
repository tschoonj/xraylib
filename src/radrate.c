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

                          
                          
