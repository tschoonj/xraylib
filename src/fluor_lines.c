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
#include <stddef.h>
#include "xrayglob.h"
#include "xraylib.h"
#include "xraylib-error-private.h"

#define KL1 -(int)KL1_LINE-1
#define KL2 -(int)KL2_LINE-1
#define KL3 -(int)KL3_LINE-1
#define KM1 -(int)KM1_LINE-1
#define KM2 -(int)KM2_LINE-1
#define KM3 -(int)KM3_LINE-1
#define KP5 -(int)KP5_LINE-1

static struct {int line; int shell;} lb_pairs[] = {
  {LB1_LINE, L2_SHELL},
  {LB2_LINE, L3_SHELL},
  {LB3_LINE, L1_SHELL},
  {LB4_LINE, L1_SHELL},
  {LB5_LINE, L3_SHELL},
  {LB6_LINE, L3_SHELL},
  {LB7_LINE, L3_SHELL},
  {LB9_LINE, L1_SHELL},
  {LB10_LINE, L1_SHELL},
  {LB15_LINE, L3_SHELL},
  {LB17_LINE, L2_SHELL},
  {L3N6_LINE, L3_SHELL},
  {L3N7_LINE, L3_SHELL},
};

static double LineEnergyComposed(int Z, int line1, int line2, xrl_error **error) {
  double line_tmp1 = LineEnergy(Z, line1, NULL);
  double line_tmp2 = LineEnergy(Z, line2, NULL);
  double rate_tmp1 = RadRate(Z, line1, NULL);
  double rate_tmp2 = RadRate(Z, line2, NULL);
  double rv = line_tmp1 * rate_tmp1 + line_tmp2 * rate_tmp2;

  if (rv > 0.0) {
    return rv/(rate_tmp1 + rate_tmp2);
  }
  else if ((line_tmp1 + line_tmp2) > 0.0) {
    return (line_tmp1 + line_tmp2)/2.0; /* in case of both radiative rates missing, use the average of both line energies. */
  }
  xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, INVALID_LINE);
  return 0.0;
}

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//                    Fluorescent line energy (keV)                 //
//                                                                  //
//          Z : atomic number                                       //
//          line :                                                  //
//            KA_LINE 0                                             //
//            KB_LINE 1                                             //
//            LA_LINE 2                                             //
//            LB_LINE 3                                             //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
      
double LineEnergy(int Z, int line, xrl_error **error)
{
  double line_energy;
  double lE, rr;
  double tmp=0.0, tmp1=0.0, tmp2=0.0;
  int i;
  int temp_line;
  
  if (Z < 1 || Z > ZMAX) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, Z_OUT_OF_RANGE);
    return 0;
  }
  
  if (line == KA_LINE || line == KB_LINE) {
    if (line == KA_LINE) {
      for (i = KL1; i <= KL3 ; i++) {
        lE = LineEnergy_arr[Z][i];
        rr = RadRate_arr[Z][i];
        tmp1 += rr;
        tmp += lE * rr;
      }
    }
    else if (line == KB_LINE) {
      for (i = KM1; i < KP5; i++) {
        lE = LineEnergy_arr[Z][i];
        rr = RadRate_arr[Z][i];
        tmp1 += rr;
        tmp += lE * rr;
      }
    }
    if (tmp1 > 0) {
      return tmp / tmp1;
    }
    else {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, INVALID_LINE);
      return 0.0;
    }
  }
  
  if (line == LA_LINE) {
    return LineEnergyComposed(Z, L3M4_LINE, L3M5_LINE, error);
  }
  else if (line == LB_LINE) {
    tmp2 = 0.0;
    tmp = 0.0;

    for (i = 0 ; i < sizeof(lb_pairs)/sizeof(lb_pairs[0]) ; i++) {
      tmp1 = CS_FluorLine(Z, lb_pairs[i].line, EdgeEnergy(Z, lb_pairs[i].shell, NULL) + 0.1, NULL);
      tmp2 += tmp1;
      tmp += LineEnergy(Z, lb_pairs[i].line, NULL) * tmp1;
    }

    if (tmp2 > 0) {
      return tmp / tmp2;
    }
    else {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, INVALID_LINE);
      return 0.0;
    }
  }
  /*
   * special cases for composed lines
   */
  else if (line == L1N67_LINE) {
    return LineEnergyComposed(Z, L1N6_LINE, L1N7_LINE, error);
  }
  else if (line == L1O45_LINE) {
    return LineEnergyComposed(Z, L1O4_LINE, L1O5_LINE, error);
  }
  else if (line == L1P23_LINE) {
    return LineEnergyComposed(Z, L1P2_LINE, L1P3_LINE, error);
  }
  else if (line == L2P23_LINE) {
    return LineEnergyComposed(Z, L2P2_LINE, L2P3_LINE, error);
  }
  else if (line == L3O45_LINE) {
    return LineEnergyComposed(Z, L3O4_LINE, L3O5_LINE, error);
  }
  else if (line == L3P23_LINE) {
    return LineEnergyComposed(Z, L3O4_LINE, L3O5_LINE, error);
  }
  else if (line == L3P45_LINE) {
    return LineEnergyComposed(Z, L3P4_LINE, L3P5_LINE, error);
  }
  /*
   * KO_LINE and KP_LINE only have entries in the radrate database, not in the fluor_lines one.
   * So to get the line energies, we should map to a macro that will work as long as there is an appropriate
   * line energy in the fluor_lines database.
   */
  else if (line == KO_LINE) {
    line = KO1_LINE;
  }
  else if (line == KP_LINE) {
    line = KP1_LINE;
  }

  
  line = -line - 1;
  if (line < 0 || line >= LINENUM) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNKNOWN_LINE);
    return 0;
  }
  
  line_energy = LineEnergy_arr[Z][line];

  if (line_energy <= 0.) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, INVALID_LINE);
    return 0;
  }
  return line_energy;
}


                          
                          





