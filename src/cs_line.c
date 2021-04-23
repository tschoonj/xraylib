/*
Copyright (c) 2009,2010, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "config.h"
#include "xrayglob.h"
#include "xraylib.h"
#include "xraylib-error-private.h"
#include <stddef.h>

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//                    Fluorescent line cross section (cm2/g)        //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//          line :                                                  //
//            KA_LINE 0                                             //
//            KB_LINE 1                                             //
//            LA_LINE 2                                             //
//            LB_LINE 3                                             //
//                                                                  //
// Ref: M. O. Krause et. al. "X-Ray Fluorescence Cross Sections     //
// for K and L X Rays of the Elements", ORNL 53                     //
/////////////////////////////////////////////////////////////////// */

/* the next three methods correspond to the math shown in the third page of the Brunetti et al 2004 xraylib manuscript */

static double Jump_from_L1(int Z, double E, xrl_error **error)
{
  double Factor = 1.0, JumpL1, JumpK;
  double edgeK = EdgeEnergy(Z, K_SHELL, NULL);
  double edgeL1 = EdgeEnergy(Z, L1_SHELL, NULL);
  double yield;

  if (E > edgeK && edgeK > 0.0) {
    JumpK = JumpFactor(Z, K_SHELL, NULL);
    if (JumpK == 0.0) {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_JUMP_FACTOR);
      return 0.0;
    }
    Factor /= JumpK ;
  }
 
  if (E > edgeL1 && edgeL1 > 0.0) {
    JumpL1 = JumpFactor(Z, L1_SHELL, NULL);
    if (JumpL1 == 0.0) {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_JUMP_FACTOR);
      return 0.0;
    }
    yield = FluorYield(Z, L1_SHELL, NULL);
    if (yield == 0.0) {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_FLUOR_YIELD);
      return 0.0;
    }
    Factor *= ((JumpL1 - 1) / JumpL1) * yield;
  }
  else {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, TOO_LOW_EXCITATION_ENERGY);
    return 0.0;
  }

  return Factor;
}

static double Jump_from_L2(int Z, double E, xrl_error **error)
{
  double Factor = 1.0, JumpL1, JumpL2, JumpK;
  double TaoL1 = 0.0, TaoL2 = 0.0;
  double edgeK = EdgeEnergy(Z, K_SHELL, NULL);
  double edgeL1 = EdgeEnergy(Z, L1_SHELL, NULL);
  double edgeL2 = EdgeEnergy(Z, L2_SHELL, NULL);
  double ck_L12, yield;

  if (E > edgeK && edgeK > 0.0) {
    JumpK = JumpFactor(Z, K_SHELL, NULL);
    if (JumpK == 0.0) {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_JUMP_FACTOR);
      return 0.0;
    }
    Factor /= JumpK ;
  }

  JumpL1 = JumpFactor(Z, L1_SHELL, NULL);
  JumpL2 = JumpFactor(Z, L2_SHELL, NULL);

  if (E > edgeL1 && edgeL1 > 0.0) {
    if (JumpL1 == 0.0 || JumpL2 == 0.0) {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_JUMP_FACTOR);
      return 0.0;
    }
    TaoL1 = (JumpL1 - 1) / JumpL1 ;
    TaoL2 = (JumpL2 - 1) / (JumpL2 * JumpL1) ;
  }
  else if (E > edgeL2 && edgeL2 > 0.0) {
    if (JumpL2 == 0.0) {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_JUMP_FACTOR);
      return 0.0;
    }
    TaoL1 = 0.0;
    TaoL2 = (JumpL2 - 1) / JumpL2;
  }
  else {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, TOO_LOW_EXCITATION_ENERGY);
    return 0.0;
  }

  ck_L12 = CosKronTransProb(Z, FL12_TRANS, NULL);
  if (TaoL1 > 0 && ck_L12 == 0.0) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_CK);
    return 0.0;
  }

  yield = FluorYield(Z, L2_SHELL, NULL);
  if (yield == 0.0) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_FLUOR_YIELD);
    return 0.0;
  }

  Factor *= (TaoL2 + TaoL1 * ck_L12) * yield;

  return Factor;
}

static double Jump_from_L3(int Z, double E, xrl_error **error)
{
  double Factor=1.0, JumpL1, JumpL2, JumpL3, JumpK;
  double TaoL1=0.0, TaoL2=0.0, TaoL3=0.0;
  double edgeK = EdgeEnergy(Z, K_SHELL, NULL);
  double edgeL1 = EdgeEnergy(Z, L1_SHELL, NULL);
  double edgeL2 = EdgeEnergy(Z, L2_SHELL, NULL);
  double edgeL3 = EdgeEnergy(Z, L3_SHELL, NULL);
  double ck_L23, ck_L13, ck_LP13, ck_L12;
  double yield;

  if (E > edgeK && edgeK > 0.0) {
    JumpK = JumpFactor(Z, K_SHELL, NULL);
    if (JumpK == 0.0) {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_JUMP_FACTOR);
      return 0.0;
    }
    Factor /= JumpK ;
  }
  JumpL1 = JumpFactor(Z, L1_SHELL, NULL);
  JumpL2 = JumpFactor(Z, L2_SHELL, NULL);
  JumpL3 = JumpFactor(Z, L3_SHELL, NULL);
  if (E > edgeL1 && edgeL1 > 0.0) {
    if (JumpL1 == 0.0 || JumpL2 == 0.0 || JumpL3 == 0.0) {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_JUMP_FACTOR);
      return 0.0;
    }
    TaoL1 = (JumpL1 - 1) / JumpL1 ;
    TaoL2 = (JumpL2 - 1) / (JumpL2 * JumpL1) ;
    TaoL3 = (JumpL3 - 1) / (JumpL3 * JumpL2 * JumpL1) ;
  }
  else if (E > edgeL2 && edgeL2 > 0.0) {
    if (JumpL2 == 0.0 || JumpL3 == 0.0) {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_JUMP_FACTOR);
      return 0.0;
    }
    TaoL1 = 0.0;
    TaoL2 = (JumpL2 - 1) / (JumpL2) ;
    TaoL3 = (JumpL3 - 1) / (JumpL3 * JumpL2) ;
  }
  else if (E > edgeL3 && edgeL3 > 0.0) {
    TaoL1 = 0.0;
    TaoL2 = 0.0;
    if (JumpL3 == 0.0) {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_JUMP_FACTOR);
      return 0.0;
    }
    TaoL3 = (JumpL3 - 1) / JumpL3 ;
  }
  else {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, TOO_LOW_EXCITATION_ENERGY);
    return 0.0;
  }

  ck_L23 = CosKronTransProb(Z, FL23_TRANS, NULL);
  ck_L13 = CosKronTransProb(Z, FL13_TRANS, NULL);
  ck_LP13 = CosKronTransProb(Z, FLP13_TRANS, NULL);
  ck_L12 = CosKronTransProb(Z, FL12_TRANS, NULL);

  if (TaoL2 > 0.0 && ck_L23 == 0.0) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_CK);
    return 0.0;
  }

  if (TaoL1 > 0.0 && (ck_L13 + ck_LP13 == 0.0 || ck_L12 == 0.0 || ck_L23 == 0.0)) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_CK);
    return 0.0;
  }

  Factor *= TaoL3 + TaoL2 * ck_L23 + TaoL1 * (ck_L13 + ck_LP13 + ck_L12 * ck_L23);

  yield = FluorYield(Z, L3_SHELL, NULL);
  if (yield == 0.0) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNAVAILABLE_FLUOR_YIELD);
    return 0.0;
  }

  Factor *= yield;
  return Factor;
}

static double Jump_from_K(int Z, double E, xrl_error **error)
{
  double edgeK = EdgeEnergy(Z, K_SHELL, error);
  double Factor;
  if (E > edgeK && edgeK > 0.0) {
    double yield;
    double JumpK = JumpFactor(Z, K_SHELL, error);
    if (JumpK == 0.0) {
      return 0.0;
    }
    yield = FluorYield(Z, K_SHELL, error);
    if (yield == 0.0) {
      return 0.0;
    }
    Factor = ((JumpK - 1)/JumpK) * yield;
  }
  else if (edgeK == 0.0) {
    return 0.0;
  }
  else {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, TOO_LOW_EXCITATION_ENERGY);
    return 0.0;
  }
  return Factor;
}

static double (*jumpers[])(int, double, xrl_error **) = {Jump_from_K, Jump_from_L1, Jump_from_L2, Jump_from_L3};

double CS_FluorShell(int Z, int shell, double E, xrl_error **error)
{
  double cs = 0.0;
  double Factor = 0.0;

  if (Z < 1 || Z > ZMAX) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, Z_OUT_OF_RANGE);
    return 0.0;
  }

  if (E <= 0.0) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, NEGATIVE_ENERGY);
    return 0.0;
  }

  if (shell < K_SHELL || shell > L3_SHELL) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, INVALID_SHELL);
    return 0.0;
  }

  Factor = jumpers[shell](Z, E, error);
  if (Factor == 0.0) {
    return 0.0;
  }
  
  cs = CS_Photo(Z, E, error);
  if (cs == 0.0) {
    return 0.0;
  }

  return cs * Factor;
}

double CS_FluorLine(int Z, int line, double E, xrl_error **error)
{
  double Factor = 1.0;
  double rr;

  if (line >= KP5_LINE && line <= KB_LINE) {
    rr = RadRate(Z, line, error);
    if (rr == 0.0) {
      return 0.0;
    }

    Factor = CS_FluorShell(Z, K_SHELL, E, error);
    if (Factor == 0.0) {
      return 0.0;
    }

    return rr * Factor;
  }
  else if ((line <= L1L2_LINE && line >= L3Q1_LINE) || line == LA_LINE) {
    rr = RadRate(Z, line, error);
    if (rr == 0.0) {
      return 0.0;
    }

    if (line >= L1P5_LINE && line <= L1L2_LINE) {
      Factor = CS_FluorShell(Z, L1_SHELL, E, error);
    }
    else if (line >= L2Q1_LINE && line <= L2L3_LINE)  {
      Factor = CS_FluorShell(Z, L2_SHELL, E, error);
    }
    /*
     * it's safe to use LA_LINE since it's only composed of 2 L3-lines
     */
    else if (line <= L3M1_LINE || line == LA_LINE) {
      Factor = CS_FluorShell(Z, L3_SHELL, E, error);
    }

    if (Factor == 0.0) {
      return 0.0;
    }
    return rr * Factor;
  }
  else if (line == LB_LINE) {
    /*
     * b1->b17
     */
    double cs;
    double cs_line = Jump_from_L2(Z, E, NULL) * (RadRate(Z, L2M4_LINE, NULL) + RadRate(Z, L2M3_LINE, NULL)) +
      Jump_from_L3(Z, E, NULL) * (RadRate(Z, L3N5_LINE, NULL) + RadRate(Z, L3O4_LINE, NULL) + RadRate(Z, L3O5_LINE, NULL) + RadRate(Z, L3O45_LINE, NULL) + RadRate(Z, L3N1_LINE, NULL) + RadRate(Z, L3O1_LINE, NULL) + RadRate(Z, L3N6_LINE, NULL) + RadRate(Z, L3N7_LINE, NULL) + RadRate(Z, L3N4_LINE, NULL)) +
      Jump_from_L1(Z, E, NULL) * (RadRate(Z, L1M3_LINE, NULL) + RadRate(Z, L1M2_LINE, NULL) + RadRate(Z, L1M5_LINE, NULL) + RadRate(Z, L1M4_LINE, NULL));

    if (cs_line == 0.0) {
      xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, TOO_LOW_EXCITATION_ENERGY);
      return 0.0;
    }
    cs = CS_Photo(Z, E, error);
    if (cs == 0.0) {
      return 0.0;
    }
    return cs_line * cs;
  }
  else {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, INVALID_LINE);
    return 0.0;
  }
  return 0.0;
}            
