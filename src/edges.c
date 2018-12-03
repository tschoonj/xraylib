/*
Copyright (c) 2009-2018, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
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

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//                    Absorption edge energy (keV)                  //
//                                                                  //
//          Z : atomic number                                       //
//          shell :                                                 //
//            K_SHELL  0                                            //
//            L1_SHELL 1                                            //
//            L2_SHELL 2                                            //
//            L3_SHELL 3                                            //
//            M1_SHELL 4                                            //
//            M2_SHELL 5                                            //
//            M3_SHELL 6                                            //
//            M4_SHELL 7                                            //
//            M5_SHELL 8                                            //
//             .......                                              //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
      
double EdgeEnergy(int Z, int shell, xrl_error **error)
{
  double edge_energy;

  if (Z < 1 || Z > ZMAX) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, Z_OUT_OF_RANGE);
    return 0;
  }
  if (shell < 0 || shell >= SHELLNUM) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, UNKNOWN_SHELL);
    return 0;
  }

  edge_energy = EdgeEnergy_arr[Z][shell];

  if (edge_energy <= 0.) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, INVALID_SHELL);
    return 0;
  }

  return edge_energy;
}

                          
                          
