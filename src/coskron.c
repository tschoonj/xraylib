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

/*////////////////////////////////////////////////////////////////////
//                                                                  //
//                    Coster-Kronig transition probability          //
//          Z : atomic number                                       //
//          transition type :                                       //
//            F1_TRANS  0                                           //
//            F12_TRANS 1                                           //
//            F13_TRANS 2                                           //
//            FP13_TRANS 3                                          //
//            F23_TRANS 4                                           //
//                                                                  //
/////////////////////////////////////////////////////////////////// */
      
double CosKronTransProb(int Z, int trans)
{
  double trans_prob;

  if (Z<1 || Z>ZMAX){
    ErrorExit("Z out of range in function CosKronTransProb");
    return 0;
  }

  if (trans<0 || trans>=TRANSNUM) {
    ErrorExit("Transition not available in function CosKronTransProb");
    return 0;
  }

  trans_prob = CosKron_arr[Z][trans];
  if (trans_prob <= 0.) {
    ErrorExit("Transition not available in function CosKronTransProb");
    return 0;
  }

  return trans_prob;
}

                          
                          
