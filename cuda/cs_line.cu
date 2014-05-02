/*
Copyright (c) 2014, Tom Schoonjans and Antonio Brunetti
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans and Antonio Brunetti ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans and Antonio Brunetti BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "xraylib-cuda.h"
#include "xraylib-cuda-private.h"
#include "xraylib.h"

      
static __device__ double Jump_from_L1(int Z,double E) {
  double Factor=1.0,JumpL1,JumpK;
	if( E > EdgeEnergy_cu(Z,K_SHELL) ) {
	  JumpK = JumpFactor_cu(Z,K_SHELL) ;
	  if( JumpK <= 0. )
		return 0. ;
	  Factor /= JumpK ;
	}
	if (E > EdgeEnergy_cu(Z, L1_SHELL)) {
	  JumpL1 = JumpFactor_cu(Z, L1_SHELL);
	  if (JumpL1 <= 0.0) return 0.0;
	  Factor *= ((JumpL1-1)/JumpL1) * FluorYield_cu(Z, L1_SHELL);
	}
	else
	  return 0.;
  return Factor;

}

static __device__ double Jump_from_L2(int Z,double E)
{
  double Factor=1.0,JumpL1,JumpL2,JumpK;
  double TaoL1=0.0,TaoL2=0.0;
	if( E > EdgeEnergy_cu(Z,K_SHELL) ) {
	  JumpK = JumpFactor_cu(Z,K_SHELL) ;
	  if( JumpK <= 0. )
		return 0. ;
	  Factor /= JumpK ;
	}
	JumpL1 = JumpFactor_cu(Z,L1_SHELL) ;
	JumpL2 = JumpFactor_cu(Z,L2_SHELL) ;
	if(E>EdgeEnergy_cu(Z,L1_SHELL)) {
	  if( JumpL1 <= 0.|| JumpL2 <= 0. )
		return 0. ;
	  TaoL1 = (JumpL1-1) / JumpL1 ;
	  TaoL2 = (JumpL2-1) / (JumpL2*JumpL1) ;
	}
	else if( E > EdgeEnergy_cu(Z,L2_SHELL) ) {
	  if( JumpL2 <= 0. )
		return 0. ;
	  TaoL1 = 0. ;
	  TaoL2 = (JumpL2-1)/(JumpL2) ;
	}
	else
	  Factor = 0;
	Factor *= (TaoL2 + TaoL1*CosKronTransProb_cu(Z,F12_TRANS)) * FluorYield_cu(Z,L2_SHELL) ;

	return Factor;

}


static __device__ double Jump_from_L3(int Z,double E )
{
  double Factor=1.0,JumpL1,JumpL2,JumpL3,JumpK;
  double TaoL1=0.0,TaoL2=0.0,TaoL3=0.0;

	if( E > EdgeEnergy_cu(Z,K_SHELL) ) {
	  JumpK = JumpFactor_cu(Z,K_SHELL) ;
	  if( JumpK <= 0. )
	return 0.;
	  Factor /= JumpK ;
	}
	JumpL1 = JumpFactor_cu(Z,L1_SHELL) ;
	JumpL2 = JumpFactor_cu(Z,L2_SHELL) ;
	JumpL3 = JumpFactor_cu(Z,L3_SHELL) ;
	if( E > EdgeEnergy_cu(Z,L1_SHELL) ) {
	  if( JumpL1 <= 0.|| JumpL2 <= 0. || JumpL3 <= 0. )
	return 0. ;
	  TaoL1 = (JumpL1-1) / JumpL1 ;
	  TaoL2 = (JumpL2-1) / (JumpL2*JumpL1) ;
	  TaoL3 = (JumpL3-1) / (JumpL3*JumpL2*JumpL1) ;
	}
	else if( E > EdgeEnergy_cu(Z,L2_SHELL) ) {
	  if( JumpL2 <= 0. || JumpL3 <= 0. )
	return 0. ;
	  TaoL1 = 0. ;
	  TaoL2 = (JumpL2-1) / (JumpL2) ;
	  TaoL3 = (JumpL3-1) / (JumpL3*JumpL2) ;
	}
	else if( E > EdgeEnergy_cu(Z,L3_SHELL) ) {
	  TaoL1 = 0. ;
	  TaoL2 = 0. ;
	  if( JumpL3 <= 0. )
	return 0. ;
	  TaoL3 = (JumpL3-1) / JumpL3 ;
	}
	else
	  Factor = 0;
	Factor *= (TaoL3 + TaoL2 * CosKronTransProb_cu(Z,F23_TRANS) +
		TaoL1 * (CosKronTransProb_cu(Z,F13_TRANS) + CosKronTransProb_cu(Z,FP13_TRANS)
		+ CosKronTransProb_cu(Z,F12_TRANS) * CosKronTransProb_cu(Z,F23_TRANS))) ;
	Factor *= (FluorYield_cu(Z,L3_SHELL) ) ;
	return Factor;

}

__device__ double CS_FluorLine_cu(int Z, int line, double E)
{
  double JumpK;
  double cs_line, Factor = 1.;

  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  if (E <= 0.) {
    return 0;
  }

  if (line>=KN5_LINE && line<=KB_LINE) {
    if (E > EdgeEnergy_cu(Z, K_SHELL)) {
      JumpK = JumpFactor_cu(Z, K_SHELL);
      if (JumpK <= 0.)
	return 0.;
      Factor = ((JumpK-1)/JumpK) * FluorYield_cu(Z, K_SHELL);
    }
    else
      return 0.;                               
    cs_line = CS_Photo_cu(Z, E) * Factor * RadRate_cu(Z, line) ;
  }

  else if (line>=L1P5_LINE && line<=L1L2_LINE) {
	Factor=Jump_from_L1(Z,E);
	cs_line = CS_Photo_cu(Z, E) * Factor * RadRate_cu(Z, line) ;
  }
  
  else if (line>=L2Q1_LINE && line<=L2L3_LINE)  {
	Factor=Jump_from_L2(Z,E);
	cs_line = CS_Photo_cu(Z, E) * Factor * RadRate_cu(Z, line) ;
  }
  /*
   * it's safe to use LA_LINE since it's only composed of 2 L3-lines
   */
  else if ((line>=L3Q1_LINE && line<=L3M1_LINE) || line==LA_LINE) {
	Factor=Jump_from_L3(Z,E);
	cs_line = CS_Photo_cu(Z, E) * Factor * RadRate_cu(Z, line) ;
  }
  else if (line==LB_LINE) {
   	/*
	 * b1->b17
	 */
   	cs_line=Jump_from_L2(Z,E)*(RadRate_cu(Z,L2M4_LINE)+RadRate_cu(Z,L2M3_LINE))+
		   Jump_from_L3(Z,E)*(RadRate_cu(Z,L3N5_LINE)+RadRate_cu(Z,L3O4_LINE)+RadRate_cu(Z,L3O5_LINE)+RadRate_cu(Z,L3O45_LINE)+RadRate_cu(Z,L3N1_LINE)+RadRate_cu(Z,L3O1_LINE)+RadRate_cu(Z,L3N6_LINE)+RadRate_cu(Z,L3N7_LINE)+RadRate_cu(Z,L3N4_LINE)) +
		   Jump_from_L1(Z,E)*(RadRate_cu(Z,L1M3_LINE)+RadRate_cu(Z,L1M2_LINE)+RadRate_cu(Z,L1M5_LINE)+RadRate_cu(Z,L1M4_LINE));
   	cs_line*=CS_Photo_cu(Z, E);
  }

  else {
    return 0;
  }
  
  
  return (cs_line);
}            
