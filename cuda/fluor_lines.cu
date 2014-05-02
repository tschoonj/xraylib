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

#include "xraylib.h"
#include "xraylib-cuda.h"
#include "xraylib-cuda-private.h"

#define KL1 -(int)KL1_LINE-1
#define KL2 -(int)KL2_LINE-1
#define KL3 -(int)KL3_LINE-1
#define KM1 -(int)KM1_LINE-1
#define KM2 -(int)KM2_LINE-1
#define KM3 -(int)KM3_LINE-1
#define KP5 -(int)KP5_LINE-1

__device__ double LineEnergy_arr_d[(ZMAX+1)*LINENUM];

__device__ double LineEnergy_cu(int Z, int line) {
  double line_energy;
  double lE[50],rr[50];
  double tmp=0.0,tmp1=0.0,tmp2=0.0;
  int i;
  int temp_line;
  
  if (Z<1 || Z>ZMAX) {
    return 0;
  }
  
  if (line>=KA_LINE && line<LA_LINE) {
    if (line == KA_LINE) {
	for (i = KL1; i <= KL3 ; i++) {
	 lE[i] = LineEnergy_arr_d[Z*LINENUM+i];
	 rr[i] = RadRate_arr_d[Z*LINENUM+i];
	 tmp1+=rr[i];
	 tmp+=lE[i]*rr[i];

	 if (lE[i]<0.0 || rr[i]<0.0) {
	  return 0;
	 }
	}
    }
    else if (line == KB_LINE) {
    	for (i = KM1; i < KP5; i++) {
	 lE[i] = LineEnergy_arr_d[Z*LINENUM+i];
	 rr[i] = RadRate_arr_d[Z*LINENUM+i];
	 tmp1+=rr[i];
	 tmp+=lE[i]*rr[i];
	 if (lE[i]<0.0 || rr[i]<0.0) {
	  return 0;
	 }
	}
    }
   if (tmp1>0)   return tmp/tmp1;  else return 0.0;
  }
  
  if (line == LA_LINE) {
	temp_line = L3M5_LINE;
	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
	tmp2=tmp1;
	tmp=LineEnergy_arr_d[Z*LINENUM+temp_line]*tmp1;
	temp_line = L3M4_LINE;
	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
	tmp2+=tmp1;
	tmp+=LineEnergy_arr_d[Z*LINENUM+temp_line]*tmp1;
  	if (tmp2>0)   return tmp/tmp2;  else return 0.0;
  }
  else if (line == LB_LINE) {
	temp_line = L2M4_LINE;     /* b1 */
	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L2_SHELL)+0.1);
	tmp2=tmp1;
	tmp=LineEnergy_arr_d[Z*LINENUM+temp_line]*tmp1;

	temp_line = L3N5_LINE;     /* b2 */
	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
	tmp2+=tmp1;
	tmp+=LineEnergy_arr_d[Z*LINENUM+temp_line]*tmp1;

	temp_line = L1M3_LINE;     /* b3 */
	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L1_SHELL)+0.1);
	tmp2+=tmp1;
	tmp+=LineEnergy_arr_d[Z*LINENUM+temp_line]*tmp1;

	temp_line = L1M2_LINE;     /* b4 */
	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L1_SHELL)+0.1);
	tmp2+=tmp1;
	tmp+=LineEnergy_arr_d[Z*LINENUM+temp_line]*tmp1 ;

	temp_line = L3O3_LINE;     /* b5 */
	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
	tmp2+=tmp1;
	tmp+=LineEnergy_arr_d[Z*LINENUM+temp_line]*tmp1 ;

	temp_line = L3O4_LINE;     /* b5 */
	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
	tmp2+=tmp1;
	tmp+=LineEnergy_arr_d[Z*LINENUM+temp_line]*tmp1 ;

	temp_line = L3N1_LINE;     /* b6 */
	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
	tmp2+=tmp1;
	tmp+=LineEnergy_arr_d[Z*LINENUM+temp_line]*tmp1 ;
  	if (tmp2>0)   return tmp/tmp2;  else return 0.0;
  }
  /*
   * special cases for composed lines
   */
  else if (line == L1N67_LINE) {
 	return (LineEnergy_arr_d[Z*LINENUM+ L1N6_LINE]+LineEnergy_arr_d[Z*LINENUM+L1N7_LINE])/2.0; 
  }
  else if (line == L1O45_LINE) {
 	return (LineEnergy_arr_d[Z*LINENUM+ L1O4_LINE]+LineEnergy_arr_d[Z*LINENUM+L1O5_LINE])/2.0; 
  }
  else if (line == L1P23_LINE) {
 	return (LineEnergy_arr_d[Z*LINENUM+ L1P2_LINE]+LineEnergy_arr_d[Z*LINENUM+L1P3_LINE])/2.0; 
  }
  else if (line == L2P23_LINE) {
 	return (LineEnergy_arr_d[Z*LINENUM+ L2P2_LINE]+LineEnergy_arr_d[Z*LINENUM+L2P3_LINE])/2.0; 
  }
  else if (line == L3O45_LINE) {
 	return (LineEnergy_arr_d[Z*LINENUM+ L3O4_LINE]+LineEnergy_arr_d[Z*LINENUM+L3O5_LINE])/2.0; 
  }
  else if (line == L3P23_LINE) {
 	return (LineEnergy_arr_d[Z*LINENUM+ L3P2_LINE]+LineEnergy_arr_d[Z*LINENUM+L3P3_LINE])/2.0; 
  }
  else if (line == L3P45_LINE) {
 	return (LineEnergy_arr_d[Z*LINENUM+ L3P4_LINE]+LineEnergy_arr_d[Z*LINENUM+L3P5_LINE])/2.0; 
  }
  
  line = -line - 1;
  if (line<0 || line>=LINENUM) {
    return 0;
  }
  
  line_energy = LineEnergy_arr_d[Z*LINENUM+line];
  if (line_energy < 0.) {
    return 0;
  }
  return line_energy;
}
