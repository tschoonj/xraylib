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
#include "xraylib-defs.h"
#include "xraylib.h"
      
#define KL1 -(int)KL1_LINE-1
#define KL2 -(int)KL2_LINE-1
#define KL3 -(int)KL3_LINE-1
#define KM2 -(int)KM2_LINE-1
#define KM3 -(int)KM3_LINE-1

__device__ double RadRate_arr_d[(ZMAX+1)*LINENUM];

__device__ double RadRate_cu(int Z, int line) {
  double rad_rate, rr;
  int i;

  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  if (line == KA_LINE) {
        rr=0.0;
    	for (i=KL1 ; i <= KL3 ; i++)
		rr += RadRate_arr_d[Z*LINENUM+i];
    	if (rr == 0.0) {
      		return 0.0;
    	}
    	return rr;
  }
  else if (line == KB_LINE) {
    	/*
	 * we assume that RR(Ka)+RR(Kb) = 1.0
	 */
        rr=0.0;
    	for (i=KL1 ; i <= KL3 ; i++)
		rr += RadRate_arr_d[Z*LINENUM+i];
    	if (rr == 1.0) {
      		return 0.0;
    	}
	else if (rr == 0.0) {
		return 0.0;
	}
    	return 1.0-rr;
  }
  else if (line == LA_LINE) {
	line = -L3M5_LINE-1;
	rr=RadRate_arr_d[Z*LINENUM+line];
	line = -L3M4_LINE-1;
	rr+=RadRate_arr_d[Z*LINENUM+line];
	return rr;
  }
  /*
   * in Siegbahn notation: use only KA, KB and LA. The radrates of other lines are nonsense
   */

  line = -line - 1;
  if (line<0 || line>=LINENUM) {
    return 0;
  }

  rad_rate = RadRate_arr_d[Z*LINENUM+line];
  if (rad_rate < 0.) {
    return 0;
  }

  return rad_rate;
}
