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

__device__ int Nq_Rayl_d[ZMAX+1];
__device__ double q_Rayl_arr_d[(ZMAX+1)*NQ_RAYL_MAX];
__device__ double FF_Rayl_arr_d[(ZMAX+1)*NQ_RAYL_MAX];
__device__ double FF_Rayl_arr2_d[(ZMAX+1)*NQ_RAYL_MAX];

__device__ int Nq_Compt_d[ZMAX+1];
__device__ double q_Compt_arr_d[(ZMAX+1)*NQ_COMPT_MAX];
__device__ double SF_Compt_arr_d[(ZMAX+1)*NQ_COMPT_MAX];
__device__ double SF_Compt_arr2_d[(ZMAX+1)*NQ_COMPT_MAX];


__device__ double FF_Rayl_cu(int Z, double q) {
  double FF;

  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  if (q == 0) return Z;

  if (q < 0.) {
    return 0;
  }

  FF = splint_cu(q_Rayl_arr_d+Z*NQ_RAYL_MAX-1, FF_Rayl_arr_d+Z*NQ_RAYL_MAX-1, FF_Rayl_arr2_d+Z*NQ_RAYL_MAX-1,
	 Nq_Rayl_d[Z], q);

  return FF;
}

__device__ double SF_Compt_cu(int Z, double q) {
  double SF;

  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  if (q <= 0.) {
    return 0;
  }

  SF = splint_cu(q_Compt_arr_d+Z*NQ_COMPT_MAX-1, SF_Compt_arr_d+Z*NQ_COMPT_MAX-1, SF_Compt_arr2_d+Z*NQ_COMPT_MAX-1,
	 Nq_Compt_d[Z], q);

  return SF;
}

__device__ double DCS_Thoms_cu(double theta) { 
  double cos_theta;

  cos_theta = cos(theta);

  return (RE2/2.0) * (1.0 + cos_theta*cos_theta);
}

__device__ double DCS_KN_cu(double E, double theta) { 
  double cos_theta, t1, t2;

  if (E <= 0.) {
    return 0;
  }

  cos_theta = cos(theta);
  t1 = (1.0 - cos_theta) * E / MEC2 ;
  t2 = 1.0 + t1;
  
  return (RE2/2.) * (1.0 + cos_theta*cos_theta + t1*t1/t2) /t2 /t2;
}

__device__ double DCS_Rayl_cu(int Z, double E, double theta) { 
  double F, q ;                                                      
                                                        
  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  if (E <= 0.) {
    return 0;
  }

  q = MomentTransf_cu(E, theta);
  F = FF_Rayl_cu(Z, q);
  return  AVOGNUM / AtomicWeight_arr_d[Z] * F*F * DCS_Thoms_cu(theta);
}

__device__ double DCS_Compt_cu(int Z, double E, double theta) { 
  double S, q ;                                                      
                                                        
  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  if (E <= 0.) {
    return 0;
  }

  q = MomentTransf_cu(E, theta);
  S = SF_Compt_cu(Z, q);
  return  AVOGNUM / AtomicWeight_arr_d[Z] * S * DCS_KN_cu(E, theta);
}

__device__ double MomentTransf_cu(double E, double theta) {
  if (E <= 0.) {
    return 0;
  }
  
  return E / KEV2ANGST * sin(theta / 2.0) ;
}

__device__ double CS_KN_cu(double E) { 
  double a, a3, b, b2, lb;
  double sigma;

  if (E <= 0.) {
    return 0;
  }

  a = E / MEC2;
  a3 = a*a*a;
  b = 1 + 2*a;
  b2 = b*b;
  lb = log(b);

  sigma = 2*PI*RE2*( (1+a)/a3*(2*a*(1+a)/b-lb) + 0.5*lb/a - (1+3*a)/b2); 
  return sigma;
}

__device__ double ComptonEnergy_cu(double E0, double theta) { 
  double cos_theta, alpha;

  if (E0 <= 0.) {
    return 0;
  }

  cos_theta = cos(theta);
  alpha = E0/MEC2;

  return E0 / (1 + alpha*(1 - cos_theta));
}
