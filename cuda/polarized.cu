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

__device__ double DCSP_Rayl_cu(int Z, double E, double theta, double phi) {
  double F, q;                                                      
                                                        
  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  if (E <= 0.) {
    return 0;
  }

  q = MomentTransf_cu(E , theta);
  F = FF_Rayl_cu(Z, q);
  return  AVOGNUM / AtomicWeight_cu(Z) * F*F * DCSP_Thoms_cu(theta, phi);
}                                                                              

__device__ double  DCSP_Compt_cu(int Z, double E, double theta, double phi) { 
  double S, q;                                                      
                                                        
  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  if (E <= 0.) {
    return 0;
  }

  q = MomentTransf_cu(E, theta);
  S = SF_Compt_cu(Z, q);
  return  AVOGNUM / AtomicWeight_cu(Z) * S * DCSP_KN_cu(E, theta, phi);
}

__device__ double DCSP_KN_cu(double E, double theta, double phi) { 
  double k0_k, k_k0, k_k0_2, cos_th, sin_th, cos_phi;
  
  if (E <= 0.) {
    return 0;
  }

  cos_th = cos(theta);
  sin_th = sin(theta);
  cos_phi = cos(phi);
  
  k0_k = 1.0 + (1.0 - cos_th) * E / MEC2 ;
  k_k0 = 1.0 / k0_k;
  k_k0_2 = k_k0 * k_k0;
  
  return (RE2/2.) * k_k0_2 * (k_k0 + k0_k - 2 * sin_th * sin_th 
			      * cos_phi * cos_phi);
} 

__device__ double  DCSP_Thoms_cu(double theta, double phi) {
  double sin_th, cos_phi ;

  sin_th = sin(theta) ;
  cos_phi = cos(phi);
  return RE2 * (1.0 - sin_th * sin_th * cos_phi * cos_phi);
}



