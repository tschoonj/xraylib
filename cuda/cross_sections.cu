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

__device__ int NE_Photo_d[ZMAX+1];
__device__ double E_Photo_arr_d[(ZMAX+1)*NE_PHOTO_MAX];
__device__ double CS_Photo_arr_d[(ZMAX+1)*NE_PHOTO_MAX];
__device__ double CS_Photo_arr2_d[(ZMAX+1)*NE_PHOTO_MAX];

__device__ int NE_Rayl_d[ZMAX+1];
__device__ double E_Rayl_arr_d[(ZMAX+1)*NE_RAYL_MAX];
__device__ double CS_Rayl_arr_d[(ZMAX+1)*NE_RAYL_MAX];
__device__ double CS_Rayl_arr2_d[(ZMAX+1)*NE_RAYL_MAX];

__device__ int NE_Compt_d[ZMAX+1];
__device__ double E_Compt_arr_d[(ZMAX+1)*NE_COMPT_MAX];
__device__ double CS_Compt_arr_d[(ZMAX+1)*NE_COMPT_MAX];
__device__ double CS_Compt_arr2_d[(ZMAX+1)*NE_COMPT_MAX];

__device__ int NE_Energy_d[ZMAX+1];
__device__ double E_Energy_arr_d[(ZMAX+1)*NE_ENERGY_MAX];
__device__ double CS_Energy_arr_d[(ZMAX+1)*NE_ENERGY_MAX];
__device__ double CS_Energy_arr2_d[(ZMAX+1)*NE_ENERGY_MAX];

__device__ double CS_Total_cu(int Z, double E) {
  if (Z<1 || Z>ZMAX || NE_Photo_d[Z]<0 || NE_Rayl_d[Z]<0 || NE_Compt_d[Z]<0) {
    return 0;
  }

  if (E <= 0.) {
    return 0;
  }

  return CS_Photo_cu(Z, E) + CS_Rayl_cu(Z, E) + CS_Compt_cu(Z, E);
}

__device__ double CS_Photo_cu(int Z, double E) {
  double ln_E, ln_sigma, sigma;

  if (Z<1 || Z>ZMAX || NE_Photo_d[Z]<0) {
    return 0;
  }

  if (E <= 0.) {
    return 0;
  }

  ln_E = log(E * 1000.0);

  ln_sigma = splint_cu(E_Photo_arr_d+Z*NE_PHOTO_MAX-1, CS_Photo_arr_d+Z*NE_PHOTO_MAX-1, CS_Photo_arr2_d+Z*NE_PHOTO_MAX-1,
	 NE_Photo_d[Z], ln_E);

  sigma = exp(ln_sigma);

  return sigma;
}

__device__ double CS_Rayl_cu(int Z, double E) {
  double ln_E, ln_sigma, sigma;

  if (Z<1 || Z>ZMAX || NE_Rayl_d[Z]<0) {
    return 0;
  }

  if (E <= 0.) {
    return 0;
  }

  ln_E = log(E * 1000.0);

  ln_sigma = splint_cu(E_Rayl_arr_d+Z*NE_RAYL_MAX-1, CS_Rayl_arr_d+Z*NE_RAYL_MAX-1, CS_Rayl_arr2_d+Z*NE_RAYL_MAX-1,
	 NE_Rayl_d[Z], ln_E);
  sigma = exp(ln_sigma);

  return sigma;
}

__device__ double CS_Compt_cu(int Z, double E) {
  double ln_E, ln_sigma, sigma;

  if (Z<1 || Z>ZMAX || NE_Compt_d[Z]<0) {
    return 0;
  }

  if (E <= 0.) {
    return 0;
  }

  ln_E = log(E * 1000.0);

  ln_sigma = splint_cu(E_Compt_arr_d+Z*NE_COMPT_MAX-1, CS_Compt_arr_d+Z*NE_COMPT_MAX-1, CS_Compt_arr2_d+Z*NE_COMPT_MAX-1,
	 NE_Compt_d[Z], ln_E);

  sigma = exp(ln_sigma);

  return sigma;
}

__device__ double CS_Energy_cu(int Z, double E) {
	double ln_E, ln_sigma, sigma;
	if (Z < 1 || Z > 92 || NE_Energy_d[Z] < 0) {
		return 0;
	}
	if (E <= 0.0) {
		return 0;
	}
	ln_E = log(E);
	ln_sigma = splint_cu(E_Energy_arr_d+Z*NE_ENERGY_MAX-1, CS_Energy_arr_d+Z*NE_ENERGY_MAX-1, CS_Energy_arr2_d+Z*NE_ENERGY_MAX-1, NE_Energy_d[Z], ln_E);

	sigma = exp(ln_sigma);

	return sigma;
}


