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

#ifndef XRAYLIB_CUDA_H
#define XRAYLIB_CUDA_H

#include <stdio.h>



#ifdef __cplusplus
extern "C" {
#endif




//host functions

/*
 *
 * Initializes cuda xraylib
 * Copies all the relevant datasets to the GPU device memory
 *
 */
int CudaXRayInit();


/*
 * device functions
 */
__device__ double AtomicLevelWidth_cu(int Z, int shell);
__device__ double AtomicWeight_cu(int Z);
__device__ double AugerRate_cu(int Z, int auger_trans);
__device__ double AugerYield_cu(int Z, int shell);
__device__ double ComptonProfile_cu(int Z, double pz);
__device__ double ComptonProfile_Partial_cu(int Z, int shell, double pz);
__device__ double CS_Total_cu(int Z, double E);
__device__ double CS_Photo_cu(int Z, double E);
__device__ double CS_Rayl_cu(int Z, double E);
__device__ double CS_Compt_cu(int Z, double E);
__device__ double CS_Energy_cu(int Z, double E);


__device__ double splint_cu(double *xa, double *ya, double *y2a, int n, double x);
__device__ double CS_Photo_cu(int Z, double E);
__device__ double  FluorYield_cu(int Z, int shell);
__device__ double EdgeEnergy_cu(int Z, int shell);
//__device__ double LineEnergy_cu(int Z, int line);
__device__ double JumpFactor_cu(int Z, int shell);
__device__ double CosKronTransProb_cu(int Z, int trans);
__device__ double RadRate_cu(int Z, int line);










#ifdef __cplusplus
}
#endif
#endif
