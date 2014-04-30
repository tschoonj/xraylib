/*
Copyright (c) 2014, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef XRAYLIB_CUDA_PRIVATE_H
#define XRAYLIB_CUDA_PRIVATE_H

#include "xraylib-defs.h"
#define NPZ 31

extern __device__ double AtomicLevelWidth_arr_d[(ZMAX+1)*SHELLNUM];
extern __device__ double AtomicWeight_arr_d[ZMAX+1];
extern __device__ double Auger_Rates_d[(ZMAX+1)*AUGERNUM];
extern __device__ double Auger_Yields_d[(ZMAX+1)*SHELLNUM_A];
extern __device__ int Npz_ComptonProfiles_d[ZMAX+1];
extern __device__ int NShells_ComptonProfiles_d[ZMAX+1];
extern __device__ double UOCCUP_ComptonProfiles_d[(ZMAX+1)*SHELLNUM_C];
extern __device__ double pz_ComptonProfiles_d[(ZMAX+1)*NPZ];
extern __device__ double Total_ComptonProfiles_d[(ZMAX+1)*NPZ];
extern __device__ double Total_ComptonProfiles2_d[(ZMAX+1)*NPZ];
extern __device__ double Partial_ComptonProfiles_d[(ZMAX+1)*SHELLNUM_C*NPZ];
extern __device__ double Partial_ComptonProfiles2_d[(ZMAX+1)*SHELLNUM_C*NPZ];


#endif