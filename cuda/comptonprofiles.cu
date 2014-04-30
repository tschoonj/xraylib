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

#include "xraylib-cuda.h"
#include "xraylib-cuda-private.h"
#include "xraylib-defs.h"


__device__ int Npz_ComptonProfiles_d[ZMAX+1];
__device__ int NShells_ComptonProfiles_d[ZMAX+1];
__device__ double UOCCUP_ComptonProfiles_d[(ZMAX+1)*SHELLNUM_C];
__device__ double pz_ComptonProfiles_d[(ZMAX+1)*NPZ];
__device__ double Total_ComptonProfiles_d[(ZMAX+1)*NPZ];
__device__ double Total_ComptonProfiles2_d[(ZMAX+1)*NPZ];
__device__ double Partial_ComptonProfiles_d[(ZMAX+1)*SHELLNUM_C*NPZ];
__device__ double Partial_ComptonProfiles2_d[(ZMAX+1)*SHELLNUM_C*NPZ];

__device__ double ComptonProfile_cu(int Z, double pz) {
        double q, ln_q;
        double ln_pz;

        if (Z < 1 || Z > ZMAX || NShells_ComptonProfiles_d[Z] < 0) {
                return 0;
        }

        if (pz < 0.0) {
                return 0;
        }

        ln_pz = log(pz + 1.0);

        ln_q = splint_cu(pz_ComptonProfiles_d+Z*NPZ-1, Total_ComptonProfiles_d+Z*NPZ-1, Total_ComptonProfiles2_d+Z*NPZ-1,  Npz_ComptonProfiles_d[Z],ln_pz);

        q = exp(ln_q);

        return q;
}

__device__ double ComptonProfile_Partial_cu(int Z, int shell, double pz) {
        double q, ln_q;
        double ln_pz;


        if (Z < 1 || Z > ZMAX || NShells_ComptonProfiles_d[Z] < 0) {
                return 0;
        }
        if (shell >= NShells_ComptonProfiles_d[Z] || UOCCUP_ComptonProfiles_d[Z*SHELLNUM_C+shell] == 0.0 ) {
                return 0;
        }

        ln_pz = log(pz + 1.0);

        ln_q = splint_cu(pz_ComptonProfiles_d+Z*NPZ-1, Partial_ComptonProfiles_d+NPZ*(SHELLNUM_C*Z+shell)-1,Partial_ComptonProfiles2_d+NPZ*(SHELLNUM_C*Z+shell)-1, Npz_ComptonProfiles_d[Z],ln_pz);

        q = exp(ln_q);

        return q;
}
