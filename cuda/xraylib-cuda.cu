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


#define CUDA_ERROR_CHECK
#include "xraylib-cuda.h"
#include "xraylib-cuda-private.h"
#include "xraylib.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "xrayglob.h"



int CudaXRayInit() {
	int deviceCount, device;
	int gpuDeviceCount = 0;
	cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	int Z, shell;
	if (cudaResultCode != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount returned error\n");
        	deviceCount = 0;
	}
   	/* machines with no GPUs can still report one emulation device */
	for (device = 0; device < deviceCount; ++device) {
        	cudaGetDeviceProperties(&properties, device);
        	if (properties.major != 9999) /* 9999 means emulation only */
            		++gpuDeviceCount;
    	}

    	/* don't just return the number of gpus, because other runtime cuda
       	errors can also yield non-zero return values */
    	if (gpuDeviceCount == 0) {
		fprintf(stderr,"No CUDA enabled devices found\nAborting\n");
        	return 0;
	}



	/* start memcpy'ing */
  	CudaSafeCall(cudaMemcpyToSymbol(AtomicLevelWidth_arr_d, AtomicLevelWidth_arr, sizeof(double)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(AtomicWeight_arr_d, AtomicWeight_arr, sizeof(double)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(Auger_Rates_d, Auger_Rates, sizeof(double)*(ZMAX+1)*AUGERNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(Auger_Yields_d, Auger_Yields, sizeof(double)*(ZMAX+1)*SHELLNUM_A, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(Npz_ComptonProfiles_d, Npz_ComptonProfiles, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(NShells_ComptonProfiles_d, NShells_ComptonProfiles, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(CosKron_arr_d, CosKron_arr, sizeof(double)*(ZMAX+1)*TRANSNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(NE_Photo_d, NE_Photo, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(NE_Rayl_d, NE_Rayl, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(NE_Compt_d, NE_Compt, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(NE_Energy_d, NE_Energy, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(ElementDensity_arr_d, ElementDensity_arr, sizeof(double)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(EdgeEnergy_arr_d, EdgeEnergy_arr, sizeof(double)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(NE_Fi_d, NE_Fi, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(NE_Fii_d, NE_Fii, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(FluorYield_arr_d, FluorYield_arr, sizeof(double)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(JumpFactor_arr_d, JumpFactor_arr, sizeof(double)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(RadRate_arr_d, RadRate_arr, sizeof(double)*(ZMAX+1)*LINENUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(Nq_Rayl_d, Nq_Rayl, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(Nq_Compt_d, Nq_Compt, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(NE_Photo_Total_Kissel_d, NE_Photo_Total_Kissel, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(Electron_Config_Kissel_d, Electron_Config_Kissel, sizeof(double)*(ZMAX+1)*SHELLNUM_K, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(EdgeEnergy_Kissel_d, EdgeEnergy_Kissel, sizeof(double)*(ZMAX+1)*SHELLNUM_K, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(NE_Photo_Partial_Kissel_d, NE_Photo_Partial_Kissel, sizeof(int)*(ZMAX+1)*SHELLNUM_K, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol(LineEnergy_arr_d, LineEnergy_arr, sizeof(double)*(ZMAX+1)*LINENUM, (size_t) 0,cudaMemcpyHostToDevice));


	for (Z = 1; Z <= ZMAX; Z++) {
		if (NE_Photo[Z] > 0) {
			CudaSafeCall(cudaMemcpyToSymbol(E_Photo_arr_d, E_Photo_arr[Z], sizeof(double)*NE_Photo[Z], (size_t) Z*NE_PHOTO_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(CS_Photo_arr_d, CS_Photo_arr[Z], sizeof(double)*NE_Photo[Z], (size_t) Z*NE_PHOTO_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(CS_Photo_arr2_d, CS_Photo_arr2[Z], sizeof(double)*NE_Photo[Z], (size_t) Z*NE_PHOTO_MAX*sizeof(double), cudaMemcpyHostToDevice));
		}
		if (NE_Rayl[Z] > 0) {
			CudaSafeCall(cudaMemcpyToSymbol(E_Rayl_arr_d, E_Rayl_arr[Z], sizeof(double)*NE_Rayl[Z], (size_t) Z*NE_RAYL_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(CS_Rayl_arr_d, CS_Rayl_arr[Z], sizeof(double)*NE_Rayl[Z], (size_t) Z*NE_RAYL_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(CS_Rayl_arr2_d, CS_Rayl_arr2[Z], sizeof(double)*NE_Rayl[Z], (size_t) Z*NE_RAYL_MAX*sizeof(double), cudaMemcpyHostToDevice));
		}
		if (NE_Compt[Z] > 0) {
			CudaSafeCall(cudaMemcpyToSymbol(E_Compt_arr_d, E_Compt_arr[Z], sizeof(double)*NE_Compt[Z], (size_t) Z*NE_COMPT_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(CS_Compt_arr_d, CS_Compt_arr[Z], sizeof(double)*NE_Compt[Z], (size_t) Z*NE_COMPT_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(CS_Compt_arr2_d, CS_Compt_arr2[Z], sizeof(double)*NE_Compt[Z], (size_t) Z*NE_COMPT_MAX*sizeof(double), cudaMemcpyHostToDevice));
		}
		if (NE_Energy[Z] > 0) {
			CudaSafeCall(cudaMemcpyToSymbol(E_Energy_arr_d, E_Energy_arr[Z], sizeof(double)*NE_Energy[Z], (size_t) Z*NE_ENERGY_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(CS_Energy_arr_d, CS_Energy_arr[Z], sizeof(double)*NE_Energy[Z], (size_t) Z*NE_ENERGY_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(CS_Energy_arr2_d, CS_Energy_arr2[Z], sizeof(double)*NE_Energy[Z], (size_t) Z*NE_ENERGY_MAX*sizeof(double), cudaMemcpyHostToDevice));
		}
		if (NE_Fi[Z] > 0) {
			CudaSafeCall(cudaMemcpyToSymbol(E_Fi_arr_d, E_Fi_arr[Z], sizeof(double)*NE_Fi[Z], (size_t) Z*NE_FI_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(Fi_arr_d, Fi_arr[Z], sizeof(double)*NE_Fi[Z], (size_t) Z*NE_FI_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(Fi_arr2_d, Fi_arr2[Z], sizeof(double)*NE_Fi[Z], (size_t) Z*NE_FI_MAX*sizeof(double), cudaMemcpyHostToDevice));
		}
		if (NE_Fii[Z] > 0) {
			CudaSafeCall(cudaMemcpyToSymbol(E_Fii_arr_d, E_Fii_arr[Z], sizeof(double)*NE_Fii[Z], (size_t) Z*NE_FII_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(Fii_arr_d, Fii_arr[Z], sizeof(double)*NE_Fii[Z], (size_t) Z*NE_FII_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(Fii_arr2_d, Fii_arr2[Z], sizeof(double)*NE_Fii[Z], (size_t) Z*NE_FII_MAX*sizeof(double), cudaMemcpyHostToDevice));
		}
		if (Nq_Rayl[Z] > 0) {
			CudaSafeCall(cudaMemcpyToSymbol(q_Rayl_arr_d, q_Rayl_arr[Z], sizeof(double)*Nq_Rayl[Z], (size_t) Z*NQ_RAYL_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(FF_Rayl_arr_d, FF_Rayl_arr[Z], sizeof(double)*Nq_Rayl[Z], (size_t) Z*NQ_RAYL_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(FF_Rayl_arr2_d, FF_Rayl_arr2[Z], sizeof(double)*Nq_Rayl[Z], (size_t) Z*NQ_RAYL_MAX*sizeof(double), cudaMemcpyHostToDevice));
		}
		if (Nq_Compt[Z] > 0) {
			CudaSafeCall(cudaMemcpyToSymbol(q_Compt_arr_d, q_Compt_arr[Z], sizeof(double)*Nq_Compt[Z], (size_t) Z*NQ_COMPT_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(SF_Compt_arr_d, SF_Compt_arr[Z], sizeof(double)*Nq_Compt[Z], (size_t) Z*NQ_COMPT_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(SF_Compt_arr2_d, SF_Compt_arr2[Z], sizeof(double)*Nq_Compt[Z], (size_t) Z*NQ_COMPT_MAX*sizeof(double), cudaMemcpyHostToDevice));
		}
		if (Npz_ComptonProfiles[Z] > 0) {
			CudaSafeCall(cudaMemcpyToSymbol(pz_ComptonProfiles_d, pz_ComptonProfiles[Z], sizeof(double)*Npz_ComptonProfiles[Z], (size_t) Z*NPZ*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(Total_ComptonProfiles_d, Total_ComptonProfiles[Z], sizeof(double)*Npz_ComptonProfiles[Z], (size_t) Z*NPZ*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(Total_ComptonProfiles2_d, Total_ComptonProfiles2[Z], sizeof(double)*Npz_ComptonProfiles[Z], (size_t) Z*NPZ*sizeof(double), cudaMemcpyHostToDevice));
			for (shell = K_SHELL ; shell < NShells_ComptonProfiles[Z] ; shell++) {
				if (UOCCUP_ComptonProfiles[Z][shell] > 0.0) {
					CudaSafeCall(cudaMemcpyToSymbol(Partial_ComptonProfiles_d, Partial_ComptonProfiles[Z][shell], sizeof(double)*Npz_ComptonProfiles[Z], (size_t) NPZ*(SHELLNUM_C*Z+shell)*sizeof(double), cudaMemcpyHostToDevice));
					CudaSafeCall(cudaMemcpyToSymbol(Partial_ComptonProfiles2_d, Partial_ComptonProfiles2[Z][shell], sizeof(double)*Npz_ComptonProfiles[Z], (size_t) NPZ*(SHELLNUM_C*Z+shell)*sizeof(double), cudaMemcpyHostToDevice));

				}
			}
			if (NShells_ComptonProfiles[Z] > 0.0) {
				CudaSafeCall(cudaMemcpyToSymbol(UOCCUP_ComptonProfiles_d, UOCCUP_ComptonProfiles[Z], sizeof(double)*NShells_ComptonProfiles[Z], (size_t) Z*SHELLNUM_C*sizeof(double), cudaMemcpyHostToDevice));
			}
		}
		if (NE_Photo_Total_Kissel[Z] > 0) {
			CudaSafeCall(cudaMemcpyToSymbol(E_Photo_Total_Kissel_d, E_Photo_Total_Kissel[Z], sizeof(double)*NE_Photo_Total_Kissel[Z], (size_t) Z*NE_PHOTO_TOTAL_KISSEL_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(Photo_Total_Kissel_d, Photo_Total_Kissel[Z], sizeof(double)*NE_Photo_Total_Kissel[Z], (size_t) Z*NE_PHOTO_TOTAL_KISSEL_MAX*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(Photo_Total_Kissel2_d, Photo_Total_Kissel2[Z], sizeof(double)*NE_Photo_Total_Kissel[Z], (size_t) Z*NE_PHOTO_TOTAL_KISSEL_MAX*sizeof(double), cudaMemcpyHostToDevice));
			for (shell = K_SHELL ; shell < SHELLNUM_K ; shell++) {
				if (NE_Photo_Partial_Kissel[Z][shell] > 0.0) {
        				int offset = NE_PHOTO_PARTIAL_KISSEL_MAX*(Z*SHELLNUM_K+shell);
					CudaSafeCall(cudaMemcpyToSymbol(E_Photo_Partial_Kissel_d, E_Photo_Partial_Kissel[Z][shell], sizeof(double)*NE_Photo_Partial_Kissel[Z][shell], (size_t) offset*sizeof(double), cudaMemcpyHostToDevice));
					CudaSafeCall(cudaMemcpyToSymbol(Photo_Partial_Kissel_d, Photo_Partial_Kissel[Z][shell], sizeof(double)*NE_Photo_Partial_Kissel[Z][shell], (size_t) offset*sizeof(double), cudaMemcpyHostToDevice));
					CudaSafeCall(cudaMemcpyToSymbol(Photo_Partial_Kissel2_d, Photo_Partial_Kissel2[Z][shell], sizeof(double)*NE_Photo_Partial_Kissel[Z][shell], (size_t) offset*sizeof(double), cudaMemcpyHostToDevice));
					
				}
			}
		}
	}




	



	return 1;
}



