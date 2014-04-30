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

__device__ double LineEnergy_arr_d[(ZMAX+1)*LINENUM];


#define KL1 -(int)KL1_LINE-1
#define KL2 -(int)KL2_LINE-1
#define KL3 -(int)KL3_LINE-1
#define KM1 -(int)KM1_LINE-1
#define KM2 -(int)KM2_LINE-1
#define KM3 -(int)KM3_LINE-1
#define KP5 -(int)KP5_LINE-1

//device functions

__device__ double splint_cu(double *xa, double *ya, double *y2a, int n, double x)
{
	int klo, khi, k;
	double h, b, a, y;

	if (x >= xa[n]) {
	  y = ya[n];
	  return y;
	}

	if (x <= xa[1]) {
	  y = ya[1];
	  return y;
	}

	klo = 1;
	khi = n;
	while (khi-klo > 1) {
		k = (khi + klo) >> 1;
		if (xa[k] > x) khi = k;
		else klo = k;
	}

	h = xa[khi] - xa[klo];
	if (h == 0.0) {
	  y = (ya[klo] + ya[khi])/2.0;
	  return y;
	}
	a = (xa[khi] - x) / h;
	b = (x - xa[klo]) / h;
	y = a*ya[klo] + b*ya[khi] + ((a*a*a-a)*y2a[klo]
	     + (b*b*b-b)*y2a[khi])*(h*h)/6.0;
	return y;
}

//__device__ double LineEnergy_cu(int Z, int line) {
//  double line_energy;
//  double lE[50],rr[50];
//  double tmp=0.0,tmp1=0.0,tmp2=0.0;
//  int i;
//  int temp_line;
//  
//  if (Z<1 || Z>ZMAX) {
//    ErrorExit("Z out of range in function LineEnergy");
//    return 0;
//  }
//  
//  if (line>=KA_LINE && line<LA_LINE) {
//    if (line == KA_LINE) {
//	for (i = KL1; i <= KL3 ; i++) {
//	 lE[i] = LineEnergy_arr_d[Z*LINENUM+i];
//	 rr[i] = RadRate_arr_d[Z*LINENUM+i];
//	 tmp1+=rr[i];
//	 tmp+=lE[i]*rr[i];

//	 if (lE[i]<0.0 || rr[i]<0.0) {
//	  ErrorExit("Line not available in function LineEnergy");
//	  return 0;
//	 }
//	}
//    }
//    else if (line == KB_LINE) {
//    	for (i = KM1; i < KP5; i++) {
//	 lE[i] = LineEnergy_arr_d[Z*LINENUM+i];
//	 rr[i] = RadRate_arr_d[Z*LINENUM+i];
//	 tmp1+=rr[i];
//	 tmp+=lE[i]*rr[i];
//	 if (lE[i]<0.0 || rr[i]<0.0) {
//	  ErrorExit("Line not available in function LineEnergy");
//	  return 0;
//	 }
//	}
//    }
//   if (tmp1>0)   return tmp/tmp1;  else return 0.0;
//  }
//  
//  if (line == LA_LINE) {
//	temp_line = L3M5_LINE;
//	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
//	tmp2=tmp1;
//	tmp=LineEnergy_cu(Z,temp_line)*tmp1;
//	temp_line = L3M4_LINE;
//	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
//	tmp2+=tmp1;
//	tmp+=LineEnergy_cu(Z,temp_line)*tmp1 ;
//  	if (tmp2>0)   return tmp/tmp2;  else return 0.0;
//  }
//  else if (line == LB_LINE) {
//	temp_line = L2M4_LINE;     /* b1 */
//	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L2_SHELL)+0.1);
//	tmp2=tmp1;
//	tmp=LineEnergy_cu(Z,temp_line)*tmp1;

//	temp_line = L3N5_LINE;     /* b2 */
//	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
//	tmp2+=tmp1;
//	tmp+=LineEnergy_cu(Z,temp_line)*tmp1 ;

//	temp_line = L1M3_LINE;     /* b3 */
//	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L1_SHELL)+0.1);
//	tmp2+=tmp1;
//	tmp+=LineEnergy_cu(Z,temp_line)*tmp1 ;

//	temp_line = L1M2_LINE;     /* b4 */
//	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L1_SHELL)+0.1);
//	tmp2+=tmp1;
//	tmp+=LineEnergy_cu(Z,temp_line)*tmp1 ;

//	temp_line = L3O3_LINE;     /* b5 */
//	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
//	tmp2+=tmp1;
//	tmp+=LineEnergy_cu(Z,temp_line)*tmp1 ;

//	temp_line = L3O4_LINE;     /* b5 */
//	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
//	tmp2+=tmp1;
//	tmp+=LineEnergy_cu(Z,temp_line)*tmp1 ;

//	temp_line = L3N1_LINE;     /* b6 */
//	tmp1=CS_FluorLine_cu(Z, temp_line,EdgeEnergy_cu(Z,L3_SHELL)+0.1);
//	tmp2+=tmp1;
//	tmp+=LineEnergy_cu(Z,temp_line)*tmp1 ;
//  	if (tmp2>0)   return tmp/tmp2;  else return 0.0;
//  }
//  /*
//   * special cases for composed lines
//   */
//  else if (line == L1N67_LINE) {
// 	return (LineEnergy_cu(Z, L1N6_LINE)+LineEnergy_cu(Z,L1N7_LINE))/2.0; 
//  }
//  else if (line == L1O45_LINE) {
// 	return (LineEnergy_cu(Z, L1O4_LINE)+LineEnergy_cu(Z,L1O5_LINE))/2.0; 
//  }
//  else if (line == L1P23_LINE) {
// 	return (LineEnergy_cu(Z, L1P2_LINE)+LineEnergy_cu(Z,L1P3_LINE))/2.0; 
//  }
//  else if (line == L2P23_LINE) {
// 	return (LineEnergy_cu(Z, L2P2_LINE)+LineEnergy_cu(Z,L2P3_LINE))/2.0; 
//  }
//  else if (line == L3O45_LINE) {
// 	return (LineEnergy_cu(Z, L3O4_LINE)+LineEnergy_cu(Z,L3O5_LINE))/2.0; 
//  }
//  else if (line == L3P23_LINE) {
// 	return (LineEnergy_cu(Z, L3P2_LINE)+LineEnergy_cu(Z,L3P3_LINE))/2.0; 
//  }
//  else if (line == L3P45_LINE) {
// 	return (LineEnergy_cu(Z, L3P4_LINE)+LineEnergy_cu(Z,L3P5_LINE))/2.0; 
//  }
//  
//  line = -line - 1;
//  if (line<0 || line>=LINENUM) {
//    return 0;
//  }
//  
//  line_energy = LineEnergy_arr_d[Z*LINENUM+line];
//  if (line_energy < 0.) {
//    return 0;
//  }
//  return line_energy;
//}





int CudaXRayInit() {



	int deviceCount, device;
	int gpuDeviceCount = 0;
	cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	int Z, shell;
	if (cudaResultCode != cudaSuccess) 
        	deviceCount = 0;
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
	}


  	CudaSafeCall(cudaMemcpyToSymbol(LineEnergy_arr_d, LineEnergy_arr, sizeof(double)*(ZMAX+1)*LINENUM, (size_t) 0,cudaMemcpyHostToDevice));


	



	return 1;
}



