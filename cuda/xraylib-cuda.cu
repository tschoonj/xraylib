/*
Copyright (c) 2014, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "xraylib-cuda.h"
#include "xraylib.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "xrayglob.h"


#define KL1 -KL1_LINE-1
#define KL2 -KL2_LINE-1
#define KL3 -KL3_LINE-1
#define KM1 -KM1_LINE-1
#define KM2 -KM2_LINE-1
#define KM3 -KM3_LINE-1
#define KP5 -KP5_LINE-1

__device__ double FluorYield_arr_d[(ZMAX+1)*SHELLNUM];
__device__ double AtomicWeight_arr_d[ZMAX+1];
__device__ double EdgeEnergy_arr_d[(ZMAX+1)*SHELLNUM];
__device__ double LineEnergy_arr_d[(ZMAX+1)*LINENUM];
__device__ double JumpFactor_arr_d[(ZMAX+1)*SHELLNUM];
__device__ double CosKron_arr_d[(ZMAX+1)*TRANSNUM];
__device__ double RadRate_arr_d[(ZMAX+1)*LINENUM];
__device__ double AtomicLevelWidth_arr_d[(ZMAX+1)*SHELLNUM];

__device__ int NE_Photo_d[ZMAX+1];
__device__ double E_Photo_arr_d[(ZMAX+1)*91];
__device__ double CS_Photo_arr_d[(ZMAX+1)*91];
__device__ double CS_Photo_arr2_d[(ZMAX+1)*91];


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

__device__ double CS_Photo_cu(int Z, double E)
{
  double ln_E, ln_sigma, sigma;


  if (Z<1 || Z>ZMAX || NE_Photo_d[Z]<0) {
    return 0;
  }

  if (E <= 0.) {
    return 0;
  }

  ln_E = log(E * 1000.0);


  ln_sigma = splint_cu(E_Photo_arr_d+(Z*91)-1, CS_Photo_arr_d+(Z*91)-1, CS_Photo_arr2_d+(Z*91)-1,
	 NE_Photo_d[Z], ln_E);

  sigma = exp(ln_sigma);

  return sigma;
}

__device__ double  FluorYield_cu(int Z, int shell) {
  double fluor_yield;


  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  if (shell<0 || shell>=SHELLNUM) {
    return 0;
  }

  fluor_yield = FluorYield_arr_d[Z*SHELLNUM+shell];
  if (fluor_yield < 0.) {
    return 0;
  }

  return fluor_yield;
}

__device__ double AtomicWeight_cu(int Z) {
  double atomic_weight;

  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  atomic_weight = AtomicWeight_arr_d[Z];
  if (atomic_weight < 0.) {
    return 0;
  }
  return atomic_weight;
}


__device__ double EdgeEnergy_cu(int Z, int shell) {
  double edge_energy;

  if (Z<1 || Z>ZMAX) {
    return 0;
  }
  if (shell<0 || shell>=SHELLNUM) {
    return 0;
  }
  edge_energy = EdgeEnergy_arr_d[Z*SHELLNUM+shell];

  if (edge_energy < 0.) {
    return 0;
  }

  return edge_energy;
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


__device__ double JumpFactor_cu(int Z, int shell) {
  double jump_factor;

  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  if (shell<0 || shell>=SHELLNUM) {
    return 0;
  }

  jump_factor = JumpFactor_arr_d[Z*SHELLNUM+shell];
  if (jump_factor < 0.) {
    return 0;
  }

  return jump_factor;
}

__device__ double CosKronTransProb_cu(int Z, int trans) {
  double trans_prob;

  if (Z<1 || Z>ZMAX){
    return 0;
  }

  if (trans<0 || trans>=TRANSNUM) {
    return 0;
  }

  trans_prob = CosKron_arr_d[Z*TRANSNUM+trans];
  if (trans_prob < 0.) {
    return 0;
  }

  return trans_prob;
}

__device__ double RadRate_cu(int Z, int line) {
  double rad_rate, rr;
  int i;

  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  if (line>=KA_LINE && line<LA_LINE) {
    if (line == KA_LINE) {
        rr=0.0;
    	for (i=KL1 ; i <= KL3 ; i++)
		rr += RadRate_arr_d[Z*LINENUM+i];
    }
    else if (line == KB_LINE) {
        rr=0.0;
    	for (i=KL1 ; i <= KL3 ; i++)
		rr += RadRate_arr_d[Z*LINENUM+i];
    	/*
	 * we assume that RR(Ka)+RR(Kb) = 1.0
	 */
    	return 1.0 - rr;
    }
    if (rr == 0.0 || rr == 1.0) {
      return 0.0;
    }
    return rr;
  }

  if (line == LA_LINE) {
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

__device__ double AtomicLevelWidth_cu(int Z, int shell) {
  double atomic_level_width;

  if (Z<1 || Z>ZMAX) {
    return 0;
  }
  if (shell<0 || shell>=SHELLNUM) {
    return 0;
  }
  atomic_level_width = AtomicLevelWidth_arr_d[Z*SHELLNUM+shell];

  if (atomic_level_width < 0.) {
    return 0;
  }

  return atomic_level_width;
}








int CudaXRayInit() {



	int deviceCount, device;
	int gpuDeviceCount = 0;
	cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	int Z;
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
  	CudaSafeCall(cudaMemcpyToSymbol( FluorYield_arr_d, FluorYield_arr, sizeof(double)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( AtomicWeight_arr_d, AtomicWeight_arr, sizeof(double)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( EdgeEnergy_arr_d, EdgeEnergy_arr, sizeof(double)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( LineEnergy_arr_d, LineEnergy_arr, sizeof(double)*(ZMAX+1)*LINENUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( JumpFactor_arr_d, JumpFactor_arr, sizeof(double)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( CosKron_arr_d, CosKron_arr, sizeof(double)*(ZMAX+1)*TRANSNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( RadRate_arr_d, RadRate_arr, sizeof(double)*(ZMAX+1)*LINENUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( AtomicLevelWidth_arr_d, AtomicLevelWidth_arr, sizeof(double)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( NE_Photo_d, NE_Photo, sizeof(int)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
	for (Z = 1; Z <= ZMAX; Z++) {
		if (NE_Photo[Z] > 0) {
			CudaSafeCall(cudaMemcpyToSymbol(E_Photo_arr_d, E_Photo_arr[Z], sizeof(double)*NE_Photo[Z], (size_t) Z*91*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(CS_Photo_arr_d, CS_Photo_arr[Z], sizeof(double)*NE_Photo[Z], (size_t) Z*91*sizeof(double), cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpyToSymbol(CS_Photo_arr2_d, CS_Photo_arr2[Z], sizeof(double)*NE_Photo[Z], (size_t) Z*91*sizeof(double), cudaMemcpyHostToDevice));
		}
	}

	//cudaThreadSynchronize();

	



	return 1;
}



