
#ifndef XRAYLIB_CUDA_H
#define XRAYLIB_CUDA_H


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

__device__ float FluorYield_arr_d[(ZMAX+1)*SHELLNUM];
__device__ float AtomicWeight_arr_d[ZMAX+1];
__device__ float EdgeEnergy_arr_d[(ZMAX+1)*SHELLNUM];
__device__ float LineEnergy_arr_d[(ZMAX+1)*LINENUM];
__device__ float JumpFactor_arr_d[(ZMAX+1)*SHELLNUM];
__device__ float CosKron_arr_d[(ZMAX+1)*TRANSNUM];
__device__ float RadRate_arr_d[(ZMAX+1)*LINENUM];
__device__ float AtomicLevelWidth_arr_d[(ZMAX+1)*SHELLNUM];

//__device__ float *FluorYield_arr_d;
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


int CudaXRayFree();


//device functions

__device__ float  FluorYield_cu(int Z, int shell) {
  float fluor_yield;


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

__device__ float AtomicWeight_cu(int Z) {
  float atomic_weight;

  if (Z<1 || Z>ZMAX) {
    return 0;
  }

  atomic_weight = AtomicWeight_arr_d[Z];
  if (atomic_weight < 0.) {
    return 0;
  }
  return atomic_weight;
}


__device__ float EdgeEnergy_cu(int Z, int shell) {
  float edge_energy;

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

//__device__ float LineEnergy_cu(int Z, int line) {
//  float line_energy;
//  float lE[50],rr[50];
//  float tmp=0.0,tmp1=0.0,tmp2=0.0;
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


__device__ float JumpFactor_cu(int Z, int shell) {
  float jump_factor;

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

__device__ float CosKronTransProb_cu(int Z, int trans) {
  float trans_prob;

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

__device__ float RadRate_cu(int Z, int line) {
  float rad_rate, rr;
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

__device__ float AtomicLevelWidth_cu(int Z, int shell) {
  float atomic_level_width;

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



#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}





int CudaXRayInit() {



	int deviceCount, device;
	int gpuDeviceCount = 0;
	cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
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



	/* start malloc'ing and memcpy'ing */
  	CudaSafeCall(cudaMemcpyToSymbol( FluorYield_arr_d, FluorYield_arr, sizeof(float)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( AtomicWeight_arr_d, AtomicWeight_arr, sizeof(float)*(ZMAX+1), (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( EdgeEnergy_arr_d, EdgeEnergy_arr, sizeof(float)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( LineEnergy_arr_d, LineEnergy_arr, sizeof(float)*(ZMAX+1)*LINENUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( JumpFactor_arr_d, JumpFactor_arr, sizeof(float)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( CosKron_arr_d, CosKron_arr, sizeof(float)*(ZMAX+1)*TRANSNUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( RadRate_arr_d, RadRate_arr, sizeof(float)*(ZMAX+1)*LINENUM, (size_t) 0,cudaMemcpyHostToDevice));
  	CudaSafeCall(cudaMemcpyToSymbol( AtomicLevelWidth_arr_d, AtomicLevelWidth_arr, sizeof(float)*(ZMAX+1)*SHELLNUM, (size_t) 0,cudaMemcpyHostToDevice));
	//cudaThreadSynchronize();

	



	return 1;
}
#ifdef __cplusplus
}
#endif
#endif
