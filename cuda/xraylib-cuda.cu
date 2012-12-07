#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "xrayglob.h"


#ifdef __cplusplus
#define XRLCUDACPLUSPLUS extern "C"
#else
#define XRLCUDACPLUSPLUS
#endif



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


__device__ float *FluorYield_arr_d;



XRLCUDACPLUSPLUS int CudaXRayInit() {



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
     	CudaSafeCall(cudaMalloc((void **)&FluorYield_arr_d,sizeof(float)*(ZMAX+1)*SHELLNUM));
  	CudaSafeCall(cudaMemcpy(FluorYield_arr_d, FluorYield_arr, sizeof(float)*(ZMAX+1)*SHELLNUM, cudaMemcpyHostToDevice));

	



	return 1;
}





