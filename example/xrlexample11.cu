#include <stdio.h>
#include "xraylib.h"
#include "xraylib-cuda.h"
#include <stdlib.h>
#include <cuda_runtime.h>

//#define CUDA_ERROR_CHECK

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


__global__ void Yields(int *Z, int *shells, float *yields) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	printf("tid: %i\n", tid);

	yields[tid] = FluorYield_cu(Z[tid], shells[tid]);
	//yields[tid] = (float) tid; 
	//printf("yield: %f\n",yields[tid]);
	
	__syncthreads();
	return;
}




int main (int argc, char *argv[]) {

	fprintf(stdout,"Entering xrlexample11\n");
	
	int Z[5] = {10,15,26,79,82};
	int shells[5] = {K_SHELL, K_SHELL, K_SHELL, L3_SHELL,L1_SHELL};
	int *Zd;
	int *shellsd;

	float yields[5], *yieldsd;

	CudaXRayInit();

	//fluorescence yields example
	CudaSafeCall(cudaMalloc((void **) &Zd, 5*sizeof(int)));
	CudaSafeCall(cudaMemcpy(Zd, Z, 5*sizeof(int), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMalloc((void **) &shellsd, 5*sizeof(int)));
	CudaSafeCall(cudaMemcpy(shellsd, shells, 5*sizeof(int), cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMalloc((void **) &yieldsd, 5*sizeof(float)));

	Yields<<<1,5>>>(Zd, shellsd,yieldsd);	

	CudaCheckError();

	
	CudaSafeCall(cudaMemcpy(yields, yieldsd, 5*sizeof(float), cudaMemcpyDeviceToHost));

	fprintf(stdout,"Fluorescence yields\n");
	fprintf(stdout,"Shell   Classic   CUDA\n");
	fprintf(stdout,"Ne-K    %8f %f\n",FluorYield(10,K_SHELL), yields[0]);
	fprintf(stdout,"P-K     %8f %f\n",FluorYield(15,K_SHELL), yields[1]);
	fprintf(stdout,"Fe-K    %8f %f\n",FluorYield(26,K_SHELL), yields[2]);
	fprintf(stdout,"Au-K    %8f %f\n",FluorYield(79,L3_SHELL), yields[3]);
	fprintf(stdout,"Pb-K    %8f %f\n",FluorYield(82,L1_SHELL), yields[4]);


}
