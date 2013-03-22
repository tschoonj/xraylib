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


int CudaXRayFree();

//device functions

__device__ float splint_cu(float *xa, float *ya, float *y2a, int n, float x);
__device__ float CS_Photo_cu(int Z, float E);
__device__ float  FluorYield_cu(int Z, int shell);
__device__ float AtomicWeight_cu(int Z);
__device__ float EdgeEnergy_cu(int Z, int shell);
//__device__ float LineEnergy_cu(int Z, int line);
__device__ float JumpFactor_cu(int Z, int shell);
__device__ float CosKronTransProb_cu(int Z, int trans);
__device__ float RadRate_cu(int Z, int line);
__device__ float AtomicLevelWidth_cu(int Z, int shell);


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








#ifdef __cplusplus
}
#endif
#endif
