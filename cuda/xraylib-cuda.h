
#ifndef XRAYLIB_CUDA_H
#define XRAYLIB_CUDA_H



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
extern __device__ float *FluorYield_arr_d;

__device__ float  FluorYield_cu(int Z, int shell) {
  float fluor_yield;

  printf("Z: %i shell: %i ZMAX: %i  SHELLNUM: %i\n", Z, shell, ZMAX, SHELLNUM);

  if (Z<1 || Z>ZMAX) {
    printf("Should not get here\n");
    return 0;
  }

  if (shell<0 || shell>=SHELLNUM) {
    printf("Should not get here2\n");
    return 0;
  }

  printf("offset: %i\n",Z*SHELLNUM+shell);
  fluor_yield = FluorYield_arr_d[Z*SHELLNUM+shell];
  //printf("yield: %f\n",fluor_yield);
  //fluor_yield = 0.0;
  if (fluor_yield < 0.) {
    return 0;
  }

  return fluor_yield;
}



#ifdef __cplusplus
}
#endif

#endif
