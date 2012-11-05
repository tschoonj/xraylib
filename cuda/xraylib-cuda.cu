#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "xrayglob.h"

typedef  float  *xsection[121];
typedef  int    num_xsection_data[121];



__device__ float *d_CS_Fluor_value;


__device__ float *dEPhoto, *dCSPhoto1, 
                 *dCSPhoto2, *dECompton,
                 *dCSCompton1, *dCSCompton2,
                 *dERayleigh, *dCSRayleigh1, *dCSRayleigh2,
                 *dSF_Compton,*dSF_Compton2,*dq_Compton,
                 *dAtomicWeight;


__device__ float *pippo,*pippo1,*pippo2,*pippo_exp;

__device__ float *dE,*dlogE,*dlogE1,*dlogE2;
__device__ int   *dZ,*dNE_Photons,*dNE_Compton,*dNE_Rayleigh,*dP,*dC,*dR,*Nq_Compton;

int P[120],R[120],C[120],qC[120];

int  dimP=0,dimC=0,dimR=0,dimQC=0;

cudaArray *cuArray0, *cuArray1, *cuArray2, *cuArray4, *cuArray5, *cuArray6;   

texture<float, 1, cudaReadModeElementType> texRef;


static void InitVects(int num_block,int num_threads);

extern "C" void CudaXRayInit(int  num_block,int num_threads){


	int i;
 	int tmpP=0,tmpR=0,tmpC=0,tmpQC=0;

	int deviceCount, device;
	int gpuDeviceCount = 0;
	struct cudaDeviceProp properties;
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
        	exit(1);
	}


	for (i=1;i<=(ZMAX-25);i++){
   		dimP+=NE_Photo[i];
   		dimC+=NE_Compt[i];
   		dimR+=NE_Rayl[i];
   		//dimQC+=Nq_Compton[i];
	}

	InitVects(num_block,num_threads);
    
    	cudaMemcpy(dAtomicWeight,AtomicWeight_arr,(ZMAX-25)*sizeof(float),cudaMemcpyHostToDevice );
 
	P[0]=C[0]=R[0]=qC[0]=0;
	int j; 
	for (i=1;i<=ZMAX-25;i++) {
		cudaMemcpy(dSF_Compton+tmpQC,SF_Compt_arr[i],Nq_Compt[i]*sizeof(float),cudaMemcpyHostToDevice );
		cudaMemcpy(dSF_Compton2+tmpQC,SF_Compt_arr2[i],Nq_Compt[i]*sizeof(float),cudaMemcpyHostToDevice );
		cudaMemcpy(dq_Compton+tmpQC,q_Compt_arr[i],Nq_Compt[i]*sizeof(float),cudaMemcpyHostToDevice );

		cudaMemcpy(dEPhoto+tmpP,E_Photo_arr[i],NE_Photo[i]*sizeof(float),cudaMemcpyHostToDevice );
		cudaMemcpy(dCSPhoto1+tmpP,CS_Photo_arr[i],NE_Photo[i]*sizeof(float),cudaMemcpyHostToDevice );
		cudaMemcpy(dCSPhoto2+tmpP,CS_Photo_arr2[i],NE_Photo[i]*sizeof(float),cudaMemcpyHostToDevice );

		cudaMemcpy(dECompton+tmpC,E_Compt_arr[i],NE_Compt[i]*sizeof(float),cudaMemcpyHostToDevice );
		cudaMemcpy(dCSCompton1+tmpC,CS_Compt_arr[i],NE_Compt[i]*sizeof(float),cudaMemcpyHostToDevice );
		cudaMemcpy(dCSCompton2+tmpC,CS_Compt_arr2[i],NE_Compt[i]*sizeof(float),cudaMemcpyHostToDevice );
    
		cudaMemcpy(dERayleigh+tmpR,E_Rayl_arr[i],NE_Rayl[i]*sizeof(float),cudaMemcpyHostToDevice );
		cudaMemcpy(dCSRayleigh1+tmpR,CS_Rayl_arr[i],NE_Rayl[i]*sizeof(float),cudaMemcpyHostToDevice );
		cudaMemcpy(dCSRayleigh2+tmpR,CS_Rayl_arr2[i],NE_Rayl[i]*sizeof(float),cudaMemcpyHostToDevice );

		P[i]+=tmpP;
		C[i]+=tmpC;
		R[i]+=tmpR;
		tmpP+=NE_Photo[i];
		tmpC+=NE_Compt[i];
		tmpR+=NE_Rayl[i];
		//tmpQC+=Nq_Compton[i];
     
   //  printf("%d  %d  %d\n",NE_Photo[i],NE_Compt[i],NE_Rayl[i]);
}  
  
   
     
  
}




static void InitVects(int num_block,int num_threads){
    
    int num_data=num_block*num_threads;
    float *tappo;
    
    cudaError_t e;
    cudaEvent_t start;
    cudaEvent_t end;

   
   
     printf("dimP=%d  dimC=%d  dimR%d\n", dimP,dimC,dimR);
     printf("numdata=%d\n",num_data);
   
   
   

     cudaMalloc((void **)&dEPhoto,dimP*sizeof(float));
     if (dEPhoto==NULL){
       printf("errore allocazione dEPhoto\n");
       exit(1);
     };
     cudaMalloc((void **)&dCSPhoto1,dimP*sizeof(float));
     if (dCSPhoto1==NULL){
       printf("errore allocazione dCSPhoto1\n");
       exit(1);
     };
     cudaMalloc((void **)&dCSPhoto2,dimP*sizeof(float));
     if (dCSPhoto2==NULL){
       printf("errore allocazione dCSPhoto2\n");
       exit(1);
     }; 
     cudaMalloc((void **)&dECompton,dimC*sizeof(float));
     if (dECompton==NULL){
       printf("errore allocazione dECompton\n");
       exit(1);
     }; 
     cudaMalloc((void **)&dCSCompton1,dimC*sizeof(float));
     if (dCSCompton1==NULL){
       printf("errore allocazione dCSCompton1\n");
       exit(1);
     }; 
     cudaMalloc((void **)&dCSCompton2,dimC*sizeof(float));
     if (dCSCompton2==NULL){
       printf("errore allocazione dCSCompton2\n");
       exit(1);
     }; 
    
    

     cudaMalloc((void **)&dSF_Compton,dimQC*sizeof(float));
     if (dSF_Compton==NULL){
       printf("errore allocazione dSF_Compton\n");
       exit(1);
     }; 
   
      cudaMalloc((void **)&dSF_Compton2,dimQC*sizeof(float));
     if (dSF_Compton2==NULL){
       printf("errore allocazione dSF_Compton2\n");
       exit(1);
     }; 
    
       cudaMalloc((void **)&dq_Compton,dimQC*sizeof(float));
     if (dq_Compton==NULL){
       printf("errore allocazione dq_Compton\n");
       exit(1);
     }; 
   
    
    
     cudaMalloc((void **)&dERayleigh,dimR*sizeof(float));
     cudaMalloc((void **)&dCSRayleigh1,dimR*sizeof(float));
     cudaMalloc((void **)&dCSRayleigh2,dimR*sizeof(float));
     cudaMalloc((void **)&dZ,num_data*sizeof(int));
     cudaMalloc((void **)&dP,120*sizeof(int));

     cudaMalloc((void **)&dC,120*sizeof(int));
     cudaMalloc((void **)&dR,120*sizeof(int));
      
     cudaMalloc((void **)&dAtomicWeight,120*sizeof(float));
  
  //    cudaMalloc((void **)&dNE_Photons,num_data*sizeof(int));
  //    cudaMalloc((void **)&dNE_Compton,num_data*sizeof(int));
  //    cudaMalloc((void **)&dNE_Rayleigh,num_data*sizeof(int));
  
  //    cudaMalloc((void **)&dE,num_data*sizeof(float));
      cudaMalloc((void **)&dlogE,num_data*sizeof(float));
     
     tappo=(float *)calloc(num_data,sizeof(float));
     for (int i=0;i<num_data;i++) tappo[i]=0.0;

     
     cudaMalloc((void **)&pippo,num_data*sizeof(float));
     cudaMalloc((void **)&pippo1,num_data*sizeof(float));
     cudaMalloc((void **)&pippo2,num_data*sizeof(float));
     cudaMalloc((void **)&pippo_exp,num_data*sizeof(float));
//     cudaHostAlloc((void **)&pippo_exp,num_data*sizeof(float),cudaHostAllocPortable);
     
     if (pippo_exp==NULL) printf("pippo_exp non allocato!\n");
    
     
     
     
     
     
     
     
     
     
     
     
     
     cudaMemcpy(pippo,tappo,num_data*sizeof(float),cudaMemcpyHostToDevice );
     free(tappo);
     
    /* CUT_SAFE_CALL(cutCreateTimer(&timer)); */
    //ccudaEventCreate(&start);
    cudaEventCreate(&start);
    cudaEventCreate(&end);
   
    
    /* CUT_SAFE_CALL(cutStartTimer(timer)); */
    cudaEventRecord(start, 0);
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before initialization.\n");
	exit(1);
    }

     
     
};


void EndXrayLib(void){
     cudaFree(dEPhoto);
     cudaFree(dCSPhoto1);
     cudaFree(dCSPhoto2);
    
     cudaFree(dECompton);
     cudaFree(dCSCompton1);
     cudaFree(dCSCompton2);
    
     cudaFree(dERayleigh);
     cudaFree(dCSRayleigh1);
     cudaFree(dCSRayleigh2);

}


__device__ void cuda_splintX(float *xa, float * ya, float * y2a, int n, float x, float *y)
{
	
	int klo=0, khi=n-1, k;
	float h, b, a;
	
	while (khi-klo > 1) {
		k = (khi + klo) >> 1;//division by 2
		if (xa[k] > x) khi = k;
		else klo = k;
	  //   printf("while GPU klo=%d  khi=%d   xa[k]=%f   x=%f \n\n",klo,khi,xa[k],x);
	}
	//determine position of x in xa

	h = xa[khi] - xa[+klo];
	if (h == 0.0) {
	  *y=0;
	  //printf("splint error.\n");
	  return;
	}
	a = (xa[khi] - x) / h;
	b = (x - xa[klo]) / h;
	*y = a*ya[klo] + b*ya[khi] + ((a*a*a-a)*y2a[klo]
	     + (b*b*b-b)*y2a[khi])*(h*h)/6.0;
	     
//	   printf("GPU y=%f    klo=%d  khi=%d     h=%f\n\n",*y,klo,khi,h);

}




__global__  void cuda_splint(float *xa, float *ya, float *y2a, int n, float *x, float *y)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  cuda_splintX(xa, ya, y2a, n, x[idx], &y[idx]);
 // printf("x=%f  exp(y)=%f\n",x[idx],exp(y[idx]));
   __syncthreads();	     

  
}

__global__  void cuda_exp_splint(float *a, float *b, float *c, float *exp_d)
{
   int idx = blockIdx.x*blockDim.x + threadIdx.x;
  
    a[idx]=exp(a[idx]);
    b[idx]=exp(b[idx]);
    c[idx]=exp(c[idx]);
    exp_d[idx]=a[idx]+b[idx]+c[idx];

       

  
}

                          



__global__  void cuda_splint_old(float *xa, float *ya, float *y2a, int n, float *x, float *y)
{
 

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	int klo=0, khi=n-1, k;
	float h, b, a;
	
	while (khi-klo > 1) {
		k = (khi + klo) >> 1;//division by 2
		if (xa[k] > x[idx]) khi = k;
		else klo = k;
	  //   printf("while GPU klo=%d  khi=%d   xa[k]=%f   x=%f \n\n",klo,khi,xa[k],x);
	}
	//determine position of x in xa

	h = xa[khi] - xa[+klo];
	if (h == 0.0) {
	  *y=0;
	  //printf("splint error.\n");
	  return;
	}
	a = (xa[khi] - x[idx]) / h;
	b = (x[idx] - xa[klo]) / h;
	y[idx] = a*ya[klo] + b*ya[khi] + ((a*a*a-a)*y2a[klo]
	     + (b*b*b-b)*y2a[khi])*(h*h)/6.0;
	     
//	   printf("GPU y=%f    klo=%d  khi=%d     h=%f\n\n",*y,klo,khi,h);


 __syncthreads();	     
	     
}


__global__  void log_splint(float *x, float *y)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	y[idx]=log(x[idx]*1000);   // *1000
	__syncthreads();	
}
void CS_Photo_cuda(int num_data, int Z, int *NE_Photons, float *d_Energy, float *XSC)
{

   float ln_E,  sigma,ln_sigma;
  int i, appo; 
  
  dim3 dimGrid1(10000,1);
  dim3 dimBlock1(1024,1);   //(num_data,1)

  
 
 
 /* 
  cudaMalloc((void **)&dZ,num_data*sizeof(int));
  cudaMalloc((void **)&dP,120*sizeof(int));
  cudaMalloc((void **)&dNE_Photons,num_data*sizeof(int));
  cudaMalloc((void **)&dE,num_data*sizeof(float));
  cudaMalloc((void **)&dlogE,num_data*sizeof(float));
*/  

//  cudaMemcpy(dZ,Z,num_data*sizeof(int),cudaMemcpyHostToDevice );
  cudaMemcpy(dP,P,120*sizeof(int),cudaMemcpyHostToDevice );
  cudaMemcpy(dNE_Photons,NE_Photons,num_data*sizeof(int),cudaMemcpyHostToDevice);
//  cudaMemcpy(dE,E,num_data*sizeof(float),cudaMemcpyHostToDevice );


  
  log_splint<<<dimGrid1,dimBlock1>>>(d_Energy, dlogE);
  cudaDeviceSynchronize();




  cuda_splint<<<dimGrid1,dimBlock1>>>(dEPhoto+dP[Z],dCSPhoto1+dP[Z],dCSPhoto2+dP[Z],dNE_Photons[Z],dlogE,pippo);
  cudaDeviceSynchronize();
 
  cudaMemcpy(XSC,pippo,num_data*sizeof(float),cudaMemcpyDeviceToHost);
  
  for (i=0;i<num_data;i++)  XSC[i] = exp(XSC[i]);

  /*
  cudaFree(pippo);
  cudaFree(dNE_Photons);
  cudaFree(dZ);
  cudaFree(dP);
  cudaFree(dE);
  cudaFree(dlogE);
  */
     
 
}

/************************************************************************/
/* Get Compton                                                           */
/************************************************************************/


void CS_Compton_cuda(int num_data, int Z, int *NE_Compton, float *d_Energy, float *XSC)
{

   float ln_E,  sigma,ln_sigma;
  int i, appo; 
  
  dim3 dimGrid1(10000,1);
  dim3 dimBlock1(1024,1);   //(num_data,1)

  
 /* 
//  cudaMalloc((void **)&dZ,num_data*sizeof(int));
  cudaMalloc((void **)&dC,120*sizeof(int));
  cudaMalloc((void **)&dNE_Compton,num_data*sizeof(int));
  cudaMalloc((void **)&dE,num_data*sizeof(float));
  cudaMalloc((void **)&dlogE,num_data*sizeof(float));
*/  
//  cudaMemcpy(dZ,Z,num_data*sizeof(int),cudaMemcpyHostToDevice );
  cudaMemcpy(dC,C,120*sizeof(int),cudaMemcpyHostToDevice );
  cudaMemcpy(dNE_Compton,NE_Compton,num_data*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpy(dE,E,num_data*sizeof(float),cudaMemcpyHostToDevice );


  
  log_splint<<<dimGrid1,dimBlock1>>>(d_Energy, dlogE);
  
  cudaThreadSynchronize();


  cuda_splint<<<dimGrid1,dimBlock1>>>(dECompton+dC[Z], dCSCompton1+dC[Z],dCSCompton2+dC[Z],dNE_Compton[Z],dlogE,pippo);
  cudaThreadSynchronize();
  
  cudaMemcpy(XSC,pippo,num_data*sizeof(float),cudaMemcpyDeviceToHost);
  
  for (i=0;i<num_data;i++)
    XSC[i] = exp(XSC[i]);

  /*
   cudaFree(pippo);
 cudaFree(dNE_Compton);
  cudaFree(dZ);
  cudaFree(dC);
  cudaFree(dE);
  cudaFree(dlogE);
  */
      
 
}



/************************************************************************/
/* Get Rayleigh                                                           */
/************************************************************************/


void CS_Rayleigh_cuda(int num_data, int Z, int *NE_Rayleigh, float *d_Energy, float *XSC)
{

   float ln_E,  sigma,ln_sigma;
  int i, appo; 
  
  dim3 dimGrid1(10000,1);
  dim3 dimBlock1(1024,1);   //(num_data,1)

  
  /*
  cudaMalloc((void **)&pippo,num_data*sizeof(float));
  cudaMalloc((void **)&dZ,num_data*sizeof(int));
  cudaMalloc((void **)&dR,120*sizeof(int));
  cudaMalloc((void **)&dNE_Rayleigh,num_data*sizeof(int));
  cudaMalloc((void **)&dE,num_data*sizeof(float));
  cudaMalloc((void **)&dlogE,num_data*sizeof(float));
  */
  
//  cudaMemcpy(dZ,Z,num_data*sizeof(int),cudaMemcpyHostToDevice );
  cudaMemcpy(dR,R,120*sizeof(int),cudaMemcpyHostToDevice );
  cudaMemcpy(dNE_Rayleigh,NE_Rayleigh,num_data*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpy(dE,E,num_data*sizeof(float),cudaMemcpyHostToDevice );


  
  log_splint<<<dimGrid1,dimBlock1>>>(d_Energy, dlogE);


  cuda_splint<<<dimGrid1,dimBlock1>>>(dERayleigh+dR[Z], dCSRayleigh1+dR[Z],dCSRayleigh2+dR[Z],dNE_Rayleigh[Z],dlogE,pippo);
  
  cudaMemcpy(XSC,pippo,num_data*sizeof(float),cudaMemcpyDeviceToHost);
  
  for (i=0;i<num_data;i++)
    XSC[i] = exp(XSC[i]);

  /*
  cudaFree(dNE_Rayleigh);
  cudaFree(pippo);
  cudaFree(dZ);
  cudaFree(dP);
  cudaFree(dE);
  cudaFree(dlogE);
  */
     
 
}

/************************************************************************/
/* Get Total CS                                                           */
/************************************************************************/



/*
void  Cuda_CS_Total(num_xsection_data NE_Photo, num_xsection_data NE_Compt, num_xsection_data NE_Rayl, int  num_block,int num_threads,
                              int **Z_Total, float ***CS_Total_Vect,  int *NElem_Total,int NPhases, float **CS_Total_Photo,float **CS_Total_Compt,float **CS_Total_Rayl )


{


   int nkernels=16;
   int num_streams=nkernels;
   
   float ln_E,  sigma,ln_sigma;
   float *tmp,*tmp1,*tmp2;
  int i,j,k, appo,appoP,appoC,appoR; 
  
  int num_data=num_block*num_threads;
  
  dim3 dimGrid1(num_block,1);
  dim3 dimBlock1(num_threads,1);   //(num_data,1)



  cudaStream_t *streams=(cudaStream_t *)malloc(num_streams*sizeof(cudaStream_t));
  for (i=0;i<num_streams;i++)
    cutilSafeCall( cudaStreamCreate(&(streams[i])));


    cudaError_t e;
    
    cudaEvent_t start;
    cudaEvent_t end;


    cudaEventCreate(&start);
    cudaEventCreate(&end);


  
  
  
  
  
  
  
//  cudaMemcpyAsync(dZ,Z,num_data*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpyAsync(dP,P,120*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpyAsync(dC,C,120*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpyAsync(dR,R,120*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpyAsync(dNE_Photons,NE_Photons,num_data*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpyAsync(dNE_Compton,NE_Compton,num_data*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpyAsync(dNE_Rayleigh,NE_Rayleigh,num_data*sizeof(int),cudaMemcpyHostToDevice );
//   cudaMemcpy(dZ,Z,num_data*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpy(dNE_Photons,NE_Photons,num_data*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpy(dNE_Compton,NE_Compton,num_data*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpy(dNE_Rayleigh,NE_Rayleigh,num_data*sizeof(int),cudaMemcpyHostToDevice );
//  cudaMemcpy(dE,E,num_data*sizeof(float),cudaMemcpyHostToDevice );
// // cudaMemcpy(dP,P,120*sizeof(int),cudaMemcpyHostToDevice );   
// // cudaMemcpy(dC,C,120*sizeof(int),cudaMemcpyHostToDevice );
// // cudaMemcpy(dR,R,120*sizeof(int),cudaMemcpyHostToDevice );
//  for (i=0;i<NPhases;i++)
//   for (j=0;j<NElem_Total[i];j++)
//     printf("i=%d  j=%d dP=%d   Z=%d   NE_T=%d\n",i,j,P[Z_Total[i][j]],Z_Total[i][j],NE_Photo[Z_Total[i][j]]);
 
  
  
  
  
//  tmp=(float *)calloc(num_data,sizeof(float));
 
    
  
  log_splint<<<dimGrid1,dimBlock1>>>(d_Energy, dlogE);    //d_Energy
  //   cudaDeviceSynchronize();



   int double_num_block=2*num_block; 
    

  for (i=0;i<NPhases;i++){
   for (j=0;j<NElem_Total[i];j++){
   
     appo=Z_Total[i][j];
     appoP=P[appo];
     appoR=R[appo];
     appoC=C[appo];

   

     cuda_splint<<<double_num_block,512,0,streams[0]>>>(dEPhoto+appoP,dCSPhoto1+appoP,dCSPhoto2+appoP,NE_Photo[appo],dlogE,pippo);
     cuda_splint<<<double_num_block,512,0,streams[1]>>>(dECompton+appoC, dCSCompton1+appoC,dCSCompton2+appoC,NE_Compt[appo],dlogE,pippo1);
     cuda_splint<<<double_num_block,512,0,streams[2]>>>(dERayleigh+appoR, dCSRayleigh1+appoR,dCSRayleigh2+appoR,NE_Rayl[appo],dlogE,pippo2);
     cudaDeviceSynchronize();
     cuda_exp_splint<<<double_num_block,512>>>(pippo, pippo1,pippo2, pippo_exp);
     
     cudaMemcpy(CS_Total_Photo[appo],pippo,num_data*sizeof(float),cudaMemcpyDeviceToHost);

     cudaMemcpy(CS_Total_Vect[i][j],pippo_exp,num_data*sizeof(float),cudaMemcpyDeviceToHost);
     cudaMemcpy(CS_Total_Rayl[appo],pippo2,num_data*sizeof(float),cudaMemcpyDeviceToHost);
     cudaMemcpy(CS_Total_Compt[appo],pippo1,num_data*sizeof(float),cudaMemcpyDeviceToHost);
      
     
      }

    }
      cudaFree(pippo);
      cudaFree(pippo1);
      cudaFree(pippo2);
      cudaFree(pippo_exp);
 //     free(tmp);
 
   
}

*/
