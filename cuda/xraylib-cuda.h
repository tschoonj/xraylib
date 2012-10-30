
#ifndef XRAYLIB_CUDA_H
#define XRAYLIB_CUDA_H

void cuda_splint(float *xa, float *ya, float *y2a, int n, float x, float *y);





void CudaXRayInit(int  num_block,int num_threads);



void CS_Photo_cuda(int num_data,int *Z, int *NE_Photons, float *E,float *cuda_E_array);
void CS_Compton_cuda(int num_data, int *Z, int *NE_Compton, float *E, float *cuda_E_array);
void CS_Rayleigh_cuda(int num_data, int *Z, int *NE_Rayleigh, float *E, float *cuda_E_array);
//void Cuda_CS_Total(int num_data, int *Z, int *NE_Photons, int *NE_Rayleigh,int *NE_Compton, float *cuda_E_array);
//void   Cuda_CS_Total(int num_block,int num_threads, int **Z_Total,  float ***CS_Total_Vect,  int *NElem_Total,int NPhases,num_xsection_data NE_Photo,num_xsection_data NE_Compt,num_xsection_data NE_Rayl);


//void  Cuda_CS_Total(num_xsection_data NE_Photo, num_xsection_data NE_Compt, num_xsection_data NE_Rayl, int  num_block,int num_threads, int **Z_Total, float ***CS_Total_Vect,  int *NElem_Total,int NPhases, float **CS_Total_Photo,float **CS_Total_Compt,float **CS_Total_Rayl );



//float CS_FluorLineGPU(int Z, int line, float E, int index);



void InitVects(int num_block,int num_threads); 
void EndXrayLib(void);

//void call_cuda_DCS_Compt(int *Z,float *E, float *theta,int num_blocks,int num_threads, float *DCS_Compt_results);
//
//
#endif
