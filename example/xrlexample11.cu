#include <stdio.h>
#include "xraylib.h"
#include "xraylib-cuda.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include "xrayglob.h"

//#define CUDA_ERROR_CHECK


__global__ void Yields(int *Z, int *shells, float *yields) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;


	yields[tid] = FluorYield_cu(Z[tid], shells[tid]);
	
	__syncthreads();
	return;
}

__global__ void Weights(int *Z, float *weights) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;


	weights[tid] = AtomicWeight_cu(Z[tid]);
	
	__syncthreads();
	return;
}

__global__ void Edges(int *Z, int *shells, float *edges) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;


	edges[tid] = EdgeEnergy_cu(Z[tid], shells[tid]);
	
	__syncthreads();
	return;
}

__global__ void Jumps(int *Z, int *shells, float *jumps) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;


	jumps[tid] = JumpFactor_cu(Z[tid], shells[tid]);
	
	__syncthreads();
	return;
}

__global__ void CosKrons(int *Z, int *trans, float *coskrons) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;


	coskrons[tid] = CosKronTransProb_cu(Z[tid], trans[tid]);
	
	__syncthreads();
	return;
}

__global__ void RadRates(int *Z, int *lines, float *radrates) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;


	radrates[tid] = RadRate_cu(Z[tid], lines[tid]);
	
	__syncthreads();
	return;
}

__global__ void Widths(int *Z, int *shells, float *widths) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;


	widths[tid] = AtomicLevelWidth_cu(Z[tid], shells[tid]);
	
	__syncthreads();
	return;
}

__global__ void CS_Photos(int *Z, float *energies, float *cs) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	cs[tid] = CS_Photo_cu(Z[tid], energies[tid]);
	
	__syncthreads();
	return;
}



int main (int argc, char *argv[]) {

	fprintf(stdout,"Entering xrlexample11\n");
	
	int Z[5] = {10,15,26,79,82};
	int shells[5] = {K_SHELL, K_SHELL, K_SHELL, L3_SHELL,L1_SHELL};
	int lines[5] = {KL2_LINE, KL3_LINE, KM3_LINE, L3M5_LINE, L2M4_LINE};
	float energies[5] = {2.0, 8.0, 9.275, 15.89, 50.23};
	int Z2[3] = {56, 68, 80};
	int trans[3] = {FL13_TRANS, FL23_TRANS, FM34_TRANS};
	int *Zd, *Z2d;
	int *shellsd, *linesd, *transd;
	float *energiesd;

	float yields[5], *yieldsd;
	float weights[5], *weightsd;
	float edges[5], *edgesd;
	float lineEnergies[5], *lineEnergiesd;
	float jumps[5], *jumpsd;
	float coskrons[3], *coskronsd;
	float radrates[5], *radratesd;
	float widths[5], *widthsd; 
	float photo_cs[5], *photo_csd;


	int i;

	CudaXRayInit();

	//input variables
	CudaSafeCall(cudaMalloc((void **) &Zd, 5*sizeof(int)));
	CudaSafeCall(cudaMemcpy(Zd, Z, 5*sizeof(int), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMalloc((void **) &Z2d, 3*sizeof(int)));
	CudaSafeCall(cudaMemcpy(Z2d, Z2, 3*sizeof(int), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMalloc((void **) &shellsd, 5*sizeof(int)));
	CudaSafeCall(cudaMemcpy(shellsd, shells, 5*sizeof(int), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMalloc((void **) &linesd, 5*sizeof(int)));
	CudaSafeCall(cudaMemcpy(linesd, lines, 5*sizeof(int), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMalloc((void **) &transd, 3*sizeof(int)));
	CudaSafeCall(cudaMemcpy(transd, trans, 3*sizeof(int), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMalloc((void **) &energiesd, 5*sizeof(float)));
	CudaSafeCall(cudaMemcpy(energiesd, energies, 5*sizeof(float), cudaMemcpyHostToDevice));


	//output variables
	CudaSafeCall(cudaMalloc((void **) &yieldsd, 5*sizeof(float)));
	CudaSafeCall(cudaMalloc((void **) &weightsd, 5*sizeof(float)));
	CudaSafeCall(cudaMalloc((void **) &edgesd, 5*sizeof(float)));
//	CudaSafeCall(cudaMalloc((void **) &lineEnergiesd, 5*sizeof(float)));
	CudaSafeCall(cudaMalloc((void **) &jumpsd, 5*sizeof(float)));
	CudaSafeCall(cudaMalloc((void **) &coskronsd, 3*sizeof(float)));
	CudaSafeCall(cudaMalloc((void **) &radratesd, 5*sizeof(float)));
	CudaSafeCall(cudaMalloc((void **) &widthsd, 5*sizeof(float)));
	CudaSafeCall(cudaMalloc((void **) &photo_csd, 5*sizeof(float)));


	Yields<<<1,5>>>(Zd, shellsd,yieldsd);	
	CudaCheckError();

	Weights<<<1,5>>>(Zd, weightsd);	
	CudaCheckError();

	Edges<<<1,5>>>(Zd, shellsd, edgesd);	
	CudaCheckError();

	Jumps<<<1,5>>>(Zd, shellsd, jumpsd);	
	CudaCheckError();

	CosKrons<<<1,3>>>(Z2d, transd, coskronsd);	
	CudaCheckError();

	RadRates<<<1,5>>>(Zd, linesd, radratesd);	
	CudaCheckError();

	Widths<<<1,5>>>(Zd, shellsd, widthsd);	
	CudaCheckError();
	
	CS_Photos<<<1,5>>>(Zd, energiesd, photo_csd);	
	CudaCheckError();
	
	CudaSafeCall(cudaMemcpy(yields, yieldsd, 5*sizeof(float), cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(weights, weightsd, 5*sizeof(float), cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(edges, edgesd, 5*sizeof(float), cudaMemcpyDeviceToHost));
//	CudaSafeCall(cudaMemcpy(lineEnergies, lineEnergiesd, 5*sizeof(float), cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(jumps, jumpsd, 5*sizeof(float), cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(coskrons, coskronsd, 3*sizeof(float), cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(radrates, radratesd, 5*sizeof(float), cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(widths, widthsd, 5*sizeof(float), cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(photo_cs, photo_csd, 5*sizeof(float), cudaMemcpyDeviceToHost));


	fprintf(stdout,"Fluorescence yields\n");
	fprintf(stdout,"Shell   Classic   CUDA\n");
	fprintf(stdout,"Ne-K    %8f %f\n",FluorYield(10,K_SHELL), yields[0]);
	fprintf(stdout,"P-K     %8f %f\n",FluorYield(15,K_SHELL), yields[1]);
	fprintf(stdout,"Fe-K    %8f %f\n",FluorYield(26,K_SHELL), yields[2]);
	fprintf(stdout,"Au-L3   %8f %f\n",FluorYield(79,L3_SHELL), yields[3]);
	fprintf(stdout,"Pb-L1   %8f %f\n",FluorYield(82,L1_SHELL), yields[4]);

	fprintf(stdout,"\n\n");

	fprintf(stdout,"Atomic weights\n");
	fprintf(stdout,"Element   Classic   CUDA\n");
	fprintf(stdout,"Ne      %8f %f\n",AtomicWeight(10), weights[0]);
	fprintf(stdout,"P       %8f %f\n",AtomicWeight(15), weights[1]);
	fprintf(stdout,"Fe      %8f %f\n",AtomicWeight(26), weights[2]);
	fprintf(stdout,"Au      %8f %f\n",AtomicWeight(79), weights[3]);
	fprintf(stdout,"Pb      %8f %f\n",AtomicWeight(82), weights[4]);

	fprintf(stdout,"\n\n");

	fprintf(stdout,"Edge energies\n");
	fprintf(stdout,"Shell   Classic   CUDA\n");
	fprintf(stdout,"Ne-K    %8f %f\n",EdgeEnergy(10,K_SHELL), edges[0]);
	fprintf(stdout,"P-K     %8f %f\n",EdgeEnergy(15,K_SHELL), edges[1]);
	fprintf(stdout,"Fe-K    %8f %f\n",EdgeEnergy(26,K_SHELL), edges[2]);
	fprintf(stdout,"Au-L3   %8f %f\n",EdgeEnergy(79,L3_SHELL), edges[3]);
	fprintf(stdout,"Pb-L1   %8f %f\n",EdgeEnergy(82,L1_SHELL), edges[4]);

	fprintf(stdout,"\n\n");

	fprintf(stdout,"Jump factors\n");
	fprintf(stdout,"Shell   Classic   CUDA\n");
	fprintf(stdout,"Ne-K    %8f %f\n",JumpFactor(10,K_SHELL), jumps[0]);
	fprintf(stdout,"P-K     %8f %f\n",JumpFactor(15,K_SHELL), jumps[1]);
	fprintf(stdout,"Fe-K    %8f %f\n",JumpFactor(26,K_SHELL), jumps[2]);
	fprintf(stdout,"Au-L3   %8f %f\n",JumpFactor(79,L3_SHELL), jumps[3]);
	fprintf(stdout,"Pb-L1   %8f %f\n",JumpFactor(82,L1_SHELL), jumps[4]);

	fprintf(stdout,"\n\n");

	fprintf(stdout,"Coster-Kronig transition probabilities\n");
	fprintf(stdout,"Transition   Classic   CUDA\n");
	fprintf(stdout,"Ba L1->L3    %8f %f\n",CosKronTransProb(56,FL13_TRANS), coskrons[0]);
	fprintf(stdout,"Er L3->L2    %8f %f\n",CosKronTransProb(68,FL23_TRANS), coskrons[1]);
	fprintf(stdout,"Hg M3->M4    %8f %f\n",CosKronTransProb(80,FM34_TRANS), coskrons[2]);

	fprintf(stdout,"\n\n");

	fprintf(stdout,"Radiative rates\n");
	fprintf(stdout,"Line      Classic   CUDA\n");
	fprintf(stdout,"Ne-KL2    %8f %f\n",RadRate(10,lines[0]), radrates[0]);
	fprintf(stdout,"P-KL3     %8f %f\n",RadRate(15,lines[1]), radrates[1]);
	fprintf(stdout,"Fe-KM3    %8f %f\n",RadRate(26,lines[2]), radrates[2]);
	fprintf(stdout,"Au-L3M5   %8f %f\n",RadRate(79,lines[3]), radrates[3]);
	fprintf(stdout,"Pb-L2M4   %8f %f\n",RadRate(82,lines[4]), radrates[4]);

	fprintf(stdout,"\n\n");

	fprintf(stdout,"Atomic level widths\n");
	fprintf(stdout,"Shell   Classic   CUDA\n");
	fprintf(stdout,"Ne-K    %8f %f\n",AtomicLevelWidth(10,K_SHELL), widths[0]);
	fprintf(stdout,"P-K     %8f %f\n",AtomicLevelWidth(15,K_SHELL), widths[1]);
	fprintf(stdout,"Fe-K    %8f %f\n",AtomicLevelWidth(26,K_SHELL), widths[2]);
	fprintf(stdout,"Au-L3   %8f %f\n",AtomicLevelWidth(79,L3_SHELL), widths[3]);
	fprintf(stdout,"Pb-L1   %8f %f\n",AtomicLevelWidth(82,L1_SHELL), widths[4]);

	fprintf(stdout,"Photo ionization cross sections\n");
	fprintf(stdout,"Element   Energy(keV)   Classic   Cuda\n");
	for (i = 0 ; i < 5 ; i++) {
		fprintf(stdout,"%i      %6f    %8f %f\n", Z[i], energies[i], CS_Photo(Z[i], energies[i]), photo_cs[i]);
	}


}
