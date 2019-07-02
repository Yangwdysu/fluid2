
#include"cuda_runtime.h"
#include <iostream>
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions 
#include"device_functions.h"
#include "cuda.h"
#include"time.h"
#include<device_launch_parameters.h>
#include"device_functions.h"
#include <sstream>



#define BLOCK_SIZE 16



__global__ void Add(float* A, float*B, float* C,int M,int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i<M&&j<N)
	{
		C[i + j*N] = A[i + j*N] + B[i + j*N];
	}

}

void add(float* A, float* B, float* C, int M, int N)
{
	float* A_b;
	float* B_b;
	float* C_b;

	cudaMalloc((void**)&A_b, M * sizeof(float));
	cudaMalloc((void**)&B_b, N * sizeof(float));
	cudaMalloc((void**)&C_b, M*N * sizeof(float));

	cudaMemcpy(A_b, A, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_b, B, M * sizeof(float), cudaMemcpyHostToDevice);


	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
	Add<< <dimGrid, dimBlock >> > (A_b, B_b, C_b, M, N);
	cudaMemcpy(C, C_b, M*N * sizeof(float), cudaMemcpyDeviceToHost);

}

