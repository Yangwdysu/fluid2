#include"cuda_runtime.h"
#include <iostream>
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions 
#include"device_functions.h"
#include "cuda.h"
#include "cuda_UpdatePhi.h"
#include"time.h"
#include<device_launch_parameters.h>
#include"device_functions.h"
#include <sstream>
//#include "HybridMultiPhaseFluid.h"
using namespace mfd;
using namespace std;
//#define dsize Nx*Ny*Nz
//#define Nx 512
//#define Ny 512
#define BLOCK_SIZE 16
//#define EPSILON 1e-6

#ifdef NDEBUG
#define cuSynchronize() {}
#else
#define cuSynchronize()	{						\
		char str[200];							\
		cudaDeviceSynchronize();				\
		cudaError_t err = cudaGetLastError();	\
		if (err != cudaSuccess)					\
		{										\
			sprintf(str, "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);		\
			throw std::runtime_error(std::string(str));																\
		}																											\
	}
#endif

__global__ void AdvectForward_dd(float* phi, float* phi0, float3* v, int nx, int ny, int nz, float dt, float h)
{

	float fx, fy, fz;
	int  ix, iy, iz;
	float w000, w100, w010, w001, w111, w011, w101, w110;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	{

		int idx = i + j*nx + k*nx*ny;

		fx = i + dt*v[idx].x / h;
		fy = j + dt*v[idx].y / h;
		fz = k + dt*v[idx].z / h;
		if (fx < 1) { fx = 1; }
		if (fx > nx - 2) { fx = nx - 2; }
		if (fy < 1) {fy = 1;}
		if (fy > ny - 2) { fy = ny - 2; }
		if (fz < 1) { fz = 1; }
		if (fz > nz - 2) { fz = nz - 2; }



		ix = (int)fx;
		iy = (int)fy;
		iz = (int)fz;
		fx -= ix;
		fy -= iy;
		fz -= iz;

		float& val = phi0[idx];

		w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
		w100 = fx * (1.0f - fy)*(1.0f - fz);
		w010 = (1.0f - fx)*fy *(1.0f - fz);
		w001 = (1.0f - fx)*(1.0f - fy)*fz;
		w111 = fx * fy * fz;
		w011 = (1.0f - fx)*fy * fz;
		w101 = fx * (1.0f - fy)*fz;
		w110 = fx * fy *(1.0f - fz);

		phi[ix + iy*nx + iz*nx*ny] += val * w000;
		phi[(ix + 1) + iy*nx + iz*nx*ny] += val * w100;
		phi[ix + (iy + 1)*nx + iz*nx*ny] += val * w010;
		phi[ix + iy*nx + (iz + 1)*nx*ny] += val * w001;

		phi[(ix + 1) + (iy + 1)*nx + (iz + 1)*nx*ny] += val * w111;
		phi[ix + (iy + 1)*nx + (iz + 1)*nx*ny] += val * w011;
		phi[(ix + 1) + iy*nx + (iz + 1)*nx*ny] += val * w101;
		phi[(ix + 1) + (iy + 1)*nx + iz*nx*ny] += val * w110;

	}
}

void cuda_UpdatePhi::cuda_AdvectForwardd(float* dd, float* dd0, float3* v, int nx, int ny, int nz, float dt, float h, int dsize)
{

	//device
	float* Ddevice_phi;
	float* Ddevice_phi0;
	float3* Device_v;

	//allocate
	cudaMalloc((void**)&Ddevice_phi, dsize * sizeof(float));
	cudaMalloc((void**)&Ddevice_phi0, dsize * sizeof(float));
	cudaMalloc((void**)&Device_v, dsize * sizeof(float3));
	cudaMemset(Ddevice_phi, 0, dsize * sizeof(float));

	//copy data
    cudaMemcpy(Ddevice_phi0, dd0, dsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Device_v, v, dsize * sizeof(float3), cudaMemcpyHostToDevice);

	////computer
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
	AdvectForward_dd << <dimGrid, dimBlock >> > (Ddevice_phi, Ddevice_phi0, Device_v, nx, ny, nz, dt, h);
	cudaMemcpy(dd, Ddevice_phi, dsize * sizeof(float), cudaMemcpyDeviceToHost);

	//Free space
	cudaFree(Ddevice_phi);
	cudaFree(Ddevice_phi0);
	cudaFree(Device_v);
	cuSynchronize();
}









