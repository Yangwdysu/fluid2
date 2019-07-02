#include"cuda_runtime.h"
#include <iostream>
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions 
#include"device_functions.h"
#include "cuda.h"
#include "cudaHybridMultiPhaseFluid.h"
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


__global__ void Kenel_LinerSolve(float* phi, float* phi0, float* cp,float c, int Nx, int Ny, int Nz)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	float c1 = 1.0f / c;
	float c2 = (1.0f - c1) / 6.0f;
	if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
	{
		int k0 = i + j*Nx + k*Nx*Ny;
		phi[k0] = (c1*phi0[k0] + c2*(cp[k0 + 1] + cp[k0 - 1] + cp[k0 + Nx] + cp[k0 - Nx] + cp[k0 + Nx*Ny] + cp[k0 - Nx*Ny]));
	}

}




__global__ void Kenel_AdvectForward(float* phi, float* phi0, float3* v, int nx, int ny, int nz, float dt, float h)
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
		//Ô­×Ó²Ù×÷
		atomicAdd(&phi[ix + iy*nx + iz*nx*ny],val * w000);
		atomicAdd(&phi[(ix + 1) + iy*nx + iz*nx*ny],val * w100);
		atomicAdd(&phi[ix + (iy + 1)*nx + iz*nx*ny],val * w010);
		atomicAdd(&phi[ix + iy*nx + (iz + 1)*nx*ny], val * w001);

		atomicAdd(&phi[(ix + 1) + (iy + 1)*nx + (iz + 1)*nx*ny], val * w111);
		atomicAdd(&phi[ix + (iy + 1)*nx + (iz + 1)*nx*ny],val * w011);
		atomicAdd(&phi[(ix + 1) + iy*nx + (iz + 1)*nx*ny], val * w101);
		atomicAdd(&phi[(ix + 1) + (iy + 1)*nx + iz*nx*ny], val * w110);

	}
}

__global__ void SetScalarFieldBoundary_x(float* field, float s, int Nx, int Ny, int Nz)
{

	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
	{
		field[0 + j*Ny + k*Nx*Ny] = s*field[1 + j*Ny + k*Nx*Ny];
		field[(Nx - 1) + j*Ny + k*Nx*Ny] = s*field[(Nx - 2) + j*Ny + k*Nx*Ny];
	}
}

__global__ void SetScalarFieldBoundary_y(float* field, float s, int Nx, int Ny, int Nz)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i >= 1 && i < Nx - 1 && k >= 1 && k < Nz - 1)
	{
		field[i + 0 * Ny + k*Nx*Ny] = s*field[i + 1 * Ny + k*Nx*Ny];
		field[i + (Ny - 1)*Ny + k*Nx*Ny] = s*field[i + (Ny - 2)*Nx + k*Nx*Ny];
	}
}

__global__ void SetScalarFieldBoundary_z(float* field, float s, int Nx, int Ny, int Nz)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1)
	{
		field[i + j* Ny + 0 * Nx*Ny] = s*field[i + j * Nx + 1 * Nx*Ny];
		field[i + j*Ny + (Nz - 1)*Nx*Ny] = s*field[i + j*Ny + (Nz - 2)*Nx*Ny];
	}
}

__global__ void SetScalarFieldBoundary_yz(float* field, float s, int Nx, int Ny, int Nz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 1 && i < Nx - 1)
	{
		//x direction
		field[i + 0 * Nx + 0 * Nx*Ny] = 0.5*(field[i + 1 * Nx + 0 * Nx*Ny] + field[i + 0 * Nx + 1 * Nx*Ny]);
		field[i + (Ny - 1) * Nx + 0 * Nx*Ny] = 0.5*(field[i + (Ny - 2) * Nx + 0 * Nx*Ny] + field[i + (Ny - 1) * Nx + 1 * Nx*Ny]);
		field[i + 0 * Nx + (Nz - 1) * Nx*Ny] = 0.5*(field[i + 1 * Nx + (Nz - 1) * Nx*Ny] + field[i + 0 * Nx + (Nz - 2) * Nx*Ny]);
		field[i + (Ny - 1) * Nx + (Nz - 1)*Nx*Ny] = 0.5*(field[i + (Ny - 1) * Nx + (Nz - 1)*Nx*Ny] + field[i + (Ny - 2) * Nx + (Nz - 1)*Nx*Ny]);
	}

}

__global__ void SetScalarFieldBoundary_xz(float* field, float s, int Nx, int Ny, int Nz)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= 1 && j < Ny - 1)
	{
		//y direction
		field[0 + j * Nx + 0 * Nx*Ny] = 0.5*(field[1 + j * Nx + 0 * Nx*Ny] + field[0 + j * Nx + 1 * Nx*Ny]);
		field[0 + j * Nx + (Nz - 1)*Nx*Ny] = 0.5*(field[1 + j * Ny + (Nz - 1)*Nx*Ny] + field[0 + j * Ny + (Nz - 2)*Nx*Ny]);
		field[(Nx - 1) + j * Nx + 0 * Nx*Ny] = 0.5*(field[(Nx - 2) + j * Ny + 0 * Nx*Ny] + field[(Nx - 2) + j * Ny + 1 * Nx*Ny]);
		field[(Nx - 1) + j * Nx + (Nz - 1)*Nx*Ny] = 0.5*(field[(Nx - 2) + j * Ny + (Nz - 1)*Nx*Ny] + field[(Nx - 1) + j * Ny + (Nz - 2)*Nx*Ny]);
	}
}

__global__ void SetScalarFieldBoundary_xy(float* field,float s, int Nx, int Ny, int Nz)
{
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (k >= 1 && k < Nz - 1)
	{
		//z direction
		field[0 + 0 * Nx + k * Nx*Ny] = 0.5*(field[1 + 0 * Nx + k * Nx*Ny] + field[0 + 1 * Nx + k * Nx*Ny]);
		field[(Nx - 1) + 0 * Nx + k*Nx*Ny] = 0.5*(field[(Nx - 2) + 0 * Ny + k*Nx*Ny] + field[(Nx - 1) + 1 * Ny + k*Nx*Ny]);
		field[0 + (Ny - 1) * Nx + k*Nx*Ny] = 0.5*(field[1 + (Ny - 1) * Ny + k*Nx*Ny] + field[0 + (Ny - 2) * Ny + k*Nx*Ny]);
		field[(Nx - 1) + (Ny - 1) * Nx + k*Nx*Ny] = 0.5*(field[(Nx - 2) + (Ny - 1) * Ny + k*Nx*Ny] + field[(Nx - 1) + (Ny - 2) * Ny + k*Nx*Ny]);
	}
}

__global__ void UpdataPhi_d(float3* nGrid, float* dif_field, int Nx, int Ny, int Nz)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	float eps = 0.000001;

	if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
	{
		float3 norm;
		norm.x = dif_field[(i + 1) + j*Nx + k*Nx*Ny] - dif_field[(i - 1) + j*Nx + k*Nx*Ny];
		norm.y = dif_field[i + (j + 1)*Nx + k*Nx*Ny] - dif_field[i + (j - 1)*Nx + k*Nx*Ny];
		norm.z = dif_field[i + j*Nx + (k + 1)*Nx*Ny] - dif_field[i + j*Nx + (k - 1)*Nx*Ny];
		float norm_xy = sqrt((norm.x*norm.x) + (norm.y*norm.y)+(norm.z*norm.z)) + eps;
		norm.x /= norm_xy;
		norm.y /= norm_xy;
		norm.z /= norm_xy;

		int index = i + j*Nx + k*Nx*Ny;
		nGrid[index].x = norm.x;
		nGrid[index].y = norm.y;
		nGrid[index].z = norm.z;
	}
}
__global__ void UpdataPhi_dd(float3* nGrid, float* phi, float* phi0, float dt, float h, int Nx, int Ny, int Nz)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	float w = 1.0f*h;
	float gamma = 1.0f;
	float ceo2 = 1.0f*gamma*w / h / h;

	float weight = 1.0f;
	if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
	{
		int ix0, iy0, iz0, ix1, iy1, iz1;
		ix0 = i;   iy0 = j; iz0 = k;
		ix1 = i + 1; iy1 = j; iz1 = k;
		int idx0, idx1;
		if (ix1 < Nx - 1)
		{
			idx0 = ix0 + iy0*Nx + iz0*Nx*Ny;
			idx1 = ix1 + iy1*Nx + iz1*Nx*Ny;
			float2 c;
			c.x = phi0[idx0] * (1.0f - phi0[idx0])*nGrid[idx0].x;
			c.y = phi0[idx1] * (1.0f - phi0[idx1])*nGrid[idx1].x;
			float dc = 0.5f*weight*ceo2*dt*(c.x + c.y);
			atomicAdd(&phi[idx0], -dc);
			atomicAdd(&phi[idx1], dc);
		}

		ix0 = i; iy0 = j, iz0 = k;
		ix1 = i; iy1 = j + 1, iz1 = k;
		if (iy1 < Ny - 1)
		{
			idx0 = ix0 + iy0*Nx + iz0*Nx*Ny;
			idx1 = ix1 + iy1*Nx + iz1*Nx*Ny;
			float2 c;
			c.x = phi0[idx0] * (1.0f - phi0[idx0])*nGrid[idx0].y;
			c.y = phi0[idx1] * (1.0f - phi0[idx1])*nGrid[idx1].y;
			float dc = 0.5f*weight*ceo2*dt*(c.x + c.y);
			atomicAdd(&phi[idx0], -dc);
			atomicAdd(&phi[idx1], dc);
		}

		ix0 = i; iy0 = j, iz0 = k;
		ix1 = i; iy1 = j, iz1 = k + 1;
		if (iz1 < Nz - 1)
		{
			idx0 = ix0 + iy0*Nx + iz0*Nx*Ny;
			idx1 = ix1 + iy1*Nx + iz1*Nx*Ny;
			float2 c;
			c.x = phi0[idx0] * (1.0f - phi0[idx0])*nGrid[idx0].z;
			c.y = phi0[idx1] * (1.0f - phi0[idx1])*nGrid[idx1].z;
			float dc = 0.5f*weight*ceo2*dt*(c.x + c.y);
			atomicAdd(&phi[idx0], -dc);
			atomicAdd(&phi[idx1], dc);
		}
	}
}

void cudaHybridMultiPhaseFluid::cudaUpdatePhi(float* d, float* d0, float dt,int Nx, int Ny, int Nz,float h)
{
	int dsize = Nx*Ny*Nz;
	//device
	float3* dnGrid;
	float* ddif_field;
	float* Device_d;
	float* Device_d0;

	cudaMalloc((void**)&dnGrid, dsize * sizeof(float3));
	cudaMalloc((void**)&ddif_field, dsize * sizeof(float));
	cudaMalloc((void**)&Device_d, dsize * sizeof(float));
	cudaMalloc((void**)&Device_d0, dsize * sizeof(float));

	cudaMemcpy(ddif_field, d0, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Device_d0, d0, sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((Nx + dimBlock.x - 1) / dimBlock.x, (Ny + dimBlock.y - 1) / dimBlock.y, (Nz + dimBlock.z - 1) / dimBlock.z);
	UpdataPhi_d << <dimGrid, dimBlock >> > (dnGrid, ddif_field, Nx, Ny, Nz);
	cuSynchronize();
	UpdataPhi_dd << <dimGrid, dimBlock >> > (dnGrid, Device_d, Device_d0, dt, h, Nx, Ny, Nz);
	cuSynchronize();
	cudaMemcpy(d, Device_d, dsize * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(Device_d);
	cudaFree(ddif_field);
	cudaFree(dnGrid);
	cudaFree(Device_d0);

}



void cudaHybridMultiPhaseFluid::cudaLinerSolve(float* phi, float* phi0,float* cp, float c, int Nx, int Ny, int Nz)
{
	int dsize = Nx*Ny*Nz;
	
	//device
	float* Device_L_phi;
	float *Device_L_phi0;
	float* Device_cp;

	cudaMalloc((void**)&Device_L_phi, dsize * sizeof(float));
	cudaMalloc((void**)&Device_L_phi0, dsize * sizeof(float));
	cudaMalloc((void**)&Device_cp, dsize * sizeof(float));
	cudaMemcpy(Device_L_phi0, phi0, dsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Device_cp, cp, dsize * sizeof(float), cudaMemcpyHostToDevice);

	//computer
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((Nx + dimBlock.x - 1) / dimBlock.x, (Ny + dimBlock.y - 1) / dimBlock.y, (Nz + dimBlock.z - 1) / dimBlock.z);
	Kenel_LinerSolve << <dimGrid, dimBlock >> > (Device_L_phi, Device_L_phi0, Device_cp, c, Nx, Ny, Nz);
	cuSynchronize();
	cudaMemcpy(phi, Device_L_phi, dsize * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(Device_L_phi);
	cudaFree(Device_L_phi0);
	cudaFree(Device_cp);

}


void cudaHybridMultiPhaseFluid::cudaSetScalarFieldBoundary(float* field, float s, int Nx, int Ny, int Nz)
{
	
	int dsize = Nx*Ny*Nz;
	//device
	float* Device_field;
	cudaMalloc((void**)&Device_field, dsize * sizeof(float));
	cudaMemcpy(Device_field, field, dsize * sizeof(float), cudaMemcpyHostToDevice);
	
	//computer
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	//x=0
	dim3 dimGrid_x((Ny + dimBlock.y - 1) / dimBlock.y, (Nz + dimBlock.z - 1) / dimBlock.z);
	SetScalarFieldBoundary_x << <dimGrid_x, dimBlock >> > (Device_field, s, Nx, Ny, Nz);
	cuSynchronize();
	//y=0
	dim3 dimGrid_y((Nx + dimBlock.x - 1) / dimBlock.x, (Nz + dimBlock.z - 1) / dimBlock.z);
	SetScalarFieldBoundary_y << <dimGrid_y, dimBlock >> > (Device_field, s, Nx, Ny, Nz);
	cuSynchronize();
	//z=0
	dim3 dimGrid_z((Nx + dimBlock.x - 1) / dimBlock.x, (Ny + dimBlock.y - 1) / dimBlock.y);
	SetScalarFieldBoundary_z << <dimGrid_z, dimBlock >> > (Device_field, s, Nx, Ny, Nz);
	cuSynchronize();

	//xz=0
	dim3 dimGrid_xz((Nx + dimBlock.x - 1) / dimBlock.x);
	SetScalarFieldBoundary_xz << <dimGrid_xz, dimBlock >> > (Device_field, s, Nx, Ny, Nz);
    cuSynchronize();
	//yz=0
	dim3 dimGrid_yz((Ny + dimBlock.y - 1) / dimBlock.y);
	SetScalarFieldBoundary_yz << <dimGrid_yz, dimBlock >> > (Device_field, s, Nx, Ny, Nz);
	cuSynchronize();
	//xy=0
	dim3 dimGrid_xy((Nz + dimBlock.z - 1) / dimBlock.z);
	SetScalarFieldBoundary_xy << <dimGrid_xy, dimBlock >> > (Device_field,s, Nx, Ny, Nz);
	cuSynchronize();

    cudaMemcpy(field, Device_field, dsize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(Device_field);
	
}


void cudaHybridMultiPhaseFluid::cudaAdvectForward(float* dd, float* dd0, float3* v, int nx, int ny, int nz, float dt, float h, int dsize)
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

	//computer
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
	Kenel_AdvectForward << <dimGrid, dimBlock >> > (Ddevice_phi, Ddevice_phi0, Device_v, nx, ny, nz, dt, h);
	cuSynchronize();
	cudaMemcpy(dd, Ddevice_phi, dsize * sizeof(float), cudaMemcpyDeviceToHost);

	//Free space
	cudaFree(Ddevice_phi);
	cudaFree(Ddevice_phi0);
	cudaFree(Device_v);

}









