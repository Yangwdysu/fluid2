#pragma once

#include<stdlib.h>
#include<string.h>
#include"vector_types.h"

//#include "HybridMultiPhaseFluid.h"




namespace mfd {
#define NUM_THREAD 3
#define PHASE_SIZE 1
	class HybridMultiPhaseFluid;
	class cudaHybridMultiPhaseFluid
	{
	public:
		cudaHybridMultiPhaseFluid(void) {}
		void cudaAdvectForward(float* d, float* d0, float3* v, int Nx, int Ny, int Nz, float dt, float h, int dsize);
		void cudaLinerSolve(float* phi, float* phi0, float* cp, float c, int Nx, int Ny, int Nz);
		void cudaSetScalarFieldBoundary(float* field, float s, int Nx, int Ny, int Nz);
		void cudaUpdatePhi(float* d, float* d0, float dt, int Nx, int Ny, int Nz, float h);

	public:


	};



}