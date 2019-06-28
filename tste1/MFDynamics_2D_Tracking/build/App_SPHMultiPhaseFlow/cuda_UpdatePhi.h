#pragma once

#include<stdlib.h>
#include<string.h>
#include"vector_types.h"

//#include "HybridMultiPhaseFluid.h"




namespace mfd {
#define NUM_THREAD 3
#define PHASE_SIZE 1
	class HybridMultiPhaseFluid;
	class cuda_UpdatePhi
	{
	public:
		cuda_UpdatePhi(void) {}
		void cuda_AdvectForwardd(float* d, float* d0, float3* v, int Nx, int Ny, int Nz, float dt, float h, int dsize);




	public:


	};



}