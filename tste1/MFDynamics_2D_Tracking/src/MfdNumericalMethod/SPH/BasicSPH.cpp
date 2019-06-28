#include "BasicSPH.h"
#include <iostream>
#include <sstream>
#include <omp.h>
#include <time.h>



BasicSPH::BasicSPH(void)
{
	simItor = 0;
	N = -1;
	nFluid = -1;
	nRigid = -1;
	refRhoOfFluid = NULL;

	dataManager.AddArray("mass", &massArr);
	dataManager.AddArray("attribute", &attriArr);
	dataManager.AddArray("position", &posArr);
	dataManager.AddArray("velocity", &velArr);
	dataManager.AddArray("normal", &normalArr);


	smoothingLength = Config::samplingdistance*Config::smoothinglength;

	integratorType = Euler;

	lowBound = Vector3f( 0.0f, 0.0f, 0.0f );
	upBound = Vector3f( 1.0f, 1.0f, 1.0f );

	m_boundary = new Boundary;

//////////////////////////////////////////////////////////////////////////
#pragma omp parallel for
	for (int i = 0; i < nFluid; i++)
	{
		refRhoOfFluid[i] = 1000.0f;
	}
//////////////////////////////////////////////////////////////////////////
}

void BasicSPH::AllocMemory(int np, int nf, int nr)
{
	N = np;
	nFluid = nf;
	nRigid = nr;
	refRhoOfFluid = new float[nf];		memset(refRhoOfFluid, 0, sizeof(float)*nf);

	massArr.SetSpace(N);
	posArr.SetSpace(N);
	velArr.SetSpace(N);
	normalArr.SetSpace(N);
	FvisArr.SetSpace(N);
	FpArr.SetSpace(N);
	FsurArr.SetSpace(N);
	volArr.SetSpace(N);	
	preArr.SetSpace(N);	
	rhoArr.SetSpace(N);	
	attriArr.SetSpace(N);
}

BasicSPH::~BasicSPH(void)
{
}

bool BasicSPH::Initialize( string in_filename )
{
	InitalSceneBoundary();

	return true;
}


void BasicSPH::InitalSceneBoundary()
{
// 	DistanceField3D * df = new DistanceField3D(Config::constraint);
// 	m_boundary->increBarrier(new BarrierDistanceField3D(df));
}


void BasicSPH::ComputeNeighbors()
{

}


void BasicSPH::Advance( float dt )
{
	PreProcessing();

	cout << "---------------------------------Frame " << simItor << " Begin!--------------------------" << endl;
	clock_t t_start = clock();
	switch(integratorType){
	case Euler:
		StepEuler(dt);
		break;
	case PredictorCorrector:
		StepPredictorCorrector(dt);
		break;
	case RungeKutta4:
		StepRungeKutta4(dt);
		break;
	}
	clock_t t_end = clock();
	cout << "------------------------Costs totally " << t_end-t_start << " million seconds!----------------" << endl;

	PostProcessing();

	cout << endl << endl << endl;

	simItor++;
}

void BasicSPH::StepEuler( float dt )
{
}

void BasicSPH::StepPredictorCorrector( float dt )
{

}

void BasicSPH::StepRungeKutta4( float dt )
{

}

float BasicSPH::GetTimeStep()
{
	return timeStep;
}

void BasicSPH::BoundaryHandling()
{
	int pNum = GetParticleNumber();
#pragma omp parallel for
	for (int i = 0; i < pNum; i++)
	{
		m_boundary->Constrain(posArr[i], velArr[i]);
	}
}

void BasicSPH::SavePositions( string in_path, int in_iter )
{
	stringstream ss; ss << in_iter;
	string filename = in_path + string("pos_") + ss.str() + string(".txt");
	ofstream output_pos(filename.c_str(), ios::out|ios::binary);

	int total_num = 0;
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID && posArr[i].y < 0.91f)
		{
			total_num++;
		}
	}

	output_pos.write((char*)&total_num, sizeof(int));
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID && posArr[i].y < 0.91f)
		{
			output_pos.write((char*)&(posArr[i].x), sizeof(float));
			output_pos.write((char*)&(posArr[i].y), sizeof(float));
			output_pos.write((char*)&(posArr[i].z), sizeof(float));
		}
	}

	output_pos.close();
}

void BasicSPH::SaveVelocities( string in_path, int in_iter )
{

}




void BasicSPH::ComputeFieldStandard( Array<float>& out_field, Array<float>& in_field, Array<NeighborList>& neighbors, KernelFactory::KernelType type )
{
	if (out_field.ElementCount() != in_field.ElementCount())
		out_field.Reset(in_field.ElementCount());

	Kernel& kern = KernelFactory::CreateKernel(type);

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		float val = 0.0f;
		NeighborList& neighborlist_i = neighbors[i];
		int size_i = neighborlist_i.size;
		for ( int ne = 0; ne < size_i; ne++ )
		{
			int j = neighborlist_i.ids[ne];
			float r = neighborlist_i.distance[ne];
			float V_j = volArr[j];
			val += V_j*in_field[j]*kern.Weight(r, smoothingLength);
		}
		out_field[i] = val;
	}
}

void BasicSPH::ComputeGradientStandard( Array<Vector3f>& out_grad, Array<float>& in_field, Array<NeighborList>& neighbors, KernelFactory::KernelType type )
{
	if (out_grad.ElementCount() != in_field.ElementCount())
		out_grad.Reset(in_field.ElementCount());

	Kernel& kern = KernelFactory::CreateKernel(type);

	out_grad.Zero();
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		NeighborList& neighborlist_i = neighbors[i];
		int size_i = neighborlist_i.size;
		float V_i = volArr[i];
		for ( int ne = 0; ne < size_i; ne++ )
		{
			int j = neighborlist_i.ids[ne];
			float r = neighborlist_i.distance[ne];

			if ( r > smoothingLength*EPSILON )
			{
				float V_j = volArr[j];
				const Vector3f F_t = 0.5f*V_i*V_j*kern.Gradient(r, smoothingLength)*(in_field[i]+in_field[j])*(posArr[j]-posArr[i]) * (1.0f/r);
				out_grad[i] += F_t;
				out_grad[j] -= F_t;
			}
		}
	}
}
