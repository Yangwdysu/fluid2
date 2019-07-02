#include "stdafx.h"
#include <time.h>
#include "DeformableSolid.h"
#include "MfdNumericalMethod/UniformGridQuery.h"

//#define DEBUG_INFORMATION

DeformableSolid::DeformableSolid(void)
{
	m_uniGrid = NULL;

	dataManager.AddArray("init_pos", &initPosArr);
	dataManager.AddArray("orient", &orientArr);
	dataManager.AddArray("marker", &marker);

	Initialize();

	youngModulus = 5000000.0f;
	poissonRatio = 0.49;

	lamda = poissonRatio*youngModulus/(1.0f+poissonRatio)/(1-2.0f*poissonRatio);
	mu = 0.5f*youngModulus/(1.0f+poissonRatio);
}

DeformableSolid::~DeformableSolid(void)
{
}

bool DeformableSolid::Initialize( string in_filename /*= "NULL" */ )
{
	AllocMemory(19*5*5, 0, 1);
	neighborListsRef.SetSpace(GetParticleNumber());
	initPosArr.SetSpace(GetParticleNumber());
	orientArr.SetSpace(GetParticleNumber());
	strainArr.SetSpace(GetParticleNumber());
	stressArr.SetSpace(GetParticleNumber());
	F_e.SetSpace(GetParticleNumber());
	F_v.SetSpace(GetParticleNumber());
	accArr.SetSpace(GetParticleNumber());
	marker.SetSpace(GetParticleNumber());

	Config::reload("config3d.txt");

	densityRef = Config::density;
	timeStep = Config::timestep*0.2f;
	damp = 100.0f*Config::viscosity;
	gravity = 0.0f*Vector3f(0.0f, 1.0f*Config::gravity, 0.0f);
	surfaceTension = Config::surfacetension;
	samplingDistance = Config::samplingdistance;
	smoothingLength = Config::samplingdistance*2.5f;

	int id = 0;
	for (int nx = -9; nx <= 9; nx++)
	{
		for (int nz = -2; nz <= 2; nz++)
		{
			for (int ny = -2; ny <= 2; ny++)
			{
				SolidParticle sp = GetSolidParticle(id);

				float x = 0.5f+nx*samplingDistance;
				float y = 0.5f+ny*samplingDistance;
				float z = 0.5f+nz*samplingDistance;

				if (nx == -9)
				{
					marker[id] = 1;
				}
				else if (nx == 9)
				{
					marker[id] = 2;
				}
				else
					marker[id] = 0;
				

				sp.SetMass(1.0f);
				sp.SetPosition(Vector3f(x, y, z));
				sp.SetInitPosition(Vector3f(x, y, z));
				sp.SetVelocity(Vector3f(0.0f, 0.0f, 0.0f));
				sp.SetMaterialType(MATERIAL_ELASTIC);
				if (nx <= -9 || nx == 9)
					sp.SetMotionType(MOTION_DISABLED);
				else
					sp.SetMotionType(MOTION_ENABLED);
				
				
				id++;
			}
		}
	}

	ComputeInitalNeighbors();
	ComputeDensity();
	float rho_max = 0.0f;
	for (int i = 0; i < N; i++)
	{
		if (rhoArr[i] > rho_max) rho_max = rhoArr[i];
	}

	float ratio = densityRef/rho_max;

	int pNum = GetParticleNumber();
	for (int i = 0; i < pNum; i++)
	{
		massArr[i] *= ratio;
	}

	ComputeVolume();

	cout << "mass: " << massArr[0] << endl;

	for (int i = 0; i < pNum; i++)
	{
		orientArr[i] = Rotation3Df::Identity();
	}

	return true;
}

void DeformableSolid::ComputeInitalNeighbors()
{
	clock_t t_start, t_end;
	t_start = clock();

	neighborListsRef.Zero();

	if (m_uniGrid != NULL) delete m_uniGrid;

	m_uniGrid = new UniformGridQuery(smoothingLength, lowBound, upBound);

	m_uniGrid->Construct( initPosArr, dataManager );


	t_end =  clock();

	cout << "Timer ConstructGrid: " << t_end-t_start << endl; 


	t_start = clock();

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		m_uniGrid->GetSizedNeighbors(initPosArr[i], smoothingLength, neighborListsRef[i], 30);
	}
	t_end = clock();
	cout << "Timer QuarySizedNeighbors: " << t_end-t_start << endl;
}

void DeformableSolid::ComputeDensity()
{
	Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Spiky);

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		float rho = 0.0f;
		NeighborList& neighborlist_i = neighborListsRef[i];
		int size_i = neighborlist_i.size;
		for ( int ne = 0; ne < size_i; ne++ )
		{
			int j = neighborlist_i.ids[ne];
			float r = neighborlist_i.distance[ne];
			rho += massArr[i]*kern.Weight(r, smoothingLength);
		}
		rhoArr[i] = rho;
	}
}

void DeformableSolid::ComputeVolume()
{
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		volArr[i] = massArr[i]/densityRef;
	}
}

void DeformableSolid::ComputeStrain()
{
	strainArr.Zero();
	Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Smooth);

	int pNum = GetParticleNumber();
	for (int i = 0; i < pNum; i++)
	{
		NeighborList& neighborlist_i = neighborListsRef[i];
		int size_i = neighborlist_i.size;
		float weight = 0;
		for ( int ne = 0; ne < size_i; ne++ )
		{
			int j = neighborlist_i.ids[ne];
			float r = neighborlist_i.distance[ne];
			float V_j = volArr[j];
			if (r > EPSILON)
			{
				Vector3f u_ji = posArr[j]-posArr[i]-orientArr[i]*(initPosArr[j]-initPosArr[i]);
				Vector3f d_ji = -V_j*(initPosArr[j]-initPosArr[i])*kern.Gradient(r, smoothingLength)*(1.0f/r);
				weight += abs(V_j*r*kern.Gradient(r, smoothingLength));
				strainArr[i](0, 0) += u_ji.x*d_ji.x;
				strainArr[i](0, 1) += u_ji.y*d_ji.x;
				strainArr[i](0, 2) += u_ji.z*d_ji.x;
				strainArr[i](1, 0) += u_ji.x*d_ji.y;
				strainArr[i](1, 1) += u_ji.y*d_ji.y;
				strainArr[i](1, 2) += u_ji.z*d_ji.y;
				strainArr[i](2, 0) += u_ji.x*d_ji.z;
				strainArr[i](2, 1) += u_ji.y*d_ji.z;
				strainArr[i](2, 2) += u_ji.z*d_ji.z;
			}
		}
		if (weight > EPSILON)
		{
			strainArr[i] *= (1.0f/weight);
		}

		strainArr[i] += strainArr[i].Transpose();
		strainArr[i] *= 0.5f;

// 		cout << strainArr[i](0, 0) << ", " << strainArr[i](0, 1) << ", " << strainArr[i](0, 2) << endl;
// 		cout << strainArr[i](1, 0) << ", " << strainArr[i](1, 1) << ", " << strainArr[i](1, 2) << endl;
// 		cout << strainArr[i](2, 0) << ", " << strainArr[i](2, 1) << ", " << strainArr[i](2, 2) << endl;
	}
}

void DeformableSolid::ComputeStress()
{
	int pNum = GetParticleNumber();
	for (int i = 0; i < pNum; i++)
	{
		MatrixSq3f& strain_i = strainArr[i];
		MatrixSq3f& stress_i = stressArr[i];
		float bulk = lamda * ( strain_i(0, 0)+strain_i(1, 1)+strain_i(2, 2) );
		MatrixSq3f B = MatrixSq3f::Identity()*bulk;

		stress_i = strain_i*2.0f*mu + B;

#ifdef DEBUG_INFORMATION
		cout << stressArr[i](0, 0) << ", " << stressArr[i](0, 1) << ", " << stressArr[i](0, 2) << endl;
		cout << stressArr[i](1, 0) << ", " << stressArr[i](1, 1) << ", " << stressArr[i](1, 2) << endl;
		cout << stressArr[i](2, 0) << ", " << stressArr[i](2, 1) << ", " << stressArr[i](2, 2) << endl;
#endif

// 		if (stressArr[i].Norm2() > EPSILON)
// 		{
// 			cout << "strain " << i << ": " << stressArr[i].Norm2() << endl;
// 		}
	}
}

void DeformableSolid::ComputeElasticForce()
{
	F_e.Zero();
	Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Spiky);
	Kernel& kern2 = KernelFactory::CreateKernel(KernelFactory::Smooth);

	int pNum = GetParticleNumber();
	for (int i = 0; i < pNum; i++)
	{
		NeighborList& neighborlist_i = neighborListsRef[i];
		int size_i = neighborlist_i.size;
		float V_i = volArr[i];
		float weight = 0;
		Vector3f tmp;
		for ( int ne = 0; ne < size_i; ne++ )
		{
			int j = neighborlist_i.ids[ne];
			float r = neighborlist_i.distance[ne];
			float V_j = volArr[j];
			float dist = (posArr[i]-posArr[j]).Length();
			if (r > EPSILON)
			{
				Vector3f f = stressArr[i]*((initPosArr[i]-initPosArr[j])*kern.Gradient(r, smoothingLength)*V_i*V_j*(1.0f/r))/**0.00000001f*kern2.Gradient(dist, smoothingLength)*/;
				F_e[i] += f;
				F_e[j] -= f;
			}
		}
	}
}

void DeformableSolid::ComputeViscousForce()
{
	Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Laplacian);

	F_v.Zero();
#pragma omp parallel for
	int pNum = GetParticleNumber();
	for (int i = 0; i < pNum; i++)
	{
		NeighborList& neighborlist_i = neighborListsRef[i];
		int size_i = neighborlist_i.size;
		float V_i = volArr[i];
		for ( int ne = 0; ne < size_i; ne++ )
		{
			int j = neighborlist_i.ids[ne];

			float r = neighborlist_i.distance[ne];
			float V_j = volArr[j];
			const Vector3f F_t = 0.5f*damp*V_i*V_j*kern.Weight(r, smoothingLength)*(velArr[j]-velArr[i]);
//			cout << "damp: " << 0.5f*damp*V_i*V_j*kern.Weight(r, smoothingLength) << endl;
			F_v[i] += F_t;
			F_v[j] -= F_t;
		}
	}

// 	for (int i = 0; i < N; i++)
// 	{
// 		cout << F_v[i].x << ", " << F_v[i].y << ", " << F_v[i].z << endl;
// 	}
}


void DeformableSolid::ComputeAcceleration()
{
	int pNum = GetParticleNumber();
	for (int i = 0; i < pNum; i++)
	{
		accArr[i] = (F_e[i]+F_v[i])/massArr[i]+gravity;
	}
}


void DeformableSolid::StepEuler( float dt )
{
	accArr.Zero();
	ComputeStrain();
	ComputeStress();

	ComputeElasticForce();
//	ComputeViscousForce();

	ComputeAcceleration();

	DampVelocity();

	int pNum = GetParticleNumber();
	for (int i = 0; i < pNum; i++)
	{
		SolidParticle sp = GetSolidParticle(i);
		
		if (sp.GetMotionTpye() == MOTION_ENABLED)
		{
			velArr[i] += accArr[i]*dt;
			posArr[i] += velArr[i]*dt;
		}
	}
}

void DeformableSolid::DampVelocity()
{
	Array<Vector3f> newVel;
	newVel.Zero();
	newVel.SetSpace(velArr.ElementCount());
	int pNum = GetParticleNumber();
	for (int i = 0; i < pNum; i++)
	{
		NeighborList& neighborlist_i = neighborListsRef[i];
		int size_i = neighborlist_i.size;
		float num = 1.0f;
		for ( int ne = 0; ne < size_i; ne++ )
		{
			int j = neighborlist_i.ids[ne];

			float r = neighborlist_i.distance[ne];
			float V_j = volArr[j];
			newVel[i] += velArr[j];
			num += 1.0f;
		}
		newVel[i] /= num;
	}

	for (int i = 0; i < pNum; i++)
	{
		velArr[i] = newVel[i];
	}
}

void DeformableSolid::Invoke( unsigned char type, unsigned char key, int x, int y )
{
	cout << "DeformableSolid Key Pressed: " << key << endl;
	switch (type) {
	case 'K':
		{
			switch(key){
			case 'u':
				{
					int pNum = GetParticleNumber();
					for (int i = 0; i < pNum; i++)
					{
						if (marker[i] == 1)
						{
							posArr[i] -= Vector3f(0.001f, 0.0f, 0.0f);
						}
						if (marker[i] == 2)
						{
							posArr[i] += Vector3f(0.001f, 0.0f, 0.0f);
						}
					}
				}
				
				break;
			case 'i':
				{
					int pNum = GetParticleNumber();
					for (int i = 0; i < pNum; i++)
					{
						if (marker[i] == 1)
						{
							posArr[i] += Vector3f(0.001f, 0.0f, 0.0f);
						}
						if (marker[i] == 2)
						{
							posArr[i] -= Vector3f(0.001f, 0.0f, 0.0f);
						}
					}
				}
				break;
			default:
				break;
			}
		}
		break;

	default : 
		break;
	}
}

float DeformableSolid::GetTimeStep()
{
	return timeStep;
}
