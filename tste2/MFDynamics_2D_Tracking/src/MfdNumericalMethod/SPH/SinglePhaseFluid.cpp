#include <time.h>
#include "SinglePhaseFluid.h"
#include "MfdNumericalMethod/UniformGridQuery.h"
#include "MfdNumericalMethod/SPH/Kernels.h"

using namespace mfd;

SinglePhaseFluid::SinglePhaseFluid(void)
{
	phiArr = NULL;
	energyArr = NULL;
	color = NULL;
	m_uniGrid = NULL;
	massInSmall = 1.0f;
	massInLarge = 1.0f;
	segLow = 0.2f;
	segHigh = 0.4f;
	
	Initialize();
}


SinglePhaseFluid::~SinglePhaseFluid(void)
{
	if(phiArr != NULL) delete[] phiArr;
	if(energyArr != NULL) delete[] energyArr;
}

#define BUNNY

bool SinglePhaseFluid::Initialize(string in_filename)
{
/*	InitalSceneBoundary();
	ifstream input(in_filename.c_str(), ios::in);
	if (input.is_open())
	{
		ReadFromFile(input);
	}
	else
	{
		SPH* _sph = new SPH();
		vector<Particle>& fluidparticles = ((SPH*)_sph)->getParticles();
		const int nbparticles = fluidparticles.size();

		int nbunny = 0;

#ifdef BUNNY
		DistanceField3D * bunny = new DistanceField3D("data/bunny.set");

		Vector3f lo = Config::scenelowerbound;
		Vector3f hi = Config::sceneupperbound;
		float spacing = Config::samplingdistance;
		float dist;
		for (float x = lo.x; x < hi.x; x += spacing)
		{
			for (float y = lo.y; y < hi.y; y += spacing)
			{
				for (float z = lo.z; z < hi.z; z += spacing)
				{
					bunny->distance(Vector3f(x, y, z), dist);
					if (dist < 0.5f*spacing && dist > -3.0f*spacing)
					{
						nbunny++;
					}
				}
			}
		}
#endif
		

		cout << "Total particle number: " << nbparticles+nbunny << endl;

		AllocMemory(nbparticles+nbunny);

		for (int i = 0; i < nbparticles; i++)
		{
			massArr[i] = 1.0f;
			posArr[i] = fluidparticles[i];
			velArr[i] = fluidparticles[i].v;
			SetMaterialType(i, MATERIAL_FLUID);
			if (fluidparticles[i].m_status == UnifiedParticle::FORCE_DRIVEN)
			{
				SetDynamicType(i, DYNAMIC_POSITIVE);
			}
			else
				SetDynamicType(i, DYNAMIC_PASSIVE);
		}

#ifdef BUNNY
		int bunnyindex = nbparticles;
		for (float x = lo.x; x < hi.x; x += spacing)
		{
			for (float y = lo.y; y < hi.y; y += spacing)
			{
				for (float z = lo.z; z < hi.z; z += spacing)
				{
					bunny->distance(Vector3f(x, y, z), dist);
					if (dist < 0.5f*spacing && dist > -3.0f*spacing)
					{
						if (dist < 0.0f)
						{
							massArr[bunnyindex] = 1.0f;
						}
						else
							massArr[bunnyindex] = 1.0f-abs(dist)/spacing;
						
						posArr[bunnyindex] = Vector3f(x, y, z);
						velArr[bunnyindex] = Vector3f(0.0f, 0.0f, 0.0f);
						SetDynamicType(bunnyindex, DYNAMIC_PASSIVE);
						SetMaterialType(bunnyindex, MATERIAL_RIGID);
						bunnyindex++;
					}
				}
			}
		}
		delete bunny;
#endif

		
		ComputeNeighbors();
		ComputeDensity();
		float rhoInLarge_max = 0.0f;
		float rhoInSmall_max = 0.0f;
		for (int i = 0; i < N; i++)
		{
			if (rhoArr[i] > rhoInLarge_max) rhoInLarge_max = rhoArr[i];
			if (rhoInSmallArr[i] > rhoInSmall_max) rhoInSmall_max = rhoInSmallArr[i];
		}
		float ratioInLarge = Config::density/rhoInLarge_max;
		float ratioInSmall = Config::density/rhoInSmall_max;

		massInLarge *= ratioInLarge;
		massInSmall *= ratioInSmall;

		for (int i = 0; i < N; i++)
		{
			massArr[i] = massInLarge;
		}
		delete _sph;
	}*/


	AllocMemory(15*15*15);
	neighborLists.SetSpace(N);

	Config::reload("config3d.txt");

	timeStep = Config::timestep;
	viscosity = Config::viscosity;
	gravity = Config::gravity;
	surfaceTension = Config::surfacetension;
	samplingDistance = Config::samplingdistance;
	smoothingLength = Config::samplingdistance*Config::smoothinglength;

	lengthInSmall = samplingDistance*1.5f;
	lengthInLarge = smoothingLength;

	refRhoOfFluid[0] = Config::density;

	int id = 0;
	for (int nx = -7; nx <= 7; nx++)
	{
		for (int nz = -7; nz <=7; nz++)
		{
			for (int ny = -7; ny <= 7; ny++)
			{
				Particle p = GetParticle(id);

				float x = 0.5f+nx*samplingDistance;
				float y = 0.5f+ny*samplingDistance;
				float z = 0.5f+nz*samplingDistance;

				p.SetMass(1.0f);
				p.SetPosition(Vector3f(x, y, z));
				p.SetVelocity(Vector3f(0.0f, 0.0f, 0.0f));
				p.SetMaterialType(MATERIAL_FLUID);
				p.SetDynamicType(DYNAMIC_POSITIVE);
				
// 				massArr[id] = 1.0f;
// 				posArr[id] = Vector3f(x, y, z);
// 				velArr[id].Zero();
// 				SetMaterialType(id, MATERIAL_FLUID);
// 				SetDynamicType(id, DYNAMIC_POSITIVE				
				
				id++;
			}
		}
	}

	ComputeNeighbors();
	ComputeDensity();
	float rhoInLarge_max = 0.0f;
	float rhoInSmall_max = 0.0f;
	for (int i = 0; i < N; i++)
	{
		if (rhoArr[i] > rhoInLarge_max) rhoInLarge_max = rhoArr[i];
		if (rhoInSmallArr[i] > rhoInSmall_max) rhoInSmall_max = rhoInSmallArr[i];
	}
	float ratioInLarge = refRhoOfFluid[0]/rhoInLarge_max;
	float ratioInSmall = refRhoOfFluid[0]/rhoInSmall_max;

	massInLarge *= ratioInLarge;
	massInSmall *= ratioInSmall;

	for (int i = 0; i < N; i++)
	{
		massArr[i] = massInLarge;
	}

	cout << "eeeeeeeeeee" << endl;



	return true;
}



void SinglePhaseFluid::ComputePressureTensor()
{
	float beta = 1.0f;

	Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Smooth);

	memset(pressureTensorArr, 0, N*sizeof(MatrixSq3f));

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			if(smallScale[i])
			{
				NeighborList& neighborlist_i = neighborLists[i];
				int size_i = neighborlist_i.size;
				for ( int ne = 0; ne < size_i; ne++ )
				{
					int j = neighborlist_i.ids[ne];
					float r = neighborlist_i.distance[ne];

					float weight = kern.Weight(r, lengthInLarge);

					pressureTensorArr[i].x[0] += (float)(posArr[j].x-posArr[i].x)*(posArr[j].x-posArr[i].x)*weight;
					pressureTensorArr[i].x[1] += (float)(posArr[j].y-posArr[i].y)*(posArr[j].x-posArr[i].x)*weight;
					pressureTensorArr[i].x[2] += (float)(posArr[j].z-posArr[i].z)*(posArr[j].x-posArr[i].x)*weight;
					pressureTensorArr[i].x[3] += (float)(posArr[j].x-posArr[i].x)*(posArr[j].y-posArr[i].y)*weight;
					pressureTensorArr[i].x[4] += (float)(posArr[j].y-posArr[i].y)*(posArr[j].y-posArr[i].y)*weight;
					pressureTensorArr[i].x[5] += (float)(posArr[j].z-posArr[i].z)*(posArr[j].y-posArr[i].y)*weight;
					pressureTensorArr[i].x[6] += (float)(posArr[j].x-posArr[i].x)*(posArr[j].z-posArr[i].z)*weight;
					pressureTensorArr[i].x[7] += (float)(posArr[j].y-posArr[i].y)*(posArr[j].z-posArr[i].z)*weight;
					pressureTensorArr[i].x[8] += (float)(posArr[j].z-posArr[i].z)*(posArr[j].z-posArr[i].z)*weight;
				}
				float norm2 = pressureTensorArr[i].Norm2();
				if (smallScale[i] || norm2 < EPSILON)
					pressureTensorArr[i] *= (beta/norm2);
				else
					pressureTensorArr[i] = MatrixSq3f::Identity();
			}
			else
				pressureTensorArr[i] = MatrixSq3f::Identity();
		}
	}
}

#ifdef ENERGY_TENSION

void SinglePhaseFluid::ComputeSurfaceTension()
{
// 	ComputeSuraceTensionInMD();
// 	return;
	Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Smooth);

	memset(phiArr, 0, N*sizeof(float));
	memset(energyArr, 0, N*sizeof(float));

	Vector3f* gradientPhiArr = new Vector3f[N];	
	float * weights = new float[N];
	memset(weights, 0, N*sizeof(float));
	memset(gradientPhiArr, 0, N*sizeof(Vector3f));
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			float totalWeight = 0.0f;
			float totalWeight2 = 0.0f;
			for ( int ne = 0; ne < size_i; ne++ )
			{
				int j = neighborlist_i.ids[ne];
				if ((MATERIALTYPE(attriArr[j])) == MATERIAL_FLUID)
				{
					float r = neighborlist_i.distance[ne];
					if (r > lengthInLarge*EPSILON)
					{
						float weight = kern.Gradient(r, lengthInLarge);
						totalWeight += weight;
						totalWeight2 += kern.Weight(r, lengthInLarge);
						gradientPhiArr[i] += weight * (posArr[j]-posArr[i]) * (1.0f/r);
					}
				}
			}
			totalWeight += kern.Weight(0.0f, lengthInLarge);
			totalWeight = totalWeight > EPSILON ? totalWeight : 1.0f;

			weights[i] = totalWeight2;
			gradientPhiArr[i] /= totalWeight2;
		}
		phiArr[i] = pow(gradientPhiArr[i].Length(), 2.0f);
	}

/*	memset(gradientPhiArr, 0, N*sizeof(Vector3f));

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			float totalWeight = 0.0f;
			for ( int ne = 0; ne < size_i; ne++ )
			{
				int j = neighborlist_i.ids[ne];
				float r = neighborlist_i.distance[ne];
				if (r > lengthInLarge*EPSILON)
				{
					float weight = Kernels::smoothgradient(lengthInLarge, r);
					totalWeight += weight;
					gradientPhiArr[i] += weight * (phiArr[j]+phiArr[i])*(posArr[j]-posArr[i]) * (1.0f/r);
				}
			}
			totalWeight +=  Kernels::smoothgradient(lengthInLarge, 0.0);

			gradientPhiArr[i] /= totalWeight;//*= (massInLarge/rhoArr[i]);
			energyArr[i] = gradientPhiArr[i].getSquaredLength();

		}
	}

	float maxv = 0.0f;
	float minv = 1000.0f;
	for (int i = 0; i < N; i++)
	{
		if(energyArr[i] > maxv) maxv = energyArr[i];
		if(energyArr[i] < minv) minv = energyArr[i];
	}
	cout << "max value: " << maxv << " min value: " << minv << endl;
*/

	float ceof = 0.00000000035f;
	float airPressure = 10000.0f;
	//float ceof = 0.000000001f;
	//float ceof = 10000.0f;
//	float ceof = 50000.0f;

	

	FsurArr.Zero();
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			float V_i = volArr[i];
			float ratio = (float)neighborlist_i.size/30.0f;
			float scalev = 1.0f;//pow(ratio, 1);
			for ( int ne = 0; ne < size_i; ne++ )
			{
				int j = neighborlist_i.ids[ne];
				float r = neighborlist_i.distance[ne];
				if (r > lengthInLarge*EPSILON)
				{
					float V_j = volArr[j];
					Vector3f temp = V_i*V_j*kern.Gradient(r, lengthInLarge)*(posArr[j]-posArr[i]) * (1.0f/r);
					Vector3f Fs1, Fs2, Fs3;
					if ((MATERIALTYPE(attriArr[j])) == MATERIAL_FLUID)
					{
						Fs1 = scalev*ceof*1.0f*(phiArr[i]+phiArr[j])*temp;
						Fs2 = scalev*airPressure*temp;
						Fs3 = scalev*ceof*3.0f*(phiArr[i])*temp;
					}
// 					else
// 					{
// 						Fs1.zero();
// 						Fs2 = scalev*airPressure*temp;
// 						Fs3.zero();
// 					}
					FsurArr[i] -= (Fs1+Fs2);
					FsurArr[j] += (Fs1+Fs2);
				}
				
			}
		}
	}

	delete[] gradientPhiArr;
	delete[] weights;

	CheckScale();
}

void SinglePhaseFluid::ComputeDensity()
{
	Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Spiky);

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			float rhoInLarge = 0.0f;
			float rhoInSmall = 0.0f;
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			for ( int ne = 0; ne < size_i; ne++ )
			{
				int j = neighborlist_i.ids[ne];
				if ((MATERIALTYPE(attriArr[j])) == MATERIAL_FLUID)
				{
					float r = neighborlist_i.distance[ne];
					rhoInLarge += massInLarge*kern.Weight(r, lengthInLarge);
					if (r <= lengthInSmall)
					{
						rhoInSmall += massInSmall*kern.Weight(r, lengthInSmall);
					}
				}
			}
			rhoArr[i] = rhoInLarge;
			rhoInSmallArr[i] = rhoInSmall;
		}
	}
}

void SinglePhaseFluid::ComputePressure( float dt )
{
	float maxSmall = 0.0f;
	float maxLarge = 0.0f;
	float inv_sec = 1.0f/(segHigh-segLow);
	for (int i = 0; i < N; i++)
	{
		if(rhoArr[i] > maxLarge) maxLarge = rhoArr[i];
		if(rhoInSmallArr[i] > maxSmall) maxSmall = rhoInSmallArr[i];
	}
	cout << "Large: " << maxLarge << endl;
	cout << "Small: " << maxSmall << endl;

	preArr.Zero();
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			float t_mass = massInLarge;
			float t_density = rhoArr[i];

			float t_pressureInLarge;
			float radius = pow(0.75*t_mass/(rhoArr[i]*M_PI), 1.0/3.0);
#ifndef KERNEL_3D
			t_pressureInLarge = M_PI*radius*radius*(log(radius)-0.5)*(t_density-refRhoOfFluid[0])*t_density/(refRhoOfFluid[0]*dt*dt);
			t_pressureInLarge /= (-M_PI);
			t_pressureInLarge *= 1.25f;

#else
			t_pressureInLarge = radius*radius*(t_density-refRhoOfFluid[0])*t_density/(refRhoOfFluid[0]*dt*dt);
			t_pressureInLarge *= 300.0f;
#endif

			t_mass = massInSmall;
			t_density = rhoInSmallArr[i];

			float t_pressureInSmall;
			radius = pow(0.75*t_mass/(rhoArr[i]*M_PI), 1.0/3.0);
#ifndef KERNEL_3D
			t_pressureInSmall = M_PI*radius*radius*(log(radius)-0.5)*(t_density-refRhoOfFluid[0])*t_density/(refRhoOfFluid[0]*dt*dt);
			t_pressureInSmall /= (-M_PI);
			t_pressureInSmall *= 1.25f;

#else
			t_pressureInSmall = radius*radius*(t_density-refRhoOfFluid[0])*t_density/(refRhoOfFluid[0]*dt*dt);
			t_pressureInSmall *= 0.1f;
#endif

			// 			if (smallScale[i])
			// 				pressureArr[i] = t_pressureInSmall;
			// 			else
			// 				pressureArr[i] = t_pressureInLarge;
			float loc = phiArr[i];
			if (loc > segHigh)
				loc = segHigh;
			else if (loc < segLow)
			{
				loc = segLow;
			}
//			pressureArr[i] = inv_sec*(t_pressureInSmall*(loc-segLow)+t_pressureInLarge*(segHigh-loc));
// 			if (smallScale[i])
// 			{
// 				pressureArr[i] = t_pressureInSmall;
// 
// 			}
// 			else
			{
				preArr[i] = t_pressureInLarge;
			}

			if(preArr[i] < 0.0f) preArr[i] = 0.0f;
		}
	}
}

void SinglePhaseFluid::ComputePressureForce()
{
	Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Spiky);

	ComputePressureTensor();

	FpArr.Zero();
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			float V_i = volArr[i];
			for ( int ne = 0; ne < size_i; ne++ )
			{
				int j = neighborlist_i.ids[ne];
				float r = neighborlist_i.distance[ne];

				if ( r > smoothingLength*EPSILON && (MATERIALTYPE(attriArr[j])) == MATERIAL_FLUID)
				{
					float V_j = volArr[j];
					const Vector3f F_t = 0.5f*V_i*V_j*kern.Gradient(r, lengthInLarge)*((pressureTensorArr[i]*preArr[i]+pressureTensorArr[j]*preArr[j])*(posArr[j]-posArr[i]) * (1.0f/r));
					FpArr[i] += F_t;
					FpArr[j] -= F_t;
				}
			}
		}
	}
}
#endif

void SinglePhaseFluid::CheckScale()
{
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
//		smallScale[i] = phiArr[i] > segLow ? true : false;
		smallScale[i] = false ? true : false;
//		smallScale[i] = rhoArr[i] < Config::density ? true : false;

		if(smallScale[i]) color[i] = 1.0f;
		else color[i] = 0.0f;
	}
}

void SinglePhaseFluid::PostProcessing()
{
// #pragma omp parallel for
// 	for (int i = 0; i < N; i++)
// 	{
// 		if (posArr[i].y < 0.9f)
// 		{
// 			SetDynamicType(i, DYNAMIC_POSITIVE);
// 		}
// 	}

// 	string path = "Output/no_airpressure/";
// 	if (simItor < 3000)
// 	{
// 		int mod = 6;
// 		if (simItor % mod == 0)
// 		{
// 			SavePositions(path, simItor/mod);
// 		}
// 	}
// 	else
// 	{
// 		exit(0);
//  	}

// 	if (simItor == 100)
// 	{
// 		SaveToFile("info.txt");
// 		exit(0);
// 	}
}

void SinglePhaseFluid::InitalSceneBoundary()
{
/*	DistanceField3D * scene = new DistanceField3D(Config::scenelowerbound-0.1f, Config::sceneupperbound+0.1f, 180, 120, 180);
//	df->meshToDistanceField2(mesh);
	scene->distanceFieldToBox(Config::scenelowerbound, Config::sceneupperbound, true);
	m_boundary->increBarrier(new BarrierDistanceField3D(scene));

	df = new DistanceField3D("data/bunny.set");
	df->constructMesh();
	df->dumpPovray("bunny_pov", 0);
	//	df->meshToDistanceField2(mesh);
	//	df->boxToDistanceField();
	m_boundary->increBarrier(new BarrierDistanceField3D(df));*/
}

void SinglePhaseFluid::SaveToFile(string in_filename)
{
	ofstream output(in_filename.c_str(), ios::out);

	output << N << endl;
	output << massInLarge << endl;
	output << massInSmall << endl;
	for (int i = 0; i < N; i++)
	{
		output << "position " << posArr[i].x << " " << posArr[i].y << " " << posArr[i].z << endl;
		output << "velocity " << velArr[i].x << " " << velArr[i].y << " " << velArr[i].z << endl;
		output << "attribute " << attriArr[i] << endl;
	}

	output.close();
}

void SinglePhaseFluid::ReadFromFile( ifstream& input_stream )
{
	input_stream >> N;
	input_stream >> massInLarge;
	input_stream >> massInSmall;
	AllocMemory(N);
	
	string dummy;
	for (int i = 0; i < N; i++)
	{
		massArr[i] = massInLarge;
		input_stream >> dummy >> posArr[i].x >> posArr[i].y >> posArr[i].z;
		input_stream >> dummy >> velArr[i].x >> velArr[i].y >> velArr[i].z;
		input_stream >> dummy >> attriArr[i];
	}
}

void SinglePhaseFluid::AllocMemory( int in_nf )
{
	BasicSPH::AllocMemory(in_nf, 1, 0);

	phiArr = new float[N];			memset(phiArr, 0, N*sizeof(float));
	energyArr = new float[N];		memset(energyArr, 0, N*sizeof(float));
	smallScale = new bool[N];		memset(energyArr, 0, N*sizeof(bool));
	color = new float[N];			memset(energyArr, 0, N*sizeof(float));
	pressureTensorArr = new MatrixSq3f[N];
	rhoInSmallArr = new float[N];	memset(energyArr, 0, N*sizeof(float));
}

/*void SinglePhaseFluid::CallBackKeyboardFunc( unsigned char key, int x, int y )
{
	switch(key) {
	case 's':
		SaveToFile("info.txt");

		break;
	default:
		BasicSPH::CallBackKeyboardFunc(key, x, y);
		break;
	}
}*/

/*void SinglePhaseFluid::writeToPovray( string filename, int dumpiter )
{
	int framenumber = dumpiter-1000000;
	stringstream ss2; ss2 << framenumber;
	stringstream ss; ss << dumpiter;
	filename += string("spheres_") + ss.str() + string(".pov");
	ofstream output(filename.c_str(), ios::out);
	output << "union {" << endl;
	for (int i=0; i<N; i++) {
		//if (m_uniparticles[i].m_bfluid)
		{
			if (posArr[i].x < 0.5975f && posArr[i].x > 0.4025f && posArr[i].z < 0.5975f && posArr[i].z > 0.4025f && posArr[i].y < 0.498f)
			{
				if (DYNAMICTYPE(attriArr[i]) == DYNAMIC_POSITIVE)
				{
					output << "      sphere { <" << posArr[i].x << ", " << posArr[i].z << ", " << posArr[i].y << ">  Rad  pigment {  color DYNAMICPARTICLE } }" <<  endl;
				}
				else
				{
					output << "      sphere { <" << posArr[i].x << ", " << posArr[i].z << ", " << posArr[i].y << ">  Rad  pigment {  color FIXEDPARTICLE } }" <<  endl;
				}
			}
		}
	}
	output << endl;
	output << "no_shadow" << endl;
	output << "}" << endl;
	output.close();
}*/

void SinglePhaseFluid::ComputeNeighbors()
{
	clock_t t_start, t_end;
	t_start = clock();

	neighborLists.Zero();

	if (m_uniGrid != NULL) delete m_uniGrid;

	m_uniGrid = new UniformGridQuery(smoothingLength, lowBound, upBound);

	m_uniGrid->Construct( posArr, dataManager );


	t_end =  clock();

	cout << "Timer ConstructGrid: " << t_end-t_start << endl; 

// 	dataManager.Reordering(ridArr, N);
// 
// 	m_uniGrid->SetPositionRef(posArr.DataPtr());

	t_start = clock();

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		//		m_uniGrid->QuaryNeighbors(posArr[i], smoothingLength, neighborLists[i]);
		if (MATERIALTYPE(attriArr[i]) == MATERIAL_FLUID)
			m_uniGrid->GetSizedNeighbors(posArr[i], smoothingLength, neighborLists[i], 30);
		//		cout << neighborLists[i].size << endl;
	}
	t_end = clock();
	cout << "Timer QuarySizedNeighbors: " << t_end-t_start << endl; 

}

void SinglePhaseFluid::ComputeVolume()
{
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
			volArr[i] = massArr[i]/rhoArr[i];
		else
			volArr[i] = massArr[i]/refRhoOfFluid[0];
	}
}

void mfd::SinglePhaseFluid::StepEuler( float dt )
{
	clock_t t_start, t_end;
	t_start = clock();
	ComputeNeighbors();
	t_end = clock();
	cout << "Timer ComputeNeighbors: " << t_end-t_start << endl;

	t_start = clock();
	ComputeDensity();
	t_end = clock();
	cout << "Timer ComputeDensity: " << t_end-t_start << endl; 

	ComputeVolume();


	t_start = clock();
	ComputeSurfaceTension();
	t_end = clock();
	cout << "Timer Tension: " << t_end-t_start << endl; 
	t_start = clock();
	//ComputeSurfaceTension();
	ComputeViscousForce();
	t_end = clock();
	cout << "Timer Viscous: " << t_end-t_start << endl;
	t_start = clock();
	ComputePressure(dt);
	t_end = clock();
	cout << "Timer Pressure: " << t_end-t_start << endl;
	t_start = clock();
	ComputePressureForce();
	t_end = clock();
	cout << "Timer PressureForce: " << t_end-t_start << endl;

	Advect(dt);
	cout << "gravity: " << Config::gravity << endl;

	BoundaryHandling();
}


/*
void SinglePhaseFluid::ComputePressureForce()
{
		Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Spiky);

	FpArr.Zero();
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			float V_i = volArr[i];
			for ( int ne = 0; ne < size_i; ne++ )
			{
				int j = neighborlist_i.ids[ne];
				float r = neighborlist_i.distance[ne];

				if ( r > smoothingLength*EPSILON )
				{
					float V_j = volArr[j];
					const Vector3f F_t = 0.5f*V_i*V_j*kern.Gradient(r, smoothingLength)*(preArr[i]+preArr[j])*(posArr[j]-posArr[i]) * (1.0f/r);
					FpArr[i] += F_t;
					FpArr[j] -= F_t;
				}
			}
		}
	}
}

void SinglePhaseFluid::ComputePressure( float dt )
{
	preArr.Zero();
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			float tp;
			float radius = pow(0.75*massArr[i]/(rhoArr[i]*(float)M_PI), 1.0/3.0);
#ifndef KERNEL_3D
			tp = M_PI*radius*radius*(log(radius)-0.5)*(rhoArr[i]-refRhoOfFluid[0])*rhoArr[i]/(refRhoOfFluid[0]*dt*dt);
			tp /= (-M_PI);
			tp *= 1.25f;

#else
			tp = radius*radius*(rhoArr[i]-Config::density)*rhoArr[i]/(Config::density*dt*dt);
			tp *= 300.0f;
#endif

			if(tp < 0.0f) tp = 0.0f;
			preArr[i] = tp;
		}
	}
}*/

void SinglePhaseFluid::ComputeSuraceTensionInCurvature()
{
		FsurArr.Zero();
	Array<Vector3f> surfacetension;
	surfacetension.SetSpace(N);
	surfacetension.Zero();

	Array<float> colorfield;
	colorfield.SetSpace(N);
	colorfield.Zero();


// for 2d case
	Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Smooth);

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			float gradient = 0.0f;
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			float V_i = volArr[i];
			for ( int ne = 0; ne < size_i; ne++ )
			{
				int j = neighborlist_i.ids[ne];
				float r = neighborlist_i.distance[ne];
				float V_j = volArr[j];
				
				colorfield[i] += V_j*kern.Weight(r, smoothingLength);
			}
		}
	}

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			float gradient = 0.0f;
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			float V_i = volArr[i];
			for ( int ne = 0; ne < size_i; ne++ )
			{
				int j = neighborlist_i.ids[ne];
				float r = neighborlist_i.distance[ne];
				if (r > EPSILON*smoothingLength)
				{
					float V_j = volArr[j];
					gradient = kern.Gradient(r, smoothingLength);
					Vector3f tension = 0.5f * colorfield[j]* V_j * gradient * (posArr[i] - posArr[j]) * (1.0f / r);
					surfacetension[i] += tension;
					surfacetension[j] -= tension;
				}
			}
		}
	}

	float normalThresh = 0.000001f;
//	float normalThresh = 0.00000001f;
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if (surfacetension[i].Length() > normalThresh)
		{
			surfacetension[i].Normalize();
		}
		else
			surfacetension[i] = 0.0f;
	}

	float coef = 400.0;

#pragma omp parallel for
	for (int i=0; i<N; i++) 
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			float gradient = 0.0f;
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			float V_i = volArr[i];
			for (int ne=0; ne<size_i; ne++) 
			{
				int j = neighborlist_i.ids[ne];
				float r = neighborlist_i.distance[ne];
				if (r > EPSILON*smoothingLength)
				{
					float V_j = volArr[j];
					gradient = kern.Gradient(r, smoothingLength);
					if (surfacetension[i].Length() > 0.5f && surfacetension[j].Length() > 0.5f)
					{
					Vector3f n_color = surfacetension[j] - surfacetension[i];
					Vector3f n_ij = (posArr[i] - posArr[j]) * (1.0f / r);
					//div is positive for convex surface
					float div = (n_color.Dot(n_ij));
					//if (div > 0.0f)
					{
						Vector3f Fs = - coef * V_i * V_j * gradient * div * surfacetension[i];//upi.m_surfacetension;
						Vector3f Fp = - 0.5f*coef * V_i * V_j * gradient * n_ij;
						{
							FsurArr[i] += Fs;
							FsurArr[i] -= Fp;
							FsurArr[j] += Fp;
						}
					}
					}
				}
			}
		}

	}


/*#pragma omp parallel for
	for (int i=0; i<N; i++) 
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			float gradient = 0.0f;
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			float V_i = volArr[i];
			for (int ne=0; ne<size_i; ne++) 
			{
				int j = neighborlist_i.ids[ne];
				float r = neighborlist_i.distance[ne];
				if (r > EPSILON*smoothingLength)
				{
					float V_j = volArr[j];
					gradient = Kernels::smoothgradient(smoothingLength, r);
					Vector3f n_color = surfacetension[i].length()*surfacetension[j] - surfacetension[j].length()*surfacetension[i];
					Vector3f n_ij = (posArr[i] - posArr[j]) * (1.0f / r);
					//div is positive for convex surface
					float div = (n_color.dot(n_ij));
					//if (div > 0.0f)
					{
						Vector3f Fs = - coef * V_i * V_j * gradient * div * surfacetension[i];//upi.m_surfacetension;
						Vector3f Fp = - 1.0f*coef * V_i * V_j * gradient * n_ij;
						{
							FsurArr[i] += Fs;
							FsurArr[i] -= Fp;
							FsurArr[j] += Fp;
						}
					}
				}
			}
		}

	}*/
}

void SinglePhaseFluid::ComputeSuraceTensionInMD()
{
	Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Smooth);

	FsurArr.Zero();

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			float V_i = volArr[i];
			for ( int ne = 0; ne < size_i; ne++ )
			{
				int j = neighborlist_i.ids[ne];
				float r = neighborlist_i.distance[ne];
				if (r > EPSILON*smoothingLength)
				{
					float V_j = volArr[j];
					Vector3f dn = V_j*V_i*kern.Gradient(r, smoothingLength) * (posArr[i]-posArr[j]) * (1.0f / r);
					FsurArr[i] += 0.5f*dn;
					FsurArr[j] -= 0.5f*dn;
				}
			}
		}
	}

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			Vector3f nn = FsurArr[i];

			FsurArr[i] *= (1.0f*surfaceTension);
		}
	}
}

void SinglePhaseFluid::Advect( float dt )
{
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		Particle p = GetParticle(i);
		if (p.GetMaterialType() == MATERIAL_FLUID)
		{
			if (p.GetDynamicsType() == DYNAMIC_POSITIVE)
			{
				velArr[i] += dt/massArr[i]*(FvisArr[i]+FpArr[i]+FsurArr[i]+Vector3f(0.0f, massArr[i]*gravity, 0.0f));
				posArr[i] += dt*velArr[i];
			}
			else
			{
				posArr[i] += dt*velArr[i];
			}
		}
	}
}

void SinglePhaseFluid::ComputeViscousForce()
{
	Kernel& kern = KernelFactory::CreateKernel(KernelFactory::Laplacian);

	FvisArr.Zero();
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		if ((MATERIALTYPE(attriArr[i])) == MATERIAL_FLUID)
		{
			NeighborList& neighborlist_i = neighborLists[i];
			int size_i = neighborlist_i.size;
			float V_i = volArr[i];
			for ( int ne = 0; ne < size_i; ne++ )
			{
				int j = neighborlist_i.ids[ne];
				if ((MATERIALTYPE(attriArr[j])) == MATERIAL_FLUID)
				{
					float r = neighborlist_i.distance[ne];
					float V_j = volArr[j];
					const Vector3f F_t = 0.5f*viscosity*V_i*V_j*kern.Weight(r, smoothingLength)*(velArr[j]-velArr[i]);
					FvisArr[i] += F_t;
					FvisArr[j] -= F_t;
				}
			}
		}
	}
}
