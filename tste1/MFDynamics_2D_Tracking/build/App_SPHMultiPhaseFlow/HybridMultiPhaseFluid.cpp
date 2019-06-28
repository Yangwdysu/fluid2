#include "StdAfx.h"
#include <time.h>
#include <math.h>
#include <omp.h>
#include <sstream>
#include "HybridMultiPhaseFluid.h"
#include "MfdIO/Image.h"
#include "MfdNumericalMethod/UniformGridQuery.h"
#include"cuda_UpdatePhi.h"



#define FOR_EACH_CELL for (int i=1 ; i<Nx-1 ; i++ ) { for (int j=1 ; j<Ny-1 ; j++ ) { for (int k=1 ; k<Nz-1 ; k++ ) {
#define END_FOR }}}

/*
This code corresponds to the paper "Local Poisson SPH For Viscous Incompressible Fluids"


*/


//Using
HybridMultiPhaseFluid::HybridMultiPhaseFluid(void)
{
	m_uniGrid = NULL;
	correctionFactor = 1.0f;//矫正因子
	densityKern = KernelFactory::Spiky;//密度核
	gridNeighborSize = 0;//邻居大小

	diffuse = 0.01f;//扩散系数
	render_id = 0;//渲染id

	overall_incompressibility = 0.1f;//全局不可压性

	particle_incompressibility = 300.0f;//粒子的不可压性
	particle_sharpening = 40.0f;//粒子锐化

	seprate_incompressibility = 1.0f;//
	seprate_sharpening = 1.0f;

	Initialize();
}


//Using--初始化
bool HybridMultiPhaseFluid::Initialize(string in_filename /*= "NULL" */)
{
	Config::reload("config3d.txt");
	cout << "Total Liquid Particle: " << 15 * 15 * 15 << endl;

	timeStep = Config::timestep;//时间步
	viscosity = Config::viscosity;//粘度
	gravity = Config::gravity;//重力
	surfaceTension = Config::surfacetension;//表面张力


	samplingDistance = Config::samplingdistance;//采样距离
	smoothingLength = Config::samplingdistance*Config::smoothinglength;//光滑半径/长度
	viscosityAir = Config::viscosity;//空气粘度
	rhoLiquidRef = Config::density;//流体密度
	V_grid = pow(samplingDistance, 3);
	rhoAirRef = rhoLiquidRef / 4; //空气密度
	massAir = rhoAirRef * samplingDistance*samplingDistance*samplingDistance;//空气质量
	massLiquid = rhoLiquidRef * samplingDistance*samplingDistance*samplingDistance;//流体质量

	/*float ratio = 0.25f;
	int mid = PHASE_SIZE / 2;
	float drho = rhoLiquidRef * 0.2f;
	for (int i = 0; i < PHASE_SIZE; i++)
	{
		rho_phase[i] = rhoLiquidRef - drho * (mid - i);
		mass_phase[i] = rho_phase[i] * V_grid;
	}*/


	cout << "Liquid Particle Mass: " << massLiquid << endl;
	cout << "Liquid Density: " << rhoLiquidRef << endl;
	cout << "Air Particle Mass: " << massAir << endl;
	cout << "Air Density: " << rhoAirRef << endl;
	cout << "smoothinglength" << Config::smoothinglength << endl;
	diff = Config::diffusion;

#ifdef DEMO_SEPERATION
	InitialSeparation();
#endif
	for (int i = 1; i < Nx - 1; i++)
	{
		for (int j = 1; j < Ny - 1; j++)
		{
			for (int k = 1; k < Nz - 1; k++)
			{
				marker_phase[0](i, j, k) = true;
				//cout << marker_phase[0](i, j, k) << endl;

			}
		}
	}

	MarkSolidDomain();


	ren_massfield = (massGrid_phase[0].data);
	ren_mass = 1.0f;
	ren_marker = (marker_phase[0].data);

	return true;
}



//Using--初始化分离
void HybridMultiPhaseFluid::InitialSeparation()
{
	/*viscosityAir = 0.01f;
	surfacetension = 0.0f;
	sharpening = 0.1f;
	compressibility = 0.1f;
	velocitycoef = 0.05f;
	diffuse = 0.01f;
	overall_incompressibility = 0.2f;
	seprate_incompressibility = 0.1f;
	seprate_sharpening = 0.1f;
	particle_incompressibility = 300.0f;
	particle_sharpening = 40.0f;*/


	int valid_x = Config::dimX;
	int valid_y = Config::dimY;
	int valid_z = Config::dimZ;

	int half_row = valid_x + 1;
	int half_col = valid_y + 1;
	int half_depth = valid_z + 1;

	int row = 2 * half_row + 1;
	int col = 2 * half_col + 1;
	int depth = 2 * half_depth + 1;
	int liquid_num = 0;

	origin = Vector3f(0.5f - half_row * samplingDistance, 0.5f - half_col * samplingDistance, 0.5f - half_depth * samplingDistance);


	AllocateMemoery(liquid_num, row, col, depth);

	int pid = 0;
	int num_air = 0;

	for (int k = 0; k < depth; k++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float x = origin.x + i * samplingDistance;
				float y = origin.y + j * samplingDistance;
				float z = origin.z + k * samplingDistance;

				if (i >= 0 && i < row && j >= 0 && j < col && k >= 0 && k < depth)
				{
					posGrid_Air(i, j, k) = Vector3f(x, y, z);//equal int3 a,a=vector3 (1,2,3)
					//massGrid_phase[0](i, j, k) = 0.0f;
				}
			}
		}
	}

	rho_phase[0] = 1000.0f;
	mass_phase[0] = rho_phase[0] * V_grid;
	vis_phase[0] = 0.01f;

	vel_u_boundary.Zero();
	vel_v_boundary.Zero();
	vel_w_boundary.Zero();

	for (int j = 0; j < vel_u_boundary.ny; j++)
	{
		for (int k = 0; k < vel_u_boundary.nz; k++)
		{
			vel_u_boundary(0, j, k) = 0.0f;
			vel_u_boundary(1, j, k) = 0.0f;
			vel_u_boundary(vel_u_boundary.nx - 1, j, k) = 0.0f;
			vel_u_boundary(vel_u_boundary.nx - 2, j, k) = 0.0f;
		}
	}

	for (int i = 0; i < vel_v_boundary.nx; i++)
	{
		for (int k = 0; k < vel_v_boundary.nz; k++)
		{
			vel_v_boundary(i, 0, k) = 0.0f;
			vel_v_boundary(i, 1, k) = 0.0f;
			if (i < 75 && i > 25)
			{
				vel_v_boundary(i, vel_v_boundary.ny - 1, k) = 0.0f;
				vel_v_boundary(i, vel_v_boundary.ny - 2, k) = 0.0f;
			}
			else
			{
				vel_v_boundary(i, vel_v_boundary.ny - 1, k) = 0.0f;
				vel_v_boundary(i, vel_v_boundary.ny - 2, k) = 0.0f;
			}
		}
	}

	for (int i = 0; i < vel_w_boundary.nx; i++)
	{
		for (int j = 0; j < vel_w_boundary.ny; j++)
		{
			vel_w_boundary(i, j, 0) = 0.0f;
			vel_w_boundary(i, j, 1) = 0.0f;
			vel_w_boundary(i, j, vel_w_boundary.nz - 1) = 0.0f;
			vel_w_boundary(i, j, vel_w_boundary.nz - 2) = 0.0f;
		}
	}

	for (int k = 0; k < depth; k++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				Vector3f center = Vector3f(0.5f, 0.75f, 0.5f);//这个是球的坐标
				//df->GetDistance(posGrid_Air(i, j, k), d);
				float d = (posGrid_Air(i, j, k) - center).Length();//点到中心的距离
				if (d < 0.15f)// && !(posGrid_Air(i,j,k).x < 0.53f && posGrid_Air(i,j,k).x > 0.47f && posGrid_Air(i,j,k).y < 0.83f))
				//if ( i > 40 && i < 60 && /*j > 35 && */j > 145)
				{
					massGrid_phase[0](i, j, k) = 1.0f;
				}
				else
				{
					massGrid_phase[0](i, j, k) = 0.0f;
				}
			}
		}
	}

	Vector3f end = Vector3f(0.5f + half_row * samplingDistance, 0.5f + half_col * samplingDistance, 0.5f + half_depth * samplingDistance);
}

//Using--内存分配
void HybridMultiPhaseFluid::AllocateMemoery(int _np, int _nx, int _ny, int _nz)
{
	BasicSPH::AllocMemory(_np, 1, 0);
	liquidNeigbhors.SetSpace(_np);
	airNeigbhors.SetSpace(_np);
	rhoAirArr.SetSpace(_np);
	phiLiquid.SetSpace(_np);
	energyLiquid.SetSpace(_np);

	vel_u.SetSpace(_nx + 1, _ny, _nz);
	vel_v.SetSpace(_nx, _ny + 1, _nz);
	vel_w.SetSpace(_nx, _ny, _nz + 1);
	vel_u_boundary.SetSpace(_nx + 1, _ny, _nz);
	vel_v_boundary.SetSpace(_nx, _ny + 1, _nz);
	vel_w_boundary.SetSpace(_nx, _ny, _nz + 1);

	pre_vel_u.SetSpace(_nx + 1, _ny, _nz);
	pre_vel_v.SetSpace(_nx, _ny + 1, _nz);
	pre_vel_w.SetSpace(_nx, _ny, _nz + 1);
	coef_u.SetSpace(_nx + 1, _ny, _nz);
	coef_v.SetSpace(_nx, _ny + 1, _nz);
	coef_w.SetSpace(_nx, _ny, _nz + 1);
	marker_Air.SetSpace(_nx, _ny, _nz);
	marker_Solid.SetSpace(_nx, _ny, _nz);

	posGrid_Air.SetSpace(_nx, _ny, _nz);
	preMassGrid_Air.SetSpace(_nx, _ny, _nz);

	for (int i = 0; i < PHASE_SIZE; i++)
	{
		massGrid_phase[i].SetSpace(_nx, _ny, _nz);
		velGrid_phase[i].SetSpace(_nx, _ny, _nz);
		marker_phase[i].SetSpace(_nx, _ny, _nz);

	}

	Nx = _nx;
	Ny = _ny;
	Nz = _nz;
}



//Using--标记固体边界
void HybridMultiPhaseFluid::MarkSolidDomain()
{
	marker_Solid.Zero();
	int gNum = GetAirParticleNumber();
	for (int i = 0; i < Nx; i++)
	{
		for (int j = 0; j < Ny; j++)
		{
			for (int k = 0; k < Nz; k++)
			{
				if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1)
				{
					marker_Solid(i, j, k) = true;
				}
				else
					marker_Solid(i, j, k) = false;
			}
		}
	}
}

static float t = -0.01f;
//Using--
void HybridMultiPhaseFluid::StepEuler(float dt)
{
	clock_t total_start = clock();
	clock_t t_start = clock();
	clock_t t_end;

	float elipse = 0.0f;
	float dx = 1.0f / Nx;
	float T = 4.0f;

	//	ProjectMAC(dt);

	t_end = clock();
	cout << "Solving Pressure Costs: " << t_end - t_start << endl;

	t_start = clock();

	while (elipse < dt) {
		float substep = CFL();
		if (elipse + substep > dt)
		{
			substep = dt - elipse;
		}

		cout << "*********Substep: " << substep << " *********" << endl;

		for (int i = 1; i < Nx - 1; i++)
		{
			for (int j = 1; j < Ny - 1; j++)
			{
				for (int k = 1; k < Nz - 1; k++)
				{
					if (t < T && t > 0.0f)
					{
						float x = (i + 0.5f) / (float)Nx;
						float y = (j + 0.5f) / (float)Ny;
						if (t < T)
						{
							vel_u(i + 1, j, k) = -2.0f*sin(M_PI*y)*cos(M_PI*y)*sin(M_PI*x)*sin(M_PI*x)*cos(t*M_PI / T);
							vel_v(i, j + 1, k) = 2.0f*sin(M_PI*x)*cos(M_PI*x)*sin(M_PI*y)*sin(M_PI*y)*cos(t*M_PI / T);

						}

					}
					else
					{
						vel_u(i + 1, j, k) = 0.0f;
						vel_v(i, j + 1, k) = 0.0f;
					}
				}
			}
		}

		t_start = clock();

		for (int i = 1; i < Nx - 1; i++)
		{
			for (int j = 1; j < Ny - 1; j++)
			{
				for (int k = 1; k < Nz - 1; k++)
				{
					velGrid_phase[0](i, j, k)[0] = 0.5f*(vel_u(i - 1, j, k) + vel_u(i, j, k));
					velGrid_phase[0](i, j, k)[1] = 0.5f*(vel_v(i, j - 1, k) + vel_v(i, j, k));
				}
			}
		}

		preMassGrid_Air = massGrid_phase[0];
		AdvectForward(massGrid_phase[0], preMassGrid_Air, velGrid_phase[0], substep);

		t_end = clock();
		cout << "Advect Time: " << t_end - t_start << endl;
		preMassGrid_Air = massGrid_phase[0];
		t_start = clock();
		UpdatePhi(massGrid_phase[0], preMassGrid_Air, substep);
		t_end = clock();
		cout << "Update Time: " << t_end - t_start << endl;

		elipse += substep;
	}

	cout << dt << endl;
	t += dt;

	clock_t total_end = clock();
	cout << "Total Cost " << total_end - total_start << " million seconds!" << endl;

	if (simItor*dt > 4.01f)
	{
		exit(0);
	}

}



//Using--
template<typename T>
void HybridMultiPhaseFluid::UpdatePhi(Grid<3, T>& d, Grid<3, T>& d0, float dt)
{
	d = d0;
	SetScalarFieldBoundary(d0, true);

	float eps = 0.000001f;
	float inv_root2 = 1.0f / sqrt(2.0f);
	float inv_root3 = 1.0f / sqrt(3.0f);
	float h = samplingDistance;

	Grid<3, Vec<3, T>> nGrid;
	nGrid.SetSpace(Nx, Ny, Nz);

	Grid<3, T> dif_field, dif_field2;
	dif_field.SetSpace(Nx, Ny, Nz);
	dif_field2.SetSpace(Nx, Ny, Nz);
	dif_field2 = d0;
	dif_field = d0;

#pragma omp parallel for
	for (int i = 1; i < Nx - 1; i++)
	{
		for (int j = 1; j < Ny - 1; j++)
		{
			for (int k = 1; k < Nz - 1; k++)
			{
				float norm_x;
				float norm_y;
				float norm_z;

				norm_x = dif_field(i + 1, j, k) - dif_field(i - 1, j, k);//法向
				norm_y = dif_field(i, j + 1, k) - dif_field(i, j - 1, k);
				norm_z = dif_field(i, j, k + 1) - dif_field(i, j, k - 1);
				// 				
				float norm = sqrt(norm_x*norm_x + norm_y * norm_y + norm_z * norm_z) + eps;
				norm_x /= norm;//归一化操作，得到的是向量的长度
				norm_y /= norm;
				norm_z /= norm;
				Vec<3, T>& nijk = nGrid(i, j, k);
				nijk[0] = norm_x;
				nijk[1] = norm_y;
				nijk[2] = norm_z;
			}
		}
	}



	float w = 1.0f*h;//fro diffuse parameter
	float gamma = 1.0f;
	float ceo1 = 1.0f*gamma / h;		//for smoothing
	float ceo2 = 1.5f*gamma*w / h / h;	//for sharping
	float weight;


	//#pragma omp parallel for
	for (int i = 1; i < Nx - 1; i++)
	{
		for (int j = 1; j < Ny - 1; j++)
		{
			for (int k = 1; k < Nz - 1; k++)
			{

				int k0, k1;
				int ix0, iy0, iz0;
				int ix1, iy1, iz1;

				ix0 = i;   iy0 = j; iz0 = k;
				ix1 = i + 1; iy1 = j; iz1 = k;
				weight = 1.0f;

				if (ix1 < Nx - 1)
				{
					k0 = nGrid.Index(ix0, iy0, iz0);
					k1 = nGrid.Index(ix1, iy1, iz1);
					Vec<3, T>& n1 = nGrid[k0];
					Vec<3, T>& n2 = nGrid[k1];


					float c0 = d0[k0] * (1.0f - d0[k0])*n1.x;
					float c1 = d0[k1] * (1.0f - d0[k1])*n2.x;//这个有点像求解 phi(1-phi)
					float dc = 0.5f*weight*ceo2*dt*(c1 + c0);

					d[k0] -= dc;
					d[k1] += dc;
				}

				ix0 = i; iy0 = j;   iz0 = k;
				ix1 = i; iy1 = j + 1; iz1 = k;

				if (iy1 < Ny - 1)
				{
					k0 = nGrid.Index(ix0, iy0, iz0);
					k1 = nGrid.Index(ix1, iy1, iz1);
					Vec<3, T>& n1 = nGrid[k0];
					Vec<3, T>& n2 = nGrid[k1];

					float c1 = d0[k1] * (1.0f - d0[k1])*n2.y;
					float c0 = d0[k0] * (1.0f - d0[k0])*n1.y;
					float dc = 0.5f*weight*ceo2*dt*(c1 + c0);

					d[k0] -= dc;
					d[k1] += dc;
				}

				ix0 = i; iy0 = j; iz0 = k;
				ix1 = i; iy1 = j; iz1 = k + 1;

				if (iz1 < Nz - 1)
				{
					k0 = nGrid.Index(ix0, iy0, iz0);
					k1 = nGrid.Index(ix1, iy1, iz1);
					Vec<3, T>& n1 = nGrid[k0];
					Vec<3, T>& n2 = nGrid[k1];
					 
					float c1 = d0[k1] * (1.0f - d0[k1])*n2.z;
					float c0 = d0[k0] * (1.0f - d0[k0])*n1.z;
					float dc = 0.5f*weight*ceo2*dt*(c1 + c0);

					d[k0] -= dc;
					d[k1] += dc;
				}
			}


		}
	}

	d0 = d;
	float dif2 = (ceo1 + diff / h / h)*dt;
	LinearSolve(d, d0, dif2, 1.0f + 6.0f*dif2);

}
//No
#define INNERINDEX(m,n,l) (m-1)*(Ny-2)*(Nz-2)+(n-1)*(Nz-2)+l-1
//Using-- Cfl condition,这个能看懂
float HybridMultiPhaseFluid::CFL()
{
	float maxvel = 0.0f;
	for (int i = 0; i < vel_u.Size(); i++)
		maxvel = max(maxvel, abs(vel_u[i]));
	for (int i = 0; i < vel_v.Size(); i++)
		maxvel = max(maxvel, abs(vel_v[i]));
	for (int i = 0; i < vel_w.Size(); i++)
		maxvel = max(maxvel, abs(vel_w[i]));
	if (maxvel < EPSILON)
		maxvel = 1.0f;
	return samplingDistance / maxvel;
}

//Using--
void HybridMultiPhaseFluid::LinearSolve(Grid3f& d, Grid3f& d0, float a, float c)
{
	Grid3f cp;
	cp.SetSpace(d0.nx, d0.ny, d0.nz);
	d = d0;
	float c1 = 1.0f / c;
	float c2 = (1.0f - c1) / 6.0f;
	int nx = d0.Nx();
	int ny = d0.Ny();
	int nz = d0.Nz();
	int nxy = d0.Nx()*d0.Ny();
	for (int it = 0; it < 20; it++)
	{
		SetScalarFieldBoundary(d, true);
		cp = d;

#pragma omp parallel shared(d, d0, a, c, c1, c2, nx, ny, nz, nxy, cp)
		{

#pragma omp for
			for (int i = 1; i < nx - 1; i++)
			{
				int i0 = i;
				for (int j = 1; j < ny - 1; j++)
				{
					int j0 = i0 + j * nx;
					for (int k = 1; k < nz - 1; k++)
					{
						int k0 = j0 + k * nxy;
						d(i, j, k) = (c1*d0[k0] + c2 * (cp[k0 + 1] + cp[k0 - 1] + cp[k0 + nx] + cp[k0 - nx] + cp[k0 + nxy] + cp[k0 - nxy]));
					}
				}
			}

		}
	}
}



//Using---
void HybridMultiPhaseFluid::SetScalarFieldBoundary(Grid3f& field, bool postive)
{
	int nx = field.Nx();
	int ny = field.Ny();
	int nz = field.Nz();

	float s = postive ? 1.0f : -1.0f;

#pragma omp parallel for
	for (int j = 1; j < ny - 1; j++)
	{
		for (int k = 1; k < nz - 1; k++)
		{
			field(0, j, k) = s * field(1, j, k);
			field(nx - 1, j, k) = s * field(nx - 2, j, k);//
		}
	}

#pragma omp parallel for
	for (int i = 1; i < nx - 1; i++)
	{
		for (int k = 1; k < nz - 1; k++)
		{
			field(i, 0, k) = s * field(i, 1, k);
			field(i, ny - 1, k) = s * field(i, ny - 2, k);
		}
	}

#pragma omp parallel for
	for (int i = 1; i < nx - 1; i++)
	{
		for (int j = 1; j < ny - 1; j++)
		{
			field(i, j, 0) = s * field(i, j, 1);
			field(i, j, nz - 1) = s * field(i, j, nz - 2);
		}
	}

	for (int i = 1; i < Nx - 1; i++)
	{
		field(i, 0, 0) = 0.5f*(field(i, 1, 0) + field(i, 0, 1));
		field(i, Ny - 1, 0) = 0.5f*(field(i, Ny - 2, 0) + field(i, Ny - 1, 1));
		field(i, 0, Nz - 1) = 0.5f*(field(i, 1, Nz - 1) + field(i, 0, Nz - 2));
		field(i, Ny - 1, Nz - 1) = 0.5f*(field(i, Ny - 1, Nz - 2) + field(i, Ny - 2, Nz - 1));
	}

	for (int j = 1; j < Ny - 1; j++)
	{
		field(0, j, 0) = 0.5f*(field(1, j, 0) + field(0, j, 1));
		field(0, j, Nz - 1) = 0.5f*(field(1, j, Nz - 1) + field(0, j, Nz - 2));
		field(Nx - 1, j, 0) = 0.5f*(field(Nx - 2, j, 0) + field(Nx - 2, j, 1));
		field(Nx - 1, j, Nz - 1) = 0.5f*(field(Nx - 2, j, Nz - 1) + field(Nx - 1, j, Nz - 2));
	}

	for (int k = 1; k < Nz - 1; k++)
	{
		field(0, 0, k) = 0.5f*(field(1, 0, k) + field(0, 1, k));
		field(Nx - 1, 0, k) = 0.5f*(field(Nx - 2, 0, k) + field(Nx - 1, 1, k));
		field(0, Ny - 1, k) = 0.5f*(field(1, Ny - 1, k) + field(0, Ny - 2, k));
		field(Nx - 1, Ny - 1, k) = 0.5f*(field(Nx - 2, Ny - 1, k) + field(Nx - 1, Ny - 2, k));
	}

	field(0, 0, 0) = (field(1, 0, 0) + field(0, 1, 0) + field(0, 0, 1)) / 3.0f;
	field(0, 0, Nz - 1) = (field(1, 0, Nz - 1) + field(0, 1, Nz - 1) + field(0, 0, Nz - 2)) / 3.0f;
	field(0, Ny - 1, 0) = (field(1, Ny - 1, 0) + field(0, Ny - 2, 0) + field(0, Ny - 1, 1)) / 3.0f;
	field(Nx - 1, 0, 0) = (field(Nx - 2, 0, 0) + field(Nx - 1, 1, 0) + field(Nx - 1, 0, 1)) / 3.0f;
	field(0, Ny - 1, Nz - 1) = (field(1, Ny - 1, Nz - 1) + field(0, Ny - 2, Nz - 1) + field(0, Ny - 1, Nz - 2)) / 3.0f;
	field(Nx - 1, 0, Nz - 1) = (field(Nx - 2, 0, Nz - 1) + field(Nx - 1, 1, Nz - 1) + field(Nx - 1, 0, Nz - 2)) / 3.0f;
	field(Nx - 1, Ny - 1, 0) = (field(Nx - 2, Ny - 1, 0) + field(Nx - 1, Ny - 2, 0) + field(Nx - 1, Ny - 1, 1)) / 3.0f;
	field(Nx - 1, Ny - 1, Nz - 1) = (field(Nx - 2, Ny - 1, Nz - 1) + field(Nx - 1, Ny - 2, Nz - 1) + field(Nx - 1, Ny - 1, Nz - 2)) / 3.0f;
}


//问题就出现在这个函数上了，我只将这部分改成GPU而已，其他的都是CPU上的。
template<typename T>
void HybridMultiPhaseFluid::AdvectForward(Grid<3, T>& d, Grid<3, T>& d0, GridV3f& v, float dt)
{

	assert(d.Size() == d0.Size());
	d.Zero();
	int num = d0.Size();
	int nx = d0.Nx();
	int ny = d0.Ny();
	int nz = d0.Nz();
	float h = samplingDistance;
	int dsize = nx*ny*nz;
	//computer on GPU
	cuda_UpdatePhi Advect_Forward;
	Advect_Forward.cuda_AdvectForwardd(d.data,d0.data, (float3*)v.data,nx,ny,nz,dt,h,dsize);
	

	//computer on cpu

	//float fx, fy, fz;
	//float w000, w100, w010, w001, w111, w011, w101, w110;//???
	//int ix, iy, iz;
	//for (int i = 1; i < nx - 1; i++)
	//{
	//	for (int j = 1; j < ny - 1; j++)
	//	{
	//		for (int k = 1; k < nz - 1; k++)
	//		{

				//int idx = i + j*nx + k*nx*ny;
				//fx = i + dt * v(i, j, k).x / h;	fy = j + dt * v(i, j, k).y / h;	fz = k + dt * v(i, j, k).z / h;
				////dataFile << "i=" << i << "  fx=" << fx << "    fy=" << fy << "    fz=" << fz << endl;
				//dataFile << fy << endl;
				//if (fx < 1) fx = 1;
				//if (fx > nx - 2) fx = nx - 2;
				//if (fy < 1) fy = 1;
				//if (fy > ny - 2) fy = ny - 2;
				//if (fz < 1) fz = 1;
				//if (fz > nz - 2) fz = nz - 2;

				//ix = (int)fx[idx];		iy = (int)fy[idx];		iz = (int)fz[idx];
				//fx[idx] -= ix;			fy[idx] -= iy;			fz[idx] -= iz;
				//

				//w000 = (1.0f - fx[idx])*(1.0f - fy[idx])*(1.0f - fz[idx]);
				//w100 = fx[idx] * (1.0f - fy[idx])*(1.0f - fz[idx]);
				//w010 = (1.0f - fx[idx])*fy[idx]*(1.0f - fz[idx]);
				//w001 = (1.0f - fx[idx])*(1.0f - fy[idx])*fz[idx];
				//w111 = fx[idx] * fy[idx]*fz[idx];
				//w011 = (1.0f - fx[idx])*fy[idx]*fz[idx];
				//w101 = fx[idx] * (1.0f - fy[idx])*fz[idx];
				//w110 = fx[idx] * fy[idx]*(1.0f - fz[idx]);



				/*ix = iix[idx];		iy = iiy[idx];		iz = iiz[idx];
				T& val = d0(i, j, k);*/
				/*d(ix, iy, iz) += val * w000[idx];
				d(ix + 1, iy, iz) += val * w100[idx];
				d(ix, iy + 1, iz) += val * w010[idx];
				d(ix, iy, iz + 1) += val * w001[idx];
				d(ix + 1, iy + 1, iz + 1) += val * w111[idx];
				d(ix, iy + 1, iz + 1) += val * w011[idx];
				d(ix + 1, iy, iz + 1) += val * w101[idx];
				d(ix + 1, iy + 1, iz) += val * w110[idx];
			}
		}
	}*/
}

//Using--按键控制
void HybridMultiPhaseFluid::Invoke(unsigned char type, unsigned char key, int x, int y)
{
	cout << "HybridMultiPhaseFluid Key Pressed: " << key << endl;
	switch (type)
	{
	case 'K':
	{
		switch (key) {
		case 'u':
		{
			render_id++;
			render_id %= PHASE_SIZE;
			//vel_v(10, 10, 1) += 10.5f;

			ren_massfield = (massGrid_phase[render_id].data);
			ren_mass = (mass_phase[render_id]);
			ren_marker = (marker_phase[render_id].data);
		}

		break;

		default:
			break;
		}
	}
	break;

	default:
		break;
	}
}







HybridMultiPhaseFluid::~HybridMultiPhaseFluid(void)
{
}



//Using--断点无用，注释报错
void HybridMultiPhaseFluid::ComputeDensity()
{
	//CalculateLiquidFraction();
	//ComputeAirDensityOnGrid();
	//ComputeLiquidDensityOnParticle();
//	ComputeAdjustedLiquidDensityOnParticle();
}

//Using--断点无用，注释报错
void HybridMultiPhaseFluid::ComputeNeighbors()
{
	//ComputeLiquidNeighbors();
	//ComputeAirNeighbors();
}



