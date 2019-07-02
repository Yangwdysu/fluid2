#pragma once
#include "MfdNumericalMethod/SPH/BasicSPH.h"
#include "MfdMath/DataStructure/Grid.h"
#include "MfdSurfaceReconstruction/CIsoSurface.h"
#include "MfdMath/LinearSystem/MeshlessSolver.h"

#include "Eigen/Sparse"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/Core"

namespace mfd{

typedef Grid<3, Vector3f> GridV3f;
typedef Grid<3, bool>	Grid3b;

struct Triple
{
	int x, y, z;
};

struct Coeff
{
	float a;
	float x0;
	float x1;
	float y0;
	float y1;
	float z0;
	float z1;
};

#define NUM_THREAD 3

#define DENSITY_THRESHOLD 1e-10
//#define TWO_DIMENSION
//#define DEMO_RAIN
//#define DEMO_TEASER
//#define DEMO_DAMBREAK
//#define DEMO_MISSIBLE_BUNNY
//#define DEMO_SMOKECOLLISION
//#define DEMO_OIL
#define DEMO_SEPERATION
//#define DEMO_SURFACETENSION
//#define DEMO_SPRING
//#define DEMO_SINK
//#define DEMO_FOUR_IMISCIBLE
//#define DEMO_PIPE
//#define DEMO_BUBBLE


#define PHASE_SIZE 1


class UniformGridQuery;
class HybridMultiPhaseFluid :
	public BasicSPH
{
public:
	HybridMultiPhaseFluid(void);
	~HybridMultiPhaseFluid(void);

	virtual bool Initialize( string in_filename = "NULL" );

	void InitialSeparation();



	void InitBoundary();

	virtual void StepEuler(float dt);

	void ComputeCorrectionFactor();
//	void AssignAirNodeMass();
	void CalculateLiquidFraction();

	// querying neighbors
	void ComputeNeighbors();
	void ComputeLiquidNeighbors();
	void ComputeAirNeighbors();

	//compute density
	virtual void ComputeDensity();
	void ComputeLiquidDensityOnParticle();
	void ComputeAdjustedLiquidDensityOnParticle();
	void ComputeAirDensityOnParticle();
	void ComputeAirDensityOnGrid();
	void ComputeAirDensityOnGrid(GridV3f& pGrid);

	void ComputeLiquidVolume();
	void ComputeAirVolume();
	void ComputeAirVolumeOnGrid();

	void ComputeLiquidViscousForce();
	void ComputeLiquidPressure(float dt);
	void ComputeAirPressure(float dt);
	void ComputeLiquidPressureForce();
	void ComputeSurfaceTension();

	void PredictParticleVelocities(float dt);
	void CorrectParticleVelocities(float dt);

	template<typename T>
	void MappingParticleQauntityToGrid(Grid<3, T>& d, GridV3f& grid, Array<T>& s, Array<Vector3f>& particles);
	template<typename T>
	void MappingGridQauntityToParticle(Array<T>& d, Array<Vector3f>& particles, Grid<3, T>& s, GridV3f& grid);

	void CombineTwoVelocityField();

	void MoveParticles(float dt);
	void ConstrainParticles();

	void BoundaryConstrain(Vector3f& pos, Vector3f& vel);
	void Sharpening(Vector3f& pos, Vector3f& vel, float dt);

	void ComputeAirVelocity();

	void UpdateAirParticles(float dt);
	void AddGravityToAir();
	void AddViscousForceToAir();
	void TransferVelocityFromCenterToFace();
	void TransferVelocityFromFaceToCenter();
	void SolveAirIncompressibility(float dt);

	void AdvectAir(float dt);

	int GetAirParticleNumber() { return posGrid_Air.elementCount; }

	void AddSourceToAir();
	void ComputeSurfaceTensionOnGrid(float dt);
	void ComputeSurfaceTensionOnGridAsSPH(float dt);

	void ComputeViscousForceOnGrid(float dt);
	void AdvectVelocityAlongAxis(float dt);
	void AdvectDensity(GridV3f& vel, float dt);
	void DiffuseDensity(float dt);
	void DiffuseDensity2(float dt);
	void DiffuseDensityWithSharpening(float dt);
	void DiffuseDensityInSPH(float dt);

	void PredictAirParticles(float dt);
	void VelocityStep(float dt);
	void DensityStep(float dt);
	void Project(float dt);
	void Project2(float dt);
	void ProjectDensity(float dt);
	void ProjectVelocity(float dt);
	void LinearSolve(Grid3f& d, Grid3f& d0, float a, float c);
	void LinearSolve(Grid3f& d, Grid3f& d0, Grid3b& mark, float a, float c);
	void LinearSolve2(Grid3f& d, Grid3f& d0, Grid3b& mark, float a, float c);
	void LinearSolve3(Grid3f& d, Grid3f& d0, Grid3b& mark, float a, float c);

	void CorrectAirDensity(float dt);
	void RemapAirParticles(GridV3f& gPos);

	template<typename T>
	void AdvectForward(Grid<3, T>& d, Grid<3, T>& d0, GridV3f& v, float dt);
	template<typename T>
	void AdvectForwardWithNormalization(Grid<3, T>& d, Grid<3, T>& d0, GridV3f& v, GridV3f& grid, float dt);
	template<typename T>
	void AdvectForwardWithoutSmearout(Grid<3, T>& d, Grid<3, T>& d0, GridV3f& v, GridV3f& grid, float dt, T maximum);
	template<typename T>
	void AdvectBackward(Grid<3, T>& d, Grid<3, T>& d0, GridV3f& v, float dt);
	template<typename T>
	void AdvectStaggeredBackward(Grid<3, T>& d, Grid<3, T>& d0, Grid<3, T>& u, Grid<3, T>& v, Grid<3, T>& w, float dt, int dim);



	void AdvectForward(GridV3f& v, float dt);
	void AdvectForwardWithoutBlurring(GridV3f& v, float dt);

	void AdvectForward(GridV3f& v, GridV3f& dv, float dt);

	void MarkInteriorDomain(bool boundary);
	void PhaseMarkInteriorDomain(Grid3b& mark, Grid3f& p_mass, bool boundary);
	void MarkSolidDomain();

	void SetVelocityBoundary(GridV3f& vel);

	void SetVelocityBoundary(Grid3f& u, Grid3f& v, Grid3f& w);

	void SetScalarFieldBoundary(Grid3f& field, bool postive);
	void SetScalarFieldPeriodBoundary(Grid3f& field, bool postive);
	void SetPressureBoundary(Grid3f& field);

	void AllocateMemoery(int _np, int _nx, int _ny, int _nz);

	void Invoke( unsigned char type, unsigned char key, int x, int y );
	//virtual void PostProcessing();


	void AddExternalForceToAir(float dt);
	void PhaseAddExternForce(GridV3f& vel, Grid3b& mark, float dt);

	void PredictParticles(float dt);
	void PhasePredictParticles(Grid3f& p_mass, GridV3f& p_vel, Grid3f& flux, GridV3f& v, Grid3b& mark, float refMass, float dt);
	void PhasePredictParticlesWithoutDiffusion(GridV3f& p_realpos, Grid3f& p_mass, GridV3f& p_vel, Grid3f& flux, GridV3f& v, Grid3b& mark, float refMass, float dt);
	void PhaseDiffuseDensityWithoutBlurring(Grid3f& p_mass, GridV3f& p_vel, float dt);

	void CorrectDensity(float dt);
	void CorrectVelocity(float dt);
	void PhaseCrrectDensity(Grid3f& p_mass, GridV3f& p_vel, Grid3b& mark, float refRho, float refMass, float dt);
	void PhaseComputePressure(Grid3f& p, Grid3f& rho, float refRho, Grid3b& mark, float dt);
	void PhaseComputeDensityOnGrid(Grid3f& rho, GridV3f& newPos, Grid3f& p_mass, Grid3b& mark);
	void PhaseRemapParticles(GridV3f& newPos, Grid3f& p_mass, GridV3f& p_vel, Grid3b& mark, float refMass);
	void PhaseComputeVolume(Grid3f& p_mass, float refRho);
	void PhaseComputeSurfaceTensionOnGrid(Grid3f& p_mass, GridV3f& p_vel, Grid3b& mark, float rhoRef, float massRef, float dt);
	void PhaseComputeViscousForceOnGrid(GridV3f& p_vel, Grid3b& mark, float vis, float dt);

	void ProjectCombined(float dt);
	void ProjectSeparate(float dt);
	void PhaseProject(Grid3f& p_mass, GridV3f& p_vel, Grid3b& mark, float rhoRef, float massRef, float dt);

	float CFL();

	void AddBodyForce(float dt);

	void ProjectMAC(float dt);

	template<typename T>
	void UpdatePhi(Grid<3, T>& d, Grid<3, T>& d0, float dt);

	template<typename T>
	void AdvectWENO1rd(Grid<3, T>& d, Grid<3, T>& d0, Grid<3, T>& u, Grid<3, T>& v, Grid<3, T>& w, float dt);

	template<typename T>
	void AdvectWENO5rd(Grid<3, T>& d, Grid<3, T>& d0, Grid<3, T>& u, Grid<3, T>& v, Grid<3, T>& w, float dt);

	template<typename T>
	void Flood(Grid<3, T>& c);

	void AdvectVelocity(float dt);

	void ApplySurfaceTension(float dt);

	void ApplySemiImplicitViscosity(float dt);

	void SaveDensityField( Grid3f& density , const char *outfname );

public:
	Grid3f vel_u;
	Grid3f vel_v;
	Grid3f vel_w;

	Grid3f vel_u_boundary;
	Grid3f vel_v_boundary;
	Grid3f vel_w_boundary;

	Grid3f pre_vel_u;
	Grid3f pre_vel_v;
	Grid3f pre_vel_w;
	Grid<3, Coeff> coef_u;
	Grid<3, Coeff> coef_v;
	Grid<3, Coeff> coef_w;


	Eigen::VectorXd x0;
	//LinearSystem sys;
	
	Grid3f rhoGrid_Air;		//grid density contributed by the air
	Grid3f preGrid_Air;
	Grid3f volGrid_Air;
	Grid3f preMassGrid_Air;
	Grid3f surfaceEnergyGrid_Air;

	GridV3f preVelGrid_Air;
	GridV3f posGrid_Air;
	GridV3f accGrid_Air;

	Grid3f rhoGrid_Liquid;	//grid density contributed by the liquid
	Grid3f fraction_Liquid;
	Grid3f volGrid_Liquid;

	
	Grid3b marker_Solid;

	Array<float> rhoAirArr;
	Array<float> phiLiquid;
	Array<float> energyLiquid;

// 	Grid3f f_catche;
// 	GridV3f vec_catche;

	Vector3f origin;

	float rhoLiquidRef;

	
	float massLiquid;

	float correctionFactor;


	float rhoAirRef;
	float massAir;
	Grid3f massGrid_Air;
	GridV3f velGrid_Air;
	Grid3b marker_Air;
	

	float vis_phase[PHASE_SIZE];
	float rho_phase[PHASE_SIZE];
	float mass_phase[PHASE_SIZE];
	Grid3f massGrid_phase[PHASE_SIZE];
	GridV3f velGrid_phase[PHASE_SIZE];
	Grid3b marker_phase[PHASE_SIZE];
	Grid3f extraDiv[PHASE_SIZE];
	GridV3f realposGrid_Air[PHASE_SIZE];
	GridV3f preposGrid_Air;

	float rho1;
	float rho2;

	float vis1;
	float vis2;

	float diff;

	int Nx;
	int Ny;
	int Nz;

	int n_b;

	float V_grid;

	Array<NeighborList> liquidNeigbhors;
	Array<NeighborList> airNeigbhors;

	UniformGridQuery* m_uniGrid;

	KernelFactory::KernelType densityKern;

	int gridNeighborSize;
	Triple fixed_ids[NEIGHBOR_SIZE];
	float fixed_weights[NEIGHBOR_SIZE];

	float viscosityAir;
	
	float surfacetension;
	float sharpening;
	float compressibility;
	float velocitycoef;
	float diffuse;

	float overall_incompressibility;

	float seprate_incompressibility;
	float seprate_sharpening;

	float particle_incompressibility;
	float particle_sharpening;

	CIsoSurface<float> mc;
	CIsoSurface<float> mc2;
	Vector3f bunnyori;
	int render_id;

	float* ren_massfield;
	float ren_mass;
	bool* ren_marker;
};

}
