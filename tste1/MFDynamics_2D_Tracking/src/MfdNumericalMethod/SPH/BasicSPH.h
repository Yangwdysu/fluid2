#ifndef MFD_BASICSPH_H
#define MFD_BASICSPH_H

#include "MfdNumericalMethod/SPH/Kernels.h"
#include "MfdMath/DataStructure/Array.h"
#include "MfdNumericalMethod/ISimulation.h"
#include "MfdNumericalMethod/Boundary.h"
#include "MfdNumericalMethod/INeighborQuery.h"
#include "Particle.h"
#include "MfdMath/Common.h"

//arr: array
namespace mfd{

	//algorithm property
	enum IntegratorType
	{
		Euler,				//!< Euler 1st order.
		PredictorCorrector,	//!< Predictor Corrector.
		RungeKutta4			//!< Runge-Kutta 4th order.
	};

	enum ViscosityType
	{
		ArtificialViscosity = 1,		//!< Artificial Viscosity.
		LaminarViscosity = 2,			//!< Laminar Viscosity.
		SubParticleScaleViscosity = 3	//!< Sub-particle-scale (SPS) viscosity.
	};

	class BasicSPH : public ISimulation
	{
	public:
		BasicSPH(void);
		~BasicSPH(void);

		virtual bool Initialize( string in_filename = "NULL" );
		virtual void InitalSceneBoundary();

		virtual float GetTimeStep();
		virtual void Advance(float dt);
		virtual void StepEuler(float dt);
		virtual void StepPredictorCorrector(float dt);
		virtual void StepRungeKutta4(float dt);

		virtual void ComputeNeighbors();

		void ComputeFieldStandard(Array<float>& out_field, Array<float>& in_field, Array<NeighborList>& neighbors, KernelFactory::KernelType type);
		void ComputeGradientStandard(Array<Vector3f>& out_grad, Array<float>& in_field, Array<NeighborList>& neighbors, KernelFactory::KernelType type);


		void BoundaryHandling();

	// 	void SetMaterialType(int in_index, MaterialType in_material);
	// 	void SetMotionType(int in_index, MotionType in_motion);
	// 	void SetDynamicType(int in_index, DynamicType in_dynamic);
	// 	void SetObjectId(int in_index, ObjectID in_number);

		void SavePositions(string in_path, int in_iter);
		void SaveVelocities(string in_path, int in_iter);

		Vector3f* GetPositionPtr() { return posArr.DataPtr(); }
		int GetParticleNumber() { return N; }

		int GetIterator() { return simItor; }

		inline Particle GetParticle(unsigned int id) { return Particle(this, id); }

	protected:
		void AllocMemory(int np, int nf, int nr);

		//rendering functions
	// public:
	// 	virtual void initProjection(const Camera &camera);
	// 	virtual void initModelView(const Camera &camera);
	// 	virtual void render(const Camera &camera);
	// 	virtual void CallBackKeyboardFunc(unsigned char key, int x, int y);

	public:
		int N;					//total particle number

		int nFluid;
		int nRigid;

		float *refRhoOfFluid;						//array of size nFluid

		//needs to be rearranged in each step
		Array<float> massArr;					//particle masses
		Array<unsigned> attriArr;				//particle attributes
		Array<Vector3f> posArr;					//particle positions
		Array<Vector3f> velArr;					//particle velocities
		Array<Vector3f> normalArr;				//particle normals pointing outward
	
		//don't need to be rearranged
		Array<Vector3f> FvisArr;			//viscous force
		Array<Vector3f> FpArr;				//pressure force
		Array<Vector3f> FsurArr;			//surface force
		Array<float> volArr;						//particle volume
		Array<float> preArr;						//particle pressures
		Array<float> rhoArr;						//particle densities

		ArrayManager dataManager;

	//	NeighborList* neighborLists;

		IntegratorType integratorType;

		int simItor;

		Boundary* m_boundary;
	//	DistanceField3D * df;

		Vector3f lowBound;
		Vector3f upBound;

		float timeStep;
		float viscosity;
		float gravity;
		float surfaceTension;

		float samplingDistance;
		float smoothingLength;
	};


}


#endif