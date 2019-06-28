#ifndef MFD_DEFORMABLESOLID_H
#define MFD_DEFORMABLESOLID_H

#include "MfdNumericalMethod/SPH/BasicSPH.h"
#include "MfdNumericalMethod/SPH/Particle.h"
#include "MfdMath/DataStructure/Matrix.h"

namespace mfd{

	class UniformGridQuery;

	class DeformableSolid :
		public BasicSPH
	{
	public:
		DeformableSolid(void);
		~DeformableSolid(void);

		virtual bool Initialize( string in_filename = "NULL" );

		virtual void ComputeInitalNeighbors();
		virtual void ComputeDensity();
		virtual void ComputeVolume();
		virtual void ComputeAcceleration();

		//direct damping the velocity by smoothing the velocity field
		void DampVelocity();

		void ComputeElasticForce();
		void ComputeViscousForce();

		void ComputeStrain();
		void ComputeStress();

		virtual void StepEuler(float dt);
		virtual void Invoke(unsigned char type, unsigned char key, int x, int y);

		inline SolidParticle GetSolidParticle(unsigned int id) { return SolidParticle(this, id); }

		virtual float GetTimeStep();

	public:
		Array<NeighborList> neighborListsRef;
		Array<Vector3f> initPosArr;
		Array<Rotation3Df> orientArr;
		Array<MatrixSq3f> strainArr;
		Array<MatrixSq3f> stressArr;
		Array<int> marker;

		Array<Vector3f> F_e;
		Array<Vector3f> F_v;

		Array<Vector3f> accArr;

		UniformGridQuery* m_uniGrid;

		float poissonRatio;
		float youngModulus;
		float lamda;
		float mu;
		float damp;

		Vector3f gravity;
		float densityRef;
	};

}

#endif