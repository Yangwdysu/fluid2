#ifndef MFD_PARTICLE_H
#define MFD_PARTICLE_H

#include <cstring>
#include <string>
#include "MfdMath/DataStructure/Vec.h"
#include "MfdMath/DataStructure/Array.h"

namespace mfd {

	class BasicSPH;
	class DeformableSolid;


	//particle attribute 0x00000000: [31-30]material; [29]motion; [28]Dynamic; [27-8]undefined yet, for future use; [7-0]correspondding to the id of a fluid phase in multiphase fluid or an object in multibody system

	enum MaterialType
	{
		MATERIAL_MASK = 0xC0000000,
		MATERIAL_FLUID = 0x00000000,
		MATERIAL_RIGID = 0xA0000000,
		MATERIAL_ELASTIC = 0xB0000000,
		MATERIAL_PLASTIC = 0xC0000000
	};

	enum MotionType
	{
		MOTION_MASK = 0x20000000,
		MOTION_ENABLED = 0x00000000,
		MOTION_DISABLED = 0x20000000
	};

	enum DynamicType
	{
		DYNAMIC_MASK = 0x10000000,
		DYNAMIC_POSITIVE = 0x00000000,
		DYNAMIC_PASSIVE = 0x10000000
	};

// 	enum ObjectID
// 	{
// 		OBJECTID_MASK = 0x000000FF,
// 		OBJECTID_1 = 0x00000001,
// 		OBJECTID_2 = 0x00000002,
// 		OBJECTID_3 = 0x00000003
// 	};

#define  MATERIALTYPE(attribute) (unsigned)(attribute&MATERIAL_MASK)
#define  MOTIONTYPE(attribute) (unsigned)(attribute&MOTION_MASK)
#define  DYNAMICTYPE(attribute) (unsigned)(attribute&DYNAMIC_MASK)
#define  OBJECTID(attribute) (unsigned)(attribute&OBJECTID_MASK)


	/*!
	 *	\class	Particle
	 *	\brief	Particle class.
	 *	\todo	inline functions
	 *
	 *	This class serves as a particle properties communicator, so
	 *	calling destructor will not delete particle itself.
	 */
	class Particle
	{
	public:

		Particle(BasicSPH* simulation, unsigned int index);

		~Particle();

		MaterialType GetMaterialType();
		MotionType GetMotionTpye();
		DynamicType GetDynamicsType();

		void SetMaterialType(MaterialType type);
		void SetMotionType(MotionType type);
		void SetDynamicType(DynamicType type);

		float GetMass();
		float GetDensity();
		float GetPressure();
		Vector3f GetPosition();
		Vector3f GetVelocity();
		Vector3f GetNormal();

		void SetMass(float mass);
		void SetDensity(float density);
		void SetPressure(float pressure);
		void SetPosition(const Vector3f& position);
		void SetVelocity(const Vector3f& velocity);
		void SetNormal(const Vector3f& normal);

	protected:

		unsigned int id;
		BasicSPH *sim;

	};

	class SolidParticle
		: public Particle
	{
	public:

		SolidParticle(DeformableSolid* simulation, unsigned int index);
		~SolidParticle();

		void SetInitPosition(const Vector3f& position);
		Vector3f GetInitPosition();

	public:
		DeformableSolid *sim_solid;
	};

} // namespace mfd

#endif
