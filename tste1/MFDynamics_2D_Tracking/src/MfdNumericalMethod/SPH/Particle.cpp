#include <cstring>
#include <cfloat>
#include <climits>
#include "Particle.h"
#include "BasicSPH.h"
#include "DeformableSolid.h"

using namespace std;


Particle::Particle(BasicSPH* simulation, unsigned int index)
	: sim(simulation)
	, id(index)
{
}

Particle::~Particle()
{
}

void Particle::SetMaterialType( MaterialType type )
{
	sim->attriArr[id] = ((~MATERIAL_MASK) & sim->attriArr[id]) | type;
}

MaterialType Particle::GetMaterialType()
{
	return (MaterialType)MATERIALTYPE(sim->attriArr[id]);
}


void Particle::SetMotionType( MotionType type )
{
	sim->attriArr[id] = ((~MOTION_MASK) & sim->attriArr[id]) | type;
}

MotionType Particle::GetMotionTpye()
{
	return (MotionType)MOTIONTYPE(sim->attriArr[id]);
}

DynamicType Particle::GetDynamicsType()
{
	return (DynamicType)DYNAMICTYPE(sim->attriArr[id]);
}

void Particle::SetDynamicType( DynamicType type )
{
	sim->attriArr[id] = ((~DYNAMIC_MASK) & sim->attriArr[id]) | type;
}


float Particle::GetMass()
{
	return sim->massArr[id];
}

float Particle::GetDensity()
{
	return sim->rhoArr[id];
}

float Particle::GetPressure()
{
	return sim->preArr[id];
}

Vector3f Particle::GetPosition()
{
	return sim->posArr[id];
}

Vector3f Particle::GetVelocity()
{
	return sim->velArr[id];
}

Vector3f Particle::GetNormal()
{
	return sim->normalArr[id];
}

void Particle::SetMass( float mass )
{
	sim->massArr[id] = mass;
}

void Particle::SetDensity( float density )
{
	sim->rhoArr[id] = density;
}

void Particle::SetPressure( float pressure )
{
	sim->preArr[id] = pressure;
}

void Particle::SetPosition( const Vector3f& position )
{
	sim->posArr[id] = position;
}

void Particle::SetVelocity( const Vector3f& velocity )
{
	sim->velArr[id] = velocity;
}

void Particle::SetNormal( const Vector3f& normal )
{
	sim->normalArr[id] = normal;
}



SolidParticle::SolidParticle( DeformableSolid* simulation, unsigned int index )
	: Particle(simulation, index)
{
	sim_solid = simulation;
}

SolidParticle::~SolidParticle()
{

}

void SolidParticle::SetInitPosition( const Vector3f& position )
{
	sim_solid->initPosArr[id] = position;
}

mfd::Vector3f SolidParticle::GetInitPosition()
{
	return sim_solid->initPosArr[id];
}
