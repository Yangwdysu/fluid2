#pragma once
using namespace std;

#include <vector>
#include "MfdMath/DataStructure/Vec.h"
#include "MfdNumericalMethod/SPH/Config.h"
#include "MfdSurfaceReconstruction/DistanceField3D.h"
using namespace mfd;


class Barrier
{
public:
	Barrier(){ normalFriction = 0.95f; tangentialFriction = 0.0f; }
	~Barrier(){};

	virtual bool Inside(const Vector3f& in_pos) const;
	virtual bool Constrain(Vector3f& in_out_pos, Vector3f& in_out_vec) const;
	virtual void GetDistanceAndNormal(const Vector3f& in_pos, float& out_dist, Vector3f &out_normal) const {};
// 	inline float SmallRandomValue() const {
// 		return float(rand())/float(RAND_MAX) * Config::samplingdistance * 0.0001f;
// 	}
	// 	public:
	// 		Vector3f rot_axis;
	// 		float rot_angle;

	float normalFriction;
	float tangentialFriction;
};


class BarrierDistanceField3D : public Barrier {

public:

	// CONVENTION: normal n should point outwards, i.e., away from inside
	// of constraint
	BarrierDistanceField3D(DistanceField3D *df) : 
	  Barrier(), distancefield3d(df)  {
	  }

	  virtual void GetDistanceAndNormal(const Vector3f &p, float &dist, Vector3f &normal) const {
		  distancefield3d->GetDistance(p,dist,normal);
	  }

	  DistanceField3D * distancefield3d;
};


class Boundary
{
public:
	Boundary(void);
	~Boundary(void);

public:
	void Constrain(Vector3f& in_out_pos, Vector3f& in_out_vec);

	void increBarrier(Barrier *in_barrier) {
		m_barriers.push_back(in_barrier);
	}

	inline int size() const {
		return (int)m_barriers.size();
	}  

	vector<Barrier *> m_barriers;
};

