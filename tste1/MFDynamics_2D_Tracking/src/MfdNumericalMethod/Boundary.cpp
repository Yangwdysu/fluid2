#include "Boundary.h"


Boundary::Boundary(void)
{
}


Boundary::~Boundary(void)
{
	const int nbarriers = size();
	for (int i=0; i<nbarriers; i++) {
		delete m_barriers[i];
	}
	m_barriers.clear();
}

void Boundary::Constrain( Vector3f& in_out_pos, Vector3f& in_out_vec )
{
	for (int i = 0; i < m_barriers.size(); i++)
	{
		m_barriers[i]->Constrain(in_out_pos, in_out_vec);
	}
}

bool Barrier::Inside( const Vector3f& in_pos ) const
{
	Vector3f pos = in_pos;
// 	pos -= Config::rotation_center;
// 	pos = Vector3f(cos(angle)*pos.x - sin(angle)*pos.y, sin(angle)*pos.x + cos(angle)*pos.y, pos.z);
// 	pos += Config::rotation_center;

	float dist;
	Vector3f normal;
	GetDistanceAndNormal(pos,dist,normal);
	return (dist > 0);
}

bool Barrier::Constrain( Vector3f& in_out_pos, Vector3f& in_out_vec ) const
{
	Vector3f pos = in_out_pos;
	Vector3f vec = in_out_vec;

	float dist;
	Vector3f normal;
	GetDistanceAndNormal(pos,dist,normal);
	bool constrained = false;
	// constrain particle
	if (dist <= 0) {
		float olddist = -dist;
		dist = -dist;// + SmallRandomValue();
		// reflect position
		pos -= (dist + olddist)*normal;
		// reflect velocity
		float vlength = vec.Length();
		float vec_n = vec.Dot(normal);
		Vector3f vec_normal = vec_n*normal;
		Vector3f vec_tan = vec-vec_normal;
		if (vec_n>0) vec_normal = -vec_normal;
		vec_normal*=(1.0f-normalFriction);
		vec = vec_normal + vec_tan;
		vec*=(1.0f-tangentialFriction);
		constrained = true;
	}

	in_out_pos = pos;
	in_out_vec = vec;

	return constrained;
}

