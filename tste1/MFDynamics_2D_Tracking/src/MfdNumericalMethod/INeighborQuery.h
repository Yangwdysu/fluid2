#ifndef MFD_NEIGHBOR_H
#define MFD_NEIGHBOR_H

#include "MfdMath/DataStructure/Vec.h"

namespace mfd{

#define NEIGHBOR_SIZE	150
#define NEIGHBOR_SEGMENT 20

	class NeighborList
	{
	public:
		NeighborList() {size = 0; }
		~NeighborList(){};
	public:
		int size;
		int ids[NEIGHBOR_SIZE];
		float distance[NEIGHBOR_SIZE];
	};

	class INeighborQuery {
	public:
		virtual void GetNeighbors(Vector3f& in_pos, float in_radius, NeighborList& out_neighborList) = 0;
		virtual void GetSizedNeighbors(Vector3f& in_pos, float in_radius, NeighborList& out_neighborList, int in_maxN) = 0;
	};
}

#endif

