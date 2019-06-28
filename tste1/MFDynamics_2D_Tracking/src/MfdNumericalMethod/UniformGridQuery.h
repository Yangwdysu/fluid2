#pragma once
#include "MfdNumericalMethod/INeighborQuery.h"
#include "MfdMath/DataStructure/Vec.h"
#include "MfdMath/DataStructure/Array.h"

namespace mfd {

class UniformGridQuery : public INeighborQuery {

public:
	UniformGridQuery(float in_spacing, Vector3f lowLimit, Vector3f upLimit);
	~UniformGridQuery(void);

	virtual void GetNeighbors(Vector3f& in_pos, float in_radius, NeighborList& out_neighborList);
	virtual void GetSizedNeighbors(Vector3f& in_pos, float in_radius, NeighborList& out_neighborList, int in_maxN);

	void Construct( Array<Vector3f>& pos, ArrayManager& simData);

private:
	
	void ComputeBoundingBox();
	int ComputeGridSize();
	void ExpandBoundingBox(float in_padding);
	void AllocMemory();

	inline int GetId(Vector3f& in_pos);			// return -1 if the particle is out of the bounding box.
	inline int GetId( int in_ix, int in_iy, int in_iz );

public:
	int m_nGrid;
	int m_nx, m_ny, m_nz;
	//particle ids for each cell
	int* m_beginOfLists;					//0		5	8		grid0: 0-4; grid1: 5-7; grid2: 8-9
	int* m_endOfLists;						//5		8	10
	Vector3f lowBounding;
	Vector3f upBounding;

	Vector3f lowBoundingLimit;
	Vector3f upBoundingLimit;

	int m_nParticles;
	Vector3f* m_refPosArr;
	float m_spacing;
};

}