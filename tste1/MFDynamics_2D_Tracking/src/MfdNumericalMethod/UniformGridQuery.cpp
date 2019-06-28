#include <omp.h>
#include <iostream>
#include <string>
#include "UniformGridQuery.h"
#include "MfdMath/NumericalAlgorithm/Algorithms.h"
using namespace mfd;

UniformGridQuery::UniformGridQuery(float in_spacing, Vector3f lowLimit, Vector3f upLimit)
{
	m_nGrid = 0;
	m_beginOfLists = NULL;
	m_endOfLists = NULL;
	m_nParticles = -1;
	m_refPosArr = NULL;
	m_spacing = in_spacing;
	lowBoundingLimit = lowLimit;
	upBoundingLimit = upLimit;
}


UniformGridQuery::~UniformGridQuery(void)
{
	if (m_beginOfLists != NULL) delete[] m_beginOfLists;
	if (m_endOfLists != NULL) delete[] m_endOfLists;
}


void UniformGridQuery::AllocMemory()
{
	m_nGrid = ComputeGridSize();

	if (m_beginOfLists != NULL)	delete[] m_beginOfLists;
	if (m_endOfLists != NULL)	delete[] m_endOfLists;

	m_beginOfLists = new int[m_nGrid];
	m_endOfLists = new int[m_nGrid];
}

void UniformGridQuery::ComputeBoundingBox()
{
	int t_total = omp_get_max_threads();
	Vector3f* lo = new Vector3f[t_total];
	Vector3f* hi = new Vector3f[t_total];
	for (int i = 0; i < t_total; i++)
	{
		lo[i] = m_refPosArr[0];
		hi[i] = m_refPosArr[0];
	}

#pragma omp parallel
	{
		int nthreads = omp_get_num_threads();
		int t_id = omp_get_thread_num();
		std::cout << "Thread id: " << t_id << std::endl;
		
		for (int i = t_id; i < m_nParticles; i+=nthreads)
		{
			if (m_refPosArr[i].x < lo[t_id].x)	lo[t_id].x = m_refPosArr[i].x;
			if (m_refPosArr[i].y < lo[t_id].y)	lo[t_id].y = m_refPosArr[i].y;
			if (m_refPosArr[i].z < lo[t_id].z)	lo[t_id].z = m_refPosArr[i].z;
			if (m_refPosArr[i].x > hi[t_id].x)	hi[t_id].x = m_refPosArr[i].x;
			if (m_refPosArr[i].y > hi[t_id].y)	hi[t_id].y = m_refPosArr[i].y;
			if (m_refPosArr[i].z > hi[t_id].z)	hi[t_id].z = m_refPosArr[i].z;
		}
	}

	lowBounding = lo[0];
	upBounding = hi[0];

	std::cout << "UniformGridQuery::ComputeBoundingBox(): " << lowBounding.y << " " << upBounding.y << std::endl;

	for (int i = 0; i < t_total; i++)
	{
		if (lo[i].x < lowBounding.x)	lowBounding.x = lo[i].x;
		if (lo[i].y < lowBounding.y)	lowBounding.y = lo[i].y;
		if (lo[i].z < lowBounding.z)	lowBounding.z = lo[i].z;
		if (hi[i].x > upBounding.x)	upBounding.x = hi[i].x;
		if (hi[i].y > upBounding.y)	upBounding.y = hi[i].y;
		if (hi[i].z > upBounding.z)	upBounding.z = hi[i].z;
	}

	lowBounding = Max(lowBounding, lowBoundingLimit-0.2f);
	upBounding = Min(upBounding, upBoundingLimit+0.2f);

	ExpandBoundingBox(0.25f*m_spacing);

	delete[] lo;
	delete[] hi;
}

void UniformGridQuery::ExpandBoundingBox( float padding )
{
	lowBounding -= padding;
	upBounding += padding;
}

int UniformGridQuery::ComputeGridSize()
{
	ComputeBoundingBox();

	m_nx = (int)((upBounding.x-lowBounding.x)/m_spacing)+1;
	m_ny = (int)((upBounding.y-lowBounding.y)/m_spacing)+1;
	m_nz = (int)((upBounding.z-lowBounding.z)/m_spacing)+1;
	return m_nx*m_ny*m_nz;
}

int UniformGridQuery::GetId( Vector3f& in_pos )
{
	int ix = (int)((in_pos.x-lowBounding.x)/m_spacing);
	int iy = (int)((in_pos.y-lowBounding.y)/m_spacing);
	int iz = (int)((in_pos.z-lowBounding.z)/m_spacing);

	return GetId(ix, iy, iz);
}

int UniformGridQuery::GetId( int in_ix, int in_iy, int in_iz )
{
	if(in_ix < 0 || in_ix >= m_nx)	
		return -1;
	if(in_iy < 0 || in_iy >= m_ny)	
		return -1;
	if(in_iz < 0 || in_iz >= m_nz)	
		return -1;

	return in_ix+in_iy*m_nx+in_iz*m_nx*m_ny;
}

void UniformGridQuery::Construct( Array<Vector3f>& pos, ArrayManager& simData )
{
	m_nParticles = pos.ElementCount();
	m_refPosArr = pos.DataPtr();

	int *m_refIdsArr = new int[m_nParticles];
	int *m_refRIdsArr = new int[m_nParticles];				//particle ids and reordered ids

	int nNewGrid = ComputeGridSize();
	if (nNewGrid != m_nGrid)
	{
		if (m_beginOfLists != NULL)			delete[] m_beginOfLists;
		if (m_endOfLists != NULL)			delete[] m_endOfLists;

		m_beginOfLists = new int[nNewGrid];
		m_endOfLists = new int[nNewGrid];
		m_nGrid = nNewGrid;
	}

#pragma omp parallel for
	for (int i = 0; i < m_nParticles; i++)
	{
		m_refIdsArr[i] = GetId(m_refPosArr[i]);
	}

	Algorithms::RadixSort(m_refIdsArr, m_refRIdsArr, m_nParticles, m_beginOfLists, m_endOfLists, m_nGrid);

	simData.Reordering(m_refRIdsArr, m_nParticles);

	delete[] m_refIdsArr;
	delete[] m_refRIdsArr;
}

void UniformGridQuery::GetNeighbors(Vector3f& in_pos, float in_radius, NeighborList& out_neighborList)
{
// 	Vector3f& pos = *in_pos;
// 	int& num = *out_num;
// 	NeighborIDS& neiIds = *out_neighborIds;
// 	int ix, iy, iz;
// 	GetId(pos, ix, iy, iz);

	out_neighborList.size = 0;
	float radiusSquare = in_radius*in_radius;

	int ix_st, ix_ed;
	int iy_st, iy_ed;
	int iz_st, iz_ed;
	ix_st = (int)((in_pos.x-in_radius-lowBounding.x)/m_spacing);
	ix_ed = (int)((in_pos.x+in_radius-lowBounding.x)/m_spacing);
	iy_st = (int)((in_pos.y-in_radius-lowBounding.y)/m_spacing);
	iy_ed = (int)((in_pos.y+in_radius-lowBounding.y)/m_spacing);
	iz_st = (int)((in_pos.z-in_radius-lowBounding.z)/m_spacing);
	iz_ed = (int)((in_pos.z+in_radius-lowBounding.z)/m_spacing);
	for (int i = ix_st; i <= ix_ed; i++)
	{
		for (int j = iy_st; j <= iy_ed; j++)
		{
			for (int k = iz_st; k <= iz_ed; k++)
			{
				int gridIndex = GetId(i, j, k);
				if (gridIndex >= 0)
				{
					for (int t = m_beginOfLists[gridIndex]; t < m_endOfLists[gridIndex]; t++)
					{
						float distSquare = DistanceSq(m_refPosArr[t], in_pos);
						if (distSquare <= radiusSquare && out_neighborList.size < NEIGHBOR_SIZE)
						{
							out_neighborList.ids[out_neighborList.size] = t;
							out_neighborList.distance[out_neighborList.size] = sqrt(distSquare);
							out_neighborList.size++;
						}
					}
				}
			}
		}
	}
}

void UniformGridQuery::GetSizedNeighbors(Vector3f& in_pos, float in_radius, NeighborList& out_neighborList, int in_maxN)
{
	GetNeighbors(in_pos, in_radius, out_neighborList);

	if (out_neighborList.size > in_maxN)
	{
		int segSize[NEIGHBOR_SEGMENT] = {0};
		int segIds[NEIGHBOR_SEGMENT][NEIGHBOR_SIZE];
		float segDistances[NEIGHBOR_SEGMENT][NEIGHBOR_SIZE];
		for (int i = 0; i < out_neighborList.size; i++)
		{
			float dist = out_neighborList.distance[i];
			int index = pow(0.99f*dist/in_radius, 2)*NEIGHBOR_SEGMENT;
			segIds[index][segSize[index]] = out_neighborList.ids[i];
			segDistances[index][segSize[index]] = out_neighborList.distance[i];
			segSize[index]++;
		}

		NeighborList sizedNeighborList;
		int totalNum = 0;
		int j;
		for (j = 0; j < NEIGHBOR_SEGMENT; j++)
		{
			totalNum += segSize[j];
			if (totalNum <= in_maxN)
			{
				for (int k = 0; k < segSize[j]; k++)
				{
					sizedNeighborList.ids[sizedNeighborList.size] = segIds[j][k];
					sizedNeighborList.distance[sizedNeighborList.size] = segDistances[j][k];
					sizedNeighborList.size++;
				}
			}
			else
				break;
		}

		int remN = in_maxN + segSize[j] - totalNum;
		int* remArr = new int[remN];
		Algorithms::KMinimum(segDistances[j], segSize[j], remArr, remN);

		for (int k = 0; k < remN; k++)
		{
			sizedNeighborList.ids[sizedNeighborList.size] = segIds[j][remArr[k]];
			sizedNeighborList.distance[sizedNeighborList.size] = segDistances[j][remArr[k]];
			sizedNeighborList.size++;
		}

		out_neighborList = sizedNeighborList;

		delete[] remArr;
	}
}

