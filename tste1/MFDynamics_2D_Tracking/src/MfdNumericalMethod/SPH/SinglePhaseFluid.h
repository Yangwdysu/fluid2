#pragma once

#include <iostream>
#include "BasicSPH.h"
#include "MfdMath/DataStructure/Matrix.h"
using namespace std;

#define ENERGY_TENSION

namespace mfd{

class UniformGridQuery;
class SinglePhaseFluid : public BasicSPH
{
public:
	SinglePhaseFluid(void);
	~SinglePhaseFluid(void);

	virtual bool Initialize(string in_filename = "NULL");
	virtual void InitalSceneBoundary();
	virtual void ComputeNeighbors();
	virtual void ComputeDensity();
	virtual void ComputeVolume();

	virtual void StepEuler(float dt);

#ifdef ENERGY_TENSION
	virtual void ComputeSurfaceTension();
//	virtual void ComputeDensity();
// 	virtual void ComputePressure(float dt);
// 	virtual void ComputePressureForce();
#endif

	void ComputeSuraceTensionInMD();
	void ComputeSuraceTensionInCurvature();	

	void ComputePressure(float dt);
	void ComputePressureForce();

	void ComputeViscousForce();

	void Advect(float dt);
	
	virtual void PostProcessing();

	void AllocMemory(int in_nf);

	void SaveToFile(string in_filename);
	void ReadFromFile(ifstream& input_stream);

//	void writeToPovray(string filename, int dumpiter);

//	virtual void CallBackKeyboardFunc(unsigned char key, int x, int y);
private:
	void ComputePressureTensor();
	void CheckScale();
	bool LargeScale(int in_index);

public:
	float massInLarge;
	float massInSmall;

	float lengthInLarge;
	float lengthInSmall;

	float* rhoInSmallArr;

	float* phiArr;
	float* energyArr;
	
	bool* smallScale;

	float segLow;
	float segHigh;


	float* color;

	Array<NeighborList> neighborLists;

	MatrixSq3f* pressureTensorArr;
	UniformGridQuery* m_uniGrid;
};

}
