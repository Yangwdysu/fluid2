// App_SPHMultiPhaseFlow.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <omp.h>
#include "MfdGUI/GlutMaster.h"
#include "MfdGUI/GlutWindow.h"
#include "MfdGUI/PointRenderer.h"
#include "MfdNumericalMethod/SPH/Config.h"
#include "MfdNumericalMethod/SPH/SinglePhaseFluid.h"
#include "HybridMultiPhaseFluid.h"
#include "MfdMath/LinearSystem/MeshlessSolver.h"
#include "Eigen/Core"
#include <windows.h>

using namespace mfd;

int _tmain(int argc, _TCHAR* argv[])
{
	omp_set_num_threads(NUM_THREAD);

	Eigen::initParallel();

	HybridMultiPhaseFluid* sim = new HybridMultiPhaseFluid;
	GlutWindow* win = new GlutWindow("first", 512, 512);
	VectorRender* ren1 = new VectorRender(sim->posGrid_Air.data, sim->velGrid_phase[0].data, sim->posGrid_Air.Size(), 0.01f);
	PointRenderer* ren2 = new PointRenderer(sim->posGrid_Air.data, sim->posGrid_Air.Size());
	ren2->SetMapping(&(sim->ren_massfield), &(sim->ren_mass));
	ren2->SetMarker(&(sim->ren_marker));
	MeshRenderer* ren5 = new MeshRenderer(&(sim->mc), sim->origin);
	MeshRenderer* ren6 = new MeshRenderer(&(sim->mc2), sim->bunnyori);

// 	LinearSystem sys;
// 	AnEquation equ1;
// 	equ1.coff = 1.0f;
// 	equ1.x = 0.9;
// 	equ1.b = 3;
// 	equ1.row.push_back(AnEquation::NeighborCoff(0, 1.0f));
// 	equ1.row.push_back(AnEquation::NeighborCoff(1, 1.0f));
// 	equ1.row.push_back(AnEquation::NeighborCoff(2, 1.0f));
// 	sys.m_equations.push_back(equ1);
// 	AnEquation equ2;
// 	equ2.coff = 2.0f;
// 	equ2.x = 2.1;
// 	equ2.b = 5;
// 	equ2.row.push_back(AnEquation::NeighborCoff(0, 1.0f));
// 	equ2.row.push_back(AnEquation::NeighborCoff(1, 2.0f));
// 	equ2.row.push_back(AnEquation::NeighborCoff(2, 2.0f));
// 	sys.m_equations.push_back(equ2);
// 
// 	AnEquation equ3;
// 	equ3.coff = 3.0f;
// 	equ3.x = 2.1;
// 	equ3.b = 6;
// 	equ3.row.push_back(AnEquation::NeighborCoff(0, 1.0f));
// 	equ3.row.push_back(AnEquation::NeighborCoff(1, 2.0f));
// 	equ3.row.push_back(AnEquation::NeighborCoff(2, 3.0f));
// 	sys.m_equations.push_back(equ3);
// 
// 	vector<float> x0;
// 	x0.push_back(0.0);
// 	x0.push_back(0.0);

// 	x0.push_back(0.0);
// 	//sys.SolvorSOR(x0);
// 	sys.SolvorPCG(x0);
// 
// 	int t = sys.iterationNumber();
// 	cout << "iteration: " << t << end

	win->SetSimulation(sim);
	win->AddRenderer(ren2);
	win->AddRenderer(ren1);
	win->AddRenderer(ren5);
	win->AddRenderer(ren6);
	

	GlutMaster::instance()->CallGlutCreateWindow("firsts", win);
	GlutMaster::instance()->SetIdleToCurrentWindow();
	GlutMaster::instance()->EnableIdleFunction();
	GlutMaster::instance()->CallGlutMainLoop();
	return 0;
}

