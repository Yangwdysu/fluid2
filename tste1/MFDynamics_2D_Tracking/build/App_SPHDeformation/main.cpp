// App_SPHDeformation.cpp : Defines the entry point for the console application.
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
#include "MfdNumericalMethod/SPH/DeformableSolid.h"


int _tmain(int argc, _TCHAR* argv[])
{
	DeformableSolid* sim = new DeformableSolid;
	GlutWindow* win = new GlutWindow("first", 512, 512);
	PointRenderer* ren = new PointRenderer(sim->GetPositionPtr(), sim->GetParticleNumber());
	win->SetSimulation(sim);
	win->AddRenderer(ren);

	GlutMaster::instance()->CallGlutCreateWindow("first", win);
	GlutMaster::instance()->SetIdleToCurrentWindow();
	GlutMaster::instance()->EnableIdleFunction();
	GlutMaster::instance()->CallGlutMainLoop();
	return 0;


	return 0;
}

