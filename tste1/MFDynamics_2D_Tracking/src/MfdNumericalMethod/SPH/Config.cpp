/******************************************************************************
Copyright (c) 2007 Bart Adams (bart.adams@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software. The authors shall be
acknowledged in scientific publications resulting from using the Software
by referencing the ACM SIGGRAPH 2007 paper "Adaptively Sampled Particle
Fluids".

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
******************************************************************************/


#include "Config.h"

float Config::timestep = 0.001f;
float Config::viscosity = 10000.0f;
float Config::surfacetension = 0.1f;
float Config::density = 1000.0f;
float Config::gravity = -9.81f;
float Config::incompressibility = 800000.0f;
float Config::samplingdistance = 0.01f;
float Config::smoothinglength = 2.5f;
bool  Config::ghosts = 1;
int   Config::width = 512;
int   Config::height= 512;
float Config::controlviscosity = 1;
int   Config::dumpimageevery = 1;
int   Config::dumppovrayevery = 1;
float Config::diffusion = 0.0f;
int   Config::computesurfaceevery = 1;
int   Config::fastmarchingevery = 1;
float Config::totaltime = 5;
bool  Config::offline = 0;
float Config::tangentialfriction = 0;
float Config::normalfriction = 0;
std::string Config::scene="";
std::string Config::constraint="";
float Config::rotation_angle=0;
Vector3f Config::rotation_axis;
Vector3f Config::rotation_center;
Vector3f Config::scenelowerbound;
Vector3f Config::sceneupperbound;
bool Config::multires=0;
bool Config::vorticity=0;
int Config::initiallevel=1;
float Config::xsph=0.25f;
float Config::alpha=4;//参数
float Config::beta=6;//参数
float Config::gamma=1.2f;//参数
std::string Config::pov="pov";
std::string Config::rendered="rendered";
std::string Config::restorefile="norestore";
float Config::controlvolume = 1.0f;
int Config::dimX = 0;
int Config::dimY = 0;
int Config::dimZ = 0;


std::string Config::_file="";

   void Config::reload() {
      reload(_file);
   }

   void Config::reload(std::string file) {
         _file = file;
         std::ifstream input(file.c_str(), std::ios::in);
         std::string param;
         while (input >> param) {
         if (param == std::string("timestep")) input >> timestep;
         else if (param == std::string("viscosity")) input >> viscosity;
         else if (param == std::string("incompressibility")) input >> incompressibility;
         else if (param == std::string("surfacetension")) input >> surfacetension;
         else if (param == std::string("density")) input >> density;
         else if (param == std::string("gravity")) input >> gravity;
         else if (param == std::string("samplingdistance")) input >> samplingdistance;
         else if (param == std::string("smoothinglength")) input >> smoothinglength;
         else if (param == std::string("ghosts")) input >> ghosts;
         else if (param == std::string("width")) input >> width;
         else if (param == std::string("height")) input >> height;
         else if (param == std::string("controlviscosity")) input >> controlviscosity;
         else if (param == std::string("dumpimageevery")) input >> dumpimageevery;
         else if (param == std::string("dumppovrayevery")) input >> dumppovrayevery;
         else if (param == std::string("computesurfaceevery")) input >> computesurfaceevery;
         else if (param == std::string("fastmarchingevery")) input >> fastmarchingevery;
         else if (param == std::string("totaltime")) input >> totaltime;
         else if (param == std::string("offline")) input >> offline;
         else if (param == std::string("tangentialfriction")) input >> tangentialfriction;
         else if (param == std::string("normalfriction")) input >> normalfriction;
         else if (param == std::string("scene")) input >> scene;
         else if (param == std::string("constraint")) input >> constraint;
         else if (param == std::string("rotation_angle")) input >> rotation_angle;
         else if (param == std::string("vorticity")) input >> vorticity;
         else if (param == std::string("restorefile")) input >> restorefile;
		 else if (param == std::string("dimX")) input >> dimX;
		 else if (param == std::string("dimY")) input >> dimY;
		 else if (param == std::string("dimZ")) input >> dimZ;
		 else if (param == std::string("diffusion")) input >> diffusion;
         else if (param == std::string("rotation_axis")) {
            input >> rotation_axis.x;
            input >> rotation_axis.y;
            input >> rotation_axis.z;
         }
         else if (param == std::string("rotation_center")) {
            input >> rotation_center.x;
            input >> rotation_center.y;
            input >> rotation_center.z;
         }
         else if (param == std::string("scenelowerbound")) {
            input >> scenelowerbound.x; 
            input >> scenelowerbound.y; 
            input >> scenelowerbound.z; 
         }
         else if (param == std::string("sceneupperbound")) {
            input >> sceneupperbound.x; 
            input >> sceneupperbound.y; 
            input >> sceneupperbound.z; 
         }
         else if (param == std::string("multires")) {
            input >> multires;
         }
         else if (param == std::string("initiallevel")) {
            input >> initiallevel;
         }
         else if (param== std::string("xsph")) {
            input >> xsph;
         }
         else if (param==std::string("alpha")) input >> alpha;
         else if (param==std::string("beta")) input >> beta;
         else if (param==std::string("gamma")) input >> gamma;
         else if (param=="pov") input >> pov;
         else if (param=="rendered") input >> rendered;
      }

	  preComputing();
   }

   void Config::preComputing()
   {
	   float R = samplingdistance*smoothinglength;
#ifndef KERNEL_3D
	   controlvolume = M_PI*R*R;
#else
	   controlvolume = 4.0/3.0*M_PI*R*R*R;
#endif
		
   }
