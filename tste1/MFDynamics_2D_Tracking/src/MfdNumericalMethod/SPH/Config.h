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
#ifndef MFD_CONFIG_H
#define MFD_CONFIG_H

#include <string>
#include <fstream>
#include "MfdMath/DataStructure/Vec.h"

using namespace mfd;

class Config {
   public:

      Config() {}

      static void reload(std::string file);
      static void reload();
	  static void preComputing();

      static float timestep;
      static float viscosity;
      static float incompressibility;
      static float surfacetension;
      static float density;
      static float gravity;
      static float samplingdistance;
      static float smoothinglength;
      static bool ghosts;

      static int width;
      static int height;

      static float controlviscosity;
      static int dumpimageevery;
      static int computesurfaceevery;
      static int fastmarchingevery;
      static int dumppovrayevery;

      static float totaltime;
      static bool offline;

      static float tangentialfriction;
      static float normalfriction;

      static std::string scene;
      static std::string constraint;
      static float rotation_angle;
      static Vector3f rotation_axis;
      static Vector3f rotation_center;
      static Vector3f scenelowerbound;
      static Vector3f sceneupperbound;
      static bool multires;
      static bool vorticity;

	  static int dimX;
	  static int dimY;
	  static int dimZ;

	  static float diffusion;


      static std::string _file;
      static std::string pov;
      static std::string rendered;

      static int initiallevel;
      static float xsph;
     
      // see paper 
      static float alpha;
      static float beta;
      static float gamma;

      static std::string restorefile;
	//pre-computed values

public:		
	  static float controlvolume;
	
};
#endif
