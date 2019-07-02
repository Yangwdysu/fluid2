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

#ifndef __EDGE_H__
#define __EDGE_H__

#include "MfdMath/DataStructure/Vec.h"

namespace mfd{

class Vertex;
class Triangle;

class Edge {

   public:

      Edge() {
         t[0] = NULL;
         t[1] = NULL;
      }
  
      Vertex  *v[2];

      Triangle  *t[2];

      Vector3f normal;
      bool border;
};

}
#endif
