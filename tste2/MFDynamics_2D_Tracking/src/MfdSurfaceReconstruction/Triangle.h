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

#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include <cfloat>
#include "MfdMath/DataStructure/Vec.h"
#include "Vertex.h"

class Edge;

#define MYEPSILON  0.000000001

namespace mfd {

class Triangle {

   public:
      Vertex * v[3];
  
      Edge   * e[3];
      
      Vector3f normal;
      Vector3f center;
      float area;

      bool intersect(const Vector3f &orig, const Vector3f &dir, Vector3f &intersectionPoint, float &sqrDist, bool &vertexHit, float &t) {
         const Vertex &vert0 = *(this->v[0]);
         const Vertex &vert1 = *this->v[1];
         const Vertex &vert2 = *this->v[2];
         const Vector3f edge1 = vert1-vert0;
         const Vector3f edge2 = vert2-vert0;


         const float angle_normal_dir = normal.Dot(dir);
         const bool orthogonal = fabs(angle_normal_dir) < MYEPSILON;


         const Vector3f pvec = dir.Cross(edge2);
         const float det  =  edge1.Dot(pvec);

         t=FLT_MAX;
         vertexHit = false;
         if (!orthogonal && det>MYEPSILON) {
         const Vector3f tvec = orig-vert0;
        const float u = tvec.Dot(pvec);
        if (u< 0.0 || u>det) {
            //qDebug("u=%f det=%f", u, det);
            return false;
        }

        const Vector3f qvec = tvec.Cross(edge1);
        const float v = dir.Dot(qvec);
        if (v< 0.0 || u+v > det) {
            //qDebug("v=%f u+v=%fdet=%f", u,u+v, det);
            return false;
        }

        const float inv_det = 1.0 / det;

        /* calculate t, ray intersects triangle */
        t = edge2.Dot(qvec) * inv_det;

        const float myeps = MYEPSILON*det;
        if (u<myeps/* || (det-u)<myeps*/) vertexHit = true;
        else if (v<myeps || (det-u-v)<myeps) vertexHit = true;

        intersectionPoint = orig+t*dir;
        sqrDist = DistanceSq(intersectionPoint, orig);

        return true;
         }
    else if (!orthogonal && det < -MYEPSILON) {


        const Vector3f tvec = orig-vert0;
        const float u = tvec.Dot(pvec);
        if (u>0.0 || u<det) return false;

        const Vector3f qvec = tvec.Cross(edge1);
        const float v = dir.Dot(qvec);
        if (v>0.0 || u+v<det) return false;

        const float inv_det = 1.0 / det;

        /* calculate t, ray intersects triangle */
        t = edge2.Dot(qvec) * inv_det;

        const float myeps = -MYEPSILON*det;
        if (-u<myeps/* || (u-det)<myeps*/) vertexHit = true;
        else if (-v<myeps || (u+v-det)<myeps) vertexHit = true;

        intersectionPoint = orig+t*dir;
        sqrDist = DistanceSq(intersectionPoint, orig);

        return true;

    }

      else return false;
   }

};

}

#endif
