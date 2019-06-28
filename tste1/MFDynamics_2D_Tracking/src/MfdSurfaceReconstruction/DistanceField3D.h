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

#ifndef __DISTANCEFIELD3D_H__
#define __DISTANCEFIELD3D_H__

#include <vector>
#include <string>
#include "MfdMath/DataStructure/Vec.h"
#include "CIsoSurface.h"

using namespace std;

namespace mfd {

// 	class KdTree;
// 	class Mesh;
// 	class Triangle;

class DistanceField3D {
   public:

      DistanceField3D(string filename);

      DistanceField3D(const Vector3f p0_, const Vector3f p1_, int nbx_, int nby_, int nbz_);

      virtual ~DistanceField3D();
      
            //float distanceTo(Triangle *t,
      //                 const Vector3f& rkQ) const;

      float DistPointSegment(const Vector3f &p, const Vector3f &p0, const Vector3f &p1, Vector3f &result);
      float DistPointPlane(const Vector3f &p, const Vector3f &o, const Vector3f &n);
 //     float distanceToTriangle(const Vector3f &p, Triangle *t);
	  

//       void meshToDistanceField(Mesh &mesh);
//       void meshToDistanceField2(Mesh &mesh);
//       void meshToDistanceField3(Mesh &mesh);
//      void distanceFieldToDistanceField3D();

	  void AssignDistanceFieldFromBox(Vector3f low, Vector3f high, bool sign);

	  void DistanceFieldToBox(Vector3f& lo, Vector3f& hi, bool inverted);

	  float DistanceToSegment(Vector3f& pos, Vector3f& lo, Vector3f& hi);
	  float DistanceToSqure(Vector3f& pos, Vector3f& lo, Vector3f& hi, int axis);

	  float DistanceToBox(Vector3f& pos, Vector3f& lo, Vector3f& hi);
	  float DistanceToCylinder(Vector3f& pos, Vector3f& center, float radius, float height, int axis);
	  float DistanceToSphere(Vector3f& pos, Vector3f& center, float radius);


      void WriteToFile(string filename); 
      void ReadFromFile(string filename); 

	  void SetDistance(int index, float d) {
		  distances[index] = d;
	  }

      void SetDistance(int i, int j, int k, float d) {
         const int index = i+nbx*(j+nby*k);
         distances[index] = d;
      }

      float GetDistance(int i, int j, int k) {
         return distances[i+nbx*(j+nby*k)];
      }

      void GetDistance(const Vector3f &p, float &d);

      void GetDistance(const Vector3f &p, float &d, Vector3f &g);

      inline float Lerp(float a, float b, float alpha) const {
         return (1.0f-alpha)*a + alpha *b;
      }

 //     float inside(const Vector3f &p);

      void CheckConsistency();

      inline void Initialize();

 /*     inline void rasterize(const Particle &p) {
         const float r = p.r;
         float raster_r = p.distanceToSurface;
//         float raster_r = (p.distanceToSurface>p.r ? p.r: p.distanceToSurface);
      //   if (p.distanceToSurface>3*r) return;

         Vector3f start = (p - p0 - r)*invh; start.vfloor();
         Vector3f end   = (p - p0 + r)*invh; end.vceil();
         start.bound(Vector3f(0,0,0), Vector3f(nbx,nby,nbz));
         end.bound(Vector3f(0,0,0), Vector3f(nbx,nby,nbz));

         const float rr = r*r;
         const float inv_rr = 1.0f/rr;

         for (int k=(int)start.z; k<=(int)end.z; k++) {
            for (int j=(int)start.y; j<=(int)end.y; j++) {
               for (int i=(int)start.x; i<=(int)end.x; i++) {
                  const Vector3f grid_point = p0+Vector3f(i,j,k)*h;
                  const float sqrDist = grid_point.distanceSquared(p);
                  if (sqrDist > rr) continue;
                  const int index = i+nbx*(j+nby*k);
                  float weight = 1.0f-sqrDist*inv_rr;
                  weight =weight*weight*weight;
                  positions[index] += weight*p;
                  distances[index] += weight*raster_r;
                  weights[index] += weight;
               }
            }
         }
      }
*/
      inline void Normalize();

      //void sweep();

//      void dumpPovray(string filename, int dumpiter);

      //void getSurfacePoints(std::vector<SurfacePoint> &surface);
//       void draw(bool invert = false);
//       void drawSurface();
//       void drawBox();
//       void initSurface();


      void Translate(const Vector3f &t);
      void Scale(const Vector3f &s);
      void Invert();

      //void closestPointSweep(std::vector<Vector3f> &surface, KdTree *tree);
      //void sweepDirection(int startx, int endx, int starty, int endy, int startz, int endz, int dx, int dy, int dz, int difx, int dify, int difz);
      //void drawMedialAxis();
      //void getMedialAxis(std::vector<Vector3f> &medial);

	  Vector3f p0; // lower left front corner
	  Vector3f p1; // upper right back corner
	  Vector3f h;  // single cell sizes
	  Vector3f invh;   // inverse of single cell sizes
	  int nbx,nby,nbz; // number of cells in all dimensions

      float * distances;
      float * weights;
       Vector3f * positions;
//       Vector3f * closestPoints;
//       Vector3f * closestNormals;
//      char * types;

//       KdTree *kdtree;
//       Mesh *mesh;
//       float longestdiagonal;
//       std::vector<Vector3f> triangles;
//       std::vector<Vector3f> normals;

/*      CIsoSurface<float> mc;*/
      int _list;
};

}

#endif
