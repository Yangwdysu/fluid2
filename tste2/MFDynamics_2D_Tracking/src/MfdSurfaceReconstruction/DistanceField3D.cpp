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

#include <windows.h>
#include <GL/gl.h>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <queue>

#include "Triangle.h"
#include "Vertex.h"
#include "Edge.h"
#include "Mesh.h"
#include "DistanceField3D.h"

using namespace mfd;

float mysign(float x) {
   if (x<0) return -1;
   else return +1;
}

DistanceField3D::DistanceField3D(string filename) {
//	kdtree = NULL;
//	mesh = NULL;
//	types=NULL;
// 	closestPoints = NULL;
// 	closestNormals = NULL;
	ReadFromFile(filename);
	_list = -1;
}

DistanceField3D::DistanceField3D(const Vector3f p0_, const Vector3f p1_, int nbx_, int nby_, int nbz_) 
: p0(p0_), p1(p1_), nbx(nbx_), nby(nby_), nbz(nbz_) {
// 	kdtree = NULL;
// 	mesh = NULL;
//	types=NULL;
// 	closestPoints = NULL;
// 	closestNormals = NULL;
	h = (p1-p0)*Vector3f(1.0f/float(nbx), 1.0f/float(nby), 1.0f/float(nbz));
	invh = Vector3f(1.0f/h.x, 1.0f/h.y, 1.0f/h.z);
	distances = new float[(nbx+1)*(nby+1)*(nbz+1)];
	weights = new float[(nbx+1)*(nby+1)*(nbz+1)];
	positions = new Vector3f[(nbx+1)*(nby+1)*(nbz+1)];
	_list = -1;
}

DistanceField3D::~DistanceField3D() {
         delete[] distances;
         distances = NULL;
         delete[] weights;
         weights = NULL;
//          delete[] positions;
//          positions = NULL;
//          delete kdtree; kdtree = NULL;
//          delete mesh; mesh = NULL;
//         delete[] types; types=NULL;
//          delete[] closestPoints; closestPoints=NULL;
//          delete[] closestNormals; closestNormals=NULL;
      }


void DistanceField3D::Initialize() {
	for (int i=0; i<=nbx; i++) {
		for (int j=0; j<=nby; j++) {
			for (int k=0; k<=nbz; k++) {
				const int index = i+nbx*(j+nby*k);
				weights[index] = 0;
				distances[index] = 0;
			}
		}
	}
}

void DistanceField3D::Normalize() {
	for (int i=0; i<=nbx; i++) {
		for (int j=0; j<=nby; j++) {
			for (int k=0; k<=nbz; k++) {
				const Vector3f p = p0+Vector3f(i,j,k)*h;
				const int index = i+nbx*(j+nby*k);
				const float weight = weights[index];
				if (weight<0.00000000001f) distances[index] = -h.x/8.0f;
				else {
					float inv_weight = 1.0f/weight;
					distances[index] = distances[index] * inv_weight;
// 					// this is the final field
// 					distances[index] = (dist- Distance(pos, p));
				}
			}
		}
	}
}

/*
void DistanceField3D::meshToDistanceField(Mesh &mesh) {

   std::vector<int> triangles;

   int nbtriangles = mesh.triangles.size();
   
   std::vector<Vector3f> trianglecenters;
   float longestdiagonal = 0;
   
   // loop over triangles, compute triangle centers and longest diagonal
   for (int i=0; i<nbtriangles; i++) {
      const Vector3f center = mesh.triangles[i]->center;
      trianglecenters.push_back(center);
      for (int j=0; j<3; j++) {
         float dist = Distance(*(mesh.triangles[i]->v[j]), center); 
         if (dist>longestdiagonal) longestdiagonal=dist;
      }
   }
   // put trianles in kd-tree
   KdTree * tree = new KdTree(trianglecenters, trianglecenters.size(), 10);
   for (int i=0; i<=nbx; i++) {
      cout << (float(i))/float(nbx) * 100.0f << "%" << endl;
      for (int j=0; j<=nby; j++) {
         for (int k=0; k<=nbz; k++) {
            // real world coordinate
            Vector3f p = p0 + Vector3f(i,j,k)*h;
            // for each point on grid, compute closest triangle center
            tree->setNOfNeighbours(1);
            tree->queryPosition(p);
            float dist = sqrt(tree->getSquaredDistance(0));
            // possible closest point is at most this far away:
            float closestpossibledist = max(0.0f, dist-longestdiagonal);
            // for grid points far away we don't compute distance
            // we will do flood filling at a later stage
            /*if (closestpossibledist > 10*h.length()) {
               setDistance(i,j,k,-1); // TODO: make it a marker or a boolean
               continue;
            }*/
            // do second (range) query with found closest distance + longestdiagonal
/*            float newdist = dist + longestdiagonal;
            tree->setNOfNeighbours(10000);
            tree->queryRange(p, newdist*newdist);
//            tree->setNOfNeighbours(4);
            int nbfound = tree->getNOfFoundNeighbours();
            // loop over all found triangles and find closest point
            float mindist = FLT_MAX;
            for (int m=0; m<nbfound; m++) {
               int index = tree->getNeighbourPositionIndex(m);
               // compute closest point on triangle
               // use angle-weighted normal of closest point to define inside/outside
               float dist = distanceToTriangle(p,mesh.triangles[index]);
               if (fabs(dist)<fabs(mindist)) {
                  mindist = dist;
               }
            }
            setDistance(i,j,k,mindist);
         }
      }
   }
}
*/

void DistanceField3D::GetDistance(const Vector3f &p, float &d) {
	// get cell and lerp values
	Vector3f fp = (p-p0)*invh;
	const int i = (int)floor(fp.x);
	const int j = (int)floor(fp.y);
	const int k = (int)floor(fp.z);
	if (i<0 || i>=nbx || j<0 || j>=nby || k<0 || k>=nbz) {
		d = -10;
		return;
	}
	Vector3f ip(i,j,k);
	//ip.bound(Vector3f(0,0,0), Vector3f(nbx-1,nby-1,nbz-1));

	Vector3f alphav = fp-ip;
	float alpha = alphav.x;
	float beta = alphav.y;
	float gamma = alphav.z;

	const float d000 = GetDistance(i,j,k);
	const float d100 = GetDistance(i+1,j,k);
	const float d010 = GetDistance(i,j+1,k);
	const float d110 = GetDistance(i+1,j+1,k);
	const float d001 = GetDistance(i,j,k+1);
	const float d101 = GetDistance(i+1,j,k+1);
	const float d011 = GetDistance(i,j+1,k+1);
	const float d111 = GetDistance(i+1,j+1,k+1);

	const float dx00 = (1.0f-alpha) * d000 + alpha * d100;
	const float dx10 = (1.0f-alpha) * d010 + alpha * d110;

	const float dxy0 = (1.0f-beta) * dx00 + beta * dx10;

	const float dx01 = (1.0f-alpha) * d001 + alpha * d101;
	const float dx11 = (1.0f-alpha) * d011 + alpha * d111;

	const float dxy1 = (1.0f-beta) * dx01 + beta * dx11;

	d = (1.0f-gamma) * dxy0 + gamma * dxy1;
}

void DistanceField3D::GetDistance(const Vector3f &p, float &d, Vector3f &g) {
	// get cell and lerp values
	Vector3f fp = (p-p0)*invh;
	const int i = (int)floor(fp.x);
	const int j = (int)floor(fp.y);
	const int k = (int)floor(fp.z);
	if (i<0 || i>=nbx || j<0 || j>=nby || k<0 || k>=nbz) {
		d = +10;
		g = 0.0f;
		return;
	}
	Vector3f ip(i,j,k);

	Vector3f alphav = fp-ip;
	float alpha = alphav.x;
	float beta = alphav.y;
	float gamma = alphav.z;

	const float d000 = GetDistance(i,j,k);
	const float d100 = GetDistance(i+1,j,k);
	const float d010 = GetDistance(i,j+1,k);
	const float d110 = GetDistance(i+1,j+1,k);
	const float d001 = GetDistance(i,j,k+1);
	const float d101 = GetDistance(i+1,j,k+1);
	const float d011 = GetDistance(i,j+1,k+1);
	const float d111 = GetDistance(i+1,j+1,k+1);

	const float dx00 = Lerp(d000, d100, alpha);
	const float dx10 = Lerp(d010, d110, alpha);
	const float dxy0 = Lerp(dx00, dx10, beta);

	const float dx01 = Lerp(d001, d101, alpha);
	const float dx11 = Lerp(d011, d111, alpha);
	const float dxy1 = Lerp(dx01, dx11, beta);

	const float d0y0 = Lerp(d000, d010, beta);
	const float d0y1 = Lerp(d001, d011, beta);
	const float d0yz = Lerp(d0y0, d0y1, gamma);

	const float d1y0 = Lerp(d100, d110, beta);
	const float d1y1 = Lerp(d101, d111, beta);
	const float d1yz = Lerp(d1y0, d1y1, gamma);

	const float dx0z = Lerp(dx00, dx01, gamma);
	const float dx1z = Lerp(dx10, dx11, gamma);

	g.x = d0yz - d1yz; 
	g.y = dx0z - dx1z; 
	g.z = dxy0 - dxy1;

	float length = g.Normalize();
	if (length<0.0001f) g = 0.0f;

	d = (1.0f-gamma) * dxy0 + gamma * dxy1;
}

float DistanceField3D::DistPointSegment(const Vector3f &p, const Vector3f &p0, const Vector3f &p1, Vector3f &result) {
   Vector3f v = p1-p0;
   Vector3f w = p-p0;
   float c1 = w.Dot(v);
   if ( c1 <= 0 ) return FLT_MAX; //p.distance(p0);
   float c2 = v.Dot(v);
   if ( c2 <= c1 ) return FLT_MAX; //p.distance(p1);
   float t = c1 / c2;
   result = p0 + t * v;
   return Distance(p, result);
}

float DistanceField3D::DistPointPlane(const Vector3f &p, const Vector3f &o, const Vector3f &n) {
   return fabs(n.Dot(p-o));
}

void DistanceField3D::CheckConsistency() {
   for (int i=0; i<=nbx; i++) {
      for (int j=0; j<=nby; j++) {
         for (int k=0; k<=nbz; k++) {
            float dist = GetDistance(i,j,k);
            // border is outside!
            if (i==0 || i==nbx || j==0 || j==nby || k==0 || k==nbz) {
               dist = fabs(dist);
               SetDistance(i,j,k,-dist);
               continue;
            }
            int nbpos = 0;
            int nbneg = 0;
            bool inconsistent = false;
            for (int di=-1; di<=+1; di++) {
               for (int dj=-1; dj<=+1; dj++) {
                  for (int dk=-1; dk<=+1; dk++) {
                     float ndist = GetDistance(i+di, j+dj, k+dk);
                     if (ndist<0) nbneg++;
                     else nbpos++;
                     if (dist*ndist<0) {
                        if (fabs(dist) > (h.Dot(Vector3f(fabs((float)di),fabs((float)dj),fabs((float)dk))))) inconsistent = true;
                        if (fabs(ndist) > (h.Dot(Vector3f(fabs((float)di),fabs((float)dj),fabs((float)dk))))) inconsistent = true;
                     }
                  }
               }
            }
            if (inconsistent) {
               cout << "Inconsistent: " << dist << " " << nbpos << " " << nbneg << endl;
               if (nbpos>nbneg) SetDistance(i,j,k,fabs(dist));
               else if (nbneg>nbpos) SetDistance(i,j,k,-fabs(dist));
            }
         }
      }
   }
   WriteToFile("teapot.df");
}

void DistanceField3D::WriteToFile(string filename) {
   ofstream output(filename.c_str(), ios::out|ios::binary);
   output.write((char*)&p0.x, sizeof(float));
   output.write((char*)&p0.y, sizeof(float));
   output.write((char*)&p0.z, sizeof(float));
   output.write((char*)&p1.x, sizeof(float));
   output.write((char*)&p1.y, sizeof(float));
   output.write((char*)&p1.z, sizeof(float));
   output.write((char*)&h.x, sizeof(float));
   output.write((char*)&h.y, sizeof(float));
   output.write((char*)&h.z, sizeof(float));
   output.write((char*)&nbx, sizeof(int));
   output.write((char*)&nby, sizeof(int));
   output.write((char*)&nbz, sizeof(int));
   for (int i=0; i<=nbx; i++) {
      for (int j=0; j<=nby; j++) {
         for (int k=0; k<=nbz; k++) {
            float dist = GetDistance(i,j,k);
            output.write((char*)&dist, sizeof(float));
         }
      }
   }
   output.close();
}

void DistanceField3D::Translate(const Vector3f &t) {
   p0+=t;
   p1+=t;
}

void DistanceField3D::Scale(const Vector3f &s) {
   p0.x *= s.x;
   p0.y *= s.y;
   p0.z *= s.z;
   p1.x *= s.x;
   p1.y *= s.y;
   p1.z *= s.z;
   h.x *= s.x;
   h.y *= s.y;
   h.z *= s.z;
   invh = Vector3f(1.0f/h.x, 1.0f/h.y, 1.0f/h.z);
   for (int i=0; i<(nbx+1)*(nby+1)*(nbz+1); i++) {
      distances[i]=s.x*distances[i];
   }
}

void DistanceField3D::Invert() {
   const int nb = (nbx+1)*(nby+1)*(nbz+1);
   for (int i=0; i<nb; i++) distances[i] = -distances[i];
}

void DistanceField3D::ReadFromFile(string filename) {
   fstream input(filename.c_str(), ios::in|ios::binary);
   int xx, yy, zz;
   input.read((char*)&xx, sizeof(int));
   input.read((char*)&yy, sizeof(int));
   input.read((char*)&zz, sizeof(int));
   float t_h;
   input.read((char*)&t_h, sizeof(float));
   input.read((char*)&p0.x, sizeof(float));
   input.read((char*)&p0.y, sizeof(float));
   input.read((char*)&p0.z, sizeof(float));
   p1.x = p0.x+t_h*xx;
   p1.y = p0.y+t_h*yy;
   p1.z = p0.z+t_h*zz;

   cout << "Bunny: " << p0.x << ", " << p0.y << ", " << p0.z << endl;
   cout << "Bunny: " << p1.x << ", " << p1.y << ", " << p1.z << endl;
   
   nbx = xx/4;
   nby = yy/4;
   nbz = zz/4;
   h.x = t_h*4;
   h.y = t_h*4;
   h.z = t_h*4;
   
   distances = new float[(nbx+1)*(nby+1)*(nbz+1)];
   invh = Vector3f(1.0f/h.x, 1.0f/h.y, 1.0f/h.z);
   for (int i=0; i<xx; i++) {
	   for (int j=0; j<yy; j++) {
		   for (int k=0; k<zz; k++) {
			   float dist;
			   input.read((char*)&dist, sizeof(float));
			   if ((i%4)==0 && (j%4)==0 && (k%4)==0)
			   {
				   //int index = (i/4)*(nby+1)*(nbz+1)+(j/4)*(nbz+1)+(k/4);
				   SetDistance((i/4), (j/4), (k/4),dist*t_h);
			   }
		   }
	   }
   }
   input.close();
   weights = NULL;


   cout << "read data successful" << endl;
/*   ifstream input(filename.c_str(), ios::in|ios::binary);
   input.read((char*)&p0.x, sizeof(float));
   input.read((char*)&p0.y, sizeof(float));
   input.read((char*)&p0.z, sizeof(float));
   input.read((char*)&p1.x, sizeof(float));
   input.read((char*)&p1.y, sizeof(float));
   input.read((char*)&p1.z, sizeof(float));
   input.read((char*)&h.x, sizeof(float));
   input.read((char*)&h.y, sizeof(float));
   input.read((char*)&h.z, sizeof(float));
   input.read((char*)&nbx, sizeof(int));
   input.read((char*)&nby, sizeof(int));
   input.read((char*)&nbz, sizeof(int));
   distances = new float[(nbx+1)*(nby+1)*(nbz+1)];
   invh.set(1.0f/h.x, 1.0f/h.y, 1.0f/h.z);
   for (int i=0; i<=nbx; i++) {
	   for (int j=0; j<=nby; j++) {
		   for (int k=0; k<=nbz; k++) {
			   float dist;
			   input.read((char*)&dist, sizeof(float));
			   setDistance(i,j,k,-dist);
		   }
	   }
   }
   input.close();
   weights = NULL;
   positions = NULL;*/

}


/*
void DistanceField3D::draw(bool invert) {
#ifndef OFFLINE_SIMULATION
   glEnable(GL_LIGHTING);
   glShadeModel(GL_SMOOTH);
//  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
   glBegin(GL_TRIANGLES);
   for (unsigned int i=0; i<mc.m_nTriangles; i++) {
      if (invert) {
      for (int j=0; j<=2; j++) {
         const POINT3D &p = mc.m_ppt3dVertices[mc.m_piTriangleIndices[3*i+j]];
         const POINT3D &n = mc.m_pvec3dNormals[mc.m_piTriangleIndices[3*i+j]];
         glNormal3f(-n[0], -n[1], -n[2]);
         glVertex3f(p[0]+p0.x, p[1]+p0.y, p[2]+p0.z);
      }
      }
      else {
      for (int j=2; j>=0; j--) {
         const POINT3D &p = mc.m_ppt3dVertices[mc.m_piTriangleIndices[3*i+j]];
         const POINT3D &n = mc.m_pvec3dNormals[mc.m_piTriangleIndices[3*i+j]];
         glNormal3f(n[0], n[1], n[2]);
         glVertex3f(p[0]+p0.x, p[1]+p0.y, p[2]+p0.z);
      }
      }
   }
   glEnd();
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif
}
*/
/* 
void DistanceField3D::dumpPovray(string filename, int dumpiter) {
  // TODO: change to mesh2!
	string str;

   int framenumber = dumpiter-1000000;
   stringstream ss2; ss2 << framenumber;
   string render = string("./render.sh ") + ss2.str() + string(" ") + ss2.str() + string(" ") + Config::pov + " " + Config::rendered;

   stringstream ss; ss << dumpiter;
   filename += string("dump_") + ss.str() + string(".txt");
   ofstream output(filename.c_str());

   //ogzstream output(filename.c_str(), ios::out);
   output << "   vertex_vectors {" << endl;
   output << "      " << mc.m_nVertices << "," << endl;
   for (int i=0; i<mc.m_nVertices; i++) {
      output << "<" << mc.m_ppt3dVertices[i][0]+p0.x << "," 
                         << mc.m_ppt3dVertices[i][1]+p0.y << "," 
                         << mc.m_ppt3dVertices[i][2]+p0.z << ">";
      if (i==mc.m_nVertices-1) output << endl;
      else output << ",";
   }
   output << "   }" << endl;
   output << "   normal_vectors {" << endl;
   output << "      " << mc.m_nVertices << "," << endl;
   for (int i=0; i<mc.m_nVertices; i++) {
      const POINT3D &n = mc.m_pvec3dNormals[i];
      output << "<" << n[0] << "," 
                         << n[1] << "," 
                         << n[2] << ">";
      if (i==mc.m_nVertices-1) output << endl;
      else output << ",";
   }
   output << "   }" << endl;
   output << "   face_indices {" << endl;
   output << "      " << mc.m_nTriangles << "," << endl;
   for (int i=0; i<mc.m_nTriangles; i++) {
      int i0 = mc.m_piTriangleIndices[3*i+0];
      int i1 = mc.m_piTriangleIndices[3*i+1];
      int i2 = mc.m_piTriangleIndices[3*i+2];
      output << "<" << i0 << ","
                         << i1 << ","
                         << i2 << ">";
      if (i==mc.m_nTriangles-1) output << endl;
      else output << ",";
   }
   output << "   }" << endl;


//   output << "}" << endl;
   /*
   output << "#declare fluid_surface = mesh {" << endl;
   for (int i=0; i<mc.m_nTriangles; i++) {
      output << "   smooth_triangle {" << endl;
      for (int j=2; j>=0; j--) {
         const POINT3D &p = mc.m_ppt3dVertices[mc.m_piTriangleIndices[3*i+j]];
         const POINT3D &n = mc.m_pvec3dNormals[mc.m_piTriangleIndices[3*i+j]];
         output << "<" << p0.x+p[0] << ", " << p0.y+p[1] << ", " << p0.z+p[2] << ">, ";
         output << "<" << n[0] << ", " << n[1] << ", " << n[2] << ">";
         if (j==0) output << endl;
         else output << "," << endl;
      }
      output << "   }" << endl;
   }
   output << "}" << endl;
   
//    output.close();
//    cout << "Written: " << filename << endl;
}*/

/*
void DistanceField3D::drawBox() {
#ifndef OFFLINE_SIMULATION
   glDisable(GL_LIGHTING);
   glColor3f(1,1,1);
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glTranslatef(p0.x, p0.y, p0.z);
   glScalef(p1.x-p0.x, p1.y-p0.y, p1.z-p0.z);
   glTranslatef(0.5,0.5,0.5);
   glutWireCube(1);
   glPopMatrix();
#endif
}


void DistanceField3D::drawSurface() {
#ifndef OFFLINE_SIMULATION
   glEnable(GL_LIGHTING);
   glColor3f(0.7,0.5,0);
   if (_list==-1) {
      _list = glGenLists(1);
      glNewList(_list, GL_COMPILE);
   glBegin(GL_TRIANGLES);
   cout << normals.size() << endl;
   for (unsigned int i=0; i<normals.size(); i++) {
      Vector3f &n = normals[i];
      Vector3f &v0 = triangles[3*i+0];
      Vector3f &v1 = triangles[3*i+1];
      Vector3f &v2 = triangles[3*i+2];
      glNormal3fv(&(n.x));
      glVertex3fv(&(v0.x));
      glVertex3fv(&(v1.x));
      glVertex3fv(&(v2.x));
   }
   glEnd();
   glEndList();
   }
   else glCallList(_list);
#endif
}

*/

void DistanceField3D::DistanceFieldToBox(Vector3f& lo, Vector3f& hi, bool inverted)
{
	int sign = inverted ? 1.0f : -1.0f;
	for (int i=0; i<=nbx; i++) {

		for (int j=0; j<=nby; j++) {
			for (int k=0; k<=nbz; k++) {
				Vector3f p = p0 + Vector3f(i,j,k)*h;
				float dist  = sign*DistanceToBox(p, lo, hi);
				SetDistance(i,j,k,dist);
			}
		}
	}
}

float DistanceField3D::DistanceToBox( Vector3f& pos, Vector3f& lo, Vector3f& hi )
{
	Vector3f corner0 = Vector3f(lo.x, lo.y, lo.z);
	Vector3f corner1 = Vector3f(hi.x, lo.y, lo.z);
	Vector3f corner2 = Vector3f(hi.x, hi.y, lo.z);
	Vector3f corner3 = Vector3f(lo.x, hi.y, lo.z);
	Vector3f corner4 = Vector3f(lo.x, lo.y, hi.z);
	Vector3f corner5 = Vector3f(hi.x, lo.y, hi.z);
	Vector3f corner6 = Vector3f(hi.x, hi.y, hi.z);
	Vector3f corner7 = Vector3f(lo.x, hi.y, hi.z);
	float dist0 = Distance(pos, corner0);
	float dist1 = Distance(pos, corner1);
	float dist2 = Distance(pos, corner2);
	float dist3 = Distance(pos, corner3);
	float dist4 = Distance(pos, corner4);
	float dist5 = Distance(pos, corner5);
	float dist6 = Distance(pos, corner6);
	float dist7 = Distance(pos, corner7);
	if (pos.x < hi.x && pos.x > lo.x && pos.y < hi.y && pos.y > lo.y && pos.z < hi.z && pos.z > lo.z)
	{
		float distx = min(abs(pos.x-hi.x), abs(pos.x-lo.x));
		float disty = min(abs(pos.y-hi.y), abs(pos.y-lo.y));
		float distz = min(abs(pos.z-hi.z), abs(pos.z-lo.z));
		float mindist = min(distx, disty);
		mindist = min(mindist, distz);
		return mindist;
	}
	else
	{
		float distx1 = DistanceToSqure(pos, corner0, corner7, 0);
		float distx2 = DistanceToSqure(pos, corner1, corner6, 0);
		float disty1 = DistanceToSqure(pos, corner0, corner5, 1);
		float disty2 = DistanceToSqure(pos, corner3, corner6, 1);
		float distz1 = DistanceToSqure(pos, corner0, corner2, 2);
		float distz2 = DistanceToSqure(pos, corner4, corner6, 2);
		return -min(min(min(distx1, distx2), min(disty1, disty2)), min(distz1, distz2));
	}
}

float DistanceField3D::DistanceToSqure( Vector3f& pos, Vector3f& lo, Vector3f& hi, int axis )
{
	Vector3f n;
	Vector3f corner1, corner2, corner3, corner4;
	Vector3f loCorner, hiCorner, p;
	switch(axis)
	{
	case 0:
		corner1 = Vector3f(lo.x, lo.y, lo.z);
		corner2 = Vector3f(lo.x, hi.y, lo.z);
		corner3 = Vector3f(lo.x, hi.y, hi.z);
		corner4 = Vector3f(lo.x, lo.y, hi.z);
		n = Vector3f(1.0f, 0.0f, 0.0f);

		loCorner = Vector3f(lo.y, lo.z, 0.0f);
		hiCorner = Vector3f(hi.y, hi.z, 0.0f);
		p = Vector3f(pos.y, pos.z, 0.0f);
		break;
	case 1:
		corner1 = Vector3f(lo.x, lo.y, lo.z);
		corner2 = Vector3f(lo.x, lo.y, hi.z);
		corner3 = Vector3f(hi.x, lo.y, hi.z);
		corner4 = Vector3f(hi.x, lo.y, lo.z);
		n = Vector3f(0.0f, 1.0f, 0.0f);

		loCorner = Vector3f(lo.x, lo.z, 0.0f);
		hiCorner = Vector3f(hi.x, hi.z, 0.0f);
		p = Vector3f(pos.x, pos.z, 0.0f);
		break;
	case 2:
		corner1 = Vector3f(lo.x, lo.y, lo.z);
		corner2 = Vector3f(hi.x, lo.y, lo.z);
		corner3 = Vector3f(hi.x, hi.y, lo.z);
		corner4 = Vector3f(lo.x, hi.y, lo.z);
		n = Vector3f(0.0f, 0.0f, 1.0f);

		loCorner = Vector3f(lo.x, lo.y, 0.0f);
		hiCorner = Vector3f(hi.x, hi.y, 0.0f);
		p = Vector3f(pos.x, pos.y, 0.0f);
		break;
	}

	float dist1 = DistanceToSegment(pos, corner1, corner2);
	float dist2 = DistanceToSegment(pos, corner2, corner3);
	float dist3 = DistanceToSegment(pos, corner3, corner4);
	float dist4 = DistanceToSegment(pos, corner4, corner1);
	float dist5 = abs(n.Dot(pos-corner1));
	if (p.x < hiCorner.x && p.x > loCorner.x && p.y < hiCorner.y && p.y > loCorner.y)
		return dist5;
	else
		return min(min(dist1, dist2), min(dist3, dist4));
}

float DistanceField3D::DistanceToSegment( Vector3f& pos, Vector3f& lo, Vector3f& hi )
{
	Vector3f seg = hi-lo;
	Vector3f edge1 = pos-lo;
	Vector3f edge2 = pos-hi;
	if (edge1.Dot(seg) < 0.0f)
	{
		return edge1.Length();
	}
	if (edge2.Dot(-seg) < 0.0f)
	{
		return edge2.Length();
	}
	float length1 = edge1.LengthSq();
	seg.Normalize();
	float length2 = edge1.Dot(seg);
	return sqrt(length1 - length2*length2);
}

float DistanceField3D::DistanceToCylinder( Vector3f& pos, Vector3f& center, float radius, float height, int axis )
{
	float distR;
	float distH;
	switch(axis)
	{
	case 0:
		distH = abs(pos.x-center.x);
		distR = Vector3f(0.0f, pos.y-center.y, pos.z-center.z).Length();
		break;
	case 1:
		distH = abs(pos.y-center.y);
		distR = Vector3f(pos.x-center.x, 0.0f, pos.z-center.z).Length();
		break;
	case 2:
		distH = abs(pos.z-center.z);
		distR = Vector3f(pos.x-center.x, pos.y-center.y, 0.0f).Length();
		break;
	}

	float halfH = height/2.0f;
	if (distH <= halfH && distR <= radius)
	{
		return min(distH, distR);
	}
	else if(distH > halfH && distR <= radius)
	{
		return halfH-distH;
	}
	else if (distH <= halfH && distR > radius)
	{
		return radius-distR;
	}
	else
	{
		float l1 = distR-radius;
		float l2 = distH-halfH;
		return -sqrt(l1*l1+l2*l2);
	}
	
	
}

float DistanceField3D::DistanceToSphere( Vector3f& pos, Vector3f& center, float radius )
{
	Vector3f dir = pos-center;
	return dir.Length()-radius;
}

void DistanceField3D::AssignDistanceFieldFromBox( Vector3f low, Vector3f high, bool sign)
{
	float s = sign ? 1.0f : -1.0f;
	for (int i=0; i<=nbx; i++) {

		for (int j=0; j<=nby; j++) {
			for (int k=0; k<=nbz; k++) {
				Vector3f p = p0 + Vector3f(i,j,k)*h;
				float dist1  = DistanceToBox(p, low, high);

				SetDistance(i,j,k,s*dist1);
			}
		}
	}
}
