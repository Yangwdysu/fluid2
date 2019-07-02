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

#include <map>
#include <cfloat>
#include <cassert>
#include <iostream>
#include <fstream>
#include "Mesh.h"
#include "Vertex.h"
#include "Triangle.h"
#include "Edge.h"

using namespace mfd;

void Mesh::LoadOFF(string filename) {
   ifstream input(filename.c_str(), ios::in);
   string head;
   input >> head;
   int nbvertices, nbtriangles, nbedges;
   input >> nbvertices >> nbtriangles >> nbedges;

   cout << "Loading vertices" << endl;
   float x, y ,z;
   m = Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
   M = Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
   for (int i=0; i<nbvertices; i++) {
      input >> x >> y >> z; // TODO: change this back!!!
      vertices.push_back(new Vertex(x,y,z));
      m = Min(*vertices.back(), m);
      M = Max(*vertices.back(), M);
   }

   Vector3f size = M-m;
   float scale = size.x;
   if (size.y>scale) scale = size.y;
   if (size.z>scale) scale = size.z;
   scale = 1.0f/scale;
   Vector3f center = 0.5f*(m + M);
   center.y = m.y;
   //scale = 1;
   for (int i=0; i<nbvertices; i++) {
      (*vertices[i]) = (*vertices[i] - center)*scale;
      (*vertices[i]) = (*vertices[i] + Vector3f(0.5,0,0.5));
   }

   M = (M-center)*scale;
//   M.z-=0.5f;
   M = M+Vector3f(0.5,0,0.5);
   m = (m-center)*scale;
   m = m+Vector3f(0.5,0,0.5);
//   m.z-=0.5f;

//    m.print();
//    M.print();

   cout << "Loading triangles" << endl;
   map<int,int> edgeids;
   int index[3];
   int dummy;
   for (int i=0; i<nbtriangles; i++) {
      Triangle * t = new Triangle();
         input >> dummy;
      for (int j=0; j<3; j++) {  
         input >> index[j];
         t->v[j] = vertices[index[j]];
      }

      t->normal = -((*t->v[1]) - (*t->v[0])).Cross((*t->v[2]) - (*t->v[1]));
      t->area = 0.5f*t->normal.Normalize();
      t->center = (1.0f/3.0f) * ((*t->v[0]) + (*t->v[1]) + (*t->v[2]));

      triangles.push_back(t);

      int edge[3];
      for (int j=0; j<3; j++) {
         edge[j] = (index[j]<index[(j+1)%3]) ? 10000*index[j]+index[(j+1)%3]: 10000*index[(j+1)%3]+index[j];
      }

      for (int j=0; j<3; j++) {
         map<int,int>::iterator iter;
         iter = edgeids.find(edge[j]);
         if (iter != edgeids.end()) {
            t->e[j] = edges[iter->second];
            edges[iter->second]->t[1] = t;
         }
         else {
            Edge *e = new Edge();
            e->v[0] = t->v[j];
            e->v[1] = t->v[(j+1)%3];
            e->t[0] = t;
            t->e[j] = e;
            edges.push_back(e);
            edgeids[edge[j]] = edges.size()-1;
         }
      }
      
   }

   for (int i=0; i<edges.size(); i++) {
      Edge * e = edges[i];
      if (e->t[0] == NULL || e->t[1] == NULL) e->border = true;
      else e->border = false;
   }
}

void Mesh::LoadWRL(string filename) {
   ifstream input(filename.c_str(), ios::in);
   int nbvertices, nbtriangles;
   input >> nbvertices >> nbtriangles;

   cout << "Loading vertices" << endl;
   float x, y ,z;
   m = Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
   M = Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
   for (int i=0; i<nbvertices; i++) {
      input >> x >> y >> z; // TODO: change this back!!!
      //x*=1.5f;
      vertices.push_back(new Vertex(x,y,z));
      m = Min(*vertices.back(), m);
      M = Max(*vertices.back(), M);
   }

   Vector3f size = M-m;
   float scale = size.x;
   if (size.y>scale) scale = size.y;
   if (size.z>scale) scale = size.z;
   scale = 1.0f/scale;
   Vector3f center = 0.5f*(m + M);
   center.y = m.y;

   center = 0.0f;
   scale = 1;
   //scale = 1;
   for (int i=0; i<nbvertices; i++) {
      (*vertices[i]) = (*vertices[i] - center)*scale;
      (*vertices[i]) = *vertices[i] /*+ Vector3f(0.5,0,0.5)*/;
   }

/*   M.set((M-center)*scale);
//   M.z-=0.5f;
   M.set(M+Vector3f(0.5,0,0.5));
   m.set((m-center)*scale);
   m.set(m+Vector3f(0.5,0,0.5));*/
//   m.z-=0.5f;

//    m.print();
//    M.print();

   cout << "Loading triangles" << endl;
   map<int,int> edgeids;
   int index[3];
   for (int i=0; i<nbtriangles; i++) {
      Triangle * t = new Triangle();
      //for (int j=0; j<3; j++) {
      for (int j=2; j>=0; j--) {
         input >> index[j];
         t->v[j] = vertices[index[j]];
      }

      t->normal = -((*t->v[1]) - (*t->v[0])).Cross((*t->v[2]) - (*t->v[1]));
      t->area = 0.5f*t->normal.Normalize();
      t->center = (1.0f/3.0f) * ((*t->v[0]) + (*t->v[1]) + (*t->v[2]));

      triangles.push_back(t);

      int edge[3];
      for (int j=0; j<3; j++) {
         edge[j] = (index[j]<index[(j+1)%3]) ? 10000*index[j]+index[(j+1)%3]: 10000*index[(j+1)%3]+index[j];
      }

      for (int j=0; j<3; j++) {
         map<int,int>::iterator iter;
         iter = edgeids.find(edge[j]);
         if (iter != edgeids.end()) {
            t->e[j] = edges[iter->second];
            edges[iter->second]->t[1] = t;
         }
         else {
            Edge *e = new Edge();
            e->v[0] = t->v[j];
            e->v[1] = t->v[(j+1)%3];
            e->t[0] = t;
            t->e[j] = e;
            edges.push_back(e);
            edgeids[edge[j]] = edges.size()-1;
         }
      }
      
   }

   for (int i=0; i<edges.size(); i++) {
      Edge * e = edges[i];
      if (e->t[0] == NULL || e->t[1] == NULL) e->border = true;
      else e->border = false;
   }
}

void Mesh::ComputeNormals() {
   cout << "Computing normals" << endl;
   const int nbtriangles = triangles.size();
   const int nbvertices = vertices.size();
   const int nbedges = edges.size();

   // angle weighted normal for edges
   cout << "Computing edge normals" << endl;
   for (int i=0; i<nbedges; i++) {
      Edge * e = edges[i];
      e->normal = 0.0f;
      if (e->t[0]!=NULL) e->normal+=e->t[0]->normal;
      if (e->t[1]!=NULL) e->normal+=e->t[1]->normal;
      float length = e->normal.Normalize();
   }

   for (int i=0; i<nbvertices; i++) {
      Vertex * v = vertices[i];
      v->normal = 0.0f;
   }

   // loop over triangles and add weighted normal to vertices
   cout << "Computing vertex normals" << endl;
   for (int i=0; i<nbtriangles; i++) {
      Triangle * t = triangles[i];
      for (int j=0; j<3; j++) {
         Vector3f e1 = *t->v[(j+1)%3] - *t->v[j];
         Vector3f e2 = *t->v[(j+2)%3] - *t->v[j];
         e1.Normalize();
         e2.Normalize();
         float weight = acos(e1.Dot(e2));
         t->v[j]->normal=weight*t->normal;
      }
   }

   for (int i=0; i<nbvertices; i++) {
      Vertex * v = vertices[i];
      v->normal.Normalize();
   }
   cout << "Computing normals ended" << endl;
}

/*void Mesh::draw()  {

   glDisable(GL_LIGHTING);
   const int nbvertices = vertices.size();

   glColor3f(1,1,1);
   glPointSize(3);
   glBegin(GL_POINTS);
   for (int i=0; i<nbvertices; i++) {
      Vertex *v = vertices[i];
      glVertex3f(v->x, v->y, v->z);
   }
   glEnd();

   glColor3f(1,0,0);
   glLineWidth(1);
   glBegin(GL_LINES);
   for (int i=0; i<nbvertices; i++) {
      Vertex *v = vertices[i];
      glVertex3f(v->x, v->y, v->z);
      glVertex3f(v->x+0.1*v->normal.x, v->y+0.1*v->normal.y, v->z+0.1*v->normal.z);
   }
   glEnd();
}*/
