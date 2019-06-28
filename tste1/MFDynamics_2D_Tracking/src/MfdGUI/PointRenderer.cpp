#include "PointRenderer.h"
#include "MfdMath/DataStructure/Vec.h"

PointRenderer::PointRenderer(Vector3f* _data, int _n)
{
	data = _data;
	n = _n;
	mapping = NULL;
}


PointRenderer::~PointRenderer(void)
{
}

void PointRenderer::Render( const Camera &camera )
{
	glDisable(GL_LIGHTING);
	glLineWidth(4);
	glBegin(GL_LINES);
	glColor3f(1,0,0);
	glVertex3f(0,0,0);
	glVertex3f(1,0,0);
	glColor3f(0,1,0);
	glVertex3f(0,0,0);
	glVertex3f(0,1,0);
	glColor3f(0,0,1);
	glVertex3f(0,0,0);
	glVertex3f(0,0,1);
	glEnd();

	Color color;
// 	Vector3f color;
// 	float maxvalue = 0.0f;
// 	if (data != NULL)
// 	{
// 		for (int i = 0; i < n; i++)
// 		{
// 			if (data[i] > maxvalue)
// 			{
// 				maxvalue = data[i];
// 			}
// 		}
// 	}

	if (true) { // draw simple gl_points (fastest)
		glDisable(GL_LIGHTING);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glPointSize(4);
		glBegin(GL_POINTS);
		for (int i=0; i< n; i++) {
			if ((*marker)[i])
			{
				// 			if (mapping != NULL)
				// 			{
				// 				color.set(120.0f*((mapping[i])/maxvalue)+120.0f, 1.0, 1.0);
				// 				color.HSVtoRGB();
				//				glColor3f(color.x, color.y, color.z);
				// 			}
				// 			else
				// 				glColor3f(posArr[i].x, posArr[i].y, posArr[i].z);

				//			if (MATERIALTYPE(attriArr[i]) != MATERIAL_FLUID)
				// 			{
				// 				glColor3f(0.8f, 0.8f, 0.8f);
				// 			}
				// 			else
				// 				glColor3f(0.0f, 1.0f, 0.0f);


				if (mapping != NULL)
				{
					color.HSVtoRGB((*mapping)[i]/(*green)*120.0f+120.0f, 1.0f, 1.0f);
					glColor4f(color.r, color.g, color.b, (*mapping)[i]/(*green));
				}
				else
					glColor4f(0.0f, 1.0f, 1.0f, 1.0f);

				glVertex3f(data[i].x, data[i].y, data[i].z);
			}
			

		}
		glEnd();
	}
}

BallRenderer::BallRenderer(Vector3f* _data, int _n)
{
	data = _data;
	n = _n;
}

BallRenderer::~BallRenderer(void)
{
}

void BallRenderer::Render( const Camera &camera )
{
	glDisable(GL_LIGHTING);
	glLineWidth(4);
	glBegin(GL_LINES);
	glColor3f(1,0,0);
	glVertex3f(0,0,0);
	glVertex3f(1,0,0);
	glColor3f(0,1,0);
	glVertex3f(0,0,0);
	glVertex3f(0,1,0);
	glColor3f(0,0,1);
	glVertex3f(0,0,0);
	glVertex3f(0,0,1);
	glEnd();

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glMatrixMode(GL_MODELVIEW);
	for (int i=0; i< n; i++) {
		glColor3f(1.0f, 1.0f, 1.0f);
		glPushMatrix();
		glTranslatef(data[i].x, data[i].y, data[i].z);
		glutSolidSphere(0.004,6,6);
		glPopMatrix();
	}
}

void VectorRender::Render( const Camera &camera )
{
	glDisable(GL_LIGHTING);
	glLineWidth(4);
	glBegin(GL_LINES);
	glColor3f(1,0,0);
	glVertex3f(0,0,0);
	glVertex3f(1,0,0);
	glColor3f(0,1,0);
	glVertex3f(0,0,0);
	glVertex3f(0,1,0);
	glColor3f(0,0,1);
	glVertex3f(0,0,0);
	glVertex3f(0,0,1);
	glEnd();

	if (true) { // draw simple gl_points (fastest)
		glPointSize(4);
		glLineWidth(1);

		glColor3f(0.0f, 1.0f, 0.0f);
		glBegin(GL_POINTS);
		for (int i=0; i< n; i++) {
			glVertex3f(pos[i].x, pos[i].y, pos[i].z);
		}
		glEnd();

		glColor3f(0.0f, 0.0f, 1.0f);
		glBegin(GL_LINES);
		for (int i=0; i< n; i++) {
			glVertex3f(pos[i].x, pos[i].y, pos[i].z);
			glVertex3f(pos[i].x+s*dir[i].x, pos[i].y+s*dir[i].y, pos[i].z+s*dir[i].z);
		}
		glEnd();
	}
}

QuadRenderer::QuadRenderer( Vector3f* _data, int _m, int _n )
{
	data = _data;
	m = _m;
	n = _n;
}

void QuadRenderer::Render( const Camera &camera )
{
	glDisable(GL_LIGHTING);
	glLineWidth(4);
	glBegin(GL_LINES);
	glColor3f(1,0,0);
	glVertex3f(0,0,0);
	glVertex3f(1,0,0);
	glColor3f(0,1,0);
	glVertex3f(0,0,0);
	glVertex3f(0,1,0);
	glColor3f(0,0,1);
	glVertex3f(0,0,0);
	glVertex3f(0,0,1);
	glEnd();

	Color color;
	// 	Vector3f color;
	// 	float maxvalue = 0.0f;
	// 	if (data != NULL)
	// 	{
	// 		for (int i = 0; i < n; i++)
	// 		{
	// 			if (data[i] > maxvalue)
	// 			{
	// 				maxvalue = data[i];
	// 			}
	// 		}
	// 	}

	if (true) { // draw simple gl_points (fastest)
		glDisable(GL_LIGHTING);
		glPointSize(4);
		glBegin(GL_QUADS);
		for (int i=0; i< m-1; i++) {
			for (int j = 0; j < n-1; j++)
			{
				if (mapping != NULL)
				{
					float w;
					int ind0 = i+m*j;
					color.HSVtoRGB(mapping[ind0]/green*120.0f, 1.0f, 1.0f);
					w = mapping[ind0]/green;
					glColor3f(w, w, w);
					//glColor3f(color.r, color.g, color.b);
					
					glVertex3f(data[ind0].x, data[ind0].y, data[ind0].z);
					int ind1 = i+1+m*(j);
					color.HSVtoRGB(mapping[ind1]/green*120.0f, 1.0f, 1.0f);
					w = mapping[ind1]/green;
					glColor3f(w, w, w);
					//glColor3f(color.r, color.g, color.b);
					glVertex3f(data[ind1].x, data[ind1].y, data[ind1].z);
					int ind2 = i+1+m*(j+1);
					color.HSVtoRGB(mapping[ind2]/green*120.0f, 1.0f, 1.0f);
					w = mapping[ind2]/green;
					glColor3f(w, w, w);
					//glColor3f(color.r, color.g, color.b);
					glVertex3f(data[ind2].x, data[ind2].y, data[ind2].z);
					int ind3 = i+m*(j+1);
					color.HSVtoRGB(mapping[ind3]/green*120.0f, 1.0f, 1.0f);
					w = mapping[ind3]/green;
					glColor3f(w, w, w);
					//glColor3f(color.r, color.g, color.b);
					glVertex3f(data[ind3].x, data[ind3].y, data[ind3].z);
				}
				else
				{
					glColor3f(0.0f, 1.0f, 1.0f);
					int ind0 = i+m*j;
					
					glVertex3f(data[ind0].x, data[ind0].y, data[ind0].z);
					int ind1 = i+m*(j+1);
					
					glVertex3f(data[ind1].x, data[ind1].y, data[ind1].z);
					int ind2 = i+1+m*j;
					
					glVertex3f(data[ind2].x, data[ind2].y, data[ind2].z);
					int ind3 = i+1+m*(j+1);
					
					glVertex3f(data[ind3].x, data[ind3].y, data[ind3].z);
				}

				
			}
		}
		glEnd();
	}
}

void MeshRenderer::Render( const Camera &camera )
{
	bool invert = false;

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glShadeModel(GL_SMOOTH);
	glColor3f(0.8f, 0.8f, 0.8f);
//	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_TRIANGLES);
	for (unsigned int i=0; i<mc->m_nTriangles; i++) {
		if (invert) {
			for (int j=0; j<=2; j++) {
				const POINT3D &p = mc->m_ppt3dVertices[mc->m_piTriangleIndices[3*i+j]];
				const POINT3D &n = mc->m_pvec3dNormals[mc->m_piTriangleIndices[3*i+j]];
				glNormal3f(-n[0], -n[1], -n[2]);
				glVertex3f(p[0]+p0.x, p[1]+p0.y, p[2]+p0.z);
			}
		}
		else {
			for (int j=2; j>=0; j--) {
				const POINT3D &p = mc->m_ppt3dVertices[mc->m_piTriangleIndices[3*i+j]];
				const POINT3D &n = mc->m_pvec3dNormals[mc->m_piTriangleIndices[3*i+j]];
				glNormal3f(n[0], n[1], n[2]);
				glVertex3f(p[0]+p0.x, p[1]+p0.y, p[2]+p0.z);
			}
		}
	}
	glEnd();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
