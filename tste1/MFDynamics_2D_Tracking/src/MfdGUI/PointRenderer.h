#pragma once
#include "MfdGUI/IRenderer.h"
#include "MfdMath/DataStructure/Vec.h"
#include "MfdSurfaceReconstruction/CIsoSurface.h"


using namespace mfd;


class PointRenderer : public IRenderer
{
public:
	PointRenderer() { data = NULL; n = 0; }
	PointRenderer(Vector3f* _data, int _n);
	~PointRenderer(void);

	virtual void Render(const Camera &camera);

	void SetMapping(float** _mapping, float* _g) { mapping = _mapping; green = _g; }
	void SetMarker(bool** _marker) { marker = _marker; }

private:
	Vector3f* data;
	float** mapping;
	float* green;
	bool** marker;
	int n;
};

class BallRenderer : public IRenderer
{
public:
	BallRenderer() { data = NULL; n = 0; }
	BallRenderer(Vector3f* _data, int _n);
	~BallRenderer(void);

	virtual void Render(const Camera &camera);

private:
	Vector3f* data;
	int n;
};

class VectorRender : public IRenderer
{
public:
	VectorRender() {pos = NULL; n = 0; }
	VectorRender(Vector3f* _pos, Vector3f* _dir, int _n, float _s){pos = _pos; dir = _dir; n = _n; s = _s; }

	virtual void Render(const Camera &camera);

	Vector3f* pos;
	Vector3f* dir;
	int n;
	float s;
};

class QuadRenderer : public IRenderer
{
public:
	QuadRenderer() { data = NULL; n = 0; }
	QuadRenderer(Vector3f* _data, int _m, int _n);
	~QuadRenderer(void);

	virtual void Render(const Camera &camera);

	void SetMapping(float* _mapping, float _g) { mapping = _mapping; green = _g; }

private:
	Vector3f* data;
	float* mapping;
	float green;
	int m;
	int n;
};

class MeshRenderer : public IRenderer
{
public:
	MeshRenderer(){mc = NULL;}
	MeshRenderer(CIsoSurface<float>* _mc, Vector3f _p0){mc = _mc; p0 = _p0; }
	~MeshRenderer(){};

	virtual void Render(const Camera &camera);

private:
	CIsoSurface<float>* mc;

	Vector3f p0;
};