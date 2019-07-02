#ifndef MFD_RENDERER_H
#define MFD_RENDERER_H

#include <GL/glut.h>
#include <GL/GL.h>
#include "MfdGUI/Color.h"

class Camera;

class IRenderer {
public:
	virtual void Render(const Camera &camera) {};
};

#endif

