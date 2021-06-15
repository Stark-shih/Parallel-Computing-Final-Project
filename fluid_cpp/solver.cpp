#include <cmath>
#include <assert.h>
#include "solver.h"

float4 make_float4(float x, float y, float z, float w) {
    float4 tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.z = z;
    tmp.w = w;
    return tmp;
}

float2 make_float2(float x, float y) {
    float2 tmp;
    tmp.x = x;
    tmp.y = y;
    return tmp;
}

void swap(float4 **field1, float4 **field2) {
    float4 **tmp = field1;
    field1 = field2;
    field2 = tmp;
}

void setBoundary(float4 **field, float sc, int w, int h) {
    /* horizontal: the first line and the last line */
    for (int j=1; j<w-1; j++) {
        field[0][j] = make_float4(sc*field[1][j].x, sc*field[1][j].y, sc*field[1][j].z, sc*field[1][j].w);
        field[h-1][j] = make_float4(sc*field[h-2][j].x, sc*field[h-2][j].y, sc*field[h-2][j].z, sc*field[h-2][j].w);
    }
    /* vetrtical */
    for (int i=0; i<h; i++) {
        field[i][0] = make_float4(sc*field[i][1].x, sc*field[i][1].y, sc*field[i][1].z, sc*field[i][1].w);
        field[i][w-1] = make_float4(sc*field[i][w-2].x, sc*field[i][w-2].y, sc*field[i][w-2].z, sc*field[i][w-2].w);
    }

    // field[0][0] = field[0][1];
    // field[0][w-1] = field[0][w-2];
    // field[h-1][0] = field[h-2][0];
    // field[h-1][w-1] = field[h-2][w-1];
}

void advect(float2 pos, float dt, float rpdx, float4 **u, float4 **x ,float4 **xNew) {
    int i = (int)pos.y;
    int j = (int)pos.x;

    float oldx = pos.x - dt * u[i][j].x * rpdx;
    float oldy = pos.y - dt * u[i][j].y * rpdx;

    float w = 1 / rpdx;
    if (oldx > w-2) oldx = w-2;
    if (oldx < 1) oldx = 1;
    if (oldy > w-2) oldy = w-2;
    if (oldy < 1) oldy = 1;

    float rdx = round(oldx);
    float rdy = round(oldy);
    float a = oldy - (rdy - 0.5);
    float b = oldx - (rdx - 0.5);
    float a_ = 1.0;
    float b_ = 1.0;

    int i1 = (int) (rdy + 0.5);
    int j1 = (int) (rdx + 0.5);
    int i0 = i1 - 1;
    int j0 = j1 - 1;

    xNew[i][j].x = (a_-a)*(b_-b)*x[i0][j0].x + (a_-a)*b*x[i0][j1].x + a*(b_-b)*x[i1][j0].x + a*b*x[i1][j1].x;
    xNew[i][j].y = (a_-a)*(b_-b)*x[i0][j0].y + (a_-a)*b*x[i0][j1].y + a*(b_-b)*x[i1][j0].y + a*b*x[i1][j1].y;

}

void jacobi(float2 pos, float alpha, float rbeta, float4 **x, float4 **b, float4 ** xNew) {
    int i = (int)pos.y;
    int j = (int)pos.x;

    float4 xL = x[i][j-1];
    float4 xR = x[i][j+1];
    float4 xT = x[i-1][j];
    float4 xB = x[i+1][j];

    xNew[i][j].x = (xL.x + xR.x + xT.x + xB.x + b[i][j].x*alpha) * rbeta;
    xNew[i][j].y = (xL.y + xR.y + xT.y + xB.y + b[i][j].y*alpha) * rbeta;
    xNew[i][j].z = (xL.z + xR.z + xT.z + xB.z + b[i][j].z*alpha) * rbeta;
    xNew[i][j].w = (xL.w + xR.w + xT.w + xB.w + b[i][j].w*alpha) * rbeta;
}

void addForce(float2 pos, float2 forceOrigin, float2 forceVector, float4 **w_in, float4 **w_out) {
    int i = (int)pos.y;
    int j = (int)pos.x;
    float distance = sqrtf( (pos.x-forceOrigin.x)*(pos.x-forceOrigin.x) + (pos.y-forceOrigin.y)*(pos.y-forceOrigin.y) );
    float amp = exp(-distance);

    w_out[i][j].x = w_in[i][j].x + forceVector.x * amp;
    w_out[i][j].y = w_in[i][j].y + forceVector.y * amp;
}

void divergence(float2 pos, float halfrdx, float4 **w, float4 **div) {
    int i = (int)pos.y;
    int j = (int)pos.x;

    float4 wL = w[i][j-1];
    float4 wR = w[i][j+1];
    float4 wT = w[i-1][j];
    float4 wB = w[i+1][j];

    div[i][j].w = halfrdx * ((wR.x - wL.x) + (wT.y - wB.y));
}

void subgradient(float2 pos, float halfrdx, float4 **p, float4 **w, float4 **uNew) {
    int i = (int)pos.y;
    int j = (int)pos.x;

    float4 pL = p[i][j-1];
    float4 pR = p[i][j+1];
    float4 pT = p[i-1][j];
    float4 pB = p[i+1][j];

    uNew[i][j] = w[i][j];
    uNew[i][j].x -= halfrdx * (pR.w - pL.w);
    uNew[i][j].y -= halfrdx * (pB.w - pT.w);
}

Solver::Solver(int screenWidth, int screenHeight, int resolution)
{
    assert((resolution * screenHeight) % screenWidth == 0);
    this->screenWidth = screenWidth;
    this->screenHeight = screenHeight;
    gridSizeX = resolution;
    gridSizeY = resolution * screenHeight / screenWidth;
    minX = 1.0f;
    minY = 1.0f;
    maxX = gridSizeX - 1.0f;
    maxY = gridSizeY - 1.0f;
    dx = 1.0f / gridSizeY;
    viscosity = 1e-6f;
}

Solver::~Solver()
{
}

void Solver::reset() {
    this->u = (float4**) malloc(gridSizeY * sizeof(float4*));
    for (int i=0; i<gridSizeY; i++) {
        u[i] = (float4*) malloc(gridSizeX * sizeof(float4));
    }

    this->tmp = (float4**) malloc(gridSizeY * sizeof(float4*));
    for (int i=0; i<gridSizeY; i++) {
        tmp[i] = (float4*) malloc(gridSizeX * sizeof(float4));
    }

    this->div = (float4**) malloc(gridSizeY * sizeof(float4*));
    for (int i=0; i<gridSizeY; i++) {
        div[i] = (float4*) malloc(gridSizeX * sizeof(float4));
    }

    this->p = (float4**) malloc(gridSizeY * sizeof(float4*));
    for (int i=0; i<gridSizeY; i++) {
        p[i] = (float4*) calloc(0, gridSizeX * sizeof(float4));
    }
}

void Solver::update(float dt, float2 forceOrigin, float2 forceVector) {

    // advect
    for (int i=1; i<gridSizeY-1; i++) {
        for (int j=1; j<gridSizeX-1; j++) {
            float2 pos = make_float2(i+0.5, j+0.5);
            advect(pos, dt, 1/dx, u, u, tmp);
        }
    }
    swap(tmp, u);
    setBoundary(u, -1.0f, gridSizeX, gridSizeY);

    // diffusion
    float alpha = dx * dx / (viscosity * dt);
    float rBeta = 1 / (4 + alpha);
    for (int s=0; s<20; s++) {
        for (int i=1; i<gridSizeY-1; i++) {
            for (int j=1; j<gridSizeX-1; j++) {
                float2 pos = make_float2(i+0.5, j+0.5);
                jacobi(pos, alpha, rBeta, u, u, tmp);
            }
        }
        swap(tmp, u);
        setBoundary(u, -1.0f, gridSizeX, gridSizeY);
    }

    // external force
    for (int i=1; i<gridSizeY-1; i++) {
        for (int j=1; j<gridSizeX-1; j++) {
            float2 pos = make_float2(i+0.5, j+0.5);
            addForce(pos, forceOrigin, forceVector, u, tmp);
        }
    }
    swap(tmp, u);
    setBoundary(u, -1.0f, gridSizeX, gridSizeY);

    // projection step: divergence + pressure
    // divergence
    for (int i=1; i<gridSizeY-1; i++) {
        for (int j=1; j<gridSizeX-1; j++) {
            float2 pos = make_float2(i+0.5, j+0.5);
            divergence(pos, 0.5/dx, u, div);
            p[i][j] = make_float4(0,0,0,0);
        }
    }

    // pressure
    alpha = -dx * dx;
    rBeta = 1 / 4;
    for (int s=0; s<20; s++) {
        for (int i=1; i<gridSizeY-1; i++) {
            for (int j=1; j<gridSizeX-1; j++) {
                float2 pos = make_float2(i+0.5, j+0.5);
                jacobi(pos, alpha, rBeta, p, div, tmp);
            }
        }
        swap(tmp, p);
        setBoundary(p, 1.0f, gridSizeX, gridSizeY);
    }


    // subGradient
    for (int i=1; i<gridSizeY-1; i++) {
        for (int j=1; j<gridSizeX-1; j++) {
            float2 pos = make_float2(i+0.5, j+0.5);
            subgradient(pos, 0.5/dx, p, u, tmp);
        }
    }
    swap(tmp, u);
    setBoundary(u, -1.0f, gridSizeX, gridSizeY);

    // TODO: apply color
}
