/* solver.h */
#ifndef SOLVER_H // include guard
#define SOLVER_H

typedef struct {
    float x, y;
}float2;

typedef struct {
    float x, y, z, w;
}float4;

class Solver
{
private:
    /* data */
    int screenWidth;
    int screenHeight;
    int gridSizeX;
    int gridSizeY;

    float minX;
    float minY;
    float maxX;
    float maxY;

    float dx;
    float viscosity;

    float4 **u;
    float4 **tmp;
    float4 **div;
    float4 **p;

public:
    Solver(int width, int height, int resolution);
    ~Solver();
    void reset();
    void update(float dt, float2 forceOrigin, float2 forceVector);
};

#endif /* SOLVER_H */
