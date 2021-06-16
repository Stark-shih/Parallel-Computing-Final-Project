/* solver.h */
#ifndef SOLVER_H // include guard
#define SOLVER_H

#include <SFML/Graphics.hpp>
typedef struct {
    float x, y;
}float2;

typedef struct {
    float x, y, z, w;
}float4;

float4 make_float4(float x, float y, float z, float w);
float2 make_float2(float x, float y);

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
    void update(float dt, float2 forceOrigin, float2 forceVector, sf::Uint8 *pixels);
    void print(float4 **matrix);
    void swap(float4 **field1, float4 **field2);
};

#endif /* SOLVER_H */
