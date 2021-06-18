
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SFML/Graphics.hpp>
#include <iostream>
#include <cmath>
#include <assert.h>
#include <iomanip>
#include <stdio.h>

using namespace sf;

//solver.h
class Solver
{
private:
    /* cuda */
    int numberofblocks = 20;
    int numberofthreads = 20;
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

    float4* u;
    float4* tmp;
    float4* div;
    float4* p;

public:
    Solver(int width, int height, int resolution);
    ~Solver();
    void reset();
    void update(float dt, float2 forceOrigin, float2 forceVector, Uint8* pixels);
    void print(float4* matrix);
};
//solver.cpp

//todo
void addForce(float2 pos, float2 forceOrigin, float2 forceVector, float4* w_in, float4* w_out, int wid) {
    int i = (int)pos.y;
    int j = (int)pos.x;
    float distance = sqrtf((pos.x - forceOrigin.x) * (pos.x - forceOrigin.x) + (pos.y - forceOrigin.y) * (pos.y - forceOrigin.y));
    float amp = exp(-distance);

    w_out[i * wid + j].x = w_in[i * wid + j].x + forceVector.x * amp;
    w_out[i * wid + j].y = w_in[i * wid + j].y + forceVector.y * amp;
}

void divergence(float2 pos, float halfrdx, float4* w, float4* div, int wid) {
    int i = (int)pos.y;
    int j = (int)pos.x;

    float wL = w[i * wid + j - 1].x;
    float wR = w[i * wid + j + 1].x;
    float wT = w[(i - 1) * wid + j].y;
    float wB = w[(i + 1) * wid + j].y;

    div[i * wid + j].w = halfrdx * ((wR - wL) + (wT - wB));
}

void subgradient(float2 pos, float halfrdx, float4* p, float4* w, float4* uNew, int wid) {
    int i = (int)pos.y;
    int j = (int)pos.x;

    float4 pL = p[i * wid + j - 1];
    float4 pR = p[i * wid + j + 1];
    float4 pT = p[(i - 1) * wid + j];
    float4 pB = p[(i + 1) * wid + j];

    uNew[i * wid + j] = w[i * wid + j];
    uNew[i * wid + j].x -= halfrdx * (pR.w - pL.w);
    uNew[i * wid + j].y -= halfrdx * (pB.w - pT.w);
}
//fin
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
    int deviceId;
    int numberOfSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    this->numberofblocks = 16 * numberOfSMs;
    this->numberofthreads = 128;

    cudaMallocManaged(&(this->u), gridSizeY * gridSizeX * sizeof(float4));
    cudaMallocManaged(&(this->tmp), gridSizeY * gridSizeX * sizeof(float4));
    cudaMallocManaged(&(this->div), gridSizeY * gridSizeX * sizeof(float4));
    cudaMallocManaged(&(this->p), gridSizeY * gridSizeX * sizeof(float4));
}
void swap(float4*& field1, float4*& field2) {
    float4* temp = field1;
    field1 = field2;
    field2 = temp;
}
__global__
void setBoundary(float4* field, float sc, int w, int h) {
    /* horizontal: the first line and the last line */
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = index; j < w; j+=stride) {
        field[j] = make_float4(sc * field[w + j].x, sc * field[w + j].y, sc * field[w + j].z, sc * field[w + j].w);
        field[(h - 1) * w + j] = make_float4(sc * field[(h - 2) * w + j].x, sc * field[(h - 2) * w + j].y, sc * field[(h - 2) * w + j].z, sc * field[(h - 2) * w + j].w);
    }
    /* vetrtical */
    for (int i = index; i < h; i+=stride) {
        field[i * w] = make_float4(sc * field[i * w + 1].x, sc * field[i * w + 1].y, sc * field[i * w + 1].z, sc * field[i * w + 1].w);
        field[i * w + w - 1] = make_float4(sc * field[i * w + w - 2].x, sc * field[i * w + w - 2].y, sc * field[i * w + w - 2].z, sc * field[i * w + w - 2].w);
    }
}
__global__
void cuda_addForce(int gridSizeX, int gridSizeY, float2 forceOrigin, float2 forceVector, float4* w_in, float4* w_out) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < gridSizeY * gridSizeX; i+=stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        float2 pos = make_float2(a + 0.5, b + 0.5);
        a = (int)pos.x;
        b = (int)pos.y;

        float distance = sqrtf((pos.x - forceOrigin.x) * (pos.x - forceOrigin.x) + (pos.y - forceOrigin.y) * (pos.y - forceOrigin.y));
        float amp = exp(-distance);
        w_out[a * gridSizeX + b].x = w_in[a * gridSizeX + b].x + forceVector.x * amp;
        w_out[a * gridSizeX + b].y = w_in[a * gridSizeX + b].y + forceVector.y * amp;
    }
}
__global__
void cuda_advect(int gridSizeX, int gridSizeY, float dt, float rpdx, float4* u, float4* x, float4* xNew) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < gridSizeY * gridSizeX; i += stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        float2 pos = make_float2(a + 0.5, b + 0.5);
        a = (int)pos.x;
        b = (int)pos.y;

        float oldx = pos.x - dt * u[a * gridSizeX + b].x * rpdx;
        float oldy = pos.y - dt * u[a * gridSizeX + b].y * rpdx;
        float w = 1 / rpdx;
        if (oldx > w - 2) oldx = w - 2;
        if (oldx < 1) oldx = 1;
        if (oldy > w - 2) oldy = w - 2;
        if (oldy < 1) oldy = 1;
        int oi = (int)oldx;
        int oj = (int)oldy;
        xNew[a * gridSizeX + b].x = (u[oi * gridSizeX + oj + 1].x + u[oi * gridSizeX + oj - 1].x + u[(oi + 1) * gridSizeX + oj].x + u[(oi - 1) * gridSizeX + oj].x) / 4;
        xNew[a * gridSizeX + b].y = (u[oi * gridSizeX + oj + 1].y + u[oi * gridSizeX + oj - 1].y + u[(oi + 1) * gridSizeX + oj].y + u[(oi - 1) * gridSizeX + oj].y) / 4;
    }
}
__global__
void cuda_jacobi(int gridSizeX, int gridSizeY,  float alpha, float rbeta, float4* x, float4* bb, float4* xNew) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < gridSizeY * gridSizeX; i += stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        if (a == 0 || a == gridSizeY - 1 || b == 0 || b == gridSizeX - 1) continue;
        float2 pos = make_float2(a + 0.5, b + 0.5);
        a = (int)pos.x;
        b = (int)pos.y;

        float4 xL = x[a * gridSizeX + b - 1];
        float4 xR = x[a * gridSizeX + b + 1];
        float4 xT = x[(a - 1) * gridSizeX + b];
        float4 xB = x[(a + 1) * gridSizeX + b];
        xNew[a * gridSizeX + b].x = (xL.x + xR.x + xT.x + xB.x + bb[a * gridSizeX + b].x * alpha) * rbeta;
        xNew[a * gridSizeX + b].y = (xL.y + xR.y + xT.y + xB.y + bb[a * gridSizeX + b].y * alpha) * rbeta;
        xNew[a * gridSizeX + b].z = (xL.z + xR.z + xT.z + xB.z + bb[a * gridSizeX + b].z * alpha) * rbeta;
        xNew[a * gridSizeX + b].w = (xL.w + xR.w + xT.w + xB.w + bb[a * gridSizeX + b].w * alpha) * rbeta;
    }
}
void Solver::update(float dt, float2 forceOrigin, float2 forceVector, Uint8* pixels) {

    // external force
    cuda_addForce<<<numberofblocks, numberofthreads>>>(gridSizeX, gridSizeY, forceOrigin, forceVector, u, tmp);
    swap(tmp, u);
    setBoundary<<<numberofblocks, numberofthreads >>>(u, -1.0f, gridSizeX, gridSizeY);

    // advect
    cuda_advect<<<numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY, dt, dx, u, u, tmp);
    swap(tmp, u);
    setBoundary << <numberofblocks, numberofthreads >> > (u, -1.0f, gridSizeX, gridSizeY);

    //// diffusion
    float alpha = dx * dx / (viscosity * dt);
    float rBeta = 1 / (4 + alpha);
    for (int s = 0; s < 20; s++) {
        //for (int i = 1; i < gridSizeY - 1; i++) {
        //    for (int j = 1; j < gridSizeX - 1; j++) {
        //        float2 pos = make_float2(i + 0.5, j + 0.5);
        //        jacobi(pos, alpha, rBeta, u, u, tmp, gridSizeX);
        //    }
        //}
        cuda_jacobi << <numberofblocks, numberofthreads >> > (gridSizeX, gridSizeY, alpha, rBeta, u, u, tmp);
        swap(tmp, u);
        setBoundary << <numberofblocks, numberofthreads >> > (u, -1.0f, gridSizeX, gridSizeY);
    }

    //// projection step: divergence + pressure
    //// divergence bug
    //for (int i = 1; i < gridSizeY - 1; i++) {
    //    for (int j = 1; j < gridSizeX - 1; j++) {
    //        float2 pos = make_float2(i + 0.5, j + 0.5);
    //        divergence(pos, 0.5 / dx, u, div, gridSizeX);
    //        p[i * gridSizeX + j] = make_float4(0, 0, 0, 0);
    //    }
    //}

    //// pressure
    //alpha = -dx * dx;
    //rBeta = 1 / 4;
    //for (int s = 0; s < 20; s++) {
    //    for (int i = 1; i < gridSizeY - 1; i++) {
    //        for (int j = 1; j < gridSizeX - 1; j++) {
    //            float2 pos = make_float2(i + 0.5, j + 0.5);
    //            jacobi(pos, alpha, rBeta, p, div, tmp, gridSizeX);
    //        }
    //    }
    //    swap << <1, 1 >> > (tmp, p);
    //    setBoundary(p, 1.0f, gridSizeX, gridSizeY);
    //}


    //// subGradient
    //for (int i = 1; i < gridSizeY - 1; i++) {
    //    for (int j = 1; j < gridSizeX - 1; j++) {
    //        float2 pos = make_float2(i + 0.5, j + 0.5);
    //        subgradient(pos, 0.5 / dx, p, u, tmp, gridSizeX);
    //    }
    //}
    //swap << <1, 1 >> > (tmp, u);
    //setBoundary(u, -1.0f, gridSizeX, gridSizeY);
    cudaDeviceSynchronize();
    // apply color
    for (int i = 0; i < gridSizeY; i++) {
        for (int j = 0; j < gridSizeX; j++) {
            pixels[(i * gridSizeX + j) * 4] = 138;
            pixels[(i * gridSizeX + j) * 4 + 1] = 43;
            pixels[(i * gridSizeX + j) * 4 + 2] = 226;
            float amp = sqrtf(u[i*gridSizeX+ j].x * u[i * gridSizeX + j].x + u[i * gridSizeX + j].y * u[i * gridSizeX + j].y) * 150;
            if (amp > 255) pixels[(i * gridSizeX + j) * 4 + 3] = 255;
            else pixels[(i * gridSizeX + j) * 4 + 3] = (int) amp;
        }
    }
}

void Solver::print(float4* matrix) {
    for (int i = 0; i < gridSizeY; i++) {
        for (int j = 0; j < gridSizeX; j++) {
            float amp = sqrtf(matrix[i*gridSizeX+j].x * matrix[i * gridSizeX + j].x + matrix[i * gridSizeX + j].y * matrix[i * gridSizeX + j].y);
            std::cout << std::fixed << std::setprecision(0) << amp;
        }
        std::cout << "\n";
    }
}
//main.cpp
int main()
{
    int W = 400, H = 400;
    RenderWindow window(VideoMode(W, H), "test");
    //window.setFramerateLimit(60);

    Uint8* pixels = new Uint8[W * H * 4];
    Texture texture;
    texture.create(W, H);
    Sprite sprite(texture);
    for (register int i = 0; i < W * H * 4; i += 4) {
        pixels[i] = 0;
        pixels[i + 1] = 0;
        pixels[i + 2] = 0;
        pixels[i + 3] = 255;
    }

    Clock clock;
    Time t;
    Vector2i new_pos, old_pos;
    float2 forceVector = make_float2(0, 0);
    float2 forceOrigin = make_float2(0, 0);
    bool click_flag = false;
    Solver stableSolver(W, H, W);
    stableSolver.reset();
    float  timestep = 0.01;
    while (window.isOpen())
    {
        Event event;
        while (window.pollEvent(event))
        {
            switch (event.type) {
            case Event::Closed:
                window.close();
                break;
            case Event::Event::MouseButtonReleased:
                click_flag = false;
                break;
            case Event::MouseButtonPressed:
                click_flag = true;
                old_pos.x = event.mouseButton.x;
                old_pos.y = event.mouseButton.y;
                new_pos.x = event.mouseButton.x;
                new_pos.y = event.mouseButton.y;          
                break;
            case Event::MouseMoved:
                if (click_flag) {
                    new_pos.x = event.mouseMove.x;
                    new_pos.y = event.mouseMove.y;
                }
                break;
            default:
                break;
            }

        }


        float elapsed = clock.getElapsedTime().asSeconds();
        if (elapsed > timestep) {
            if (click_flag) {
                forceOrigin = make_float2(old_pos.y, old_pos.x);
                forceVector = make_float2(new_pos.y - old_pos.y, new_pos.x - old_pos.x);
                old_pos = new_pos;
            }
            stableSolver.update(timestep, forceOrigin, forceVector, pixels);
            forceOrigin = make_float2(0, 0);
            forceVector = make_float2(0, 0);
            clock.restart();
        }


        texture.update(pixels);
        window.clear();
        window.draw(sprite);
        window.display();
    }

    return 0;
}
