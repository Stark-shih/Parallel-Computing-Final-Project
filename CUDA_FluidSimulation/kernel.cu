
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

    cudaMemset(this->u, 0, 1000 * gridSizeY * gridSizeX * sizeof(float4));
    cudaMemset(this->tmp, 0, 1000 * gridSizeY * gridSizeX * sizeof(float4));
    cudaMemset(this->div, 0, 1000 * gridSizeY * gridSizeX * sizeof(float4));
    cudaMemset(this->p, 0, 1000 * gridSizeY * gridSizeX * sizeof(float4));
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
//這個點的速度應該要到下一個位置去
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
__global__
void cuda_divergence(int gridSizeX, int gridSizeY, float halfrdx, float4* w, float4* div) {//gridSizeX, gridSizeY, 0.5 / dx, u, div
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < gridSizeY * gridSizeX; i += stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        if (a == 0 || a == gridSizeY - 1 || b == 0 || b == gridSizeX - 1) continue;
        float2 pos = make_float2(a + 0.5, b + 0.5);
        a = (int)pos.x;
        b = (int)pos.y;

        float wL = w[a * gridSizeX + b - 1].x;
        float wR = w[a * gridSizeX + b + 1].x;
        float wT = w[(a - 1) * gridSizeX + b].y;
        float wB = w[(a + 1) * gridSizeX + b].y;
        div[a * gridSizeX + b].w = halfrdx * ((wR - wL) + (wT - wB));
    }
}
__global__
void cuda_subgradient(int gridSizeX, int gridSizeY, float halfrdx, float4* p, float4* w, float4* uNew) {//gridSizeX, gridSizeY, 0.5 / dx, p, u, tmp
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < gridSizeY * gridSizeX; i += stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        if (a == 0 || a == gridSizeY - 1 || b == 0 || b == gridSizeX - 1) continue;
        float2 pos = make_float2(a + 0.5, b + 0.5);
        a = (int)pos.x;
        b = (int)pos.y;

        float4 pL = p[a * gridSizeX + b - 1];
        float4 pR = p[a * gridSizeX + b + 1];
        float4 pT = p[(a - 1) * gridSizeX + b];
        float4 pB = p[(a + 1) * gridSizeX + b];

        uNew[a * gridSizeX + b] = w[a * gridSizeX + b];
        uNew[a * gridSizeX + b].x -= halfrdx * (pR.w - pL.w);
        uNew[a * gridSizeX + b].y -= halfrdx * (pB.w - pT.w);
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
    // diffusion
    float alpha = dx * dx / (viscosity * dt);
    float rBeta = 1 / (4 + alpha);
    for (int s = 0; s < 20; s++) {
        cuda_jacobi << <numberofblocks, numberofthreads >> > (gridSizeX, gridSizeY, alpha, rBeta, u, u, tmp);
        swap(tmp, u);
        setBoundary << <numberofblocks, numberofthreads >> > (u, -1.0f, gridSizeX, gridSizeY);
    }
    /*------------------------------------------------------------------------------------------------------------*/
    // projection step: divergence + pressure
    //divergence
    cuda_divergence << <numberofblocks, numberofthreads >> > (gridSizeX, gridSizeY, 0.5 / dx, u, div); // u -> div
    // pressure
    alpha = -dx * dx;
    rBeta = 1 / 4;
    for (int s = 0; s < 20; s++) {
        cuda_jacobi << <numberofblocks, numberofthreads >> > (gridSizeX, gridSizeY, alpha, rBeta, p, div, tmp); //div -> tmp
        swap(tmp, p);
        setBoundary << <numberofblocks, numberofthreads >> > (p, 1.0f, gridSizeX, gridSizeY);
    }
    // subGradient
    cuda_subgradient << <numberofblocks, numberofthreads >> > (gridSizeX, gridSizeY, 0.5 / dx, p, u, tmp);
    swap(tmp, u);
    setBoundary << <numberofblocks, numberofthreads >> > (u, -1.0f, gridSizeX, gridSizeY);
    //finish
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
    int W = 300, H = 300;
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
    float  timestep = 0.05;
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
