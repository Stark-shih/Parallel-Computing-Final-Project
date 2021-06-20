
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
    float viscosity;

    float4* u;
    float4* tmp;
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

    cudaMemset(this->u, 0, 1000 * gridSizeY * gridSizeX * sizeof(float4));
    cudaMemset(this->tmp, 0, 1000 * gridSizeY * gridSizeX * sizeof(float4));
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
    for (int j = index; j < w; j += stride) {
        field[j] = make_float4(sc * field[w + j].x, sc * field[w + j].y, sc * field[w + j].z, sc * field[w + j].w);
        field[(h - 1) * w + j] = make_float4(sc * field[(h - 2) * w + j].x, sc * field[(h - 2) * w + j].y, sc * field[(h - 2) * w + j].z, sc * field[(h - 2) * w + j].w);
    }
    /* vetrtical */
    for (int i = index; i < h; i += stride) {
        field[i * w] = make_float4(sc * field[i * w + 1].x, sc * field[i * w + 1].y, sc * field[i * w + 1].z, sc * field[i * w + 1].w);
        field[i * w + w - 1] = make_float4(sc * field[i * w + w - 2].x, sc * field[i * w + w - 2].y, sc * field[i * w + w - 2].z, sc * field[i * w + w - 2].w);
    }
}
__global__
void cuda_addForce(int gridSizeX, int gridSizeY, float2 forceOrigin, float2 forceVector, float4* w_in, float4* w_out) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < gridSizeY * gridSizeX; i += stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        float2 pos = make_float2(a, b);

        float distance = sqrtf((pos.x - forceOrigin.x) * (pos.x - forceOrigin.x) + (pos.y - forceOrigin.y) * (pos.y - forceOrigin.y));
        float amp = exp(-distance);
        w_out[a * gridSizeX + b].x = w_in[a * gridSizeX + b].x + forceVector.x * amp;
        w_out[a * gridSizeX + b].y = w_in[a * gridSizeX + b].y + forceVector.y * amp;
    }
}
__global__
void cuda_advect(int gridSizeX, int gridSizeY, float dt, float4* u, float4* xNew) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float oldx, oldy, dx, dy, mdx, mdy;
    int xid0, xid1, yid0, yid1;
    for (int i = index; i < gridSizeY * gridSizeX; i += stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        oldx = a - dt * u[a * gridSizeX + b].x;
        oldy = b - dt * u[a * gridSizeY + b].y;
        oldx = fmax(0.5f, fmin(gridSizeX + 0.5f, oldx));
        oldy = fmax(0.5f, fmin(gridSizeY + 0.5f, oldy));
        xid0 = (int)oldx;
        xid1 = xid0 + 1;
        yid0 = (int)oldy;
        yid1 = yid0 + 1;
        dx = oldx - xid0;
        mdx = 1 - xid0;
        dy = oldy - yid0;
        mdy = 1 - dy;
        xNew[a * gridSizeX + b].x = mdx * (mdy * u[xid0 * gridSizeX + yid0].x + dy * u[xid0 * gridSizeX + yid1].x) + dx * (mdy * u[xid1 * gridSizeX + yid0].x + dy * u[xid1 * gridSizeX + yid1].x);
        xNew[a * gridSizeX + b].y = mdx * (mdy * u[xid0 * gridSizeX + yid0].y + dy * u[xid0 * gridSizeX + yid1].y) + dx * (mdy * u[xid1 * gridSizeX + yid0].y + dy * u[xid1 * gridSizeX + yid1].y);
        xNew[a * gridSizeX + b].z = u[xid0 * gridSizeX + yid0].z;
        xNew[a * gridSizeX + b].w = 1.0;
    }
}
__global__
void cuda_divergence(int gridSizeX, int gridSizeY, float4* w, float4* div) {//gridSizeX, gridSizeY, 0.5 / dx, u, div
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
        div[a * gridSizeX + b].x = w[a * gridSizeX + b].x;
        div[a * gridSizeX + b].y = w[a * gridSizeX + b].y;
        div[a * gridSizeX + b].z = w[a * gridSizeX + b].z;
        div[a * gridSizeX + b].w = 0.5 * ((wR - wL) + (wT - wB));
    }
}
__global__
void cuda_jacobi(int gridSizeX, int gridSizeY, float alpha, float rbeta, float4* x, float4* xNew) {
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
        xNew[a * gridSizeX + b].x = x[a * gridSizeX + b].x;
        xNew[a * gridSizeX + b].y = x[a * gridSizeX + b].y;
        xNew[a * gridSizeX + b].z = (xL.z + xR.z + xT.z + xB.z + x[a * gridSizeX + b].w * alpha) * rbeta;;
        xNew[a * gridSizeX + b].w = x[a * gridSizeX + b].w;
    }
}
__global__
void cuda_subgradient(int gridSizeX, int gridSizeY, float4* u, float4* uNew) {//gridSizeX, gridSizeY, 0.5 / dx, p, u, tmp
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < gridSizeY * gridSizeX; i += stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        if (a == 0 || a == gridSizeY - 1 || b == 0 || b == gridSizeX - 1) continue;
        float2 pos = make_float2(a + 0.5, b + 0.5);
        a = (int)pos.x;
        b = (int)pos.y;

        float4 pL = u[a * gridSizeX + b - 1];
        float4 pR = u[a * gridSizeX + b + 1];
        float4 pT = u[(a - 1) * gridSizeX + b];
        float4 pB = u[(a + 1) * gridSizeX + b];

        uNew[a * gridSizeX + b] = u[a * gridSizeX + b];
        uNew[a * gridSizeX + b].x -= 0.5 * (pR.z - pL.z);
        uNew[a * gridSizeX + b].y -= 0.5 * (pB.z - pT.z);
    }
}
//adect->forceaply->applyDye->divergence->jacobiviscousdiffusion->applygradient
void Solver::update(float dt, float2 forceOrigin, float2 forceVector, Uint8* pixels) {

    // external force
    cuda_addForce << <numberofblocks, numberofthreads >> > (gridSizeX, gridSizeY, forceOrigin, forceVector, u, tmp);
    swap(tmp, u);
    setBoundary << <numberofblocks, numberofthreads >> > (u, -1.0f, gridSizeX, gridSizeY);
    // advect
    cuda_advect << <numberofblocks, numberofthreads >> > (gridSizeX, gridSizeY, dt, u, tmp);
    swap(tmp, u);
    setBoundary << <numberofblocks, numberofthreads >> > (u, -1.0f, gridSizeX, gridSizeY);
    //divergence
    cuda_divergence << <numberofblocks, numberofthreads >> > (gridSizeX, gridSizeY, u, tmp); // u -> div
    swap(tmp, u);
    setBoundary << <numberofblocks, numberofthreads >> > (u, -1.0f, gridSizeX, gridSizeY);
    // diffusion
    float alpha = -1;
    float rBeta = 1 / 4;
    for (int s = 0; s < 20; s++) {
        cuda_jacobi << <numberofblocks, numberofthreads >> > (gridSizeX, gridSizeY, alpha, rBeta, u, tmp);
        swap(tmp, u);
        setBoundary << <numberofblocks, numberofthreads >> > (u, 1.0f, gridSizeX, gridSizeY);
    }
    // subGradient
    cuda_subgradient << <numberofblocks, numberofthreads >> > (gridSizeX, gridSizeY, u, tmp);
    swap(tmp, u);
    setBoundary << <numberofblocks, numberofthreads >> > (u, -1.0f, gridSizeX, gridSizeY);
    //GPU finish
    cudaDeviceSynchronize();
    // apply color
    for (int i = 0; i < gridSizeY; i++) {
        for (int j = 0; j < gridSizeX; j++) {
            pixels[(i * gridSizeX + j) * 4] = 138;
            pixels[(i * gridSizeX + j) * 4 + 1] = 43;
            pixels[(i * gridSizeX + j) * 4 + 2] = 226;
            float amp = sqrtf(u[i * gridSizeX + j].x * u[i * gridSizeX + j].x + u[i * gridSizeX + j].y * u[i * gridSizeX + j].y) * 150;
            if (amp > 255) pixels[(i * gridSizeX + j) * 4 + 3] = 255;
            else pixels[(i * gridSizeX + j) * 4 + 3] = (int) amp;
        }
    }
}

void Solver::print(float4* matrix) {
    for (int i = 0; i < gridSizeY; i++) {
        for (int j = 0; j < gridSizeX; j++) {
            float amp = sqrtf(matrix[i * gridSizeX + j].x * matrix[i * gridSizeX + j].x + matrix[i * gridSizeX + j].y * matrix[i * gridSizeX + j].y);
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
    float  timestep = 0.1;
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
