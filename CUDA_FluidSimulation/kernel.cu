
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
    int numberofblocks;
    int numberofthreads;
    /* data */
    int screenWidth;
    int screenHeight;
    int gridSizeX;
    int gridSizeY;
    int deviceId;

    float minX;
    float minY;
    float maxX;
    float maxY;
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
    viscosity = 0.005;
}

Solver::~Solver()
{
}

void Solver::reset() {
    int numberOfSMs;
    cudaGetDevice(&(this->deviceId));
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    this->numberofblocks = 16 * numberOfSMs;
    this->numberofthreads = 128;

    cudaMallocManaged(&(this->u), gridSizeY * gridSizeX * sizeof(float4));
    cudaMallocManaged(&(this->tmp), gridSizeY * gridSizeX * sizeof(float4));
    cudaMallocManaged(&(this->div), gridSizeY * gridSizeX * sizeof(float4));
    cudaMallocManaged(&(this->p), gridSizeY * gridSizeX * sizeof(float4));
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


void swap(float4*& field1, float4*& field2) {
    float4* temp = field1;
    field1 = field2;
    field2 = temp;
}
// __device__
float clampTo_0_255(float val) {
	if (val < 0.f) val = 0;
	if (val > 255.0f) val = 255;
	return val;
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
        if (a == 0 || a == gridSizeY - 1 || b == 0 || b == gridSizeX - 1) continue;
        float2 pos = make_float2(b, a);

        float distance = sqrtf((pos.x - forceOrigin.x) * (pos.x - forceOrigin.x) + (pos.y - forceOrigin.y) * (pos.y - forceOrigin.y));
        float amp = exp(-distance/10);
        w_out[a * gridSizeX + b].x = (w_in[a * gridSizeX + b].x + forceVector.x * amp);
        w_out[a * gridSizeX + b].y = (w_in[a * gridSizeX + b].y + forceVector.y * amp);
    }
}
__global__
void cuda_advect(int gridSizeX, int gridSizeY, float dt, float4* u, float4* xNew) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float rdx, rdy, oldx, oldy, dx, dy, mdx, mdy;
    int xid0, xid1, yid0, yid1;
    for (int i = index; i < gridSizeY * gridSizeX; i += stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        if (a == 0 || a == gridSizeY - 1 || b == 0 || b == gridSizeX - 1) continue;
        oldx = (b+0.5) - dt * u[a * gridSizeX + b].x * gridSizeX;
        oldy = (a+0.5) - dt * u[a * gridSizeY + b].y * gridSizeX;
        oldx = fmax(0.5f, fmin(gridSizeX-0.5f, oldx));
        oldy = fmax(0.5f, fmin(gridSizeY-0.5f, oldy));
        rdx = round(oldx);
        rdy = round(oldy);
        xid0 = (int)(rdx - 0.5);
        xid1 = xid0 + 1;
        yid0 = (int)(rdy - 0.5);
        yid1 = yid0 + 1;
        dx = oldx - (rdx - 0.5);
        mdx = (rdx + 0.5) - oldx;
        dy = oldy - (rdy - 0.5);
        mdy = (rdy + 0.5) - oldy;
        xNew[a * gridSizeX + b].x = mdx * (mdy * u[yid0 * gridSizeX + xid0].x + dy * u[yid1 * gridSizeX + xid0].x) + dx * (mdy * u[yid0 * gridSizeX + xid1].x + dy * u[yid1 * gridSizeX + xid1].x);
        xNew[a * gridSizeX + b].y = mdx * (mdy * u[yid0 * gridSizeX + xid0].y + dy * u[yid1 * gridSizeX + xid0].y) + dx * (mdy * u[yid0 * gridSizeX + xid1].y + dy * u[yid1 * gridSizeX + xid1].y);
    
    }
}
__global__
void cuda_divergence(int gridSizeX, int gridSizeY, float4* w, float4* div, float4* p) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < gridSizeY * gridSizeX; i += stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        if (a == 0 || a == gridSizeY - 1 || b == 0 || b == gridSizeX - 1) continue;

        float wL = w[a * gridSizeX + b - 1].x;
        float wR = w[a * gridSizeX + b + 1].x;
        float wT = w[(a - 1) * gridSizeX + b].y;
        float wB = w[(a + 1) * gridSizeX + b].y;
        div[a * gridSizeX + b].w = -0.5 * ((wR - wL) + (wB - wT)) / gridSizeX;
        p[a * gridSizeX + b] = make_float4(0,0,0,0);
    }
}
__global__
void cuda_jacobi(int gridSizeX, int gridSizeY,  float alpha, float beta, float4* x, float4* b_, float4* xNew) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < gridSizeY * gridSizeX; i += stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        if (a == 0 || a == gridSizeY - 1 || b == 0 || b == gridSizeX - 1) continue;

        float4 xL = x[a * gridSizeX + b - 1];
        float4 xR = x[a * gridSizeX + b + 1];
        float4 xT = x[(a - 1) * gridSizeX + b];
        float4 xB = x[(a + 1) * gridSizeX + b];
        float4 bc = b_[a * gridSizeX + b];
        xNew[a * gridSizeX + b].z = ((xL.z + xR.z + xT.z + xB.z)*alpha + bc.w) / beta;
    }
}
__global__
void cuda_subgradient(int gridSizeX, int gridSizeY, float4* p, float4* w, float4* uNew) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < gridSizeY * gridSizeX; i += stride) {
        int a = i / gridSizeX;
        int b = i - a * gridSizeX;
        if (a == 0 || a == gridSizeY - 1 || b == 0 || b == gridSizeX - 1) continue;

        float4 pL = p[a * gridSizeX + b - 1];
        float4 pR = p[a * gridSizeX + b + 1];
        float4 pT = p[(a - 1) * gridSizeX + b];
        float4 pB = p[(a + 1) * gridSizeX + b];

        uNew[a * gridSizeX + b] = w[a * gridSizeX + b];
        uNew[a * gridSizeX + b].x -= 0.5 * (pR.z - pL.z) * gridSizeX;
        uNew[a * gridSizeX + b].y -= 0.5 * (pB.z - pT.z) * gridSizeX;
    }
}
__global__ 
void cuda_print(float4* u) {
    printf("%f\n", u[200*400+200].z);
}
//adect->forceaply->applyDye->divergence->jacobiviscousdiffusion->applygradient
void Solver::update(float dt, float2 forceOrigin, float2 forceVector, Uint8* pixels) {

    // external force
    cuda_addForce<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY, forceOrigin, forceVector, u, tmp);
    swap(tmp, u);
    setBoundary<<< numberofblocks, numberofthreads >>>(u, -1.0f, gridSizeX, gridSizeY);
    // diffussion
    for (int s = 0; s < 10; s++) {
        cuda_jacobi<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY, dt*viscosity*gridSizeX*gridSizeY, 1+4*dt*viscosity*gridSizeX*gridSizeY, u, u, tmp);
        cuda_jacobi<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY, dt*viscosity*gridSizeX*gridSizeY, 1+4*dt*viscosity*gridSizeX*gridSizeY, tmp, tmp, u);
        setBoundary<<< numberofblocks, numberofthreads >>>(u, -1.0f, gridSizeX, gridSizeY);
    }
    // -------------------- projection start----------------------
    // divergence
    cuda_divergence<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY,  u, div, p);
    setBoundary<<<numberofblocks, numberofthreads >>>(div, 1.0f, gridSizeX, gridSizeY);
    // pressure
    for (int s = 0; s < 20; s++) {
        cuda_jacobi<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY, 1, 4, p, div, tmp);
        cuda_jacobi<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY, 1, 4, tmp, div, p);
        setBoundary<<< numberofblocks, numberofthreads >>>(p, 1.0f, gridSizeX, gridSizeY);
    }
    // subGradient
    cuda_subgradient<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY, p, u, tmp);
    swap(tmp, u);
    setBoundary<<< numberofblocks, numberofthreads >>>(u, -1.0f, gridSizeX, gridSizeY);
    // -------------------- projection end ----------------------
    // advect
    cuda_advect<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY, dt, u, tmp);
    swap(tmp, u);
    setBoundary<<< numberofblocks, numberofthreads >>>(u, -1.0f, gridSizeX, gridSizeY);
    // -------------------- projection start----------------------
    // divergence
    cuda_divergence<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY,  u, div, p);
    setBoundary<<<numberofblocks, numberofthreads >>>(div, 1.0f, gridSizeX, gridSizeY);
    // pressure
    for (int s = 0; s < 20; s++) {
        cuda_jacobi<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY, 1, 4, p, div, tmp);
        cuda_jacobi<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY, 1, 4, tmp, div, p);
        setBoundary<<< numberofblocks, numberofthreads >>>(p, 1.0f, gridSizeX, gridSizeY);
    }
    // subGradient
    cuda_subgradient<<< numberofblocks, numberofthreads >>>(gridSizeX, gridSizeY, p, u, tmp);
    swap(tmp, u);
    setBoundary<<< numberofblocks, numberofthreads >>>(u, -1.0f, gridSizeX, gridSizeY);
    // -------------------- projection end ----------------------

    // cuda_print<<< 1,1 >>>(p);
    //finish
    cudaDeviceSynchronize();
    cudaMemPrefetchAsync(u, gridSizeY * gridSizeX * sizeof(float4), deviceId);

    // apply color
    for (int i = 0; i < gridSizeY; i++) {
        for (int j = 0; j < gridSizeX; j++) {
            pixels[(i * gridSizeX + j) * 4] = 138;
            pixels[(i * gridSizeX + j) * 4 + 1] = 43;
            pixels[(i * gridSizeX + j) * 4 + 2] = 226;
            float amp = sqrtf(u[i*gridSizeX+ j].x * u[i * gridSizeX + j].x + u[i * gridSizeX + j].y * u[i * gridSizeX + j].y) * 200;
            pixels[(i * gridSizeX + j) * 4 + 3] = (int) clampTo_0_255(amp);
        }
    }
}

//main.cpp
int main()
{
    int W = 400, H = 400;
    RenderWindow window(VideoMode(W, H), "stable fluid");
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
    Vector2i last_pos, now_pos;
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
                forceOrigin = make_float2(0, 0);
                forceVector = make_float2(0, 0);
                break;
            case Event::MouseButtonPressed:
                click_flag = true;
                last_pos = Mouse::getPosition(window);         
                break;
            case Event::MouseMoved:
                if (click_flag) {
                    now_pos = Mouse::getPosition(window);
                    forceOrigin = make_float2(last_pos.x, last_pos.y);
                    forceVector = make_float2(now_pos.x - last_pos.x, now_pos.y - last_pos.y);
                    last_pos = now_pos;
                }
                break;
            default:
                break;
            }

        }


        float elapsed = clock.getElapsedTime().asSeconds();
        if (elapsed > timestep) {
            stableSolver.update(timestep, forceOrigin, forceVector, pixels);
            clock.restart();
        }


        texture.update(pixels);
        window.clear();
        window.draw(sprite);
        window.display();
    }

    return 0;
}
