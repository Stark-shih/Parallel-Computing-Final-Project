// #include <iostream>
// #include <chrono>
// #include <vector>
// #include "solver.cuh"

// #define FIX_UPDATE_RATE 0.02
// #define ITERATION 20
// #define GRID_SIZE 128

// using namespace std;

// void init() {

// }

// void update() {
//     // Advection
//     // Diffuse setup
//     // Jacobi iteration
//     for (int i = 0; i < ITERATION; i++)
//     {
        
//     }
//     // Add external force
//     // Projection setup
//     // Jacobi iteration
//     for (int i = 0; i < ITERATION; i++)
//     {
        
//     }
//     // Projection finish
// }

// void render() {

// }

// int main() {

//     float2 u[GRID_SIZE][GRID_SIZE];
//     float2 p[GRID_SIZE][GRID_SIZE];
//     init();

//     double lastTime = chrono::steady_clock::now();
//     double lag = 0.0;
//     while (1) {
//         double current = chrono::steady_clock::now();
//         double elapsed = chrono::duration_cast<chrono::seconds>(current - lastTime).count();
//         lastTime = current;
//         lag += elapsed;
//         while (lag >= FIX_UPDATE_RATE) {
//             update();
//             lag -= FIX_UPDATE_RATE;
//         }       
//         render();
//     }

//     return 0;
// }