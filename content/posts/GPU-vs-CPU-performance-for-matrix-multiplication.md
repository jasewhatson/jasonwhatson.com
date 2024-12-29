---
title: "GPU vs CPU Performance for Matrix Multiplication"
date: 2024-12-28T13:41:59+11:00
draft: false
---

https://www.youtube.com/watch?v=pPStdjuYzSI - Fireship Cuda 

<!-- <iframe src="./misc/matrix-multiply-vanilla.html " width="100%" height="500px"></iframe> -->

{{< embed-html "matrix-multiply-vanilla.html" >}}

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>

__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < P) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

void matrixMulCPU(float* A, float* B, float* C, int M, int N, int P) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < P; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * P + col];
            }
            C[row * P + col] = sum;
        }
    }
}

int main() {
    const int M = 1024; // Rows in A and C
    const int N = 1024; // Columns in A, Rows in B
    const int P = 1024; // Columns in B and C

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * P * sizeof(float);
    size_t sizeC = M * P * sizeof(float);

    float* A = new float[M * N];
    float* B = new float[N * P];
    float* C = new float[M * P]; // For GPU results
    float* C_cpu = new float[M * P]; // For CPU results

    // Initialize matrices
    for (int i = 0; i < M * N; ++i) A[i] = 1.0f;
    for (int i = 0; i < N * P; ++i) B[i] = 2.0f;

    float* dev_A, * dev_B, * dev_C;
    cudaMalloc((void**)&dev_A, sizeA);
    cudaMalloc((void**)&dev_B, sizeB);
    cudaMalloc((void**)&dev_C, sizeC);

    cudaMemcpy(dev_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((P + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // GPU computation
    cudaEventRecord(start);
    matrixMulKernel << <blocksPerGrid, threadsPerBlock >> > (dev_A, dev_B, dev_C, M, N, P);
    cudaEventRecord(stop);

    cudaMemcpy(C, dev_C, sizeC, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // CPU computation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(A, B, C_cpu, M, N, P);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_ms = cpu_stop - cpu_start;

    // Output the results
    printf("GPU Computation Time: %.3f ms\n", milliseconds);
    printf("CPU Computation Time: %.3f ms\n", cpu_ms.count());

    // Cleanup
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_cpu;

    return 0;
}
```

