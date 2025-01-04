---
title: "GPU vs CPU Performance for Matrix Multiplication"
date: 2024-12-28T13:41:59+11:00
draft: false
---

Key Points in the Text
	1.	LSTMs and Their Limitations
	•	LSTMs process input sequentially, token by token, which means:
	•	They cannot leverage parallel processing, as the current state depends on the previous token’s state.
	•	This leads to slower training times, especially for long sequences.
	•	They are recursive models, meaning each token must pass through the same network repeatedly, leading to bottlenecks in efficiency.
	2.	Transformers and Parallelism
	•	Transformers, introduced in the seminal “Attention Is All You Need” paper (2017), solved this issue with:
	•	Positional embeddings: Adding positional information to represent sequence order, allowing tokens to be processed simultaneously instead of sequentially.
	•	Self-Attention Mechanism: By computing relationships between all tokens in a sequence in parallel, the model captures contextual dependencies across the entire sequence.
	•	These innovations enable the entire sequence to be processed simultaneously, allowing for parallel computation across GPUs.
	3.	Matrix Multiplication in Attention
	•	The attention mechanism relies heavily on dot products between matrices to calculate relationships between tokens (queries, keys, and values).
	•	This matrix multiplication is highly parallelizable and efficiently handled by GPUs.
	4.	Scalability of Transformers
	•	The architecture is inherently scalable because modern GPUs are optimized for matrix operations.
	•	Training large models like GPT, BERT, and modern LLMs is feasible because every step of the computation (attention, feedforward layers, etc.) is parallelized.
	5.	Compute Efficiency
	•	The ability to process sequences in parallel rather than token by token allowed researchers to train models on massive datasets in less time, leading to the explosive growth of Large Language Models (LLMs).

Why This is True
	•	LSTMs vs. Transformers:
	•	LSTMs: Sequential, slow, and hard to scale to long contexts.
	•	Transformers: Parallel, fast, and scale well with compute resources.
	•	GPU Utilization:
	•	GPUs thrive on operations like matrix multiplication, a fundamental building block of the transformer architecture (attention, positional encoding, and feedforward layers all use it).
	•	Masked Attention:
	•	In transformers (e.g., GPT models), the masked self-attention mechanism ensures causal relationships are maintained, allowing autoregressive tasks (like text generation) to also benefit from parallelism during training.

Conclusion

The text is accurate in emphasizing how transformers overcame the primary limitation of sequential models (lack of parallelism) and utilized GPU-friendly operations (matrix multiplication). This parallelism is indeed why transformers became the foundation of modern AI, enabling the training of massive LLMs like GPT-4 and beyond.

https://www.youtube.com/watch?v=h9Z4oGN89MU - How do Graphics Cards Work? Exploring GPU Architecture (Branch Education)

https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
https://en.wikipedia.org/wiki/Embarrassingly_parallel

Deep learning with Python - FRANÇOIS CHOLLET

1.3.1 Hardware
Between 1990 and 2010, off-the-shelf CPUs became faster by a factor of approximately
5,000. As a result, nowadays it’s possible to run small deep learning models on your
laptop, whereas this would have been intractable 25 years ago.
But typical deep learning models used in computer vision or speech recognition
require orders of magnitude more computational power than your laptop can deliver.
Throughout the 2000s, companies like NVIDIA and AMD invested billions of dollars
in developing fast, massively parallel chips (graphical processing units, or GPUs) to
power the graphics of increasingly photorealistic video games—cheap, single-purpose
supercomputers designed to render complex 3D scenes on your screen in real time.
This investment came to benefit the scientific community when, in 2007, NVIDIA
launched CUDA (https://developer.nvidia.com/about-cuda), a programming interface
Why deep learning? Why now?
21
for its line of GPUs. A small number of GPUs started replacing massive clusters of
CPUs in various highly parallelizable applications, beginning with physics modeling.
Deep neural networks, consisting mostly of many small matrix multiplications, are
also highly parallelizable, and around 2011 some researchers began to write CUDA
implementations of neural nets—Dan Ciresan6 and Alex Krizhevsky7 were among
the first.

Memory Bandwidth - https://www.digitalocean.com/community/tutorials/gpu-memory-bandwidth 
https://acecloud.ai/resources/blog/cuda-cores-vs-tensor-cores/


https://www.youtube.com/watch?v=pPStdjuYzSI - Fireship Cuda 

https://youtu.be/wjZofJX0v4M?si=uuc7uE_2MoTJD4wI&t=297 - 3Blue1Brown Transformer mat mul

The reason AI and NNs benefit so much from GPUs is that their operations—primarily matrix multiplications—are inherently parallel, and GPUs are purpose-built to handle massive parallelism. CPUs simply don’t have the core count or memory bandwidth to match GPUs for these types of workloads.

Vectorization: Instead of using loops, we can perform operations on entire arrays at once
GPUs have thousands of simpler cores optimized for parallel matrix math

because artificial neural networks were inspired by biological neurons.

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

