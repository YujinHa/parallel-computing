#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void MultMatGPU(float *C, float *A, float *B, int m, int p, int n);

void PrintMat(float *C, int m, int n);

void main()
{
	int m = 1024, p = 640, n = 2048;
	float *C = new float[m * n];
	float *A = new float[m * p];
	float *B = new float[p * n];
	for (int i = 0; i < m * p; ++i)
		A[i] = rand() % 3 - 1;
	for (int i = 0; i < p * n; ++i)
		B[i] = rand() % 3 - 1;

	cudaError_t cudaStatus = cudaSetDevice(0);
	float *dev_C, *dev_A, *dev_B;
	cudaStatus = cudaMalloc((void **)&dev_C, m * n * sizeof(float));
	cudaStatus = cudaMalloc((void **)&dev_A, m * p * sizeof(float));
	cudaStatus = cudaMalloc((void **)&dev_B, p * n * sizeof(float));
	cudaStatus = cudaMemcpy(dev_A, A, m * p * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_B, B, p * n * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimGrid((m - 1) / 32 + 1, (n - 1) / 32 + 1);
	dim3 dimBlock(32 * 32);

	clock_t st = clock();
	MultMatGPU << <dimGrid, dimBlock >> >(dev_C, dev_A, dev_B, m, p, n);
	cudaDeviceSynchronize();
	printf("Elapsed time: %u ms\n", clock() - st);

	cudaStatus = cudaMemcpy(dev_C, C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaStatus = cudaDeviceReset();

//	PrintMat(A, m, p);
//	PrintMat(B, p, n);
//	PrintMat(C, m, n);

	delete[] A;
	delete[] B;
	delete[] C;
	cudaFree(dev_C);
	cudaFree(dev_A);
	cudaFree(dev_B);
}

__global__ void MultMatGPU(float *C, float *A, float *B, int m, int p, int n)
{
	int i = blockIdx.y * 32 + threadIdx.y;
	int j = blockIdx.x * 32 + threadIdx.x;
	if (i < m && j < n)
	{
		float sum = 0.0;
		for (int k = 0; k < p; ++k)
			sum += A[i * p + k] * B[k * n + j];

		C[i * n + j] = sum;
	}
}

void PrintMat(float *C, int m, int n)
{
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
			printf("%.2f", C[i * n + j]);
		printf("\n");
	}
}