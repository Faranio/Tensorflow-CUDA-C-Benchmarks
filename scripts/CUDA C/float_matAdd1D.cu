#include <stdio.h>

#define M 512
#define K 128

// Implementation of the Matrix Add taken from the Lecture Notes
__global__ void matrixAdd1D(float A[M][M], float B[M][M]) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i = (int)(idx / M);
    int j = idx % M;
    A[i][j] += B[i][j];
}

void verifyMatrixAdd(float A[M][M], int check) {
    int maxError = 0;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            maxError += (int)abs(A[i][j] - check);
        }
    }
    
    printf("Maximum Error = %d\n", maxError);
}

float matrixAddHost(bool verbose) {
    float A[M][M], B[M][M];
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
        }
    }
    
    float (*dA)[M], (*dB)[M];
    cudaMalloc((void**)&dA, sizeof(float) * M * M);
    cudaMalloc((void**)&dB, sizeof(float) * M * M);
    cudaMemcpy(dA, A, sizeof(float) * M * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * M * M, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixAdd1D<<<M*M/K, K>>>(dA, dB);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(A, dA, sizeof(float) * M * M, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    if (verbose) {
        printf("Elapsed Time = %f milliseconds\n", elapsedTime);
        verifyMatrixAdd(A, 3.0);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;
}

int main() {
	int count = 100;
	float averageTime = 0;

	for (int i = 0; i < count; i++)
		averageTime += matrixAddHost(false);

	averageTime /= count;
	printf("[GPU - Float] (Matrix Add 1D) Average Elapsed Time = %f ms\n", averageTime);
	return 0;
}
