#include <stdio.h>

#define M 512
#define P 256
#define N 128
#define K 8

// Implementation of the Matrix Multiplication taken from the Lecture Notes
__global__ void globalMatrixMultiplication(float A[M][P], float B[P][N], float C[M][N]) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    for (int k = 0; k < P; k++)
        C[i][j] += A[i][k] * B[k][j];
}

void verifyMatrixMultiplication(float C[M][N], float check) {
    int maxError = 0;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            maxError += (int)abs(C[i][j] - check);
        }
    }
    
    printf("Maximum Error = %d\n", maxError);
}

float matrixMultiplicationHost(bool verbose) {
    float A[M][P], B[P][N], C[M][N];
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++)
            A[i][j] = 1.0f;
    }
    
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < N; j++)
            B[i][j] = 2.0f;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            C[i][j] = 0.0f;
    }
    
    float (*dA)[P], (*dB)[N], (*dC)[N];
    cudaMalloc((void**)&dA, sizeof(float) * M * P);
    cudaMalloc((void**)&dB, sizeof(float) * P * N);
    cudaMalloc((void**)&dC, sizeof(float) * M * N);
    cudaMemcpy(dA, A, sizeof(float) * M * P, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * P * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadPerBlocks(K, K);
    dim3 numBlocks(M/K, N/K);
    
    cudaEventRecord(start);
    
    globalMatrixMultiplication<<<numBlocks, threadPerBlocks>>>(dA, dB, dC);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    if (verbose) {
        printf("Elapsed Time = %f milliseconds\n", elapsedTime);
        verifyMatrixMultiplication(C, P * 1.0 * 2.0);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;
}

int main() {
	int count = 100;
	float averageTime = 0;

	for (int i = 0; i < count; i++)
		averageTime += matrixMultiplicationHost(false);

	averageTime /= count;
	printf("[GPU - Float] (Matrix Multiplication 2D - Global) Average Elapsed Time = %f ms\n", averageTime);
	return 0;
}
