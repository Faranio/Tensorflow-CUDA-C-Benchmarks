#include <stdio.h>

#define M 512
#define P 256
#define N 128
#define K 8

// Implementation of the Matrix Multiplication taken from the Lecture Notes
__global__ void globalMatrixMultiplication(double A[M][P], double B[P][N], double C[M][N]) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    for (int k = 0; k < P; k++)
        C[i][j] += A[i][k] * B[k][j];
}

void verifyMatrixMultiplication(double C[M][N], double check) {
    int maxError = 0;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            maxError += (int)abs(C[i][j] - check);
        }
    }
    
    printf("Maximum Error = %d\n", maxError);
}

float matrixMultiplicationHost(bool verbose) {
    double A[M][P], B[P][N], C[M][N];
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++)
            A[i][j] = 1.0;
    }
    
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < N; j++)
            B[i][j] = 2.0;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            C[i][j] = 0.0;
    }
    
    double (*dA)[P], (*dB)[N], (*dC)[N];
    cudaMalloc((void**)&dA, sizeof(double) * M * P);
    cudaMalloc((void**)&dB, sizeof(double) * P * N);
    cudaMalloc((void**)&dC, sizeof(double) * M * N);
    cudaMemcpy(dA, A, sizeof(double) * M * P, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(double) * P * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(double) * M * N, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadPerBlocks(K, K);
    dim3 numBlocks(M/K, N/K);
    
    cudaEventRecord(start);
    
    globalMatrixMultiplication<<<numBlocks, threadPerBlocks>>>(dA, dB, dC);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(C, dC, sizeof(double) * M * N, cudaMemcpyDeviceToHost);
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
	printf("[GPU - Double] (Matrix Multiplication 2D - Global) Average Elapsed Time = %f ms\n", averageTime);
	return 0;
}
