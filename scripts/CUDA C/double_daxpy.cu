#include <stdio.h>

#define M 65536
#define K 64

// Implementation of DAXPY taken from the Lecture Notes
__global__ void daxpy(double *A, double *B) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    A[i] = 2.0 * A[i] + B[i];
}

void verifyDaxpy(double A[M], int check) {
    int maxError = 0;
    
    for (int i = 0; i < M; i++) {
        maxError += (int)abs(A[i] - check);
    }
    
    printf("Maximum Error = %d\n", maxError);
}

float daxpyHost(bool verbose) {
    double A[M], B[M];
    
    for (int i = 0; i < M; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }
    
    double *dA, *dB;
    cudaMalloc((void**)&dA, sizeof(double) * M);
    cudaMalloc((void**)&dB, sizeof(double) * M);
    cudaMemcpy(dA, A, sizeof(double) * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(double) * M, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    daxpy<<<M/K, K>>>(dA, dB);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(A, dA, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    if (verbose) {
        printf("Elapsed Time = %f milliseconds\n", elapsedTime);
        verifyDaxpy(A, 4);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;
}

int main() {
	int count = 100;
	float averageTime = 0;

	for (int i = 0; i < count; i++)
		averageTime += daxpyHost(false);

	averageTime /= count;
	printf("[GPU - Double] (DAXPY) Average Elapsed Time = %f ms\n", averageTime);
	return 0;
}
