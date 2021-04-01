#include <stdio.h>

#define M 65536
#define K 128

// Implementation of SAXPY taken from Lecture Notes
__global__ void saxpy(float A[M], float B[M]) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    A[i] = 2.0 * A[i] + B[i];
}

void verifySaxpy(float A[M], int check) {
    int maxError = 0;
    
    for (int i = 0; i < M; i++) {
        maxError += abs(A[i] - check);
    }
    
    printf("Maximum Error = %d\n", maxError);
}

float saxpyHost(bool verbose) {
    float A[M], B[M];
    
    for (int i = 0; i < M; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    
    float *dA, *dB;
    cudaMalloc((void**)&dA, sizeof(float) * M);
    cudaMalloc((void**)&dB, sizeof(float) * M);
    cudaMemcpy(dA, A, sizeof(float) * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * M, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    saxpy<<<M/K, K>>>(dA, dB);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(A, dA, sizeof(float) * M, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    if (verbose) {
        printf("Elapsed Time = %f milliseconds\n", elapsedTime);
        verifySaxpy(A, 4);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;
}

int main() {
	int count = 100;
	float averageTime = 0;

	for (int i = 0; i < count; i++)
		averageTime += saxpyHost(false);

	averageTime /= count;
	printf("[GPU - Float] (SAXPY) Average Elapsed Time = %f ms\n", averageTime);
	return 0;
}
