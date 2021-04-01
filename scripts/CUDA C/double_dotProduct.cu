#include <stdio.h>

#define M 65536
#define K 64

// Implementation of the Dot Product taken from the Lecture Notes
__global__ void dotProduct(double A[M], double B[M], float *answer) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float temp[K];
    temp[threadIdx.x] = A[i] * B[i];
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float dot_result = 0;
        
        for (int j = 0; j < K; j++)
            dot_result += temp[j];

        atomicAdd(answer, dot_result);
    }
}

void verifyDotProduct(double answer, double check) {
    int maxError = (int)abs(answer - check);
    printf("Maximum Error = %d\n", maxError);
}

float dotProductHost(bool verbose) {
    double A[M], B[M];
    float answer = 0;
    
    for (int i = 0; i < M; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }
    
    double *dA, *dB;
    float *danswer;
    cudaMalloc((void**)&dA, sizeof(double) * M);
    cudaMalloc((void**)&dB, sizeof(double) * M);
    cudaMalloc((void**)&danswer, sizeof(float));
    cudaMemcpy(dA, A, sizeof(double) * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(double) * M, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    dotProduct<<<M/K, K>>>(dA, dB, danswer);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(&answer, danswer, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(danswer);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    if (verbose) {
        printf("Elapsed Time = %f milliseconds\n", elapsedTime);
        verifyDotProduct(answer, M*2);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;
}

int main() {
	int count = 100;
	float averageTime = 0;

	for (int i = 0; i < count; i++)
		averageTime += dotProductHost(false);

	averageTime /= count;
	printf("[GPU - Double] (Dot Product) Average Elapsed Time = %f ms\n", averageTime);
	return 0;
}
