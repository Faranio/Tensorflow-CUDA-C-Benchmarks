#include <stdio.h>
#include <time.h>

#define M 512
#define P 256
#define N 128

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
    		C[i][j] = 0;
    }

    clock_t start, end;
    start = clock();

    for (int i = 0; i < M; i++) {
    	for (int j = 0; j < N; j++) {
    		for (int k = 0; k < P; k++)
    			C[i][j] += A[i][k] * B[k][j];
    	}
    }

    end = clock();
    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
    elapsedTime *= 1000;
    
    if (verbose) {
        printf("Elapsed Time = %lf milliseconds\n", elapsedTime);
        verifyMatrixMultiplication(C, P * 1.0 * 2.0);
    }

    return elapsedTime;
}

int main() {
	int count = 100;
	float averageTime = 0;

	for (int i = 0; i < count; i++)
		averageTime += matrixMultiplicationHost(false);

	averageTime /= count;
	printf("[CPU - Double] (Matrix Multiplication) Average Elapsed Time = %f ms\n", averageTime);
	return 0;
}
