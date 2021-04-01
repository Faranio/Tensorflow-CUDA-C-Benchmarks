#include <stdio.h>
#include <time.h>

#define M 512

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

    clock_t start, end;
    start = clock();

    for (int i = 0; i < M; i++) {
    	for (int j = 0; j < M; j++) {
    		A[i][j] += B[i][j];
    	}
    }

    end = clock();
    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
    elapsedTime *= 1000;
    
    if (verbose) {
        printf("Elapsed Time = %lf milliseconds\n", elapsedTime);
        verifyMatrixAdd(A, 3.0);
    }

    return elapsedTime;
}

int main() {
	int count = 100;
	float averageTime = 0;

	for (int i = 0; i < count; i++)
		averageTime += matrixAddHost(false);

	averageTime /= count;
	printf("[CPU - Float] (Matrix Add) Average Elapsed Time = %f ms\n", averageTime);
	return 0;
}
