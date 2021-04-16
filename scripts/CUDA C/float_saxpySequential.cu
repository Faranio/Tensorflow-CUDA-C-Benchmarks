#include <stdio.h>
#include <time.h>

#define M 65536

void verifySaxpy(float A[M], int check) {
    int maxError = 0;
    
    for (int i = 0; i < M; i++) {
        maxError += (int)abs(A[i] - check);
    }
    
    printf("Maximum Error = %d\n", maxError);
}

float saxpyHost(bool verbose) {
    float A[M], B[M], alpha = 2.0f;
    
    for (int i = 0; i < M; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    clock_t start, end;
    start = clock();

    for (int i = 0; i < M; i++) {
    	A[i] = alpha * A[i] + B[i];
    }

    end = clock();
    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
    elapsedTime *= 1000;
    
    if (verbose) {
        printf("Elapsed Time = %lf milliseconds\n", elapsedTime);
        verifySaxpy(A, 4);
    }

    return elapsedTime;
}

int main() {
	int count = 100;
	float averageTime = 0;

	for (int i = 0; i < count; i++)
		averageTime += saxpyHost(false);

	averageTime /= count;
	printf("[CPU - Float] (SAXPY) Average Elapsed Time = %f ms\n", averageTime);
	return 0;
}
