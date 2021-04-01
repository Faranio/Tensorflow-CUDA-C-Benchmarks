#include <stdio.h>
#include <time.h>

#define M 65536

void verifyDaxpy(double A[M], int check) {
    int maxError = 0;
    
    for (int i = 0; i < M; i++) {
        maxError += (int)abs(A[i] - check);
    }
    
    printf("Maximum Error = %d\n", maxError);
}

double daxpyHost(bool verbose) {
    double A[M], B[M], alpha = 2.0;
    
    for (int i = 0; i < M; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
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
        verifyDaxpy(A, 4);
    }

    return elapsedTime;
}

int main() {
	int count = 100;
	double averageTime = 0;

	for (int i = 0; i < count; i++)
		averageTime += daxpyHost(false);

	averageTime /= count;
	printf("[CPU - Double] (DAXPY) Average Elapsed Time = %lf ms\n", averageTime);
	return 0;
}
