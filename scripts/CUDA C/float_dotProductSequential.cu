#include <stdio.h>
#include <time.h>

#define M 65536

void verifyDotProduct(float answer, float check) {
    int maxError = (int)abs(answer - check);
    printf("Maximum Error = %d\n", maxError);
}

float dotProductHost(bool verbose) {
    float A[M], B[M], answer = 0;
    
    for (int i = 0; i < M; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    clock_t start, end;
    start = clock();

    for (int i = 0; i < M; i++)
    	answer += A[i] * B[i];

    end = clock();
    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
    elapsedTime *= 1000;
    
    if (verbose) {
        printf("Elapsed Time = %lf milliseconds\n", elapsedTime);
        verifyDotProduct(answer, M*2);
    }

    return elapsedTime;
}

int main() {
	int count = 100;
	float averageTime = 0;

	for (int i = 0; i < count; i++)
		averageTime += dotProductHost(false);

	averageTime /= count;
	printf("[CPU - Float] (Dot Product) Average Elapsed Time = %f ms\n", averageTime);
	return 0;
}
