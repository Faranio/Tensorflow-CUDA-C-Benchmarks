import tensorflow as tf
import time

M = 512
P = 256
N = 128

dtype = tf.float64
device = "/cpu:0"


def verify_mm(C, check):
	maxError = 0
	
	for i in range(M):
		for j in range(N):
			maxError += abs(C[i][j] - check)
	
	print("Maximum Error = {}".format(maxError))


# My own implementation of Matrix Multiplication
def matrix_multiplication(verbose=False, device="/gpu:0"):
	deviceType = device.upper()[1:4]
	
	A = tf.ones([M, P], dtype=dtype)
	B = tf.ones([P, N], dtype=dtype) * 2
	
	start = time.perf_counter()
	
	with tf.device(device):
		A = tf.matmul(A, B)
		
	end = time.perf_counter()
	
	elapsed_time = end - start
	elapsed_time *= 1000
	
	if verbose:
		print("[{}] (Matrix Multiplication) Elapsed Time = {} ms".format(deviceType, elapsed_time))
		verify_mm(A, P * 1 * 2)
	
	return elapsed_time


def main():
	count = 100
	averageTime = 0
	
	for i in range(count):
		averageTime += matrix_multiplication(False, device)
	
	averageTime /= count
	print("[CPU - Double] (Matrix Multiplication) Average Elapsed Time = {} ms".format(averageTime))


if __name__ == "__main__":
	main()
