import tensorflow as tf
import time

M = 512

dtype = tf.float32
device = "/gpu:0"


def verify_matAdd(A, check):
	maxError = 0
	
	for i in range(M):
		for j in range(M):
			maxError += abs(A[i][j] - check)
	
	print("Maximum Error = {}".format(maxError))


# My own implementation of Matrix Add
def matrix_add(verbose=False, device="/gpu:0"):
	deviceType = device.upper()[1:4]
	
	A = tf.ones([M, M], dtype=dtype)
	B = tf.ones([M, M], dtype=dtype) * 2
	
	start = time.perf_counter()
	
	with tf.device(device):
		A = tf.add(A, B)
		
	end = time.perf_counter()
	
	elapsed_time = end - start
	elapsed_time *= 1000
	
	if verbose:
		print("[{}] (Matrix Add) Elapsed Time = {} ms".format(deviceType, elapsed_time))
		verify_matAdd(A, 3)
	
	return elapsed_time


def main():
	count = 100
	averageTime = 0
	matrix_add(False, device)
	
	for i in range(count):
		averageTime += matrix_add(False, device)
	
	averageTime /= count
	print("[GPU - Float] (Matrix Add) Average Elapsed Time = {} ms".format(averageTime))


if __name__ == "__main__":
	main()
