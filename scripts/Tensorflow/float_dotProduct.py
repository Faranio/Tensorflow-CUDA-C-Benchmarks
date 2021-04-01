import tensorflow as tf
import time

M = 65536

dtype = tf.float32
device = "/gpu:0"


def verify_dotProduct(answer, check):
	maxError = abs(answer - check)
	print("Maximum Error = {}".format(maxError))


# My own implementation of Dot Product
def dot_product(verbose=False, device="/gpu:0"):
	deviceType = device.upper()[1:4]
	
	A = tf.ones([1, M], dtype=dtype)
	B = tf.ones([M, 1], dtype=dtype) * 2
	
	start = time.perf_counter()
	
	with tf.device(device):
		A = tf.matmul(A, B)
		
	end = time.perf_counter()
	
	elapsed_time = end - start
	elapsed_time *= 1000
	
	if verbose:
		print("[{}] (Dot Product) Elapsed Time = {} ms".format(deviceType, elapsed_time))
		verify_dotProduct(A, M*2)
	
	return elapsed_time


def main():
	count = 100
	averageTime = 0
	# dot_product(False, device)
	
	for i in range(count):
		averageTime += dot_product(False, device)
	
	averageTime /= count
	print("[GPU - Float] (Dot Product) Average Elapsed Time = {} ms".format(averageTime))


if __name__ == "__main__":
	main()
