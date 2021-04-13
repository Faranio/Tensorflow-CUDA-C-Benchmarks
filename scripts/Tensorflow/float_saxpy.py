import tensorflow as tf
import time

M = 65536

dtype = tf.float32
device = "/gpu:0"


def verify_saxpy(A, check):
    maxError = 0

    for i in range(M):
        maxError += abs(A[i] - check)

    print("Maximum Error = {}".format(maxError))


# My own implementation of SAXPY
def saxpy(verbose=False, device="/gpu:0"):
    deviceType = device.upper()[1:4]

    alpha = tf.constant(2, dtype=dtype)
    A = tf.ones([M], dtype=dtype)
    B = tf.ones([M], dtype=dtype) * 2

    start = time.perf_counter()
    
    with tf.device(device):
        A = tf.add(tf.scalar_mul(alpha, A), B)
        
    end = time.perf_counter()
        
    elapsed_time = end - start
    elapsed_time *= 1000
    
    if verbose:
        print("[{}] (DAXPY) Elapsed Time = {} ms".format(deviceType, elapsed_time))
        verify_saxpy(A, 4)

    return elapsed_time


def main():
    count = 100
    averageTime = 0
    saxpy(False, device)

    for i in range(count):
        averageTime += saxpy(False, device)

    averageTime /= count
    print("[GPU - Float] (DAXPY) Average Elapsed Time = {} ms".format(averageTime))


if __name__ == "__main__":
    main()
