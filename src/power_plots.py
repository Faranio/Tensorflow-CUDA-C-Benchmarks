import matplotlib.pyplot as plt
import numpy as np


# My own plot of the power
def plot_power_graph(data, title):
	names = data.keys()
	values = data.values()
	fig = plt.figure(figsize=(15, 10))
	y_pos = np.arange(len(names))
	barlist = plt.barh(y_pos, values)
	
	for i in range(len(names)):
		if i < len(names) / 2:
			barlist[i].set_color('orange')
		else:
			barlist[i].set_color('cyan')
	
	plt.yticks(y_pos, names, fontsize=15)
	plt.xlabel('Average Power in mW', fontsize=20)
	fig.suptitle(title, fontsize=22)
	plt.tight_layout()
	plt.grid(axis='x')
	plt.show()


def main():
	# M = 65536
	saxpy_gpu_power = {
		'[GPU - Double] CUDA C (1024 blocks, 64 threads)': 11649.999999999998,
		'[GPU - Double] Tensorflow': 12194.920634920634,
		'[GPU - Float] CUDA C (512 blocks, 128 threads)': 11650.86956521739,
		'[GPU - Float] Tensorflow': 12416.923076923078
	}
	
	# M = 65536
	dot_product_gpu_power = {
		'[GPU - Double] CUDA C (1024 blocks, 64 threads)': 11796.0,
		'[GPU - Double] Tensorflow': 12207.719298245614,
		'[GPU - Float] CUDA C (1024 blocks, 64 threads)': 11692.051282051283,
		'[GPU - Float] Tensorflow': 11876.888888888889
	}
	
	# MxM = 512x512
	matrix_add_gpu_power = {
		'[GPU - Double] CUDA C (1D, 4096 blocks, 64 threads)': 13665.652173913042,
		'[GPU - Double] CUDA C (2D, 64x64 blocks, 8x8 threads)': 14368.78048780488,
		'[GPU - Double] Tensorflow': 13841.632653061226,
		'[GPU - Float] CUDA C (1D, 2048 blocks, 128 threads)': 11748.918918918918,
		'[GPU - Float] CUDA C (2D, 64x64 blocks, 8x8 threads)': 11732.051282051285,
		'[GPU - Float] Tensorflow': 11929.14893617021
	}
	
	# MxN = 512x128
	matrix_multiplication_gpu_power = {
		'[GPU - Double] CUDA C (Global, 1D, 1024 blocks, 64 threads)': 13200.512820512822,
		'[GPU - Double] CUDA C (Global, 2D, 64x16 blocks, 8x8 threads)': 15759.333333333332,
		'[GPU - Double] CUDA C (Shared, 1D, 256 blocks, 256 threads)': 11640.0,
		'[GPU - Double] CUDA C (Shared, 2D, 64x16 blocks, 8x8 threads)': 13185.000000000002,
		'[GPU - Double] Tensorflow': 12375.0,
		'[GPU - Float] CUDA C (Global, 1D, 128 blocks, 512 threads)': 11885.641025641025,
		'[GPU - Float] CUDA C (Global, 2D, 64x16 blocks, 8x8 threads)': 11535.757575757576,
		'[GPU - Float] CUDA C (Shared, 1D, 256 blocks, 256 threads)': 11657.22222222222,
		'[GPU - Float] CUDA C (Shared, 2D, 64x16 blocks, 8x8 threads)': 11774.390243902439,
		'[GPU - Float] Tensorflow': 12171.186440677966,
	}
	
	plot_power_graph(data=saxpy_gpu_power,
	           title="[GPU] SAXPY/DAXPY Inference Time for Single and Double Precision Data "
	                 "(65536 size)")
	plot_power_graph(data=dot_product_gpu_power, title="[GPU] Dot Product Inference Time for Single and Double "
	                                                    "Precision Data (65536 size)")
	plot_power_graph(data=matrix_add_gpu_power,
	           title="[GPU] Matrix Add Inference Time for Single and Double Precision "
	                 "Data (512x512 size)")
	plot_power_graph(data=matrix_multiplication_gpu_power,
	           title="[GPU] Matrix Multiplication Inference Time for Single "
	                 "and Double Precision Data (512x256 and 256x128 size)")


if __name__ == "__main__":
	main()
