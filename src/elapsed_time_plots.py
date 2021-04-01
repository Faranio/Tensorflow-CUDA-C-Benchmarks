import matplotlib.pyplot as plt
import numpy as np


# My own plot of the elapsed time
def plot_elapsed_time_graph(data, title):
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
	plt.xlabel('Average Inference Time in ms', fontsize=20)
	fig.suptitle(title, fontsize=22)
	plt.tight_layout()
	plt.grid(axis='x')
	plt.show()


def main():
	# M = 65536
	saxpy_gpu_elapsed_time = {
		'[GPU - Double] CUDA C (1024 blocks, 64 threads)': 0.0045410,
		'[GPU - Double] Tensorflow (With Warm-Up)': 0.06161928980873199,
		'[GPU - Double] Tensorflow (W/O Warm-Up)': 0.05851822021213593,
		'[GPU - Float] CUDA C (512 blocks, 128 threads)': 0.0030720,
		'[GPU - Float] Tensorflow (With Warm-Up)': 0.061056680024194065,
		'[GPU - Float] Tensorflow (W/O Warm-Up)': 0.05726448001951212,
	}
	
	saxpy_cpu_elapsed_time = {
		'[CPU - Double] CUDA C': 0.122080,
		'[CPU - Double] Tensorflow': 0.7176052799695753,
		'[CPU - Float] CUDA C': 0.136230,
		'[CPU - Float] Tensorflow': 0.4058981596608646
	}
	
	# M = 65536
	dot_product_gpu_elapsed_time = {
		'[GPU - Double] CUDA C (1024 blocks, 64 threads)': 0.006272,
		'[GPU - Double] Tensorflow (With Warm-Up)': 3.758295350125991,
		'[GPU - Double] Tensorflow (W/O Warm-Up)': 0.05195684996579075,
		'[GPU - Float] CUDA C (1024 blocks, 64 threads)': 0.004608,
		'[GPU - Float] Tensorflow (With Warm-Up)': 3.713701920096355,
		'[GPU - Float] Tensorflow (W/O Warm-Up)': 0.05077187011920614,
	}
	
	dot_product_cpu_elapsed_time = {
		'[CPU - Double] CUDA C': 0.155420,
		'[CPU - Double] Tensorflow': 0.6070597299549263,
		'[CPU - Float] CUDA C': 0.153470,
		'[CPU - Float] Tensorflow': 0.35387275009270525
	}
	
	# MxM = 512x512
	matrix_add_gpu_elapsed_time = {
		'[GPU - Double] CUDA C (1D, 4096 blocks, 64 threads)': 0.052470,
		'[GPU - Double] CUDA C (2D, 64x64 blocks, 8x8 threads)': 0.051415,
		'[GPU - Double] Tensorflow (With Warm-Up)': 0.17294299959758064,
		'[GPU - Double] Tensorflow (W/O Warm-Up)': 0.06057500013412209,
		'[GPU - Float] CUDA C (1D, 2048 blocks, 128 threads)': 0.023871,
		'[GPU - Float] CUDA C (2D, 64x64 blocks, 8x8 threads)': 0.043753,
		'[GPU - Float] Tensorflow (With Warm-Up)': 0.0588540001444926,
		'[GPU - Float] Tensorflow (W/O Warm-Up)': 0.03151900000375463,
	}
	
	matrix_add_cpu_elapsed_time = {
		'[CPU - Double] CUDA C': 0.566670,
		'[CPU - Double] Tensorflow': 1.9303801101341378,
		'[CPU - Float] CUDA C': 0.559160,
		'[CPU - Float] Tensorflow': 1.0637350901015452
	}
	
	# MxN = 512x128
	matrix_multiplication_gpu_elapsed_time = {
		'[GPU - Double] CUDA C (Global, 1D, 1024 blocks, 64 threads)': 0.33282,
		'[GPU - Double] CUDA C (Global, 2D, 64x16 blocks, 8x8 threads)': 1.1469,
		'[GPU - Double] CUDA C (Shared, 1D, 256 blocks, 256 threads)': 0.087746,
		'[GPU - Double] CUDA C (Shared, 2D, 64x16 blocks, 8x8 threads)': 0.26704,
		'[GPU - Double] Tensorflow (With Warm-Up)': 3.591447259823326,
		'[GPU - Double] Tensorflow (W/O Warm-Up)': 0.09400099997947109,
		'[GPU - Float] CUDA C (Global, 1D, 128 blocks, 512 threads)': 0.22237,
		'[GPU - Float] CUDA C (Global, 2D, 64x16 blocks, 8x8 threads)': 0.58989,
		'[GPU - Float] CUDA C (Shared, 1D, 256 blocks, 256 threads)': 0.082045,
		'[GPU - Float] CUDA C (Shared, 2D, 64x16 blocks, 8x8 threads)': 0.26831,
		'[GPU - Float] Tensorflow (With Warm-Up)': 3.596720449895656,
		'[GPU - Float] Tensorflow (W/O Warm-Up)': 0.09101244000703446,
	}
	
	matrix_multiplication_cpu_elapsed_time = {
		'[CPU - Double] CUDA C (Sequential)': 52.689537,
		'[CPU - Double] Tensorflow': 1.0858771400307887,
		'[CPU - Float] CUDA C (Sequential)': 48.004147,
		'[CPU - Float] Tensorflow': 0.7069014800072182
	}
	
	plot_elapsed_time_graph(data=saxpy_gpu_elapsed_time, title="[GPU] SAXPY/DAXPY Inference Time for Single and Double "
	                                                           "Precision Data (65536 size)")
	plot_elapsed_time_graph(data=saxpy_cpu_elapsed_time, title="[CPU] SAXPY/DAXPY Inference Time for Single and Double "
	                                                           "Precision Data (65536 size)")
	plot_elapsed_time_graph(data=dot_product_gpu_elapsed_time, title="[GPU] Dot Product Inference Time for Single and "
	                                                                 "Double Precision Data (65536 size)")
	plot_elapsed_time_graph(data=dot_product_cpu_elapsed_time, title="[CPU] Dot Product Inference Time for Single and "
	                                                                 "Double Precision Data (65536 size)")
	plot_elapsed_time_graph(data=matrix_add_gpu_elapsed_time, title="[GPU] Matrix Add Inference Time for Single and "
	                                                                "Double Precision Data (512x512 size)")
	plot_elapsed_time_graph(data=matrix_add_cpu_elapsed_time, title="[CPU] Matrix Add Inference Time for Single and "
	                                                                "Double Precision Data (512x512 size)")
	plot_elapsed_time_graph(data=matrix_multiplication_gpu_elapsed_time, title="[GPU] Matrix Multiplication Inference "
	                                                                           "Time for Single and Double Precision "
	                                                                           "Data (512x256 and 256x128 size)")
	plot_elapsed_time_graph(data=matrix_multiplication_cpu_elapsed_time, title="[CPU] Matrix Multiplication Inference "
	                                                                           "Time for Single and Double Precision "
	                                                                           "Data (512x256 and 256x128 size)")


if __name__ == "__main__":
	main()
