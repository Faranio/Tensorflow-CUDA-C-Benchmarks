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
			
	for i, v in enumerate(values):
		plt.text(0, i, "{:.5f}".format(v), color='blue', fontweight='bold', fontsize=20)
	
	plt.yticks(y_pos, names, fontsize=15)
	plt.xlabel('Average Inference Time in ms', fontsize=20)
	fig.suptitle(title, fontsize=22)
	plt.tight_layout()
	plt.grid(axis='x')
	plt.show()
	

def plot_accuracy_graph(data, title):
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
	
	for i, v in enumerate(values):
		plt.text(0, i, "{:.5f}".format(v), color='blue', fontweight='bold', fontsize=20)
	
	plt.yticks(y_pos, names, fontsize=15)
	plt.xlabel('Accuracy in %', fontsize=20)
	fig.suptitle(title, fontsize=22)
	plt.tight_layout()
	plt.grid(axis='x')
	plt.show()


def main():
	# M = 65536
	saxpy_gpu_elapsed_time = {
		'[GPU - Double] CUDA C (1024 blocks, 64 threads)': 0.0045410,
		'[GPU - Double] Tensorflow (W/O Warm-Up)': 0.05851822021213593,
		'[CPU - Double] CUDA C': 0.151320,
		'[CPU - Double] Tensorflow': 0.7176052799695753,
		'[GPU - Float] CUDA C (512 blocks, 128 threads)': 0.0030720,
		'[GPU - Float] Tensorflow (W/O Warm-Up)': 0.05726448001951212,
		'[CPU - Float] CUDA C': 0.133380,
		'[CPU - Float] Tensorflow': 0.4058981596608646
	}
	
	# M = 65536
	dot_product_gpu_elapsed_time = {
		'[GPU - Double] CUDA C (1024 blocks, 64 threads)': 0.006272,
		'[GPU - Double] Tensorflow (W/O Warm-Up)': 0.05195684996579075,
		'[CPU - Double] CUDA C': 0.218660,
		'[CPU - Double] Tensorflow': 0.6070597299549263,
		'[GPU - Float] CUDA C (1024 blocks, 64 threads)': 0.004608,
		'[GPU - Float] Tensorflow (W/O Warm-Up)': 0.05077187011920614,
		'[CPU - Float] CUDA C': 0.153470,
		'[CPU - Float] Tensorflow': 0.35387275009270525
	}
	
	# MxM = 512x512
	matrix_add_gpu_elapsed_time = {
		'[GPU - Double] CUDA C (1D, 4096 blocks, 64 threads)': 0.051115,
		'[GPU - Double] CUDA C (2D, 64x64 blocks, 8x8 threads)': 0.051415,
		'[GPU - Double] Tensorflow (W/O Warm-Up)': 0.06057500013412209,
		'[CPU - Double] CUDA C': 0.566670,
		'[CPU - Double] Tensorflow': 1.9303801101341378,
		'[GPU - Float] CUDA C (1D, 2048 blocks, 128 threads)': 0.023871,
		'[GPU - Float] CUDA C (2D, 64x64 blocks, 8x8 threads)': 0.031253,
		'[GPU - Float] Tensorflow (W/O Warm-Up)': 0.03151900000375463,
		'[CPU - Float] CUDA C': 0.559160,
		'[CPU - Float] Tensorflow': 1.0637350901015452
	}
	
	# MxN = 512x128
	matrix_multiplication_gpu_elapsed_time = {
		'[GPU - Double] CUDA C (Shared, 1D, 256 blocks, 256 threads)': 0.087746,
		'[GPU - Double] Tensorflow (W/O Warm-Up)': 0.09400099997947109,
		'[GPU - Double] CUDA C (Shared, 2D, 64x16 blocks, 8x8 threads)': 0.26704,
		'[GPU - Double] CUDA C (Global, 1D, 1024 blocks, 64 threads)': 0.33282,
		'[GPU - Double] CUDA C (Global, 2D, 64x16 blocks, 8x8 threads)': 1.1469,
		'[CPU - Double] Tensorflow': 1.0858771400307887,
		'[CPU - Double] CUDA C (Sequential)': 52.689537,
		'[GPU - Float] CUDA C (Shared, 1D, 256 blocks, 256 threads)': 0.082045,
		'[GPU - Float] Tensorflow (W/O Warm-Up)': 0.09101244000703446,
		'[GPU - Float] CUDA C (Global, 1D, 128 blocks, 512 threads)': 0.22237,
		'[GPU - Float] CUDA C (Shared, 2D, 64x16 blocks, 8x8 threads)': 0.26831,
		'[GPU - Float] CUDA C (Global, 2D, 64x16 blocks, 8x8 threads)': 0.58989,
		'[CPU - Float] Tensorflow': 0.7069014800072182,
		'[CPU - Float] CUDA C (Sequential)': 48.004147,
	}
	
	s_le_net_inference_elapsed_time = {
		'[GPU] Tensorflow (Batch Size 1)': 5308.234113,
		'[CPU] Tensorflow (Batch Size 1)': 2951.483607,
		'[GPU] CUDA C (Batch Size 1)': 407.877655
	}
	
	s_le_net_inference_elapsed_time_CUDA_C_kernels = {
		'Convolutional Layer': 0.0019280,
	    'Convolutional Bias': 0.0013010,
		'Convolutional Sigmoid': 0.0013050,
		'Subsampling Layer': 0.0011130,
		'Subsampling Bias': 0.000911,
		'Subsampling Sigmoid': 0.000936,
		'Fully Connected Layer': 0.022405,
		'Fully Connected Bias': 0.000951,
		'Fully Connected Sigmoid': 0.000979
	}
	
	s_le_net_inference_accuracy_tensorflow = {
		'[GPU] Tensorflow': 87.6800000667572,
		'[CPU] Tensorflow': 87.6800000667572,
	}
	
	matrix_multiplication_gpu_elapsed_time_tensorflow = {
		'[GPU - Double] Tensorflow (First Execution)': 405.06568500040885,
		'[GPU - Double] Tensorflow (Second Execution)': 0.09341699978904217,
		'[GPU - Float] Tensorflow (First Execution)': 396.4082520001284,
		'[GPU - Float] Tensorflow (Second Execution)': 0.08577600010539754
	}
	
	# plot_elapsed_time_graph(data=matrix_multiplication_gpu_elapsed_time_tensorflow, title="[GPU] Matrix Multiplication "
	#                                                                                       "Inference Time for Single and "
	#                                                                                       "Double Precision Data "
	#                                                                                       "(512x256 and 256x128 size)")
	plot_elapsed_time_graph(data=s_le_net_inference_elapsed_time, title='s-LeNet Inference Time for CPU and '
	                                                                    'GPU')
	# plot_elapsed_time_graph(data=s_le_net_inference_elapsed_time_CUDA_C_kernels, title='s-LeNet Inference Time for CUDA '
	#                                                                                    'C kernels')
	# plot_accuracy_graph(data=s_le_net_inference_accuracy_tensorflow, title='s-LeNet Accuracy for CPU and GPU')
	
	# plot_elapsed_time_graph(data=saxpy_gpu_elapsed_time, title="SAXPY/DAXPY Inference Time for Single and Double "
	#                                                            "Precision Data (65536 size)")
	# plot_elapsed_time_graph(data=dot_product_gpu_elapsed_time, title="Dot Product Inference Time for Single and "
	#                                                                  "Double Precision Data (65536 size)")
	# plot_elapsed_time_graph(data=matrix_add_gpu_elapsed_time, title="Matrix Add Inference Time for Single and "
	#                                                                 "Double Precision Data (512x512 size)")
	# plot_elapsed_time_graph(data=matrix_multiplication_gpu_elapsed_time, title="Matrix Multiplication Inference "
	#                                                                            "Time for Single and Double Precision "
	#                                                                            "Data (512x256 and 256x128 size)")


if __name__ == "__main__":
	main()
