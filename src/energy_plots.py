import matplotlib.pyplot as plt
import numpy as np


# My own plot of the energy
def plot_energy_graph(data, title):
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
	plt.xlabel('Average Energy in mJ', fontsize=20)
	fig.suptitle(title, fontsize=22)
	plt.tight_layout()
	plt.grid(axis='x')
	plt.show()


def main():
	# M = 65536
	saxpy_gpu_energy = {
		'[GPU - Double] CUDA C (1024 blocks, 64 threads)': 23754.350399971005,
		'[GPU - Double] Tensorflow': 38694.48145336575,
		'[GPU - Float] CUDA C (512 blocks, 128 threads)': 13072.278185512707,
		'[GPU - Float] Tensorflow': 32246.750746506914
	}
	
	# M = 65536
	dot_product_gpu_energy = {
		'[GPU - Double] CUDA C (1024 blocks, 64 threads)': 23485.832445144653,
		'[GPU - Double] Tensorflow': 34962.906160856546,
		'[GPU - Float] CUDA C (1024 blocks, 64 threads)': 22635.80976559566,
		'[GPU - Float] Tensorflow': 26675.495887756348
	}
	
	# MxM = 512x512
	matrix_add_gpu_energy = {
		'[GPU - Double] CUDA C (1D, 4096 blocks, 64 threads)': 31390.00210513239,
		'[GPU - Double] CUDA C (2D, 64x64 blocks, 8x8 threads)': 29297.94390794708,
		'[GPU - Double] Tensorflow': 33870.47547165229,
		'[GPU - Float] CUDA C (1D, 2048 blocks, 128 threads)': 21535.770529669684,
		'[GPU - Float] CUDA C (2D, 64x64 blocks, 8x8 threads)': 22736.71216231127,
		'[GPU - Float] Tensorflow': 27926.13292694091
	}
	
	# MxN = 512x128
	matrix_multiplication_gpu_energy = {
		'[GPU - Double] CUDA C (Global, 1D, 1024 blocks, 64 threads)': 25582.59022052472,
		'[GPU - Double] CUDA C (Global, 2D, 64x16 blocks, 8x8 threads)': 35300.91057427724,
		'[GPU - Double] CUDA C (Shared, 1D, 256 blocks, 256 threads)': 18414.477424621582,
		'[GPU - Double] CUDA C (Shared, 2D, 64x16 blocks, 8x8 threads)': 26264.521358013157,
		'[GPU - Double] Tensorflow': 63644.626557826996,
		'[GPU - Float] CUDA C (Global, 1D, 128 blocks, 512 threads)': 23046.259490282107,
		'[GPU - Float] CUDA C (Global, 2D, 64x16 blocks, 8x8 threads)': 18814.82109012026,
		'[GPU - Float] CUDA C (Shared, 1D, 256 blocks, 256 threads)': 31567.759912278918,
		'[GPU - Float] CUDA C (Shared, 2D, 64x16 blocks, 8x8 threads)': 24043.302722093533,
		'[GPU - Float] Tensorflow': 35953.68469933332,
	}
	
	plot_energy_graph(data=saxpy_gpu_energy,
	                 title="[GPU] SAXPY/DAXPY Inference Time for Single and Double Precision Data "
	                       "(65536 size)")
	plot_energy_graph(data=dot_product_gpu_energy, title="[GPU] Dot Product Inference Time for Single and Double "
	                                                   "Precision Data (65536 size)")
	plot_energy_graph(data=matrix_add_gpu_energy,
	                 title="[GPU] Matrix Add Inference Time for Single and Double Precision "
	                       "Data (512x512 size)")
	plot_energy_graph(data=matrix_multiplication_gpu_energy,
	                 title="[GPU] Matrix Multiplication Inference Time for Single "
	                       "and Double Precision Data (512x256 and 256x128 size)")


if __name__ == "__main__":
	main()
