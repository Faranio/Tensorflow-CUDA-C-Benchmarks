import numpy as np

from datetime import datetime


# Power and Energy benchmarks were based on the method written in the README.md file of the following GitHub repository:
# https://github.com/abr/power_benchmarks
def main(start_time=None, end_time=None):
	with open("profiling.csv", 'r') as file:
		text = file.read().split("\n")
		
	del text[0]
	del text[-1]

	start_idx = -1
	end_idx = -1

	for i in range(len(text)):
		text[i] = text[i].split(" ")
		text[i][1] = text[i][1][:-1]

		if text[i][1] == start_time:
			start_idx = i
		elif text[i][1] == end_time:
			end_idx = i

	data = []

	for i in range(start_idx, end_idx + 1):
		data.append(float(text[i][2]))

	start_date = datetime.strptime(start_time, "%H:%M:%S.%f").timestamp()
	end_date = datetime.strptime(end_time, "%H:%M:%S.%f").timestamp()
	elapsed_time = (end_date - start_date) * 1000

	mean_power = np.mean(data) * 1000
	energy = mean_power * 0.001 * elapsed_time

	print("Elapsed Time: {} ms".format(elapsed_time))
	print("Mean Power: {} mW".format(mean_power))
	print("Energy: {} mJ".format(energy))


if __name__ == "__main__":
	main(start_time="13:59:45.295", end_time="13:59:52.765")
