# write fib function with and without numba

from numba import njit
from time import perf_counter as pc
import matplotlib.pyplot as plt

def fib_py(n):
	if n <= 1:
		return n
	else:
		return (fib_py(n-1) + fib_py(n-2))

@njit
def fib_numba(n):
	if n <= 1:
		return n
	else:
		return (fib_numba(n-1) + fib_numba(n-2))

def main():
	x_py = []
	y_py = []
	for i in range(20, 31):
		x_py.append(i)
		start_py = pc()
		fib_py(i)
		end_py = pc()
		time_py = round(end_py-start_py, 2)
		y_py.append(time_py)
	plt.plot(x_py, y_py)
	plt.savefig('pure_py.png', format = 'png')
	plt.clf()
	x_numba = []
	y_numba = []
	for j in range(20, 31):
		x_numba.append(j)
		start_numba = pc()
		fib_numba(j)
		end_numba = pc()
		time_numba = round(end_numba-start_numba, 2)
		y_numba.append(time_numba)
	plt.plot(x_numba, y_numba)
	plt.savefig('py_numba.png', format = 'png')

if __name__ == '__main__':
	main()
