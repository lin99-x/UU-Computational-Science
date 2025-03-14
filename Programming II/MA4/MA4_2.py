#!/usr/bin/env python3.9

import matplotlib.pyplot as plt
from time import perf_counter as pc
from person import Person

def main():
	f = Person(5)
	print(f.get())
	f.set(7)
	print(f.get())
	print(f.fib())
	x = []
	y = []
	for n in range(30, 46):
		f.set(n)
		x.append(n)
		start = pc()
		f.fib()
		end = pc()
		time = round(end-start, 2)
		y.append(time)
	plt.plot(x, y)
	plt.savefig('C++.png', format = "png")

if __name__ == '__main__':
	main()
