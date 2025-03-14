import random
import math
import functools
import numpy as np
from time import perf_counter as pc
import matplotlib.pyplot as plt

# MA4-1.1
n = {1000, 10000, 100000}
def approximation_pi(n):
    coordinates = []
    for i in range (n):
        (x, y) = (random.uniform(-1, 1), random.uniform(-1, 1))
        coordinates.append((x, y))
    nclst = [x for x in coordinates if (x[0]**2 + x[1]**2) <= 1]
    nc = len(nclst)
    pai = 4*nc/n
    return pai
print(approximation_pi(400))

# #################################################################
# MA4-1.2
# n is the number of random points
# d is the number of dimensions

def power2(a):
    result = a**2
    return result

def HypersphereVolume(n, d):
    count_in_sphere = 0
    for i in range (n):
        point = np.random.uniform(-1, 1, d)
        po = map(power2, point)
        sum = functools.reduce(lambda x, y : x + y, po)
        if sum <= 1:
            count_in_sphere += 1
    return np.power(2, d)*(count_in_sphere/n)

def exactvolume(de):
    result = (np.power(math.pi, de/2))/(math.gamma((de/2) + 1))
    return result
# print(HypersphereVolume(100000,11))
# print(exactvolume(11))

if __name__ == "__main__":
    start = pc()
    print(HypersphereVolume(10000000, 11))
    print(exactvolume(11))
    end = pc()
    print(f"Process took {round(end-start, 2)} seconds")

# #################################################################
# MA-1.3


