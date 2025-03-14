import concurrent.futures as future
from time import perf_counter as pc
import functools
import numpy as np
from MA4_1 import power2, HypersphereVolume         # import the function

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
    
if __name__ == "__main__":
    start = pc()                      # mark the start
    
    with future.ProcessPoolExecutor() as ex:            # use map to simplify
        p1 = ex.submit(HypersphereVolume, 1000000, 11)
        p2 = ex.submit(HypersphereVolume, 1000000, 11)
        p3 = ex.submit(HypersphereVolume, 1000000, 11)
        p4 = ex.submit(HypersphereVolume, 1000000, 11)
        p5 = ex.submit(HypersphereVolume, 1000000, 11)
        p6 = ex.submit(HypersphereVolume, 1000000, 11)
        p7 = ex.submit(HypersphereVolume, 1000000, 11)
        p8 = ex.submit(HypersphereVolume, 1000000, 11)
        p9 = ex.submit(HypersphereVolume, 1000000, 11)
        p10 = ex.submit(HypersphereVolume, 1000000, 11)
        r1 = p1.result()
        r2 = p2.result()
        r3 = p3.result()
        r4 = p4.result()
        r5 = p5.result()
        r6 = p6.result()
        r7 = p7.result()
        r8 = p8.result()
        r9 = p9.result()
        r10 = p10.result()

    end = pc()                        # mark the end 
    print(f"Process took {round(end-start, 2)} seconds")

'''
Time the code without parallel is about 87.03 seconds.
Time the code with parallel is about 19.4 seconds.
So use parallel is way faster than without using parallel, save 77.7%
time. Because using parallel can decrease the loop numbers to 1000000
and compute it 10 times in the same time, so it is way faster than 
calculate it 10000000 times in one core.
''' 