import matplotlib.pyplot as plt
import re

size = []
size1 = []
bw_128cuplas = []
MUPD_128cuplas = []
bw_128my = []
MUPD_128my = []
bw_128 = []
MUPD_128 = []
bw_256 = []
MUPD_256 = []
bw_512 = []
MUPD_512 = []
bw_bmy = []
MUPD_bmy = []
bw_bcu = []
MUPD_bcu = []
bw_cmy = []
bw_ccu = []
gflops_cu = []
gflops_my = []
gflops_cpu = []

output_128cublas = open("128_cublas.out")
output_128my = open("128_my.out")
output_128 = open("128.out")
output_256 = open("256.out")
output_512 = open("512.out")
output_bcu = open("2-b-cublas.out")
output_bmy = open("2-b-my.out")
output_cmy = open("2c-my.out")
output_ccu = open("2c_cublas.out")
matmul_my = open("task4_my.out")
matmul_cu = open("task4_cu.out")
matmul_cpu = open("task4_cpu.out")


pattern = '(^[A-Za-z\s]+)([\d]+)\s([\d]+)'
pattern2 = r'([e\+\d+\.]+) GB/s'
pattern3 = r'([e\+\d+\.]+) GFlop/s'
pattern4 = '(^[A-Za-z\s]+)([\d]+)'
pattern5 = '(^[A-Za-z\s]+)([\d]+)'
pattern6 = r'(\d+\.\d+)\sGFLOP/s'


# for line in output_128cublas:
#     result1 = re.search(pattern, line)
#     result2 = re.search(pattern2, line)
#     result3 = re.search(pattern3, line)
#     size.append(int(result1.group(2)))
#     bw_128cuplas.append(float(result2.group(1)))
#     MUPD_128cuplas.append(float(result3.group(1)))
    
    
# for line in output_128my:
#     result4 = re.search(pattern2, line)
#     result5 = re.search(pattern3, line)
#     bw_128my.append(float(result4.group(1)))
#     MUPD_128my.append(float(result5.group(1)))
    
    
# for line in output_128:
#     result6 = re.search(pattern, line)
#     result7 = re.search(pattern2, line)
#     result8 = re.search(pattern3, line)
#     size.append(int(result6.group(2)))
#     bw_128.append(float(result7.group(1)))
#     MUPD_128.append(float(result8.group(1)))

# for line in output_256:
#     result9 = re.search(pattern2, line)
#     result10 = re.search(pattern3, line)
#     bw_256.append(float(result9.group(1)))
#     MUPD_256.append(float(result10.group(1)))
    
# for line in output_512:
#     result11 = re.search(pattern2, line)
#     result12 = re.search(pattern3, line)
#     bw_512.append(float(result11.group(1)))
#     MUPD_512.append(float(result12.group(1)))
    
# for line in output_bmy:
#     result13 = re.search(pattern2, line)
#     result14 = re.search(pattern, line)
#     bw_bmy.append(float(result13.group(1)))
#     size.append(float(result14.group(2)))
    
# for line in output_bcu:
#     result15 = re.search(pattern2, line)
#     result16 = re.search(pattern3, line)
#     bw_bcu.append(float(result15.group(1)))
#     MUPD_bcu.append(float(result16.group(1)))

# for line in output_cmy:
#     result6 = re.search(pattern, line)
#     result17 = re.search(pattern2, line)
#     size.append(int(result6.group(3)))
#     bw_cmy.append(float(result17.group(1)))
    
# for line in output_ccu:
#     result18 = re.search(pattern2, line)
#     bw_ccu.append(float(result18.group(1)))


for line in matmul_cu:
    result19 = re.search(pattern5, line)
    result20 = re.search(pattern6, line)
    size.append(int(result19.group(2)))
    gflops_cu.append(float(result20.group(1)))
    
    
for line in matmul_my:
    result21 = re.search(pattern3, line)
    gflops_my.append(float(result21.group(1)))

for line in matmul_cpu:
    result23 = re.search(pattern4, line)
    result22 = re.search(pattern3, line)
    size1.append(int(result23.group(2)))
    gflops_cpu.append(float(result22.group(1)))
    
# # plot the comparison of cublas and my implementation for matrix matrix multiplication
# plt.xscale('log')
# plt.yscale('log')
# plt.plot(size, gflops_cu, 'o', linestyle = '--', label = "cuBLAS")
# plt.plot(size, gflops_my, 'x', linestyle = '--', label = "Self Implementation")
# plt.xlabel("Matrix Size")
# plt.ylabel("Number of floating-point operations per sec (GFLOP/S)")
# plt.legend()
# plt.show()

plt.xscale('log')
plt.yscale('log')
plt.plot(size, gflops_cu, '*', linestyle = '--', label = "CuBLAS")
plt.plot(size, gflops_my, 'o', linestyle = '--', label = "Self Implementation")
plt.plot(size1, gflops_cpu, 'x', linestyle = '--', label = "Cpu")
plt.xlabel("Matrix Size")
plt.ylabel("Number of floating-point operations per sec (GFLOP/S)")
plt.legend()
plt.show()

# plot the comparison of cublas and my implementation for matrix vector multiplication
# plt.xscale('log')
# # plt.yscale('log')
# plt.plot(size, bw_128cuplas, '-', linestyle = '--', label = "cuBLAS")
# plt.plot(size, bw_128my, 'x', linestyle = '--', label = "Self Implementation")
# plt.xlabel("Matrix Size")
# plt.ylabel("Memory Throughput (GB/s)")
# plt.legend()
# plt.show()

# # plt.xscale('log')
# plt.plot(size, MUPD_128cuplas, '-', linestyle = '--', label = "cuBLAS")
# plt.plot(size, MUPD_128my, 'x', linestyle = '--', label = "Self Implementation")
# plt.xlabel("Matrix Size")
# plt.ylabel("Million updates per second (MUPD/s)")
# plt.legend()
# plt.show()



# plot the comparison of different block size for matrix vector multiplication
# plt.xscale('log')
# plt.plot(size, bw_128, '.', linestyle = '--', label = "block size = 128")
# plt.plot(size, bw_256, 'x', linestyle = '--', label = "block size = 256")
# plt.plot(size, bw_512, 'o', linestyle = '--', label = "block size = 512")
# plt.xlabel("Matrix Size")
# plt.ylabel("Memory Throughput (GB/s)")
# plt.legend()
# plt.show()

# plt.xscale('log')
# plt.plot(size, MUPD_128, '.', linestyle = '--', label = "block size = 128")
# plt.plot(size, MUPD_256, 'x', linestyle = '--', label = "block size = 256")
# plt.plot(size, MUPD_512, 'o', linestyle = '--', label = "block size = 512")
# plt.xlabel("Matrix Size")
# plt.ylabel("Million updates per second (MUPD/s)")
# plt.legend()
# plt.show()

# # N = 10000, change M, task 2 (b)
# plt.xscale('log')
# plt.yscale('log')
# plt.plot(size, bw_bmy, '.', linestyle = '--', label = "Self Implementation")
# plt.plot(size, bw_bcu, 'x', linestyle = '--', label = "cuBLAS")
# plt.xlabel("Matrix Size")
# plt.ylabel("Memory Throughput (GB/s)")
# plt.legend()
# plt.show()

# # M = 16384, change N, task 2 (c)
# plt.xscale('log')
# plt.plot(size, bw_cmy, '.', linestyle = '--', label = "Self Implementation")
# plt.plot(size, bw_ccu, 'x', linestyle = '--', label = "cuBLAS")
# plt.xlabel("Matrix Size N")
# plt.ylabel("Memory Throughput (GB/s)")
# plt.legend()
# plt.show()

# plt.xscale('log')
# plt.yscale('log')
# plt.plot(size, bw_32, '-', linestyle = '--', label = "block size = 32")
# plt.plot(size, bw_64, 'x', linestyle = '--', label = "block size = 64")
# plt.plot(size, bw_128, 'o', linestyle = '--', label = "block size = 128")
# plt.plot(size, bw_256, '*', linestyle = '--', label = "block size = 256")
# plt.plot(size, bw_512, '.', linestyle = '--', label = "block size = 512")
# plt.plot(size, bw_1024, '+', linestyle = '--', label = "block size = 1024")
# plt.xlabel("Matrix Size")
# plt.ylabel("Memory Throughput (GB/s)")
# plt.legend()
# plt.show()

# plt.xscale('log')
# plt.plot(size, MUPD_32, 'x', linestyle = '--', label = "block size = 32")
# plt.plot(size, MUPD_64, 'o', linestyle = '--', label = "block size = 64")
# plt.plot(size, MUPD_128, '*', linestyle = '--', label = "block size = 128")
# plt.plot(size, MUPD_256, '.', linestyle = '--', label = "block size = 256")
# plt.plot(size, MUPD_512, '+', linestyle = '--', label = "block size = 512")
# plt.plot(size, MUPD_1024, '-', linestyle = '--', label = "block size = 1024")
# plt.xlabel("Matrix Size")
# plt.ylabel("Million updates per second (MUPD/s)")
# plt.legend()
# plt.show()

# plt.xscale('log')
# plt.plot(size, bw_128, 'o', linestyle = '--', label = "block size = 128")
# plt.plot(size, bw_cublas, 'o', linestyle = '--', label = "block size = 128")
# plt.xlabel("Matrix Size")
# plt.ylabel("Memory Throughput (GB/s)")
# plt.legend()
# plt.show()
    