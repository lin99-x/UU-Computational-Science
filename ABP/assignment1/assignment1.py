import matplotlib.pyplot as plt
import re

size_O2 = []
BW_O2 = []
size_O3 = []
BW_O3 = []
size_na = []
BW_na = []
size_vec = []
BW_vec = []
BW_vecno = []
BW_cuda = []
BW_single = []
BW_aling = []
BW_noalign = []
size_aling = []
O2_output = open("slurm-8209870.out")
O3_output = open("slurm-8209871.out")
native = open("native.txt")
vec_align = open("align.out")
vec_noalign = open("noalign.out")
cuda = open("cuda.txt")
single = open("single.out")
double = open("double.out")
align = open("local_align.out")
noalign = open("local_noalign.out")

BW_size1 = []
BW_size128 = []
BW_size256 = []
BW_size1024 = []
BW_double = []
MUPD_double = []
MUPD_single = []
size1 = open("1.out")
size128 = open("128.out")
size256 = open("256.out")
size1024 = open("1024.out")

pattern = '(^[A-Za-z\s]+)([\d]+)'
pattern2 = r'([\d+\.]+) GB/s'
pattern3 = r'([\d+\.]+) MUPD/s'

# # get the vector size N and memory throughput GB/s
# for line in O2_output:
#     result1 = re.search(pattern, line)
#     result2 = re.search(pattern2, line)
#     size_O2.append(int(result1.group(2)))
#     BW_O2.append(float(result2.group(1)))

# for line1 in O3_output:
#     result3 = re.search(pattern, line1)
#     result4 = re.search(pattern2, line1)
#     size_O3.append(int(result3.group(2)))
#     BW_O3.append(float(result4.group(1)))
    
# for line2 in native:
#     result5 = re.search(pattern, line2)
#     result6 = re.search(pattern2, line2)
#     size_na.append(int(result5.group(2)))
#     BW_na.append(float(result6.group(1)))

# for line3 in vec_align:
#     result7 = re.search(pattern2, line3)
#     result8 = re.search(pattern, line3)
#     BW_vec.append(float(result7.group(1)))
#     size_vec.append(int(result8.group(2)))

# for line4 in vec_noalign:
#     result9 = re.search(pattern2, line4)
#     BW_vecno.append(float(result9.group(1)))

# for line5 in cuda:
#     result10 = re.search(pattern2, line5)
#     result17 = re.search(pattern3, line5)
#     BW_cuda.append(float(result10.group(1)))
    
# for line6 in size1:
#     result11 = re.search(pattern2, line6)
#     BW_size1.append(float(result11.group(1)))
    
# for line7 in size128:
#     result12 = re.search(pattern2, line7)
#     BW_size128.append(float(result12.group(1)))
    
# for line8 in size256:
#     result13 = re.search(pattern2, line8)
#     BW_size256.append(float(result13.group(1)))
    
# for line9 in size1024:
#     result14 = re.search(pattern2, line9)
#     BW_size1024.append(float(result14.group(1)))

# for line10 in double:
#     result15 = re.search(pattern2, line10)
#     result16 = re.search(pattern3, line10)
#     BW_double.append(float(result15.group(1)))
#     MUPD_double.append(float(result16.group(1)))
    
# for line11 in single:
#     result18 = re.search(pattern2, line11)
#     result19 = re.search(pattern3, line11)
#     BW_single.append(float(result18.group(1)))
#     MUPD_single.append(float(result19.group(1)))

for line12 in align:
    result19 = re.search(pattern, line12)
    result20 = re.search(pattern2, line12)
    size_aling.append(float(result19.group(2)))
    BW_aling.append(float(result20.group(1)))
    
for line13 in noalign:
    result21 = re.search(pattern2, line13)
    BW_noalign.append(float(result21.group(1)))
    
# # plot the figure

plt.xscale('log')
plt.plot(size_aling, BW_aling, 'x', color = 'red', linestyle = '-', label = 'Align memory')
plt.plot(size_aling, BW_noalign, 'o', color = 'green', linestyle = '-', label = 'Not Align memory')
plt.xlabel("vector size N")
plt.ylabel("memory throughput [GB/s]")
plt.legend()
plt.show()

# plt.xscale('log')
# plt.plot(size_O2, BW_O2, 'x', color = 'red', label = 'O2 optimization', linestyle = '--')
# plt.plot(size_O3, BW_O3, 'o', color = 'blue', label = 'O3 optimization', linestyle = '--')
# plt.xlabel("vector size N")
# plt.ylabel("memory throughput [GB/s]")
# plt.grid()
# plt.legend()
# plt.show()

# plt.xscale('log')
# plt.plot(size_na, BW_na, 'x', color = 'green', linestyle = '--', label = 'Apple M1 pro')
# plt.plot(size_O3, BW_O3, 'o', color = 'blue', label = 'O3 optimization', linestyle = '--')
# plt.xlabel("vector size N")
# plt.ylabel("memory throughput [GB/s]")
# plt.grid()
# plt.legend()
# plt.show()

# plt.xscale('log')
# plt.plot(size_vec, BW_vec, 'x', color = 'red', linestyle = '-', label = 'Align memory')
# plt.plot(size_vec, BW_vecno, 'o', color = 'green', linestyle = '-', label = 'Not Align memory')
# plt.xlabel("vector size N")
# plt.ylabel("memory throughput [GB/s]")
# plt.legend()
# plt.show()

# plt.xscale('log')
# plt.plot(size_O3, BW_O3, 'x', color = 'red', linestyle = '-', label = 'CPU with O3 optimization')
# plt.plot(size_O3, BW_single, 'o', color = 'green', linestyle = '-', label = 'GPU')
# plt.xlabel("vector size N")
# plt.ylabel("memory throughput [GB/s]")
# plt.legend()
# plt.show()

# plt.xscale('log')
# plt.plot(size_O3, BW_size1, 'x', color = 'red', linestyle = '-', label = 'block size 1')
# plt.plot(size_O3, BW_size128, 'o', color = 'green', linestyle = '-', label = 'block size 128')
# plt.plot(size_O3, BW_size256, 'x', color = 'blue', linestyle = '-', label = 'block size 256')
# plt.plot(size_O3, BW_single, 'o', color = 'black', linestyle = '-', label = 'block size 512')
# plt.plot(size_O3, BW_size1024, 'o', color = 'yellow', linestyle = '-', label = 'block size 1024')
# plt.legend()
# plt.show()

# plt.xscale('log')
# plt.yscale('log')
# plt.plot(size_O3, BW_single, 'x', color = 'red', linestyle = '-', label = 'single precision')
# plt.plot(size_O3, BW_double, 'o', color = 'green', linestyle = '-', label = 'double precision')
# plt.xlabel("vector size N")
# plt.ylabel("memory throughput [GB/s]")
# plt.legend()
# plt.show()

# print(MUPD_single)
# print(len(MUPD_double))
# plt.xscale('log')
# plt.plot(size_O3, MUPD_single, 'x', color = 'red', linestyle = '-', label = 'single precision')
# plt.plot(size_O3, MUPD_double, 'o', color = 'green', linestyle = '-', label = 'double precision')
# plt.xlabel("vector size N")
# plt.ylabel("million updates [MUPD/s]")
# plt.legend()
# plt.show()

