import matplotlib.pyplot as plt

# x = [128, 256, 512, 1024, 2048]
# y = [0.003482, 0.024507, 0.189256, 1.478168, 11.922703]

# plt.plot(x, y, label = 'Real complexity')

# plt.xlabel('Dimension of Matrix')
# plt.ylabel('Time in Seconds')
# plt.title('Time Complexity of Serial Strassen Algorithm')
# plt.show()
# y1 = [0.001990, 0.009776, 0.053051, 0.335193, 2.295471] #O1 optimazation
# y2 = [0.001980, 0.009533, 0.051243, 0.320785, 2.307269] #O2 optimazation
# y3 = [0.002001, 0.008990, 0.047698, 0.296018, 2.094269] #O3 optimisation

# plt.plot(x, y1, label = 'O1 optimazation')
# plt.plot(x, y2, label = 'O2 optimazation')
# plt.plot(x, y3, label = 'O3 optimazation')
# plt.legend()
# plt.xlabel('Dimension of Matrix')
# plt.ylabel('Time in Seconds')
# plt.show()

# # n = 2048
# x1 = [1, 2, 4, 6, 8, 10, 12, 14, 16]
# y4 = [12.022137, 8.209211, 4.281975, 2.746756, 2.299503, 1.922941, 1.658651, 1.539527, 1.322469]
# speedup = []
# for i in range (0, len(y4)):
#     speedup.append(y4[0] / y4[i])
# for i in speedup:
#     print(i)
# plt.plot(x1, x1, '--x', label = 'Ideal speedup')
# plt.plot(x1, speedup, 'o', label = 'Actual speedup')
# plt.legend()
# plt.xlabel('Number of threads')
# plt.ylabel('Speedup')
# plt.title('Parallel Strassen Algorithm Speedup')
# plt.show()

# n = 512
x1 = [1, 2, 4, 6, 8, 10, 12, 14, 16]
y4 = [0.189798, 0.124482, 0.071243, 0.052760, 0.047986, 0.043893, 0.041064, 0.038112, 0.037874]
speedup = []
for i in range (0, len(y4)):
    speedup.append(y4[0] / y4[i])
for i in speedup:
    print(i)
plt.plot(x1, x1, '--x', label = 'Ideal speedup')
plt.plot(x1, speedup, 'o', label = 'Actual speedup')
plt.legend()
plt.xlabel('Number of threads')
plt.ylabel('Speedup')
plt.title('Parallel Strassen Algorithm Speedup')
plt.show()


# # plt.title("Time Complexity for serial Strassen algorithm")
# plt.plot(x, y, label='Without any compiler optimisation')
# plt.plot(x, y1, label='Using -O1')
# plt.plot(x, y2, label='Using -O2')
# plt.plot(x, y3, label='Using -O3')
# plt.legend()
# plt.xlabel('Dimension of matrix')
# plt.ylabel('Time in seconds')
# plt.show()


# x1 = [2, 4, 6, 8, 10, 12, 14, 16]
# y1 = [38.842497, 22.669171, 20.452446, 14.393130, 13.961746, 14.469783, 14.515356, 14.417251]
# y2 = []
# for i in range(0, len(y1)):
#     y2.append(28.906329 / y1[i])
# plt.plot(x1, y2)
# plt.xlabel('Number of threads')
# plt.ylabel('Speedup')
# plt.show()