import matplotlib.pyplot as plt

# filename = 'part-r-00000.txt'
# X, Y = [], []
# with open(filename, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         print(line.split())
#         name = str(line.split()[0])
#         value = float(line.split()[1])
#         X.append(name)
#         Y.append(value)
#
#
# plt.plot(X, Y)
# plt.show()

X = ["den", "denna", "denne", "det", "han", "hen", "hon"]
Y = [1623354, 31683, 6665, 594561, 834277, 46321, 396532]

Y = [Y[i]/2341577 for i in range (len(Y))]
print(Y)
plt.bar(X, Y)
plt.show()