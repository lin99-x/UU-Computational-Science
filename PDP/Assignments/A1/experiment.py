import matplotlib.pyplot as plt;
import numpy as np;
from scipy.interpolate import make_interp_spline;

def func(x, a, b, c):
    return a * np.exp(-b * x) + c
# strong scaling

# number of threads
x = np.array([1, 2, 4, 8, 12, 16, 20, 24, 28, 32])
y_ideal = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]

# time
y = [0.084504, 0.042836, 0.022002, 0.011358, 0.007423, 0.006044, 0.004814, 0.005042, 0.003545, 0.003507]
y_new = [0.062233, 0.032617, 0.051279, 0.109516, 0.116023, 0.016735, 0.018247, 0.018849, 0.017040, 0.018050]
y_weak = [0.062131, 0.047390, 0.085957, 0.133552, 0.145713, 0.061524, 0.119788, 0.127015, 0.123166, 0.127056]
speeduplst = []

# strong scaling speedup
# for i in range(len(y)):
#     speedup = y_new[0] / y_new[i]
#     speeduplst.append(speedup)
    
    

# weak scaling speedup
for i in range(len(y)):
    speedup = y_weak[0] * x[i] / y_weak[i]
    speeduplst.append(speedup)

print(speeduplst)


# X_ = np.linspace(x.min(), x.max(), 500)
# Y_ = spline(X_)

# plt.plot(X_, Y_)
    
plt.plot(x, speeduplst, 'ro', label='Actual Speedup')
plt.plot(x, y_ideal, linestyle='dashed', label='Ideal Speedup')
# optimizedParameters, prov = opt.curve_fit(func, x, speeduplst)
plt.xlabel('Number of Processors')
plt.ylabel('Speedup')
plt.legend()
plt.show()

# weak scaling
