import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt

def model(t, T_0, T_b, t_0, tau):
    return T_0 * np.exp(-(t - t_0) / tau) + T_b

t = np.array([3500, 5750, 7500, 9750, 12500, 15000, 18000, 24250], dtype = float)
T = np.array([15, 14.6, 14.3, 14, 13.6, 13.2, 13, 12.6], dtype = float)

pIn = (0, 0, 0, 1000)
plt.plot(t, T, 'bo', label = 'data')
popt, pcov = spo.curve_fit(model, t, T, pIn, maxfev = 1000000)
print "\nT_0 = %5.1f \nT_b = %5.1f \nt_0 = %5.1f \ntau = %5.3f" % tuple(popt)
plt.plot(t, model(t, *popt), 'r-', label = 'fit')
plt.legend()
plt.show()

