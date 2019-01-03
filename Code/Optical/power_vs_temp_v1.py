# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:23:04 2018

@author: zatkins
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

## define constants
h = 6.6261e-34
k = 1.3806e-23

##define data
# define midbands and bandwidths
bands = np.array([[30,30], [90, 45], [150, 45], [220, 60], [270, 60]])
bands = bands * 1e9

#define T
Ts = np.arange(.1, 25 + .1, .1)

## define functions
def old_power_integrand(v, f, T):
    return f(v) * h * v / (np.exp(h * v / (k * T)) - 1)
    
def old_power(f, T, cap = 1e12, lims = None):
    if lims is not None:
        res = integrate.quad(old_power_integrand, *lims, args = (f, T))
        if res [1] / res[0] > 1e-3:
            raise ValueError("Integration error > 1e-3")
        return res[0]
    else:
        res = integrate.quad(old_power_integrand, 0, cap, args = (f, T))
        if res [1] / res[0] > 1e-3:
            raise ValueError("Integration error > 1e-3")
        return res[0]
    
def simple_power(f, T, cap = 1e12, lims = None):
    if lims is not None:
        res = integrate.quad(f, *lims)
        if res [1] / res[0] > 1e-3:
            raise ValueError("Integration error > 1e-3")
        return (k * T) * res[0]
    else:
        res = integrate.quad(f, 0, cap)
        if res [1] / res[0] > 1e-3:
            raise ValueError("Integration error > 1e-3")
        return (k * T) * res[0]
    
def f_tophat(v1, v2):
    def f(v):
        return (v1 <= v) * (v <= v2)
    return f, v1, v2
    
##main loop
out = np.zeros((len(Ts), len(bands), 3))

for i in range(len(Ts)):
    for j in range(len(bands)):
                
        f_res = f_tophat(bands[j, 0] - bands[j, 1]/2, bands[j, 0] + bands[j, 1]/2)
        f = f_res[0]     
        v1 = f_res[1]
        v2 = f_res[2]
        
        out[i, j, 0] = old_power(f, Ts[i], lims = (v1, v2)) * 1e12
        out[i, j, 1] = simple_power(f, Ts[i], lims = (v1, v2)) * 1e12
        out[i, j, 2] = out[i, j, 0] / out[i, j ,1] 
        
##plot
#figure 1, act power vs. linear power
fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (8, 6))        
for j in range(len(bands)):
    ax[0].plot(Ts, out[:, j, 0], label = "{0: .0f} GHz".format(bands[j][0] / 1e9))
    ax[1].plot(Ts, out[:, j, 1], label = "{0: .0f} GHz".format(bands[j][0] / 1e9))

ax[0].legend(ncol = 2, fontsize = 12, loc = 2)
ax[0].set_title("ACT Power")
ax[0].set_ylabel("$P_{opt}/\epsilon f_{fill}$ [pW]")
ax[0].grid()

ax[1].legend(ncol = 2, fontsize = 12, loc = 2)
ax[1].set_title("Linear Power")
ax[1].set_ylabel("$P_{opt}/\epsilon f_{fill}$ [pW]")
ax[1].grid()

ax[1].set_xlabel("$T$ [K]")
ax[1].set_xlim(left = 0, right = 25)
fig.savefig("../../../Figures/Optical/Power_Abs", bbox_inches = "tight")

#figure 2, ratio of act power to linear power
fig, ax = plt.subplots(figsize = (8, 6))
for j in range(len(bands)):
    ax.plot(Ts, out[:, j, 2], label = "{0: .0f} GHz".format(bands[j][0] / 1e9))

ax.legend(ncol = 2, fontsize = 12, loc = 4)
ax.set_title("ACT Power / Linear Power")
ax.set_xlabel("$T$ [K]")
ax.set_ylabel("[a.u.]")
ax.grid()
ax.set_xlim(left = 0, right = 25)

fig.savefig("../../../Figures/Optical/Power_Rel", bbox_inches = "tight")



        
        
        
    
       