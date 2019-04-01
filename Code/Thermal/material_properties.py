# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 18:45:39 2018

@author: zatkins
"""

import numpy as np
import matplotlib.pyplot as plt

def cr110(T, prop):
    T = np.array(T)
    if prop == "k":
        return .08 * np.ones(T.size)
    if prop == "c":
        return .6 * T ** 2.05
    if prop == "rho":
        return 1.6e3
        
def mf117(T, prop):
    T = np.array(T)
    if prop == "k":
        to_return = np.zeros(T.size)
        mask = cr110(T, "k") < cr124(T, "k")
        to_return[mask] = cr110(T[mask], "k")
        to_return[~mask] = cr124(T[~mask], "k")
        return to_return
    if prop == "c":
        return .12 * T ** 2.06
    if prop == "rho":
        return cr110(T, prop) + (5. / 6) * (cr124(T, prop) - cr110(T, prop))
    
def cr124(T, prop):
    T = np.array(T)
    if prop == "k":
        to_return = np.zeros(T.size)
        mask = T <= 2.3
        to_return[mask] = .0038 * T[mask] ** 2
        to_return[~mask] = .0057 * T[~mask] ** 1.4  #discontinous but whatevs
        return to_return
    if prop == "c":
        return .132 * T + .0053 * T ** 3
    if prop == "rho":
        return 4.6e3
    
def steelcast(T, prop):
    T = np.array(T)
    if prop == "k":
        return .0075 * T
    if prop == "c":
        return .55 * T ** 2
    if prop == "rho":
        return 3.9e3
    
def tkram(T, prop):  #this is just polypropylene
    T = np.array(T)
    if prop == "k":
        return .14 * np.ones(T.size)
    if prop == "c":
        return 1900 * np.ones(T.size)
    if prop == "rho":
        return .855e3
        
def Al_6061_T6(T, prop):
    q = np.log10(np.array(T))
    if prop == "c":
        Q = 46.6467 - 314.292*q + 866.662*q**2 - 1298.3*q**3 + 1162.27*q**4 - 637.795*q**5 + 210.351*q**6 - 38.3094*q**7 + 2.96344*q**8
        return 10**Q
    if prop == "k":
        Q = 0.07918 + 1.0957*q - 0.07277*q**2 + 0.08084*q**3 + 0.02803*q**4 - 0.09464*q**5 + 0.04179*q**6 - 0.00571*q**7
        return 10**Q
    if prop == "rho":
        return 2.71e3
        
def Cu(T, prop, RRR = 50):
    RRR_array = np.transpose(np.array([[50,100,150,300,500],
[1.8743,2.2154,2.3797,1.357,2.8075],
[-0.41538,-0.47461,-0.4918,0.3981,-0.54074],
[-0.6018,-0.88068,-0.98615,2.669,-1.2777],
[0.13294,0.13871,0.13942,-0.1346,0.15362],
[0.26426,0.29505,0.30475,-0.6683,0.36444],
[-0.0219,-0.02043,-0.019713,0.01342,-0.02105],
[-0.051276,-0.04831,-0.046897,0.05773,-0.051727],
[0.0014871,0.001281,0.0011969,0.0002147,0.0012226],
[0.003723,0.003207,0.0029988,0,0.0030964]]))
    RRR_dict = dict()
    for i in range(np.size(RRR_array, 0)):
        RRR_dict[RRR_array[i, 0]] = RRR_array[i, 1:]
                                    
    if prop == "c":
        q = np.log10(np.array(T))
        Q = -1.91844 - 0.15973*q + 8.61013*q**2 - 18.996*q**3 + 21.9661*q**4 - 12.7328*q**5 + 3.54322*q**6 - 0.3797*q**7
        return 10**Q
    if prop == "k":
        a, b, c, d, e, f, g, h, i = RRR_dict[RRR]
        q = np.array(T)
        Q = (a + c*q**0.5 + e*q + g*q**1.5 + i*q**2) / (1 + b*q**0.5 + d*q + f*q**1.5 + h*q**2)
        return 10**Q
    if prop == "rho":
        return 8.96e3
        
def Nylon(T, prop):
    q = np.log10(np.array(T))
    if prop == "k":
        Q = -2.6135 + 2.3239*q - 4.7586*q**2 + 7.1602*q**3 - 4.9155*q**4 + 1.6324*q**5 - 0.2507*q**6 + 0.0131*q**7
        return 10**Q
    
def Manganin(T, prop):
    q = np.log10(np.array(T))
    k_coeffs = np.array([-1.124829e+00,  1.389400e+00,  3.331603e-01, -6.333958e-01,
       -5.728524e-01,  1.213246e+00,  4.618653e-01, -1.193638e+00,
        1.859476e-02,  5.463064e-01, -1.777981e-01, -7.122556e-02,
        5.432831e-02, -1.182376e-02,  8.901838e-04])
    if prop == "k":
        Q = 0
        for i in range(len(k_coeffs)):
            Q += k_coeffs[i] * q**i
        return 10**Q
    if prop == "res":
        return 48e-8

###
        
if __name__ == "__main__":    
        
    T = np.arange(4, 50 + .1, .1)
    materials = [Al_6061_T6, Cu, mf117]
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (8, 6))
    
    props = ["c"]
    for m in materials:
        for p in props:
            ax[0].plot(T, m(T, p) * m(T, "rho"), label = "{}_{}v".format(m.__name__, p))
    ax[0].set_title("Volumetric Heat Capacity")
    ax[0].set_ylabel("$C_{v}$ [J $\mathregular{K^{-1}m^{-3}}$]")
    ax[0].semilogy()
    #ax[0].ticklabel_format(axis = "y", style = "sci", scilimits = (0, 0))
    ax[0].legend()
    ax[0].grid()
    
    props = ["k"]
    for m in materials:
        for p in props:
            ax[1].plot(T, m(T, p), label = "{}_{}".format(m.__name__, p))       
    ax[1].set_title("Thermal Conductivity")
    ax[1].set_ylabel("$k$ [W $\mathregular{m^{-1}K^{-1}}$]")
    ax[1].semilogy()
    ax[1].set_xlabel("$T$ [K]")
    ax[1].set_xlim(4, 25)
    ax[1].legend()
    ax[1].grid()