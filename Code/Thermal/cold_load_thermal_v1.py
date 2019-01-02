# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 19:15:06 2018

@author: zatkins
"""

import numpy as np
import scipy.constants as spc
import matplotlib.pyplot as plt
import material_properties as mp
import scipy.integrate as integrate

##define functions
def C_load(T, d = 180/1e3, h = 57.15/1e3, f = 0, th = 3.175/1e3, absorber = mp.mf117, 
           Al = mp.Al_6061_T6, Cu = mp.Cu): 
    T = np.array(T)    
    
    V_absorber = spc.pi * (d / 2)**2 * (h / 3)                          #pyramid
    c_absorber = absorber(T, "c") * absorber(T, "rho") * V_absorber * (1-f) #Al filling frac
    
    V_Al = spc.pi * (d / 2)**2 * th 
    c_Al = Al(T, "c") * Al(T, "rho") * (V_Al + f * V_absorber)
    
    V_Cu = V_Al
    c_Cu = Cu(T, "c") * Cu(T, "rho") * V_Cu
    
    return c_absorber + c_Al + c_Cu
    
def G_13SP217(T, L = 25.4/1e3, n_standoff = 4, material = mp.Nylon, **kwargs):
    T = np.array(T)
    
    A = spc.pi * ((.5/2)**2 - (.26/2)**2) * (2.54/100)**2
    
    return n_standoff * material(T, "k", **kwargs) * A / L

def G_solver_A(T, L, tau, material = mp.Cu, **kwargs):
    T = np.array(T)
    
    G  = C_load(T, **kwargs) / tau
    A = G * L / material(T, prop = "k")
    
    return 2 * np.sqrt(A / spc.pi), A
    
def P_stage(T_c, T_h, A, L, k, *args): #dimensions in mm
    return A/L * integrate.quad(k, T_c, T_h, args = args)[0]

##define data
T = np.arange(4, 25 + .1, .1)
f = np.array([0.0, 0.2, 0.4, 0.8])

D = 234.0 / 1e3        #m
G = G_13SP217(T)

filebase = "/Users/zatkins/Desktop/2018/SO/Cold Load/Figures/Thermal/Load_v2/"

#Load Thermal Properties assuming 4x, inch-long 13SP217 Nylon spacers
fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (8, 6))
for i in range(len(f)):
    ax[0].plot(T, C_load(T, f = f[i], d = D), label = "Al frac = {}".format(f[i]))
    ax[1].plot(T, C_load(T, f = f[i], d = D) / G, label = "Al frac = {}".format(f[i]))
ax[0].legend(loc = 2)
ax[0].set_title("$C_{load}$")
ax[0].set_ylabel("$C$ [J $\mathregular{K^{-1}}$]")
ax[0].grid()

ax[1].legend(loc = 2)
ax[1].set_title(r"$\tau_{load}$, 4x25.4mm 13SP217 Spacers")
ax[1].set_ylabel(r"$\tau$ [s]")
ax[1].set_xlabel("$T$ [K]")
#ax[1].semilogy()
ax[1].set_xlim(4, 25)
ax[1].grid()

fig.savefig(filebase + "Load_thermal_{}".format(int(D)), bbox_inches = "tight")

#Required square side lengths for thermal conductance of given material to achieve tau
fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (8, 6))
for i in range(len(f)):
    ax[0].plot(T, G_solver_A(T, 25.4/1e3, 180, f = f[i], d = D)[0]*1e3, label = "Al frac = {}".format(f[i]))
    
    out = np.zeros(len(T))
    for j in range(len(T)):   
        out[j] = P_stage(4, T[j], G_solver_A(T[j], 25.4/1e3, 180, f = f[i], d = D)[1],
        25.4/1e3, mp.Cu, "k")
    
    ax[1].plot(T, out, label = "Al frac = {}".format(f[i]))

ax[0].legend(loc = 2)
ax[0].set_ylabel("Diameter [mm]")
ax[0].set_ylim(bottom = 0)
ax[0].set_title(r"Diameter to Achieve $\tau = 180s$ for RRR = 50 Cu wire")
ax[0].grid()

ax[1].legend(loc = 2)
ax[1].set_xlabel("$T$ [K]")
ax[1].set_ylabel("Power [W]")
ax[1].set_title(r"Power Source at Constant $T$")
ax[1].set_xlim(4, 25)
ax[1].grid()

fig.savefig(filebase + "Standoff_Thermal_{}".format(int(D)), bbox_inches = "tight")


   