# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 23:54:19 2018

@author: zatkins
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import material_properties as mp

##define data
#define T
Ts = np.arange(1,25 + .25,.25) #Ts in K

#define thicknesses
ts = np.arange(1,10 + .25,.25)
ts = ts * 1e-3            #ts in mm

#define materials
materials = {"cr110": [mp.cr110, "CR 110"], "cr124": [mp.cr124, "CR 124"], 
"steelcast": [mp.steelcast, "Steelcast"], "tkram": [mp.tkram, "TK RAM"]}

#define i/o
fileout = "../../../Figures/Thermal/Absorber/"
if not os.path.exists(fileout):
    os.makedirs(fileout)

##plot
#figure 1, c(T) and k(T) for materials
fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (8, 6))
for material, (func, name)  in sorted(materials.items(), key = lambda elem: elem[1][1]):
    ax[0].plot(Ts, func(Ts, "c"), label = name)
    ax[1].plot(Ts, func(Ts, "k"), label = name)
    
ax[0].plot(Ts, mp.mf117(Ts, "c"), "k--", label = "MF 117")
ax[1].plot(Ts, mp.mf117(Ts, "k"), "k--", label = "MF 117")
    
ax[0].semilogx()
ax[0].semilogy()

ax[0].legend(ncol = 2, fontsize = 12, loc = 4)
ax[0].set_title("Specific Heat Capacity")
ax[0].set_ylabel("$c$ [J $\mathregular{kg^{-1}K^{-1}}$]")
ax[0].grid()

c1, c2 = mp.cr124(Ts, "c"), mp.cr110(Ts, "c")
ax[0].fill_between(Ts, c1, c2, where = c1 <= c2, color = ".75")

ax[1].semilogx()
ax[1].semilogy()

ax[1].legend(ncol = 2, fontsize = 12, loc = 4)
ax[1].set_title("Thermal Conductivity")
ax[1].set_ylabel("$k$ [W $\mathregular{m^{-1}K^{-1}}$]")
ax[1].grid()

ax[1].set_xlabel("$T$ [K]")
ax[1].set_xlim(left = 1, right = 25)

fig.savefig("../../../Figures/Thermal/Absorber/Thermal_Properties", bbox_inches = "tight")

#figure 2, tau/t^2 (T) for materials
fig, ax = plt.subplots(figsize = (8, 6))
for material, (func, name) in sorted(materials.items(), key = lambda elem: elem[1][1]):
    c = func(Ts, "c")
    rho = func(Ts, "rho")
    k = func(Ts, "k")
    ax.plot(Ts, c * rho * 1e-3 ** 2 / k, label = name) #normalized to 1mm

ax.plot(Ts, mp.mf117(Ts, "c") * mp.mf117(Ts, "rho") * 1e-3 ** 2 / mp.mf117(Ts, "k"), "k--", label ="MF 117")    

ax.semilogx()
ax.set_xlim(left = 1, right = 25)
ax.semilogy()

c1, c2 = mp.cr124(Ts, "c"), mp.cr110(Ts, "c")
k1, k2 = mp.cr124(Ts, "k"), mp.cr110(Ts, "k")
rho1, rho2 = mp.cr124(Ts, "rho"), mp.cr110(Ts, "rho")
ax.fill_between(Ts, c1 * rho1 * 1e-3 ** 2 / k1, c2 * rho2 * 1e-3 ** 2 / k2, 
                where = c1 * rho1 / k1 <= c2 * rho2 / k2, color = ".75")

ax.legend(ncol = 2, fontsize = 12, loc = 4)
ax.set_title(" Normalized Characteristic Thermal Time ")
ax.set_xlabel("$T$ [K]")
#ax.set_xlim(left = 4, right = 25)
ax.set_ylabel(r"$\tau_{settle}/t^2$ [s $\mathregular{mm^{-2}}$]")
ax.grid()

fig.savefig("../../../Figures/Thermal/Absorber/Thermal_Times_Temp", bbox_inches = "tight")

#figure 3, tau(t) at T = 15K for materials
fig, ax = plt.subplots(figsize = (8, 6))
T = [15]                      #arbitrary
for material, (func, name) in sorted(materials.items(), key = lambda elem: elem[1][1]):
    c = func(T, "c")
    rho = func(T, "rho")
    k = func(T, "k")
    ax.plot(ts * 1e3, c * rho * ts ** 2 / k, label = name) #normalized to 1mm 

ax.plot(ts * 1e3, mp.mf117(T, "c") * mp.mf117(T, "rho") * ts ** 2 / mp.mf117(T, "k"), "k--", label = "MF 117")     

ax.set_xlim(left = ts[0] * 1e3, right = ts[-1] * 1e3)
ax.semilogx()
ax.semilogy()

c1, c2 = mp.cr124(T, "c"), mp.cr110(T, "c")
k1, k2 = mp.cr124(T, "k"), mp.cr110(T, "k")
rho1, rho2 = mp.cr124(T, "rho"), mp.cr110(T, "rho")
ax.fill_between(ts * 1e3, c1 * rho1 * ts ** 2 / k1, c2 * rho2 * ts ** 2 / k2, 
                where = c1 * rho1 * ts ** 2 / k1 <= c2 * rho2 * ts ** 2 / k2, color = ".75")

ax.legend(ncol = 2, fontsize = 12, loc = 4)
ax.set_title("Characteristic Thermal Time at {}K".format(T[0]))
ax.set_xlabel("$t$ [mm]")
ax.set_ylabel(r"$\tau_{settle}$ [s]")
ax.grid()

fig.savefig("../../../Figures/Thermal/Absorber/Thermal_Times_t", bbox_inches = "tight")  
    
