#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:23:13 2019

@author: zatkins
"""

import numpy as np
import cold_load_thermal_v1 as cl
import material_properties as mp
import matplotlib.pyplot as plt
import time

# useful shortcuts

pi = np.pi
Al = mp.Al_6061_T6
SS = mp.SS_304

# define data for G plot

socl_Al_A = pi * (6e-3 / 2)**2
socl_Al_L = 25.4e-3

act_Al_A = 8 * pi * (.35e-3 / 2)**2
act_Al_L = 1.5 * 3.25 * 25.4e-3

act_SS_A = 4 * pi * (6.35e-3 / 2)**2
act_SS_L = 3.25 * 25.4e-3

T = np.linspace(4, 300, 100)
T_cold = np.linspace(4, 25, 100)

# define data for cooldown time plot

feb18_t0 = time.mktime((2018, 2, 16, 11, 47, 16, 4, 47, 0))
jun19_t0 = time.mktime((2019, 6, 6, 10, 29, 48, 3, 157, 1))

feb18 = np.genfromtxt("/Users/zatkins/Desktop/2018-19/SO/Cold Load/Cold-Load/logs/cold_load_20180217.dat",
                      delimiter = "\t", names = True)
jun19 = np.genfromtxt("/Users/zatkins/Desktop/2018-19/SO/Cold Load/Cold-Load/logs/20190606_155939.csv",
                      delimiter = ",", names = True)
jun19b = np.genfromtxt("/Users/zatkins/Desktop/2018-19/SO/Cold Load/Cold-Load/logs/20190606_172538.csv",
                      delimiter = ",", names = True)

# define data for cooldown speed plot

win = 50

tcl = jun19['Control_T_K']
t = jun19['t']

dtcl = ((np.roll(tcl, -1) - tcl) / (np.roll(t, -1) - t))[:-1]
mdtcl = np.convolve(dtcl, np.ones(win)/win, mode = "valid")

# define data for G zoom plot

a = (.491/33.2)**2 * 25.2

b = (.759/33.2)**2 * 25.2

c = (.864/33.2)**2 * 25.2

A = a / (14.5 - 5.7)

B = (b - a) / (20 - 14.5)

C = (c - b) / (22 - 20)

# outputs

so_G = socl_Al_A / socl_Al_L * Al(T, "k")
act_G = act_Al_A / act_Al_L * Al(T, "k") + act_SS_A / act_SS_L * SS(T, "k")
act_G_cold = act_Al_A / act_Al_L * Al(T_cold, "k") + act_SS_A / act_SS_L * SS(T_cold, "k")

# cooldown time plot

fig, ax = plt.subplots()

ax.plot(T, so_G, label = "SO")
ax.plot(T, act_G, label = "ACT")
ax.plot(T, act_G / so_G, label = "ACT / SO Ratio")
ax.grid(which = "both")
ax.legend()
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("G [W/K] or Ratio [a.u.]")
ax.set_title("ACT vs. SO Cold Load Thermal Link")

# cooldown speed plot

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 6))

ax[0].plot(t[win//2:-win//2] - jun19_t0, mdtcl * 60e3)
ax[0].set_ylabel("dT/dt [mK / min]")
ax[0].set_xlabel("Cooldown Time [s]")
ax[0].set_ylim(-150, 0)
ax[0].grid(which = "both")

ax[1].plot(tcl[win//2:-win//2], mdtcl * 60e3, marker = ".", linestyle = "")
ax[1].set_ylabel("dT/dt [mK / min]")
ax[1].set_xlabel("Load Temperature [K]")
ax[1].semilogx()
ax[1].set_ylim(-150, 0)
ax[1].grid(which = "both")

fig.subplots_adjust(hspace = 0.3)

fig.suptitle("Cooldown Speed")

# G plot

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6), sharex = True)

ax.plot(feb18['time'] - feb18_t0, feb18['diode_A'], label = "CL, 2018 Feb 17")
ax.plot(jun19['t'] - jun19_t0, jun19['Control_T_K'], label = "CL, 2019 Jun 06")
ax.plot(jun19['t'] - jun19_t0, jun19['Sensor_T_K'], label = "4K filter clamp, 2019 Jun 06")
ax.plot(jun19b['t'] - jun19_t0, jun19b['Control_T_K'], label = "1K filter clamp, 2019 Jun 06")
ax.set_xlim(0, 5e5)
ax.set_ylabel("Temperature [K]")
ax.set_ylim(0, 300)
ax.legend()
ax.grid(which = 'both')

#x[0].set_title("Linear")

#ax.set_title("Log")
ax.semilogy()
ax.set_ylim(1, 300)
ax.set_xlabel("Cooldown Time [s]")
ax.set_title("Feb 2018 vs. Jun 2019 ACT Load Cooldown Time")

# G zoom plot

fig, ax = plt.subplots()
ax.plot(T_cold, act_G_cold, label = "$G_{theory}$")
ax.bar([5.7, 14.5, 20], [A, B, C], width = [14.5 - 5.7, 20 - 14.5, 22 - 20], align = "edge", 
       alpha = 0.25, edgecolor = "k", label = "$G_{meas,avg}$")
ax.grid(which = "both")
ax.legend()
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("G [W/K]")
ax.set_title("ACT Cold Load Thermal Link")

# tau plot

fig, ax = plt.subplots()
mask = mdtcl >= 0
ax.plot(tcl[win//2:-win//2][~mask], -1 / (mdtcl[~mask] / tcl[win//2:-win//2][~mask]),
        marker = ".", linestyle = "")
ax.set_ylim(1e4, 5e5)
ax.grid(which = "both")
ax.set_xlabel("Temperature [K]")
ax.set_ylabel(r"$\tau_{load}$ [s]")
ax.set_title("Characteristic Cooldown Time")
ax.semilogx()
ax.semilogy()