#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:53:11 2019

@author: zatkins
"""

import time
start = time.perf_counter()

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

import solid_angle as sa

###
#define state variables

det = "UHF_F"
freq = "315"
phi = "both"

###
#prepare data structures

filein = "../../../Beams/"
fileout = "../../../Figures/Optical/Load_v2/beams/"
if not os.path.exists(fileout):
    os.makedirs(fileout)

det_names = {"LF_F": "LF_AdvACT_mag.csv", "MF_F": "v11_3_mag.csv", "UHF_F": "mag_UHF_v2_2.csv",
             "MF_L": "Realized Gain Plot 1 5p6mm pixel Single Layer AR.csv"}
det_regexes = {"LF_F": "mag\(rEL3X\) \[V\] - Freq='(.+)GHz' Phi='(.+)deg'", 
               "MF_F": "mag\(rEL3X\) \[V\] - Freq='(.+)GHz' Phi='(.+)deg'",
               "UHF_F": "mag\(rEL3X\) \[V\] - Freq='(.+)GHz' Phi='(.+)deg'",
               "MF_L": "RealizedGainTotal \[\] - Freq='(.+)GHz' LensMachineAngle='90' LR='0.48' Phi='(.+)deg'"}
det_mags = {"LF_F": True, "MF_F": True, "UHF_F": True, "MF_L": False}
name = det_names[det]
regex = det_regexes[det]
mag = det_mags[det]

beam = sa.make_beam(sa.load_beams(filein, name, regex), freq, phi, mag = mag)

###
#outputs

thetas = np.array(list(beam.keys())).astype(float)
out = np.sin(pi/180 * thetas) * np.array(list(beam.values()))

name = "beam_viewer_{}_{}GHz_{}phi".format(det, freq, phi)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
fig.suptitle("Convolved {}GHz Beam vs. Frequency (det = {}, phi = {})".format(freq, det, phi), fontsize = 16)

ax[0].plot(thetas[np.logical_and(0 <= thetas, thetas < 60)], out[np.logical_and(0 <= thetas, thetas < 60)])
ax[0].set_title("Near-Field")
ax[0].text(0.7, 0.8, "avg = {:.3f}".format(np.mean(out[np.logical_and(0 <= thetas, thetas < 60)])),
  transform=ax[0].transAxes)

ax[1].plot(thetas[thetas >= 60], out[thetas >= 60])
ax[1].set_title("Far-Field")
ax[1].text(0.7, 0.8, "avg = {:.3f}".format(np.mean(out[thetas >= 60])), transform=ax[1].transAxes)

for a in ax:
    a.set_ylabel("Convolved Power [a.u.]")
    a.set_xlabel("Theta [deg]")
    a.grid()
    
fig.savefig(fileout + name, bbox_inches = "tight")
