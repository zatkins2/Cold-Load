# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 10:37:32 2019

@author: zatki
"""

import time
start = time.perf_counter()

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
from scipy.constants import pi

import solid_angle as sa

#define input/output
filein = "../../../Beams/"
name = "v11_3_mag.csv"
regex = "mag\(rEL3X\) \[V\] - Freq='(.+)GHz' Phi='(.+)deg'"

fileout = "../../../Figures/Optical/Load_v2/f_beam/"
##
beams = sa.load_beams(filein, name, regex)
if not os.path.exists(fileout):
    os.makedirs(fileout)
    
#define data
phi = "0"
freq_set_spacing = 5

D = 198.0
H = 45.0

##
freqs_str = np.array(list(beams.keys()))[::freq_set_spacing]
freqs = freqs_str.astype(float)

#define parameters
N = int(1e5)
N_sims = int(1e2)
domain = "omega"

max_r = 72
num_rs = 4

min_D = 180
max_D = 240
D_spacing = 10
slice_r = 72

##
rs = np.linspace(0, max_r, num_rs)
Ds = np.arange(min_D, max_D + D_spacing, D_spacing)

#data structures
out_r = np.zeros((num_rs, len(freqs)))
sigma_out_r = np.zeros((num_rs, len(freqs)))

out_D = np.zeros((len(Ds), len(freqs)))
sigma_out_D = np.zeros((len(Ds), len(freqs)))

#simulate
for freq in range(len(freqs)):
    points = sa.point_sample(N, domain = domain)
    planar_points = np.zeros(points.shape)
    
    r = H * np.tan(points[:, 0])
    planar_points[:, 0] = r * np.cos(points[:, 1])
    planar_points[:, 1] = r * np.sin(points[:, 1])
        
    beam = {k: v**2 for k, v in beams[freqs_str[freq]][phi].items()}
            
    for i in range(len(rs)):
        circ = path.Path.circle(center = (rs[i], 0.0), radius = D/2)
        
        mask = circ.contains_points(planar_points)
        mask[points[:, 0] > pi/2] = False
        
        out_r[i, freq] = sa.normalized_int(sa.f_beam, points, mask, domain = domain,
            f_kwargs = {"beam": beam})
        sigma_out_r[i, freq] = sa.sample_sigma_normalized_int(N_sims, sa.f_beam, points,
                 mask, domain = domain, f_kwargs = {"beam": beam})[0]
        
    for i in range(len(Ds)):
        circ = path.Path.circle(center = (slice_r, 0.0), radius = Ds[i]/2)
        
        mask = circ.contains_points(planar_points)
        mask[points[:, 0] > pi/2] = False
        
        out_D[i, freq] = sa.normalized_int(sa.f_beam, points, mask, domain = domain,
            f_kwargs = {"beam": beam})
        sigma_out_D[i, freq] = sa.sample_sigma_normalized_int(N_sims, sa.f_beam, points,
                 mask, domain = domain, f_kwargs = {"beam": beam})[0]
        
out_r *= 100
sigma_out_r *= 100

out_D *= 100
sigma_out_D *= 100

##
#plot f_beam(r, freq)
name = "f_beam(r, freq)_MF_{}phi_{}mm_{}_{}".format(phi, int(D), N, domain)
fig, ax = plt.subplots(figsize = (8, 6))
ax.set_title("Cold Load $f_{{beam}}$ vs. Frequency, Detector Location (CL_D = {}, N = {})".format(int(D), N))
for i in range(len(rs)):
    ax.plot(freqs, out_r[i], label = "Detector Location Radius = {} mm".format(rs[i]))
    ax.fill_between(freqs, out_r[i] - sigma_out_r[i], out_r[i] + sigma_out_r[i], alpha = 0.5)
ax.set_ylabel("$f_{beam}$ [%]")
ax.grid()
ax.set_xlim(np.min(freqs), np.max(freqs))
ax.set_xlabel("Frequency [GHz]")
plt.legend(loc = 4)
fig.savefig(fileout + name, bbox_inches = "tight")

#plot f_beam(D, freq, r = 72)
name = "f_beam(D, freq)_MF_{}mm_{}phi_{}_{}".format(slice_r, phi, N, domain)
fig, ax = plt.subplots(figsize = (8, 6))
ax.set_title("Cold Load $f_{{beam}}$ vs. Frequency, Load Diameter (DL_R = {}, N = {})".format(int(slice_r), N))
for i in range(len(Ds)):
    ax.plot(freqs, out_D[i], label = "Load Diameter = {} mm".format(Ds[i]))
    ax.fill_between(freqs, out_D[i] - sigma_out_D[i], out_D[i] + sigma_out_D[i], alpha = 0.5)
ax.set_ylabel("$f_{beam}$ [%]")
ax.grid()
ax.set_xlim(np.min(freqs), np.max(freqs))
ax.set_xlabel("Frequency [GHz]")
plt.legend(loc = 4)
fig.savefig(fileout + name, bbox_inches = "tight")

end = time.perf_counter()
print("Runtime = ", end - start)