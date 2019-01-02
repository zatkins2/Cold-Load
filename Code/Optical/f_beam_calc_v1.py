# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 22:38:27 2018

@author: zatkins
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
from scipy.constants import pi

import solid_angle as sa

#define data
vertices = np.array([[0, 100, 100, 40, 100, 100, 0, 0, 60, 0, 0],
                     [0, 0, 40, 40, 160, 200, 200, 160, 160, 40, 0]]).T
D = 234.0                    #load diameter in mm

max_r = 100.0                #radii in mm
H = 45.0                     #detector-load normal distance in mm

#define parameters
N = int(1e5)
domains = ["omega", "theta"]

freq, phi = ["150", "0"]
f = sa.f_beam

#define input/output
filein = "/Users/zatkins/Desktop/2018/SO/Cold Load/Beams/"
name = "v11_3_mag.csv"
regex = "mag\(rEL3X\) \[V\] - Freq='(.+)GHz' Phi='(.+)deg'"

colors = ['b', 'r']
fileout = "/Users/zatkins/Desktop/2018/SO/Cold Load/Figures/Optical/Load_v2/f_beam/"

##
#data structures
beam = sa.load_beams(filein, name, regex)[freq][phi]**2

rs = np.linspace(0, max_r, max_r + 1)

out_circ = np.zeros((len(domains), len(rs)))
sigma_out_circ = np.zeros((len(domains), len(rs)))

out_poly = np.zeros((len(domains), len(rs)))
sigma_out_poly = np.zeros((len(domains), len(rs)))

points = np.zeros((len(domains), N, 2))
planar_points = np.zeros((len(domains), N, 2))

mask_circ = np.zeros((len(domains), N), dtype = bool)
mask_poly = np.zeros((len(domains), N), dtype = bool)

#iterate over domains
for k in range(len(domains)):
    for i in range(len(rs)):
        #transform points
        points[k] = sa.point_sample(N, domain = domains[k])
        
        r = H * np.tan(points[k, :, 0])
        
        planar_points[k, :, 0] = r * np.cos(points[k, :, 1])
        planar_points[k, :, 1] = r * np.sin(points[k, :, 1])
        
        #simulate
        circ_path = path.Path.circle(center = (rs[i], 0.0), radius = D/2)       
        poly_path = path.Path(np.array([vertices[:, 0] + rs[i], vertices[:, 1]]).T, 
                              closed = True)        
                
        mask_circ[k] = circ_path.contains_points(planar_points[k])
        mask_poly[k] = poly_path.contains_points(planar_points[k])
        
        mask_circ[k][points[k, :, 0] > pi/2] = False
        mask_poly[k][points[k, :, 0] > pi/2] = False
        
        out_circ[k, i] = sa.normalized_int(f, points[k], mask_circ[k], domain = domains[k],
            f_kwargs = {"beam": beam})
        sigma_out_circ[k, i] = sa.sigma_normalized_int(f, out_circ[k, i], points[k], 
            domain = domains[k], f_kwargs = {"beam": beam})
    
out_circ *= 100
sigma_out_circ *= 100
    
##
#plot f_beam vs r
name = "f_beam(r)_MF_{}GHz_{}phi_{}mm_{}".format(freq, phi, int(D), N)
fig, ax = plt.subplots(figsize = (8, 6))
ax.set_title("Cold Load $f_{{beam}}$ vs. Detector Location (N = {})".format(N))
for i in range(len(domains)):
    ax.plot(rs, out_circ[i], color = colors[i])
    ax.fill_between(rs, out_circ[i] - sigma_out_circ[i], out_circ[i] + sigma_out_circ[i],
                    color = colors[i], alpha = 0.5)
ax.set_ylabel("$f_{beam}$ [%]")
ax.grid()
ax.set_xlim(np.min(rs), np.max(rs))
ax.set_xlabel("Radius (mm)")
fig.savefig(fileout + name, bbox_inches = "tight")

#plot graphical checks
sub_N = int(1e3)
subset = np.random.choice(N, size = sub_N, replace = False)
name = "Circle projection_MF_{}GHz_{}phi_{}mm_{}".format(freq, phi, int(D), sub_N)
fig, ax = plt.subplots(nrows = len(domains), ncols = 1, figsize = (8, 6))
fig.suptitle("Planar Projection Check, Circle (N = {})".format(sub_N), fontsize = 16)
for i in range(len(domains)):
    subsubset = np.intersect1d(np.where(mask_circ[i])[0], subset)
    ax[i].plot(planar_points[i, subsubset, 0], planar_points[i, subsubset, 1],
    color = colors[i], marker = "o", linestyle = '')
    ax[i].add_patch(patches.PathPatch(circ_path, fill = False))
    ax[i].axis("equal")
    ax[i].set_ylabel("$y$ (mm)")
    ax[i].set_title("Uniform sample over {}".format(domains[i]))
    ax[i].grid()
ax[-1].set_xlabel("$x$ (mm)")
fig.savefig(fileout + name, bbox_inches = "tight")

#Z
sub_N = int(5e3)
subset = np.random.choice(N, size = sub_N, replace = False)
name = "Polygon projection_{MF_{}GHz_{}phi_{}mm_{}".format(freq, phi, int(D), sub_N)
fig, ax = plt.subplots(nrows = len(domains), ncols = 1, figsize = (8, 6))
fig.suptitle("Planar Projection Check, Polygon (N = {})".format(sub_N), fontsize = 16)
for i in range(len(domains)):
    subsubset = np.intersect1d(np.where(mask_poly[i])[0], subset)
    ax[i].plot(planar_points[i, subsubset, 0], planar_points[i, subsubset, 1],
    color = colors[i], marker = "o", linestyle = '')
    ax[i].add_patch(patches.PathPatch(poly_path, fill = False))
    ax[i].axis("equal")
    ax[i].set_ylabel("$y$ (mm)")
    ax[i].set_title("Uniform sample over {}".format(domains[i]))
    ax[i].grid()
ax[-1].set_xlabel("$x$ (mm)")
fig.savefig(fileout + name, bbox_inches = "tight")