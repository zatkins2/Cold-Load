# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 15:21:15 2018

@author: zatkins
"""

import numpy as np
from scipy.constants import pi
import csv
import re

#define functions
def theta_inv_cdf(y):
    return np.arccos(1 - 2*y)
    
def point_sample(N, domain = "omega"):
    points = np.zeros((N, 2))
    if domain == "omega":
        points[:, 0] = theta_inv_cdf(np.random.random(N))   #theta
        points[:, 1] = np.random.random(N) * 2*pi           #phi
    elif domain == "theta":
        points[:, 0] = np.random.random(N) * pi             #theta
        points[:, 1] = np.random.random(N) * 2*pi           #phi 
    else:
        raise ValueError("domain must be 'omega' or 'theta'")
    return points
    
def jacobian(points, domain = "omega"):
    N = points.shape[0]
    if domain == "omega":
        return np.ones(N)
    elif domain == "theta":
        return np.sin(points[:, 0])
    else:
        raise ValueError("domain must be 'omega' or 'theta'")
           
def normalized_int(f, points, mask, domain = "omega", f_kwargs = {}):
    num = np.sum(f(*points.T, **f_kwargs)[mask] * jacobian(points, domain = domain)[mask])
    den = np.sum(f(*points.T, **f_kwargs) * jacobian(points, domain = domain)) 
    return num / den
    
def sigma_normalized_int(f, p, points, domain = "omega", f_kwargs = {}):
    num = np.sum((f(*points.T, **f_kwargs) * jacobian(points, domain = domain))**2 * p * (1 - p))
    den = np.sum(f(*points.T, **f_kwargs) * jacobian(points, domain = domain))**2
    return np.sqrt(num / den)

def sample_sigma_normalized_int(N_sims, f, points, mask, domain = "omega", f_kwargs = {}):
    out = np.zeros(N_sims)
    sub_N = points.shape[0] // N_sims
    
    for i in range(N_sims):
        out[i] = normalized_int(f, points[i * sub_N : (i + 1) * sub_N],
           mask = mask[i * sub_N : (i + 1) * sub_N], domain = domain, f_kwargs = f_kwargs)
    
    return (np.std(out, ddof = 1) / np.sqrt(N_sims), out)
    
def load_beams(fpath, fname, header_re):
    out = {}
    with open (fpath + fname) as csv_data:
        reader_data = csv.DictReader(csv_data)
        for row in reader_data:
            theta = row["Theta [deg]"]
            
            for col in row:
                if col != "Theta [deg]":
                    if re.search(header_re, col) is not None:
                        freq, phi = re.search(header_re, col).groups()
                        if freq not in out:
                            out[freq] = {}
                        if phi not in out[freq]:
                            out[freq][phi] = {}
                        out[freq][phi][theta] = float(row[col])
    return out

def make_beam(beams, freq, phi):
    if phi == "both":
        beam_E = {k: v**2 for k, v in beams[freq]["0"].items()}
        beam_H = {k: v**2 for k, v in beams[freq]["90"].items()}
        beam_out = {}
        if beam_E.keys() != beam_H.keys():
            raise ValueError("beam_E keys != beam_H keys")
            
        for k in beam_E:
            beam_out[k] = beam_E[k] + beam_H[k]
        
    else:
        beam_out = {k: v**2 for k, v in beams[freq][phi].items()}
    return beam_out
    

def f_beam(theta, phi, beam = None, **kwargs):
    if type(theta) is not np.ndarray:
        theta_deg = np.array([180 / pi * theta])
    else:
        theta_deg = 180 / pi * theta
       
    out = np.zeros(len(theta_deg))
    low = np.floor(theta_deg)
    high = np.floor(theta_deg + 1)
    f_low = np.zeros(len(low))
    f_high = np.zeros(len(high))
    
    mask = theta_deg != low
    
    f_low = np.array([beam[t] for t in low.astype(int).astype(str)])
    f_high[mask] = np.array([beam[t] for t in high[mask].astype(int).astype(str)])
    
    out[mask] = f_low[mask] + (f_high[mask] - f_low[mask]) / (high[mask] - low[mask]) * (theta_deg[mask] - low[mask])
    out[~mask] = f_low[~mask]
    
    return out
    
def f_I(theta, phi, **kwargs):
    N = theta.shape[0]
    return np.ones(N)