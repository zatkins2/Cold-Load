#!usr/bin/python3
#author: zaktins
#date: 2019/07/12
#
#this script takes a generic thermometer logfile and compares it to an oxford extracted logfile using vcl_extraction_20190704.py to generate a thermometer calibration curve. script must be supplied with the path to the oxford files, the oxford filenames, the path to the thermometer raw file, the thermometer raw filenames, and any offset between the file clocks. file formatting is handled below at helper function level

import numpy as np
import numpy.ma as ma
from scipy import interpolate
import time
import sys
import os

#helper functions

def raw2target_interp_inds(raw_t, target_t):
    """
    Returns the indices of target data points on or between which raw data points lie.
    """
    out = ma.empty((len(raw_t), 2), dtype = int)
    out.mask = False
    
    r_inds = ma.arange(len(raw_t))
    r_inds.mask = False
    
    t_inds = np.arange(len(target_t))
    
    #if the same time, want index of that exact time, not a pair of distinct times
    bullzeyes, r_i, t_i = np.intersect1d(raw_t, target_t, return_indices = True)
    out[:, 0][r_i] = t_i
    out[:, 1][r_i] = t_i
    r_inds[r_i] = ma.masked

    #if diff times, want indeces of bounding times
    ##WANT TO MAKE MORE PYTHONIC##
    for i in np.nonzero(~r_inds.mask)[0]:
        dt = raw_t[i] - target_t
        
        if np.all(dt > 0) or np.all(dt < 0):
            out[i] = ma.masked
            continue
        
        out[i, 0] = t_inds[dt > 0][np.argmin(dt[dt > 0])]
        out[i, 1] = t_inds[dt < 0][np.argmax(dt[dt < 0])]
        
    return out

def raw_target2interpolated_raw(raw, target, method = 'linear', **interp1d_kwargs):
    """
    Using target col0, col1 as interpolation generating data, interpolate the raw col0 values.
    
    raw: shape = (N, 2) array
    target: same as raw
    method: optional, shares domain with 'kind' kwarg in scipy.interpolate.interp1d
    interp1d_kwargs: optional, any other kwargs to pass to scipy.interpolate.interp1d
    """
    f = interpolate.interp1d(target[:, 0], target[:, 1], kind = method, **interp1d_kwargs)
    return np.stack((raw[:, 0], f(raw[:, 0]))).T

def raw_target2masked(raw, target, raw_t_to_subtract = 0, dt_min = 0, dT_max = np.inf, return_masks = False):
    """
	Reads in raw timeseries and calibration target timeseries, and returns a calibrated temp vs. raw sensor value scatter plot.

	Optional dt_min, dT_max arguments filter the raw and target data based on the target data alone. Masks data points when all previous temperatures were not within dT_max Kelvin for at least the previous dt_min seconds.

	raw: 2d numpy array, axis-0 = data points, axis-1 = (time, value) pairs, with time in seconds
	target: same format as raw, with temp in Kelvin
	raw_t_to_subtract: optional, float
    dt_min: optional, float
    dT_max: optional, float
    return_masks: optional, bool. if True, returns raw, target, mask_raw, mask_target
    """
    
    target_t = target[:, 0]
    target_T = target[:, 1]
    raw_t = raw[:, 0]
    
    #correct for clock offset
    raw[:, 0] -= raw_t_to_subtract
    
    #mask target data (inclusive in both t and T bounds)
    mask_target = np.full(target.shape[0], False)
    
    mask_target[target_t - target_t[0] < dt_min] = True
    
    ##WANT TO MAKE MORE PYTHONIC##
    for i in np.nonzero(~mask_target)[0]:
        dt = target_t[i] - target_t
        mask_target[i] = np.any(np.absolute(target_T[i] - target_T[np.logical_and(dt > 0, 
        dt <= dt_min)]) > dT_max)
    
    #mask raw data. raw data valid only if timestamp between (or on) two (one) valid target timestamps. 
    mask_raw = np.full(raw.shape[0], False)
   
    raw_interp_inds = raw2target_interp_inds(raw_t, target_t)
    mask_raw[raw_interp_inds.mask[:, 0]] = True
    
    mask_raw[~mask_raw] = np.any(mask_target[raw_interp_inds[~raw_interp_inds.mask[:, 0]]], axis = 1)
    
    if not return_masks:
        return raw[~mask_raw], target[~mask_target]
    else: 
        return raw[~mask_raw], target[~mask_target], mask_raw, mask_target
    
def raw_target2cal_scatter(raw, target, raw_t_to_subtract = 0, dt_min = 0, dT_max = np.inf, interpolation_method = 'linear'):
    """
    
    interpolation_method: method of projecting raw data onto target timeseries when the timestamps are different, string

    """