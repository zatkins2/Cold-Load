#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 13:58:35 2019

@author: zatkins
"""

import numpy as np
import cold_load_thermal_v1 as CL_therm

def time_step(T, delta_t, T_c, A, L, C_load, material = mp.Cu):
    P = -P_stage(T_c, T, A, L, material)
    Q = P * delta_t
    delta_T = Q / C_load(T_h)
    return delta_T
    
    