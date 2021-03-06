# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 19:15:06 2018

@author: zatkins
"""

import sys
mat_props_path = "/Users/zatkins/Desktop/2018-19/SO/Cold Load/Cold-Load/Code"
if not mat_props_path in sys.path:
    sys.path.append(mat_props_path)

import Thermal.material_properties as mp

import os
import numpy as np
import scipy.constants as spc
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import minimize as minimize

run = True

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
    
def mat_dummy(T, mat, mat_args, mat_kwargs):
    return mat(T, *mat_args, **mat_kwargs)

def P_stage(T_c, T_h, A, L, mat, mat_args = (), mat_kwargs = {}): #dimensions in m
    return A / L * integrate.quad(mat_dummy, T_c, T_h, args = (mat, ("k", mat_args), mat_kwargs))[0]

def T_stage_series(P, T_c_0, A, L, mat, mat_args = (), mat_kwargs = {}): #dimensions in m
    
    def target(T_h, P, T_c, A, L, mat, mat_args, mat_kwargs):
        return (P - P_stage(T_c, T_h, A, L, mat, mat_args = mat_args, mat_kwargs = mat_kwargs)) ** 2
    
    if type(A) == int or type(A) == float:
        A = np.array([A])
    else:
        A = np.array(A)
        
    if type(L) == int or type(L) == float:
        L = np.array([L])
    else:
        L = np.array(L)
    
    if A.shape != L.shape or len(A.shape) != 1:
        raise ValueError("A and L must be 1-d sequence type if not scalar")
    
    n_stages = A.size
    out = np.full(n_stages, T_c_0, dtype = float)
    
    for i in range(n_stages):
        T_c = out[i - 1]
        res = minimize(target, 2 * T_c, args = (P, T_c, A[i], L[i], mat, mat_args, mat_kwargs))
        
        if res.success:
            out[i] = res.x[0]
        else:
            raise ValueError("minimize failed for P = {}, T_c = {}, A = {}, L = {}, mat = {}, mat_args = {}, mat_kwargs = {}"
                             .format(P, T_c, A[i], L[i], mat, mat_args, mat_kwargs))
    
    return out

def Q_stage(T_c, T_h, V, mat):
    return V * mat(T_h, "rho") * integrate.quad(mat, T_c, T_h, args = ("c"))[0]

def res_thermal_integrand(T, mat, mat_args, mat_kwargs):
    return mat(T, "k", *mat_args, **mat_kwargs) * mat(T, "res", *mat_args, **mat_kwargs)

def res_thermal(T_c, T_h, A, L, mat, mat_args = (), mat_kwargs = {}):
    return L / A * (integrate.quad(res_thermal_integrand, T_c, T_h, args = (mat, mat_args, 
        mat_kwargs))[0] / integrate.quad(mat_dummy, T_c, T_h, args = (mat, ("k", mat_args),
        mat_kwargs))[0])
    
###

if __name__ == "__main__" and run:
    
    ##define data
    T = np.arange(4, 25 + .1, .1)
    f = np.array([0.0, 0.2, 0.4, 0.8])
    
    wire_D = 6 / 1e3    #m
    mat = "Al"
    L = 25.4 / 1e3      #m
    A = spc.pi * (wire_D / 2)**2
    
    D = 198.0 / 1e3     #m
    G_spacer = G_13SP217(T, L = L)
    
    set_tau = 180       #sec

    filebase = "../../../Figures/Thermal/Load_v2/"
    if not os.path.exists(filebase):
        os.mkdirs(filebase)
    
    #
    m_dict = {"Cu": mp.Cu, "Al": mp.Al_6061_T6}
    
    #Load Thermal Properties assuming 4x, inch-long 13SP217 Nylon spacers
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (8, 6))
    for i in range(len(f)):
        ax[0].plot(T, C_load(T, f = f[i], d = D), label = "Al frac = {}".format(f[i]))
        ax[1].plot(T, C_load(T, f = f[i], d = D) / G_spacer, label = "Al frac = {}".format(f[i]))
    ax[0].legend(loc = 2)
    ax[0].set_title("$C_{load}$")
    ax[0].set_ylabel("$C$ [J $\mathregular{K^{-1}}$]")
    ax[0].semilogy()
    ax[0].grid(which = "both")
    
    ax[1].legend(loc = 2)
    ax[1].set_title(r"$\tau_{load}$, 4x25.4mm 13SP217 Spacers")
    ax[1].set_ylabel(r"$\tau$ [s]")
    ax[1].set_xlabel("$T$ [K]")
    #ax[1].semilogy()
    ax[1].set_xlim(4, 25)
    ax[1].grid(which = "both")
    
    fig.savefig(filebase + "Load_thermal_{}".format(int(D * 1e3)), bbox_inches = "tight")
    
    #Required diameters for thermal conductance of given material to achieve tau
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (8, 6))
    for i in range(len(f)):
        ax[0].plot(T, G_solver_A(T, L, set_tau, f = f[i], d = D,
          material = m_dict[mat])[0]*1e3, label = "Al frac = {}".format(f[i]))
        
        out = np.zeros(len(T))
        for j in range(len(T)):   
            out[j] = P_stage(4, T[j], G_solver_A(T[j], L, set_tau, f = f[i], d = D,
               material = m_dict[mat])[1], L, m_dict[mat])
        ax[1].plot(T, out, label = "Al frac = {}".format(f[i]))
    
    ax[0].legend(loc = 2)
    ax[0].set_ylabel("Diameter [mm]")
    ax[0].set_ylim(bottom = 0)
    ax[0].set_title(r"{} Wire Diameter to Achieve $\tau = {}s$".format(mat, set_tau))
    ax[0].grid(which = "both")
    
    ax[1].legend(loc = 2)
    ax[1].set_xlabel("$T$ [K]")
    ax[1].set_ylabel("Power [W]")
    ax[1].semilogy()
    ax[1].set_title(r"Power Source at Constant $T$ ($\tau = {}s$)".format(set_tau))
    ax[1].set_xlim(4, 25)
    ax[1].semilogx()
    ax[1].grid(which = "both")
    
    fig.savefig(filebase + "{}_Standoff_Thermal_{}_const_tau".format(mat, int(D * 1e3)),
                bbox_inches = "tight")
    
    #Power and Tau at fixed wire
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (8, 6))
    out = np.zeros(len(T))
    for j in range(len(T)):   
        out[j] = P_stage(4, T[j], A, L, m_dict[mat])
    ax[0].plot(T, out)
     
    for i in range(len(f)):
        ax[1].plot(T, C_load(T, f = f[i], d = D) / (A / L * m_dict[mat](T, "k")),
          label = "Al frac = {}".format(f[i]))
    
    ax[0].set_ylabel("Power [W]")
    ax[0].semilogy()
    ax[0].set_title("Power Source at Constant $T$ ({} Wire Diameter = {} mm)".format(mat, 
      wire_D * 1e3))
    ax[0].grid(which = "both")
    
    ax[1].legend(loc = 4)
    ax[1].set_xlabel("$T$ [K]")
    ax[1].set_ylabel(r"$\tau$ [s]")
    ax[1].semilogy()
    ax[1].set_title(r"$\tau_{{load}}$ ({} Wire Diameter = {} mm)".format(mat, wire_D * 1e3))
    ax[1].set_xlim(4, 25)
    ax[1].grid(which = "both")
    
    fig.savefig(filebase + "{}_Standoff_Thermal_{}_const_wire".format(mat, int(D * 1e3)),
                bbox_inches = "tight")
    
       