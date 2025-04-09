# Jupiter Mercury Earth

#imports
import numpy as np
import matplotlib.pyplot as plt

#Constants
Msun = 2e30 # [kg]
Mearth = 6e24 # [kg]
G = 6.67e-11 # [N m^2 kg^-2]

#Governing Equations:

def Newton(M,m,r):
    return(G*M*m/(r^2))

def acc(r,M,m,t):
    pos = r[0]
    vel = r[1]
    d_pos = vel
    d_vel = Newton(M,m,pos)
    return np.array([d_pos, d_vel], float)