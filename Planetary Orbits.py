# Jupiter Mercury Earth

#imports
import numpy as np
import matplotlib.pyplot as plt

#Constants
Msun = 2e30 # [kg]
Mearth = 6e24 # [kg]
Mjup = 2e27
Mmer = 3.3e23
G = 6.67e-11 # [N m^2 kg^-2]

#Governing Equations:

def Newton(M,m,x,y):
    return(G*M*m/(np.sqrt(x**2 + y**2))**3)

def acc(r,M,m,t):
    pos = r[0]
    vel = r[1]
    d_pos = vel
    d_vel = Newton(M,m,pos)
    return np.array([d_pos, d_vel], float)

class Body(object):
    def __init__(self,xpos,ypos,xvel,yvel,M):
        self.xpos = xpos
        self.ypos = ypos
        self.xvel = xvel
        self.yvel = yvel
        self.M = M
    def acc(self,r,t):
        pos = r[0]
        vel = r[1]
        d_pos = vel
        d_vel = (Newton(Msun,self.M,self.xpos,self.ypos)/self.M) * ([self.xpos,self.ypos])
        return np.array([d_pos, d_vel], float)

Sun = Body(0,0,0,0,Msun)
Earth = Body(1,0,0,0,Mearth)
Mercury = Body(0.39,0,0,0,Mmer)
Jupiter = Body(5.2,0,0,0,Mjup)

tlin = np.linspace(0,100,100)

Ex = Earth.acc([1,0],tlin)

print(Ex)


plt.scatter(Sun.xpos,Sun.ypos)
plt.scatter(Earth.xpos,Earth.ypos)
plt.scatter(Mercury.xpos,Mercury.ypos)
plt.scatter(Jupiter.xpos,Jupiter.ypos)
plt.show()