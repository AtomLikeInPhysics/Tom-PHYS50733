# Jupiter Mercury Earth

#imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Constants
Msun = 2e30 # [kg]
Mearth = 6e24 # [kg]
Mjup = 2e27
Mmer = 3.3e23
G = 6.67e-11 # [N m^2 kg^-2]

#Governing Equations:

def NewtonX(M,x,y):
    return(-G*M/((np.sqrt(x**2 + y**2)**3)) * x)
def NewtonY(M,x,y):
    return(-G*M/((np.sqrt(x**2 + y**2)**3)) * y)
    
def acc(r,t):
        posX = r[0]
        velX = r[1]
        posY = r[2]
        velY = r[3]
        d_posX = velX
        d_posY = velY
        d_velX = NewtonX(Msun,posX,posY)
        d_velY = NewtonY(Msun,posX,posY)
        return np.array([d_posX, d_velX, d_posY, d_velY], float)

def RK(r):
    tstart = 0
    tstop = 12*(3600*24*360)
    N = (3600*24)
    h = (tstop-tstart)/N
    xpnt = []
    ypnt = []
    t = np.linspace(tstart,tstop,N)
    # r = np.array([1.5e11,0,0,29e3],float)

    for i in t:
        xpnt.append(r[0])
        ypnt.append(r[2])
        k1 = h*acc(r,i)
        k2 = h*acc(r + 0.5*k1, i + 0.5*h)
        k3 = h*acc(r + 0.5*k2, i + 0.5*h)
        k4 = h*acc(r + k3, i + h)
        r += 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return(xpnt,ypnt)

EarthPos = RK([1.5e11,0,0,29e3])
MercuryPos = RK([5.79e10,0,0,47.36e3])
JupPos = RK([7.78e11,0,0,13.06e3])

fig, ax = plt.subplots()


def update(frame):
    # for each frame, update the data stored on each artist.
    x1 = EarthPos[0][:frame]
    y1 = EarthPos[1][:frame]
    # update the scatter plot:
    data = np.stack([x1, y1]).T
    Earthplt.set_offsets(data)

    x2 = MercuryPos[0][:frame]
    y2 = MercuryPos[1][:frame]
    # update the scatter plot:
    data = np.stack([x2, y2]).T
    Mercplt.set_offsets(data)

    x3 = JupPos[0][:frame]
    y3 = JupPos[1][:frame]
    # update the scatter plot:
    data = np.stack([x3, y3]).T
    Jupplt.set_offsets(data)

    # update the line plot:
    return (Earthplt,Mercplt,Jupplt)

alpha = 1

Sunplt = ax.scatter(0,0)
Earthplt = ax.scatter(*EarthPos)
Mercplt = ax.scatter(*MercuryPos)
Jupplt = ax.scatter(*JupPos)

print(len(EarthPos[1]))

ani = animation.FuncAnimation(fig=fig, func=update, frames=86400, interval=1)
plt.show()