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

def NewtonX(M,x,y):         #Newton gravitation in x
    return(-G*M/((np.sqrt(x**2 + y**2)**3)) * x)
def NewtonY(M,x,y):         #Newton gravitation in y
    return(-G*M/((np.sqrt(x**2 + y**2)**3)) * y)
    
def acc(r,t):               #acceleration equation for RK
        posX = r[0]
        velX = r[1]
        posY = r[2]
        velY = r[3]
        d_posX = velX
        d_posY = velY
        d_velX = NewtonX(Msun,posX,posY)
        d_velY = NewtonY(Msun,posX,posY)
        return np.array([d_posX, d_velX, d_posY, d_velY], float)


#Setting up the time steps
tstart = 0
tstop = 12*(3600*24*360)        #12 years (to allow jupiter to make one orbit)
N = (3600*24)       #One day time step
h = (tstop-tstart)/N
t = np.linspace(tstart,tstop,N)

def RK(r):          #Runge-Kutta IVP method
    xpnt = []
    ypnt = []
    VelX = []
    VelY = []
    for i in t:
        xpnt.append(r[0])
        ypnt.append(r[2])
        k1 = h*acc(r,i)
        k2 = h*acc(r + 0.5*k1, i + 0.5*h)
        k3 = h*acc(r + 0.5*k2, i + 0.5*h)
        k4 = h*acc(r + k3, i + h)
        r += 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        VelX.append(r[1])
        VelY.append(r[3])
    return([xpnt,ypnt],[VelX,VelY])

#Setting initial values in the form: [xpos,xvel,ypos,yvel]

EarthPos = np.array(RK([-1.27e11,15636,-8.21e10,-25181])[0])
EarthVel = np.array(RK([-1.27e11,15636,-8.21e10,-25181])[1])

MercuryPos = np.array(RK([2.12e9,38912,-6.96e10,4514])[0])
MercuryVel = np.array(RK([2.12e9,38912,-6.96e10,4514])[1])

JupPos = np.array(RK([3.03e10,-13206,7.64e11,1140])[0])
JupVel = np.array(RK([3.03e10,-13206,7.64e11,1140])[1])

#Solving for the kinetic and potential energy at each timestep

KinE = 0.5*Mearth*(np.sqrt(EarthVel[0]**2 + EarthVel[1]**2))**2
PEE = -G*(Mearth*Msun)/(np.sqrt(EarthPos[0]**2 + EarthPos[1]**2))

KinM = 0.5*Mmer*(np.sqrt(MercuryVel[0]**2 + MercuryVel[1]**2))**2
PEM = -G*(Mmer*Msun)/(np.sqrt(MercuryPos[0]**2 + MercuryPos[1]**2))

KinJ = 0.5*Mjup*(np.sqrt(JupVel[0]**2 + JupVel[1]**2))**2
PEJ = -G*(Mjup*Msun)/(np.sqrt(JupPos[0]**2 + JupPos[1]**2))

#Ploting the energies:

plt.plot(t,KinE,label = 'Kinetic Energy')
plt.plot(t,PEE, label = 'Potential Energy')
plt.plot(t,PEE+KinE, label = 'Total Energy')
plt.title('Energy Distribution of Earth')
plt.ylabel('Energy [J]')
plt.xlabel('Time')
plt.legend()
plt.savefig('Energy of Earth.png')
plt.show()


plt.plot(t,KinJ,label = 'Kinetic Energy')
plt.plot(t,PEJ, label = 'Potential Energy')
plt.plot(t,PEJ+KinJ, label = 'Total Energy')
plt.title('Energy Distribution of Jupiter')
plt.ylabel('Energy [J]')
plt.xlabel('Time')
plt.legend()
plt.savefig('Energy of Jupiter.png')
plt.show()

plt.plot(t,KinM,label = 'Kinetic Energy')
plt.plot(t,PEM, label = 'Potential Energy')
plt.plot(t,PEM+KinM, label = 'Total Energy')
plt.title('Energy Distribution of Mercury')
plt.ylabel('Energy [J]')
plt.xlabel('Time')
plt.legend()
plt.savefig('Energy of Mercury.png')
plt.show()


plt.scatter(0,0, c='b')
x, y = EarthPos[0], EarthPos[1]
plt.scatter(x, y, c='yellow')
plt.scatter(*MercuryPos, c='green')
plt.scatter(*JupPos,c='r')
plt.savefig('Static orbits.png')

#Making plots and frames for animation

fig, ax = plt.subplots()

Sunplt = ax.scatter(0,0)
x, y = EarthPos[0], EarthPos[1]
Earthplt = ax.scatter(x, y)
Mercplt = ax.scatter(*MercuryPos)
Jupplt = ax.scatter(*JupPos)

def update(frame):
    x1 = EarthPos[0][frame]
    y1 = EarthPos[1][frame]
    # update the scatter plot:
    data1 = [x1, y1]
    Earthplt.set_offsets(data1)

    x2 = MercuryPos[0][frame]
    y2 = MercuryPos[1][frame]
    # update the scatter plot:
    data2 = [x2, y2]
    Mercplt.set_offsets(data2)

    x3 = JupPos[0][frame]
    y3 = JupPos[1][frame]
    # update the scatter plot:
    data3 = [x3, y3]
    Jupplt.set_offsets(data3)
    return(Earthplt,Mercplt,Jupplt)

ani = animation.FuncAnimation(fig=fig, func=update, frames=N, interval=10)

# The writer is commented out because it makes it take like 15 minutes to run the code compared to 20 seconds. It does work if it is uncommented.

# writer = animation.PillowWriter(fps=60,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save('orbits.gif', writer=writer)
plt.show()