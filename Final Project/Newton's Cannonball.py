# Jupiter Mercury Earth

#imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Constants

Mearth = 6e24 # [kg]
Rearth = 6378.137e3 #[m]

G = 6.67e-11 # [N m^2 kg^-2]

d = 0.10*Rearth + Rearth    #Height of the mountain

vorb = np.sqrt(G*Mearth / d)    #Calculated orbital velocity
vesc = np.sqrt(2) * vorb        #Derived escape velocity(check paper)

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
        d_velX = NewtonX(Mearth,posX,posY)
        d_velY = NewtonY(Mearth,posX,posY)
        return np.array([d_posX, d_velX, d_posY, d_velY], float)

tstart = 0
tstop = (3600*2)
N = (3600)
h = (tstop-tstart)/N
t = np.linspace(tstart,tstop,N)

def RK(r):
    xpnt = []
    ypnt = []
    VelX = []
    VelY = []
    # r = np.array([1.5e11,0,0,29e3],float)
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


pos1, vel1 = np.array(RK([0, vorb, d, 0]))
pos2, vel2 = np.array(RK([0, vesc, d, 0]))
pos3, vel3 = np.array(RK([0, vorb*0.75, d, 0]))



plt.plot(t,np.sqrt(pos1[0]**2 + pos1[1]**2),label = 'Orbiting')
plt.plot(t,np.sqrt(pos2[0]**2 + pos2[1]**2),label = 'Escape')
plt.plot(t,np.sqrt(pos3[0]**2 + pos3[1]**2), label = '75% Orbital Velocity')
plt.axhline(y=Rearth, c='r')
plt.xlabel('Time [s]')
plt.ylabel('Height')
plt.title('Height of Newtons Cannonball')
plt.legend()
plt.savefig('Cannonball Height')
plt.show()

Earth = plt.Circle((0,0),Rearth)    #Making Earth

fig, ax = plt.subplots()

ax.set_aspect('equal')
ax.set_xlim(-2*Rearth, 2*Rearth)
ax.set_ylim(-2*Rearth, 2*Rearth)

x1, y1 = np.array(pos1[0]), np.array(pos1[1])
x2, y2 = np.array(pos2[0]), np.array(pos2[1])
x3, y3 = np.array(pos3[0]), np.array(pos3[1])

Cannonplt1 = ax.scatter(x1,y1, c='r')
Cannonplt2 = ax.scatter(x2,y2, c='g')
Cannonplt3 = ax.scatter(x3,y3, c='orange')
ax.add_patch(Earth)

crashed1 = False
crashed2 = False
crashed3 = False

crashed = {'1': False, '2': False, '3': False}

def update(frame):
    if np.sqrt(pos1[0][frame]**2 + pos1[1][frame]**2) > Rearth:
        if not crashed['1']:
            x1 = pos1[0][frame]
            y1 = pos1[1][frame]
            # update the scatter plot:
            data1 = [x1, y1]
            Cannonplt1.set_offsets(data1)
        else:
            data1 = [0, 0]
            Cannonplt1.set_offsets(data1)
    else:
        crashed['1'] = True

    if np.sqrt(pos2[0][frame]**2 + pos2[1][frame]**2) > Rearth:
        if not crashed['2']:
            x2 = pos2[0][frame]
            y2 = pos2[1][frame]
            # update the scatter plot:
            data2 = [x2, y2]
            Cannonplt2.set_offsets(data2)
        else:
            data2 = [0, 0]
            Cannonplt2.set_offsets(data2)
    else:
        crashed['2'] = True

    if np.sqrt(pos3[0][frame]**2 + pos3[1][frame]**2) > Rearth:
        if not crashed['3']:
            x3 = pos3[0][frame]
            y3 = pos3[1][frame]
            # update the scatter plot:
            data3 = [x3, y3]
            Cannonplt3.set_offsets(data3)
        else:
            data3 = [0, 0]
            Cannonplt3.set_offsets(data3)
    else:
        crashed['3'] = True
    return(Cannonplt1,Cannonplt2,Cannonplt3)

ani = animation.FuncAnimation(fig=fig, func=update, frames=3600, interval=0.1, repeat=False)
crashed = {'1': False, '2': False, '3': False}
# writer = animation.PillowWriter(fps=60,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=-1)
ani.save('CannonBall.gif', writer='pillow')
plt.show()