import andrewtom.Integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

def func1(t):
    return(3*np.exp(t**2))
def func2(t):
    return(np.sin(t)*np.cos(2*t))
def func3(t):
    return(np.sin(t)*np.exp(t+4))

xvals = np.arange(0,5,0.1)

int1 = integrate.simpson(func1,0,xvals,1000)
int2 = integrate.trapRule(func2,0,xvals,1000)
int3 = integrate.simpson(func3,0,xvals,1000)


plt.plot(xvals,int1)
plt.show()
plt.plot(xvals,int2)
plt.show()
plt.plot(xvals,int3)
plt.show()
