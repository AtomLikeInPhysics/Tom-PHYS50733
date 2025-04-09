def trapRule(f,a,b,N):
   h = (b-a)/N
   s = 0.5 * (f(a)+f(b))
   for i in range(1,N):
        s += f(a + i*h)
   return(h*s)

def simpson(f,a,b,N):
    h = (b-a)/N
    s = (f(a)+f(b))
    t = 0
    for i in range(2,N,2):
        s += 2 * f(a + i*h)
    for i in range(1,N,2):
        t += 2/3 * f(a + i*h)
    return(h* (1/3 * s + 2*t))