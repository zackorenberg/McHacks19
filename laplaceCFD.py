# Simple Numerical Laplace Equation Solution using Finite Difference Method
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
pyplot = plt
numpy = np
# Boundary Condition
Tleft = 7
Tright = 20
Pop = 101325 # Pa
Runiv = 8.314 #J/molK
ArMol = 39.948 #molar weight of argon

# Set maximum iteration
maxIter = 50

# Set Dimension and delta
realX = 2.6 # centimeter
realY = 260
lenX = 20
lenY = 20 #we set it rectangular
deltaX = 1
deltaY = 1

# Boundary condition


# Initial guess of interior grid
Tguess = 0 #(Tleft + Tright)/2

# Set colour interpolation and colour map
colorinterpolation = 50
colourMap = plt.cm.jet #you can try: colourMap = plt.cm.coolwarm

# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX, deltaX), np.arange(0, lenY, deltaY))
Y = Y * (realY/lenY)
X = X * (realX/lenX)
# Set array size and set the interior value with Tguess
T = np.empty((lenX, lenY))
T.fill(Tguess)

# Set Boundary condition
#T[(lenY-1):, :] = Ttop
#T[:1, :] = Tbottom
T[:1, :] = T[(lenY-1):,:] = np.linspace(Tleft, Tright, lenX)
T[:, (lenX-1):] = Tright
T[:, :1] = Tleft

# Iteration (We assume that the iteration is convergence in maxIter = 500)
print("Please wait for a moment")
for iteration in range(0, maxIter):
    for i in range(1, lenX-1, deltaX):
        for j in range(1, lenY-1, deltaY):
            T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])
    #T[:1, :] = T[(lenY-1):,:] = np.linspace(Tleft, Tright, lenX)

print("Iteration finished")

# Configure the contour
plt.title("Contour of Temperature across double pane Window")

cp = plt.contourf(X, Y, T, colorinterpolation, vmin=0, vmax=25, cmap=colourMap)
plt.clim(0,25)
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
# Set Colorbar
cbar = plt.colorbar(cp, label="Temperature (Celsius)", extend='max')

Xcond = X
Ycond = Y

#plt.savefig("images/fig1.png")
# Show the result in the plot window
#plt.show()

density = (Pop)/((Runiv/ArMol)*T)


nx = 41
ny = 41
nt = 50
nit = 20
c = 1
norm = 2
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
X, Y = numpy.meshgrid(x, y)

rho = 1
nu = .1
dt = .001

u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx)) 
b = numpy.zeros((ny, nx))


def build_up_b(b, rho, dt, u, v, dx, dy):
    
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b


def pressure_poisson(p, dx, dy, b):
    pn = numpy.empty_like(p)
    pn = p.copy()
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])

        
        p[0,:] = p[1,:]
        p[-1,:] = p[-2,:]
		
		#p[-1, :] = 25# Thot  p[:, -2] # dp/dx = 0 at x = 2
        p[:,-1] = 25#p[:,1]   # dp/dy = 0 at y = 0
        #p[0,:] = 2#  Tcold   p[:, 1]   # dp/dx = 0 at x = 0
        p[:,0] = 5#p[:,-2] = 0    # p = 0 at y = 2
        
    return p


def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = numpy.empty_like(u)
    vn = numpy.empty_like(v)
    b = numpy.zeros((ny, nx))
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :]  = Tleft
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = Tright   # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
        
        
    return u, v, p


u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx))
b = numpy.zeros((ny, nx))
nt = 50
u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)


X = X
Y = Y
u = u
v = v

# normalize it to before plot
multiplier = np.max(Xcond)/norm
fig = pyplot.figure(figsize=(11,7), dpi=100)
# plotting the pressure field as a contour
#pyplot.contourf(X, Y, p, alpha=0.5, cmap=colourMap)  
#pyplot.colorbar()
pyplot.contourf(Xcond , Ycond, T, colorinterpolation, vmin=0, vmax=25, cmap=colourMap)
# plotting the pressure field outlines
#pyplot.contour(X, Y, p, cmap=colourMap)  
# plotting velocity field
#pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
pyplot.streamplot(X*multiplier,Y*multiplier*100 ,u*multiplier,v*multiplier*100)
pyplot.xlabel('X')
pyplot.ylabel('Y')

pyplot.show()



fig = pyplot.figure(111,figsize=(11,7), dpi=100)
# plotting the pressure field as a contour
#pyplot.contourf(X, Y, p, alpha=0.5, cmap=colourMap)  
#pyplot.colorbar()
pyplot.contourf(Xcond / multiplier, Ycond / (multiplier*100), T, colorinterpolation, vmin=0, vmax=25, cmap=colourMap)
# plotting the pressure field outlines
#pyplot.contour(X, Y, p, cmap=colourMap)  
# plotting velocity field
#pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
pyplot.streamplot(X,Y,u,v)
pyplot.xlabel('X')
pyplot.ylabel('Y')

pyplot.show()
