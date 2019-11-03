# Simple Numerical Laplace Equation Solution using Finite Difference Method
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import LocalVars

def PlottableAxes(Tleft, Tright):


    # Set maximum iteration
    maxIter = 50

    # Set Dimension and delta
    realX = LocalVars.PaneWidth # centimeter
    realY = LocalVars.PaneHeight
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

    print("Iteration finished")
    return X, Y, T
    # Configure the contour
    plt.title("Contour of Temperature across double pane Window")

    cp = plt.contourf(X, Y, T, colorinterpolation, vmin=0, vmax=25, cmap=colourMap)
    plt.clim(0,25)
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    # Set Colorbar
    cbar = plt.colorbar(cp, label="Temperature (Celsius)", extend='max')



    plt.savefig("images/fig1.png")
    # Show the result in the plot window
    plt.show()

    print("")

def PlottableAxesBoth(Tleft, Tright, rho = 1, nu = 0.1, magnefying=50):


    # Set maximum iteration
    maxIter = 75

    # Set Dimension and delta
    realX = LocalVars.PaneWidth # centimeter
    realY = LocalVars.PaneHeight
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
    T = np.empty((lenX+2, lenY+2))
    T.fill(Tguess)

    # Set Boundary condition
    #T[(lenY-1):, :] = Ttop
    #T[:1, :] = Tbottom
    #T[:1, :] = T[(lenY-1):,:] = np.linspace(Tleft, Tright, lenX)
    T[:, (lenX-1):] = Tright
    T[:, :1] = Tleft

    # Iteration (We assume that the iteration is convergence in maxIter = 500)
    print("Please wait for a moment")
    for iteration in range(0, maxIter):
        for i in range(1, lenX, deltaX):
            for j in range(1, lenY, deltaY):
                T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])
        T[:0,:] = T[2,:]
        T[:(lenX+1),:] = T[(lenX-1),:]

    Tprime = T.copy()
    dTx = np.zeros((lenX, lenY))
    dTy = np.zeros((lenX, lenY))
    for i in range(1, lenX, deltaX):
        for j in range(1, lenY, deltaY):
            dTx[i - 1, j - 1] = (Tprime[i + 1][j] - Tprime[i - 1][j]) / (2 * realX)
            dTy[i - 1, j - 1] = (Tprime[i][j + 1] - Tprime[i][j - 1]) / (2 * realY)
    print("Iteration finished")
    conduction = (X, Y, T[1:-1,1:-1])

    # convection part
    nx = 40
    ny = 40
    nt = 50
    nit = 20
    c = 1
    norm = 2
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    #rho = 2

    dt = .001

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    def build_up_b(b, rho, dt, u, v, dx, dy):

        b[1:-1, 1:-1] = (rho * (1 / dt *
                                ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                                 (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                                ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
                                2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                                ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))

        return b

    def pressure_poisson(p, dx, dy, b):
        pn = np.empty_like(p)
        pn = p.copy()

        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 +
                              (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) /
                             (2 * (dx ** 2 + dy ** 2)) -
                             dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                             b[1:-1, 1:-1])

            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]

            # p[-1, :] = 25# Thot  p[:, -2] # dp/dx = 0 at x = 2
            p[:, -1] = Tright # p[:,1]   # dp/dy = 0 at y = 0
            # p[0,:] = 2#  Tcold   p[:, 1]   # dp/dx = 0 at x = 0
            p[:, 0] = Tleft  # p[:,-2] = 0    # p = 0 at y = 2

        return p

    def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
        # discretization provided by CFD using Navier-Stokes BVP
        un = np.empty_like(u)
        vn = np.empty_like(v)
        b = np.zeros((ny, nx))

        for n in range(nt):
            un = u.copy()
            vn = v.copy()

            b = build_up_b(b, rho, dt, u, v, dx, dy)
            p = pressure_poisson(p, dx, dy, b)

            u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * dt / dx *
                             (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * dt / dy *
                             (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                             dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                             nu * (dt / dx ** 2 *
                                   (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                                   dt / dy ** 2 *
                                   (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

            v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * dt / dx *
                             (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * dt / dy *
                             (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                             dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                             nu * (dt / dx ** 2 *
                                   (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                                   dt / dy ** 2 *
                                   (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            u[0, :] = Tleft
            u[:, 0] = 0
            u[:, -1] = 0
            u[-1, :] = Tright  # set velocity on cavity lid equal to 1
            v[0, :] = 0
            v[-1, :] = 0
            v[:, 0] = 0
            v[:, -1] = 0

        return u, v, p

    multiplier = (np.max(conduction[0])/norm)
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    nt = 50
    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

    # do de-normalization

    convection = (X * multiplier, Y * multiplier * 100, u * multiplier, v * multiplier * 100)
    conduction = (conduction[0], conduction[1],  conduction[2] - magnefying*((u[::2,::2] * dTx) + (v[::2,::2] * dTy)))
    return conduction, convection

# messed up somewhere
def PlottableAxesBoth_messed(Tleft, Tright):


    # Set maximum iteration
    maxIter = 50

    # Set Dimension and delta
    realX = LocalVars.PaneWidth # centimeter
    realY = LocalVars.PaneHeight
    lenX = 20
    lenY = 20 #we set it rectangular, 2 ghost points for BVP
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
    T = np.empty((lenX+2, lenY+2))
    T.fill(Tguess)

    # Set Boundary condition
    #T[(lenY-1):, :] = Ttop
    #T[:1, :] = Tbottom
    #T[:1, :] = T[(lenY-1):,:] = np.linspace(Tleft, Tright, lenX)
    T[:, (lenX - 1):] = Tright
    T[:, :1] = Tleft

    # Iteration (We assume that the iteration is convergence in maxIter = 500)
    print("Please wait for a moment")
    for iteration in range(0, maxIter):
        for i in range(1, lenX, deltaX):
            for j in range(1, lenY, deltaY):
                T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])
        T[(lenY+1),:] = T[(lenY-1),:]
        T[0,:] = T[2,:]

    print("Iteration finished")
    conduction = (X, Y, T[1:-1,1:-1])

    Tprime = T.copy()
    dTx = np.zeros((lenX,lenY))
    dTy = np.zeros((lenX,lenY))
    for i in range(1, lenX, deltaX):
        for j in range(1, lenY, deltaY):
            dTx[i-1,j-1] = (Tprime[i+1][j]-Tprime[i-1][j])/(2*realX)
            dTy[i-1,j-1] = (Tprime[i][j+1]-Tprime[i][j-1])/(2*realY)



    # convection part
    nx = 41
    ny = 41
    nt = 50
    nit = 20
    c = 1
    norm = 2
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    rho = 1
    nu = .1
    dt = .001

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    def build_up_b(b, rho, dt, u, v, dx, dy):

        b[1:-1, 1:-1] = (rho * (1 / dt *
                                ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                                 (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                                ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
                                2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                                ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))

        return b

    def pressure_poisson(p, dx, dy, b):
        pn = np.empty_like(p)
        pn = p.copy()

        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 +
                              (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) /
                             (2 * (dx ** 2 + dy ** 2)) -
                             dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                             b[1:-1, 1:-1])

            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]

            # p[-1, :] = 25# Thot  p[:, -2] # dp/dx = 0 at x = 2
            p[:, -1] = Tright # p[:,1]   # dp/dy = 0 at y = 0
            # p[0,:] = 2#  Tcold   p[:, 1]   # dp/dx = 0 at x = 0
            p[:, 0] = Tleft  # p[:,-2] = 0    # p = 0 at y = 2

        return p

    def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
        # discretization provided by CFD using Navier-Stokes BVP
        un = np.empty_like(u)
        vn = np.empty_like(v)
        b = np.zeros((ny, nx))

        for n in range(nt):
            un = u.copy()
            vn = v.copy()

            b = build_up_b(b, rho, dt, u, v, dx, dy)
            p = pressure_poisson(p, dx, dy, b)

            u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * dt / dx *
                             (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * dt / dy *
                             (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                             dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                             nu * (dt / dx ** 2 *
                                   (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                                   dt / dy ** 2 *
                                   (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

            v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * dt / dx *
                             (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * dt / dy *
                             (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                             dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                             nu * (dt / dx ** 2 *
                                   (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                                   dt / dy ** 2 *
                                   (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            u[0, :] = Tleft
            u[:, 0] = 0
            u[:, -1] = 0
            u[-1, :] = Tright  # set velocity on cavity lid equal to 1
            v[0, :] = 0
            v[-1, :] = 0
            v[:, 0] = 0
            v[:, -1] = 0

        return u, v, p

    multiplier = (np.max(conduction[0])/norm)
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    nt = 50
    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

    # do de-normalization

    convection = (X * multiplier, Y * multiplier * 100, u * multiplier, v * multiplier * 100)

    return conduction, convection

# backup of good boundary
def PlottableAxesBoth_Temp(Tleft, Tright):


    # Set maximum iteration
    maxIter = 500

    # Set Dimension and delta
    realX = LocalVars.PaneWidth # centimeter
    realY = LocalVars.PaneHeight
    lenX = 20
    lenY = 20 #we set it rectangular, 2 ghost points for BVP
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
    T = np.empty((lenX+2, lenY+2))
    T.fill(Tguess)

    # Set Boundary condition
    #T[(lenY-1):, :] = Ttop
    #T[:1, :] = Tbottom
    #T[:1, :] = T[(lenY-1):,:] = np.linspace(Tleft, Tright, lenX)
    T[:, -1:] = T[:, -3:] = Tright
    T[:, :1] = T[:,:3] = Tleft

    # Iteration (We assume that the iteration is convergence in maxIter = 500)
    print("Please wait for a moment")
    for iteration in range(0, maxIter):
        for i in range(1, lenX, deltaX):
            for j in range(1, lenY, deltaY):
                T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])
        T[(lenY+1),:] = T[(lenY-1),:]
        T[0,:] = T[2,:]

    print("Iteration finished")
    conduction = (X, Y, T[1:-1,1:-1])

    """Tprime = T.copy()
    dTx = np.zeros((lenX,lenY))
    dTy = np.zeros((lenX,lenY))
    for i in range(1, lenX, deltaX):
        for j in range(1, lenY, deltaY):
            dTx[i-1,j-1] = (Tprime[i+1][j]-Tprime[i-1][j])/(2*realX)
            dTy[i-1,j-1] = (Tprime[i][j+1]-Tprime[i][j-1])/(2*realY)"""



    # convection part
    nx = 20
    ny = 20
    nt = 50
    nit = 20
    c = 1
    norm = 2
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    rho = 1
    nu = .1
    dt = .001

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    def build_up_b(b, rho, dt, u, v, dx, dy):

        b[1:-1, 1:-1] = (rho * (1 / dt *
                                ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                                 (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                                ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
                                2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                                ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))

        return b

    def pressure_poisson(p, dx, dy, b):
        pn = np.empty_like(p)
        pn = p.copy()

        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 +
                              (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) /
                             (2 * (dx ** 2 + dy ** 2)) -
                             dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                             b[1:-1, 1:-1])

            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]

            # p[-1, :] = 25# Thot  p[:, -2] # dp/dx = 0 at x = 2
            p[:, -1] = Tright # p[:,1]   # dp/dy = 0 at y = 0
            # p[0,:] = 2#  Tcold   p[:, 1]   # dp/dx = 0 at x = 0
            p[:, 0] = Tleft  # p[:,-2] = 0    # p = 0 at y = 2

        return p

    def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
        # discretization provided by CFD using Navier-Stokes BVP
        un = np.empty_like(u)
        vn = np.empty_like(v)
        b = np.zeros((ny, nx))

        for n in range(nt):
            un = u.copy()
            vn = v.copy()

            b = build_up_b(b, rho, dt, u, v, dx, dy)
            p = pressure_poisson(p, dx, dy, b)

            u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * dt / dx *
                             (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * dt / dy *
                             (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                             dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                             nu * (dt / dx ** 2 *
                                   (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                                   dt / dy ** 2 *
                                   (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

            v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * dt / dx *
                             (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * dt / dy *
                             (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                             dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                             nu * (dt / dx ** 2 *
                                   (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                                   dt / dy ** 2 *
                                   (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            u[0, :] = Tleft
            u[:, 0] = 0
            u[:, -1] = 0
            u[-1, :] = Tright  # set velocity on cavity lid equal to 1
            v[0, :] = 0
            v[-1, :] = 0
            v[:, 0] = 0
            v[:, -1] = 0

        return u, v, p

    multiplier = (np.max(conduction[0])/norm)
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    nt = 50
    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

    # do de-normalization

    convection = (X * multiplier, Y * multiplier * 100, u * multiplier, v * multiplier * 100)

    #conduction = (conduction[0],conduction[1], conduction[2]+np.abs((u*dTx)/multiplier+(v*dTy)/(multiplier*100)))
    return conduction, convection

# original math
def PlottableAxesBoth_BadBoundary(Tleft, Tright):


    # Set maximum iteration
    maxIter = 50

    # Set Dimension and delta
    realX = LocalVars.PaneWidth # centimeter
    realY = LocalVars.PaneHeight
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

    print("Iteration finished")
    conduction = (X, Y, T)

    # convection part
    nx = 41
    ny = 41
    nt = 50
    nit = 20
    c = 1
    norm = 2
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    rho = 1
    nu = .1
    dt = .001

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    def build_up_b(b, rho, dt, u, v, dx, dy):

        b[1:-1, 1:-1] = (rho * (1 / dt *
                                ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                                 (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                                ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
                                2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                                ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))

        return b

    def pressure_poisson(p, dx, dy, b):
        pn = np.empty_like(p)
        pn = p.copy()

        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 +
                              (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) /
                             (2 * (dx ** 2 + dy ** 2)) -
                             dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                             b[1:-1, 1:-1])

            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]

            # p[-1, :] = 25# Thot  p[:, -2] # dp/dx = 0 at x = 2
            p[:, -1] = Tright # p[:,1]   # dp/dy = 0 at y = 0
            # p[0,:] = 2#  Tcold   p[:, 1]   # dp/dx = 0 at x = 0
            p[:, 0] = Tleft  # p[:,-2] = 0    # p = 0 at y = 2

        return p

    def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
        # discretization provided by CFD using Navier-Stokes BVP
        un = np.empty_like(u)
        vn = np.empty_like(v)
        b = np.zeros((ny, nx))

        for n in range(nt):
            un = u.copy()
            vn = v.copy()

            b = build_up_b(b, rho, dt, u, v, dx, dy)
            p = pressure_poisson(p, dx, dy, b)

            u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * dt / dx *
                             (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * dt / dy *
                             (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                             dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                             nu * (dt / dx ** 2 *
                                   (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                                   dt / dy ** 2 *
                                   (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

            v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * dt / dx *
                             (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * dt / dy *
                             (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                             dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                             nu * (dt / dx ** 2 *
                                   (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                                   dt / dy ** 2 *
                                   (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            u[0, :] = Tleft
            u[:, 0] = 0
            u[:, -1] = 0
            u[-1, :] = Tright  # set velocity on cavity lid equal to 1
            v[0, :] = 0
            v[-1, :] = 0
            v[:, 0] = 0
            v[:, -1] = 0

        return u, v, p

    multiplier = (np.max(conduction[0])/norm)
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    nt = 50
    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

    # do de-normalization

    convection = (X * multiplier, Y * multiplier * 100, u * multiplier, v * multiplier * 100)

    return conduction, convection