# Simple Numerical Laplace Equation Solution using Finite Difference Method
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import LocalVars

NUMPYARGS = {"dtype":'float64'}
def PlottableAxes(Tleft, Tright):



    # Set maximum iteration
    maxIter = 50

    # Set Dimension and delta
    realX = LocalVars.PaneWidth # centimeter
    realY = LocalVars.PaneHeight
    lenX = LocalVars.LenX
    lenY = LocalVars.LenY
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
    T = np.empty((lenX, lenY),**NUMPYARGS)
    T.fill(Tguess)

    # Set Boundary condition
    #T[(lenY-1):, :] = Ttop
    #T[:1, :] = Tbottom
    T[:1, :] = T[(lenY-1):,:] = np.linspace(Tleft, Tright, lenX,**NUMPYARGS)
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

def PlottableAxesBoth(Tleft, Tright, rho = 1, nu = 0.1, magnefying=50,panes=2):
    # multiple of grids to use with convection
    precision_multi = panes + 1
    p_prime = panes-1 # useful for calculations
    # Set maximum iteration
    maxIter = 75
    print("Viscosity: %f"%nu)
    # Set Dimension and delta
    realX = LocalVars.PaneWidth # centimeter
    realY = LocalVars.PaneHeight
    lenX = LocalVars.LenX
    lenY = LocalVars.LenY
    deltaX = 1
    deltaY = 1
    # flip as we assume heat source is on left side
    flip = False
    if Tleft < Tright:
        flip = True
        # heat will dissipate from hot to cold. the way we calculate it factors transfer from left to right, so if
        # left temperature is smaller than right, we must calculate heat as if it were right to left and reverse it

        Tleft = Tleft + Tright
        Tright = Tleft - Tright
        Tleft = Tleft - Tright

        #T = T[:, ::-1]
        #dTx = dTx[::-1]
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
    T = np.empty((lenX+2, lenY+2),**NUMPYARGS)
    T.fill(Tguess)

    # Set Boundary condition
    #T[(lenY-1):, :] = Ttop
    #T[:1, :] = Tbottom
    #T[:1, :] = T[(lenY-1):,:] = np.linspace(Tleft, Tright, lenX)
    T[:, (lenX-1):] = Tright
    T[:, :1] = Tleft

    # Iteration (We assume that the iteration is convergence in maxIter = 500)
    #print("Please wait for a moment")
    for iteration in range(0, maxIter):
        for i in range(1, lenX, deltaX):
            for j in range(1, lenY, deltaY):
                T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])
        T[:0,:] = T[2,:]
        T[:(lenX+1),:] = T[(lenX-1),:]

    Tprime = T.copy()
    dTx = np.zeros((lenX, lenY),**NUMPYARGS)
    dTy = np.zeros((lenX, lenY),**NUMPYARGS)
    for i in range(1, lenX, deltaX):
        for j in range(1, lenY, deltaY):
            #dTx[i - 1, j - 1] = (Tprime[i + 1][j] - Tprime[i - 1][j]) / (2 * realX)
            #dTx[i - 1, j - 1] = (Tprime[i + 1][j] - Tprime[i - 1][j]) / (2 * realX)
            dTy[i, j] = (Tprime[i][j + 1] - Tprime[i][j - 1]) / (2 * realY)
            dTy[i, j] = (Tprime[i][j + 1] - Tprime[i][j - 1]) / (2 * realY)
    if flip:
        conduction = (X, Y, T[::-1,::-1][1:-1,1:-1])
    else:
        conduction = (X, Y, T[1:-1,1:-1])

    # convection part





    #nx = precision_multi * lenX + 2
    nx = precision_multi * lenX
    #ny = precision_multi * lenY + 2
    ny = precision_multi * lenY
    nt = 50
    nit = 20
    c = 10
    norm = 4
    dx = norm / (nx - 1)
    dy = norm / (ny - 1)

    x = np.linspace(0, norm, nx,**NUMPYARGS)
    y = np.linspace(0, norm, ny,**NUMPYARGS)

    X, Y = np.meshgrid(x, y)

    gravity = 1

    dt = .001

    u = np.zeros((ny, nx),**NUMPYARGS)
    v = np.zeros((ny, nx),**NUMPYARGS)
    p = np.zeros((ny, nx),**NUMPYARGS)
    b = np.zeros((ny, nx),**NUMPYARGS)

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
        p[:, -2] = Tright  # p[:,1]   # dp/dy = 0 at y = 0
        #p[-2,:] = Tright # p[:,1]   # dp/dy = 0 at y = 0
        #p[0,:] = 2#  Tcold   p[:, 1]   # dp/dx = 0 at x = 0
        p[:, 1] = Tleft  # p[:,-2] = 0    # p = 0 at y = 2
        #p[1,:] = Tleft  # p[:,-2] = 0    # p = 0 at y = 2
        #if p_prime > 2:
        #    for k in range(1, p_prime):
        #        p[:, k * int(nx / p_prime)-1] = np.mean(T[:,k * int(lenX / p_prime)-1])
        #        p[:, k * int(nx / p_prime)+1] = np.mean(T[:,k * int(lenX / p_prime)+1])
        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 +
                              (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) /
                             (2 * (dx ** 2 + dy ** 2)) -
                             (dx ** 2) * (dy ** 2) / (2 * (dx ** 2 + dy ** 2)) *
                             b[1:-1, 1:-1]) #* direction

            #p[0, :] = p[1, :]
            #p[-1, :] = p[-2, :]
            # reset the boundary conditions, where 0 pressure at window pain
            #p[0,:] = 0
            #p[-1,:] = 0
            #p[:,0] = 0
            #p[:,-1] = 0
            ##### NEW ADDITION #####
            p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
            #p[:, -2] = p[:, -3]  # dp/dx = 0 at x = 2
            #p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
            p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
            #p[-1, :] = p[-2,:]# Thot  p[:, -2] # dp/dx = 0 at x = 2
            if p_prime > 2:
                for k in range(1,p_prime):
                    p[:,k*int(nx/p_prime)] = 0#p[:,k*int(nx/p_prime)+1]
                    #p[:,k*int(nx/p_prime)+1] = 0

                #p[:,k*int(nx/p_prime)] = p[:,k*int(nx/p_prime)-1]

        return p

    def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
        # discretization provided by CFD using Navier-Stokes BVP
        un = np.empty_like(u)
        vn = np.empty_like(v)
        b = np.zeros((ny, nx),**NUMPYARGS)
        u[1, :] = Tleft
        u[-2, :] = Tright  # set initial conditions to mirror those of the right
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
                             nu * (dt / (dx ** 2) *
                                   (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                                   dt / (dy ** 2) *
                                   (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))) #* direction

            v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * dt / dx *
                             (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * dt / dy *
                             (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                             dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                             nu * (dt / (dx ** 2) *
                                   (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                                   dt / (dy ** 2) *
                                   (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))#* direction
            #print(u, v)

            # reset the boundary conditions, where 0 velocity at border for both directions
            u[:, 0] = 0
            u[:, -1] = 0
            u[0,:] = 0
            u[-1,:] = 0
            v[0, :] = 0
            v[-1, :] = 0
            v[:, 0] = 0
            v[:, -1] = 0
            if p_prime > 1:

                for k in range(1,p_prime):
                    u[:,k*int(nx/p_prime)] = 0
                    v[:,k*int(nx/p_prime)] = 0



        return u, v, p

    multiplier = ((np.max(conduction[0])-np.min(conduction[0]))/norm + np.min(conduction[0]))
    u = np.zeros((ny, nx),**NUMPYARGS)
    v = np.zeros((ny, nx),**NUMPYARGS)
    #v[:,:] = -0.000001
    p = np.zeros((ny, nx),**NUMPYARGS)
    p[:,0] = Tleft
    p[:,-1] = Tright
    #p[:,:]=10
    b = np.zeros((ny, nx),**NUMPYARGS)
    nt = 50
    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

    def flip_back(nparr, axis=0):
        return np.flip(nparr, axis=axis)
    # do de-normalization
    if flip:
        # flip back if necessary
        pass
        #u = flip_back(u)
        #v = flip_back(v)

        # T = T[::-1, ::-1]
        u = -flip_back(u,0)
        u = flip_back(u,1)
        #u = -u
        v = flip_back(v,0)
        v = -flip_back(v,1)
        #u = u[::-1,:]
        #v = v[::-1]
        #print(u)
        dTx = flip_back(dTx,0)
        dTx = flip_back(dTx,1)

        dTy = flip_back(dTy, 0)
        dTy = flip_back(dTy, 1)
    else:
        u = flip_back(u,0)
        v = -flip_back(v,0)
    #convection = (X[1:-1] * multiplier, Y[1:-1] * multiplier * 100, u[1:-1] * multiplier, v[1:-1] * multiplier * 100)
    #convection = (X[1:-1] * multiplier, Y[1:-1] * multiplier * 100, u[1:-1] * multiplier, v[1:-1] * multiplier * 100)
    convection = (X * multiplier, Y * multiplier * 100, u * multiplier, v * multiplier * 100)
    conduction = (conduction[0], conduction[1],  conduction[2] - magnefying*((u[::precision_multi,::precision_multi]**2 * dTx) + (v[::precision_multi,::precision_multi]**2 * dTy)))
    #conduction = (conduction[0], conduction[1],  conduction[2] - magnefying*((u[1:-1,1:-1][::precision_multi,::precision_multi] * dTx) + (v[1:-1,1:-1][::precision_multi,::precision_multi] * dTy)))
    """
    from matplotlib import pyplot
    import time

    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.contourf(X, Y, p, alpha=0.5)
    pyplot.colorbar()
    pyplot.contour(X, Y, p)
    pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    pyplot.xlabel('X')
    pyplot.ylabel('Y')

    pyplot.show()

    #time.sleep(10)

    """
    return conduction, convection

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
def PlottableAxesBoth_OG(Tleft, Tright, rho = 1, nu = 0.1, magnefying=50):


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
            u[-1, :] = Tright  # set initial disturbance in x direction
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