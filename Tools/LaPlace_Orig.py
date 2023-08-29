# Simple Numerical Laplace Equation Solution using Finite Difference Method
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Boundary Condition
Tleft = 7
Tright = 20


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
Y = Y * (260/20)
X = X * (2.6/20)
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

# Configure the contour
plt.title("Contour of Temperature across double pane Window")

cp = plt.contourf(X, Y, T, colorinterpolation, vmin=0, vmax=25, cmap=colourMap)
plt.clim(0,25)
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
# Set Colorbar
cbar = plt.colorbar(cp, label="Temperature (Celsius)", extend='max')



# plt.savefig("images/fig1.png")
# Show the result in the plot window
plt.show()

print("")