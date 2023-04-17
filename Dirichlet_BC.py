import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import root

# solving d^2u/dx^2 = 0 with Dirichlet boundary conditions

# number of grid points
N = 50

# boundary conditions
gamma_1 = 0
gamma_2 = 1

# initial conditions
a = 0
b = 1

# define the grid
grid = np.linspace(a, b, N-1)
dx = (b-a)/(N-1)

# define the dirichlet function
def dirichlet(u, N, dx, gamma_1, gamma_2):
    F = np.zeros(N-1)
    F[0] = (u[1] - 2*u[0] + gamma_1)/(dx**2)

    for i in range(1, N-2):
        F[i] = (u[i+1] - 2*u[i] + u[i-1])/(dx**2)
    
    F[-1] = (gamma_2 - 2*u[-1] + u[-2])/(dx**2)

    return F

# initial guess
u = np.zeros(N-1)

# solve the system
sol = root(dirichlet, u, args=(N, dx, gamma_1, gamma_2))

# plot the solution
plt.plot(grid, sol.x, 'o')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Numerical solution')
plt.show()

# plot the true solution
true_y = (grid - a)/(b - a)
plt.plot(grid, true_y)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('True solution')
plt.show()

