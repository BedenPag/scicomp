import numpy as np
from matplotlib import pyplot as plt

# import the Grid, BoundaryCondition, and construct_A_and_b functions
from BVP_uberclass import Grid, BoundaryCondition, construct_A_and_b

def q(x):
    return np.ones(np.size(x))

# create the finite-difference grid
grid = Grid(N=21, a=0, b=1)
dx = grid.dx
x = grid.x

# create two Dirichlet boundary conditions
bc_left = BoundaryCondition("Dirichlet", 0.0)
bc_right = BoundaryCondition("Dirichlet", 0.0)

# create the matrix A and the vector b
A, b = construct_A_and_b(grid, bc_left, bc_right, "Dirichlet")

# solve the linear system
u = np.linalg.solve(A, -b - dx**2 * q(x[1:-1]))


# plot the solution and the true solution
u_exact = 1/2 * x * (1 - x)
plt.plot(x[1:-1], u, 'o', label = 'Finite-difference solution')
plt.plot(x, u_exact, 'r', label = 'Exact solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Finite-difference solution with Dirichlet BC')
plt.legend()
plt.show()

# Neumann boundary condition example

# update the right boundary condition
bc_right = BoundaryCondition("Neumann", 0.0)

# update the matrix A and the vector b
A_DN, b_DN = construct_A_and_b(grid, bc_left, bc_right,"Neumann")


# solve the linear system
u_DN = np.linalg.solve(A_DN, -b_DN - dx**2 * q(x[1:]))

# plot the solution and the true solution
u_exact = 1/2 * x * (1 - x)
plt.plot(x[1:], u_DN, 'o', label = 'Finite-difference solution')
plt.plot(x, u_exact, 'r', label = 'Exact solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Finite-difference solution with Neumann BC')
plt.legend()
plt.show()

# Robin boundary condition example

# update the right boundary condition
bc_right = BoundaryCondition("Robin", 0.0, 1.0)

# update the matrix A and the vector b
A_DR, b_DR = construct_A_and_b(grid, bc_left, bc_right, "Robin")

# solve the linear system
u_DR = np.linalg.solve(A_DR, -b_DR - dx**2 * q(x[1:-1]))

# plot the solution and the true solution
u_exact = 1/2 * x * (1 - x)
plt.plot(x[1:-1], u_DR, 'o', label = 'Finite-difference solution')
plt.plot(x, u_exact, 'r', label = 'Exact solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Finite-difference solution with Robin BC')
plt.legend()
plt.show()
