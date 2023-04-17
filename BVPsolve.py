import numpy as np
from matplotlib import pyplot as plt

# import the Grid, BoundaryCondition, and construct_A_and_b functions
from BVP_uberclass import Grid, BoundaryCondition, SourceTerm, construct_A_and_b

# create the finite-difference grid with 51 points (N = 51)
grid = Grid(N=51, a=0, b=1)
dx = grid.dx
x = grid.x

# create two Dirichlet boundary conditions
bc_left = BoundaryCondition("Neumann", 0.0)
bc_right = BoundaryCondition("Neumann", 0.0)

# create the matrix A and the vector b
A, b = construct_A_and_b(grid, bc_left, bc_right)

# create a SourceTerm object
q = lambda x: np.ones(np.size(x))
source_term = SourceTerm(q)

# solve the linear system
u = np.linalg.solve(A, -b - dx**2 * source_term.evaluate(grid))

# plot the solution
plt.plot(x[1:-1], u, 'o')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Finite-difference solution')
plt.show()
