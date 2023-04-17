import numpy as np
from matplotlib import pyplot as plt
from BVP_uberclass import Grid, BoundaryCondition, construct_A_and_b

def solve_dirichlet_bc(N, a, b, alpha, beta, q):
    # create the finite-difference grid
    grid = Grid(N=N, a=a, b=b)
    dx = grid.dx
    x = grid.x


    # create two Dirichlet boundary conditions
    bc_left = BoundaryCondition("Dirichlet", 0.0, alpha, beta)
    bc_right = BoundaryCondition("Dirichlet", 0.0, alpha, beta)

    # create the matrix A and the vector b
    A, b = construct_A_and_b(grid, bc_left, bc_right)

    print(A.shape)
    print(b.shape)
    print(q(x[1:-1]).shape)

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

def solve_neumann_bc(N, a, b,alpha,beta,delta, q):
    # create the finite-difference grid
    grid = Grid(N=N, a=a, b=b)
    dx = grid.dx
    x = grid.x

    # create one Dirichlet and one Neumann boundary condition
    bc_left = BoundaryCondition("Dirichlet", 0.0, alpha, beta)
    bc_right = BoundaryCondition("Neumann", 0.0, alpha, delta)

    # create the matrix A and the vector b
    A, b = construct_A_and_b(grid, bc_left, bc_right)

    print(A.shape)
    print(b.shape)
    print(q(x[1:]).shape)

    # solve the linear system
    u = np.linalg.solve(A, -b - dx**2 * q(x[1:]))

    # plot the solution and the true solution
    u_exact = 1/2 * x * (1 - x)
    plt.plot(x[1:], u, 'o', label = 'Finite-difference solution')
    plt.plot(x, u_exact, 'r', label = 'Exact solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Finite-difference solution with Neumann BC')
    plt.legend()
    plt.show()

def solve_robin_bc(N, a, b, alpha, beta,gamma, q):
    # create the finite-difference grid
    grid = Grid(N=N, a=a, b=b)
    dx = grid.dx
    x = grid.x

    # create one Dirichlet and one Robin boundary condition
    bc_left = BoundaryCondition("Dirichlet", 0.0, alpha, beta)
    bc_right = BoundaryCondition("Robin", 0.0, alpha, gamma)

    # create the matrix A and the vector b
    A, b = construct_A_and_b(grid, bc_left, bc_right)

    # solve the linear system
    u = np.linalg.solve(A, -b - dx**2 * q(x[1:-1]))

    # plot the solution and the true solution
    u_exact = 1/2 * x * (1 - x)
    plt.plot(x[1:-1], u, 'o', label = 'Finite-difference solution')
    plt.plot(x, u_exact, 'r', label = 'Exact solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Finite-difference solution with Robin BC')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    N = 21
    a = 0
    b = 1
    q = lambda x: np.ones(np.size(x))
    solve_dirichlet_bc(N, a, b,1,1, q)
    solve_neumann_bc(N, a, b,1,1,1, q)
    solve_robin_bc(N, a, b, 1, 1, 1 ,q)