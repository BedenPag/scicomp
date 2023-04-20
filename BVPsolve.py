import numpy as np
from matplotlib import pyplot as plt
from BVP_uberclass import Grid, BoundaryCondition, construct_A_and_b

def solve_bvp(N, a, b, alpha, beta, q, bc_type, delta=None, gamma=None):
    """
    Solves a boundary value problem (BVP) using finite-difference method with different types of boundary conditions.

    Args:
        N (int): Number of grid points.
        a (float): Left boundary of the interval.
        b (float): Right boundary of the interval.
        alpha (float): Coefficient of the Dirichlet boundary condition at x=a.
        beta (float): Coefficient of the Dirichlet boundary condition at x=b.
        q (function): Function representing the coefficient added to the right-hand side of the PDE.
        bc_type (str): Type of boundary condition. Supported values: "dirichlet", "neumann", "robin".
        delta (float, optional): Coefficient of the Neumann boundary condition at x=b. Required for "neumann" BC. Default: None.
        gamma (float, optional): Coefficient of the Robin boundary condition at x=b. Required for "robin" BC. Default: None.

    Returns:
        None
    """
    # create the finite-difference grid
    grid = Grid(N=N, a=a, b=b)
    dx = grid.dx
    x = grid.x

    if bc_type == "dirichlet":
        # create two Dirichlet boundary conditions
        bc_left = BoundaryCondition("Dirichlet", 0.0, alpha, beta,[],[])
        bc_right = BoundaryCondition("Dirichlet", 0.0, alpha, beta,[],[])
    elif bc_type == "neumann":
        # create one Dirichlet and one Neumann boundary condition
        bc_left = BoundaryCondition("Dirichlet", 0.0, alpha, beta,[],[])
        bc_right = BoundaryCondition("Neumann", 0.0, alpha,[],[],delta)
    elif bc_type == "robin":
        # create one Dirichlet and one Robin boundary condition
        bc_left = BoundaryCondition("Dirichlet", 0.0, alpha, beta,[],[])
        bc_right = BoundaryCondition("Robin", 0.0, alpha,[],gamma,[])
    else:
        raise ValueError("Invalid boundary condition type. Supported values are 'dirichlet', 'neumann', and 'robin'.")

    # create the matrix A and the vector b
    A, b = construct_A_and_b(grid, bc_left, bc_right)

    # plot the solution and the true solution
    if bc_type == "dirichlet":
        # solve the linear system
        u = np.linalg.solve(A, -b - dx**2 * q(x[1:-1]))
        plt.plot(x[1:-1], u, 'o', label = 'Finite-difference solution')
        u_exact = 1/2 * x * (1 - x)
        plt.plot(x, u_exact, 'r', label = 'Exact solution')
    elif bc_type == "neumann":
        # solve the linear system
        u = np.linalg.solve(A, -b - dx**2 * q(x[1:]))
        plt.plot(x[1:], u, 'o', label = 'Finite-difference solution')
    else:
        # solve the linear system
        u = np.linalg.solve(A, -b - dx**2 * q(x[1:-1]))
        plt.plot(x[1:-1], u, 'o', label = 'Finite-difference solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Finite-difference solution with {} BC'.format(bc_type.capitalize()))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    N = 21
    a = 0
    b = 1
    alpha = 0
    beta = 0
    gamma = 1
    delta = 0
    q = lambda x: np.ones(np.size(x))
    solve_bvp(N, a, b, alpha, beta, q, "dirichlet")
    solve_bvp(N, a, b, alpha, beta, q, "neumann", delta=delta)
    solve_bvp(N, a, b, alpha, beta, q, "robin", gamma=gamma)
