import numpy as np
from matplotlib import pyplot as plt
from MethodOfLines_class import Grid, BoundaryCondition, construct_A_and_b

def solve_bvp(N, a, b, alpha, beta, q, bc_type, gamma=None, delta=None):
    """
    Solves a boundary value problem d^2u/dx^2 + q(x) = 0 using finite-difference method with different types of boundary conditions.

    Args:
        N (int): Number of grid points.
        a (float): Left boundary of the interval.
        b (float): Right boundary of the interval.
        alpha (float): Coefficient of the Dirichlet boundary condition at x=a.
        beta (float): Coefficient of the Dirichlet boundary condition at x=b.
        q (function): Function representing the coefficient added to the right-hand side of the PDE.
        bc_type (str): Type of boundary condition. Supported values: "dirichlet", "neumann", "robin".
        gamma (float, optional): Coefficient of the Robin boundary condition at x=b. Required for "robin" BC. Default: None.
        delta (float, optional): Coefficient of the Neumann boundary condition at x=b. Required for "neumann" BC. Default: None.

    Returns:
        None - plots the solution.
    """
    if N <= 1:
        raise ValueError("N must be greater than or equal to 1.")
    if not isinstance(N, int):
        raise TypeError("N must be an integer.")
    if a >= b:
        raise ValueError("a must be less than b.")
    if a == b:
        raise ValueError("a and b cannot be equal.")
    if bc_type == "robin" and gamma is None:
        raise ValueError("gamma must be provided for Robin boundary condition.")
    if bc_type == "neumann" and delta is None:
        raise ValueError("delta must be provided for Neumann boundary condition.")
    if not callable(q):
        raise TypeError("q must be a function.")
    if alpha is None or beta is None:
        raise ValueError("alpha and beta must be provided for boundary conditions.")
    if gamma is not None and not isinstance(gamma, (float, int)) or delta is not None and not isinstance(delta, (float, int)) or not isinstance(alpha, (float, int)) or not isinstance(beta, (float, int)):
        raise TypeError("alpha, beta, gamma, and delta must be numbers.")
    if q is None:
        raise ValueError("q must be provided. If q is constant, use lambda x: constant.")
    if bc_type not in ["dirichlet", "neumann", "robin"]:
        raise ValueError("Invalid boundary condition type. Supported values are 'dirichlet', 'neumann', and 'robin'.")
    
    # create the finite-difference grid
    grid = Grid(N=N, a=a, b=b)
    dx = grid.dx
    x = grid.x

    if bc_type == "dirichlet":
        # create two Dirichlet boundary conditions
        bc = BoundaryCondition("Dirichlet", 0.0, alpha, beta,[],[])
    elif bc_type == "neumann":
        # create one Dirichlet and one Neumann boundary condition
        bc = BoundaryCondition("Neumann", 0.0, alpha,[],[],delta)
    elif bc_type == "robin":
        # create one Dirichlet and one Robin boundary condition
        bc = BoundaryCondition("Robin", 0.0, alpha,[],gamma,[])

    # create the matrix A and the vector b
    A, b = construct_A_and_b(grid, bc)

    # plot the solution and the true solution
    if bc_type == "dirichlet":
        # solve the linear system
        u = np.linalg.solve(A, -b - dx**2 * q(x[1:-1]))
        plt.plot(x[1:-1], u, 'o', label = 'Finite-difference solution')
    elif bc_type == "neumann":
        # solve the linear system
        u = np.linalg.solve(A, -b - dx**2 * q(x[1:]))
        plt.plot(x[1:], u, 'o', label = 'Finite-difference solution')
    elif bc_type == "robin":
        # solve the linear system
        u = np.linalg.solve(A, -b - dx**2 * q(x[1:]))
        plt.plot(x[1:], u, 'o', label = 'Finite-difference solution')
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
    solve_bvp(N, a, b, alpha, beta, q, "dirichlet", gamma, delta)
    solve_bvp(N, a, b, alpha, beta, q, "neumann", gamma, delta)
    solve_bvp(N, a, b, alpha, beta, q, "robin", gamma, delta)