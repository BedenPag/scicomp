import numpy as np
from matplotlib import pyplot as plt
from MethodOfLines_class import Grid, BoundaryCondition, construct_A_and_b

def solve_pde(N, a, b, D, u0, t_max, dt, bc_type, method, q, alpha, beta, delta=None, gamma=None):
    """
    Solves a partial differential equation (PDE) du/dt = D*d^2u/dx^2 + q(x) using finite-difference method with different types of boundary conditions.

    Args:
        N (int): Number of grid points.
        a (float): Left boundary of the interval.
        b (float): Right boundary of the interval.
        D (float): Diffusion coefficient.
        u0 (function): Initial condition.
        t_max (float): Maximum time.
        bc_type (str): Type of boundary condition. Supported values: "dirichlet", "neumann", "robin".
        method (str): Numerical method to solve the system of ODEs. Supported values: "explicit", "implicit", "crank-nicolson".
        q (function, optional): Function representing the coefficient added to the right-hand side of the PDE. Required for "explicit" and "crank-nicolson" methods. Default: None.
        alpha (float): Coefficient of the Dirichlet boundary condition at x=a.
        beta (float): Coefficient of the Dirichlet boundary condition at x=b.
        delta (float, optional): Coefficient of the Neumann boundary condition at x=b. Required for "neumann" BC. Default: None.
        gamma (float, optional): Coefficient of the Robin boundary condition at x=b. Required for "robin" BC. Default: None.

    Returns:
        None
    """
    # create the finite-difference grid
    grid = Grid(N=N, a=a, b=b)
    dx = grid.dx
    x = grid.x
    C = D*dt/dx**2
    print('C = ', C)

    # create the time grid
    t = np.ceil(t_max/dt)
    t = int(t)

    if bc_type == "dirichlet" and method == "explicit":
        # create the boundary conditions
        bc= BoundaryCondition("Dirichlet", 0.0, 0.0, 0.0,[],[])
        # construct the matrix A and the vector b
        A, b = construct_A_and_b(grid, bc)
        # solve the PDE
        if dt > dx**2/(2*D):
            raise ValueError("The time step is too big. The maximum value is {}".format(dx**2/(2*D)))
        u = np.zeros((t+1, N-2))
        u[0, :] = u0(x[1:-1], 0.0)
        for i in range(t):
            u[i+1] = u[i] + C*(A.dot(u[i]) + b) + dt*q(x[1:-1], i*dt, u[i], 2)
    elif bc_type == "dirichlet" and method == "implicit":
        # create the boundary conditions
        bc= BoundaryCondition("Dirichlet", 0.0, 0.0, 0.0,[],[])
        # construct the matrix A and the vector b
        A, b = construct_A_and_b(grid, bc)
        u = np.zeros((t+1, N-2))
        u[0, :] = u0(x, 0.0)
        LHS = np.eye(N-2) - C*A
        for i in range(t):
            RHS = u[i] + C*b + dt*(q(x[1:-1], i*dt, u[i], 2))
            u[i+1] = np.linalg.solve(LHS, RHS)
    elif bc_type == "dirichlet" and method == "crank-nicolson":
        # create the boundary conditions
        bc= BoundaryCondition("Dirichlet", 0.0, 0.0, 0.0,[],[])
        # construct the matrix A and the vector b
        A, b = construct_A_and_b(grid, bc)
        u = np.zeros((t+1, N-2))
        u[0, :] = u0(x[1:-1], 0.0)
        LHS = np.eye(N-2) - 0.5*C*A
        for i in range(t):
            RHS = (np.eye(N-2) + 0.5*C*A).dot(u[i]) + 0.5*dt*(q(x[1:-1], i*dt, u[i], 2) + q(x[1:-1], (i+1)*dt, u[i+1], 2))
            u[i+1] = np.linalg.solve(LHS, RHS)
    elif bc_type == "neumann" and method == "explicit":
        # create the boundary conditions
        bc= BoundaryCondition("Neumann", 0.0, 0.0,[],[],delta)
        # construct the matrix A and the vector b
        A, b = construct_A_and_b(grid, bc)
        # solve the PDE
        if dt > dx**2/(2*D):
            raise ValueError("The time step is too big. The maximum value is {}".format(dx**2/(2*D)))
        u = np.zeros((t+1, N-1))
        u[0, :] = u0(x[1:], 0.0)
        for i in range(t):
            u[i+1] = u[i] + C*(A.dot(u[i])) + dt*q(x[1:], i*dt, u[i], 2)
    
    elif bc_type == "neumann" and method == "implicit":
        # create the boundary conditions
        bc= BoundaryCondition("Neumann", 0.0, 0.0,[],[],delta)
        # construct the matrix A and the vector b
        A, b = construct_A_and_b(grid, bc)
        u = np.zeros((t+1, N-1))
        u[0, :] = u0(x[1:], 0.0)
        LHS = np.eye(N-1) - C*A
        for i in range(t):
            RHS = u[i] + C*b + dt*(q(x[1:], i*dt, u[i], 2))
            u[i+1] = np.linalg.solve(LHS, RHS)
    elif bc_type == "neumann" and method == "crank-nicolson":
        # create the boundary conditions
        bc= BoundaryCondition("Neumann", 0.0, 0.0,[],[],delta)
        # construct the matrix A and the vector b
        A, b = construct_A_and_b(grid, bc)
        u = np.zeros((t+1, N-1))
        u[0, :] = u0(x[1:], 0.0)
        LHS = np.eye(N-1) - 0.5*C*A
        for i in range(t):
            RHS = (np.eye(N-1) + 0.5*C*A).dot(u[i]) + 0.5*dt*(q(x[1:], i*dt, u[i], 2) + q(x[1:], (i+1)*dt, u[i+1], 2))
            u[i+1] = np.linalg.solve(LHS, RHS)
    elif bc_type == "robin" and method == "explicit":
        # create the boundary conditions
        bc= BoundaryCondition("Robin", 0.0, 0.0,[],gamma,[])
        # construct the matrix A and the vector b
        A, b = construct_A_and_b(grid, bc)
        # solve the PDE
        if dt > dx**2/(2*D):
            raise ValueError("The time step is too big. The maximum value is {}".format(dx**2/(2*D)))
        u = np.zeros((t+1, N-1))
        u[0, :] = u0(x[1:], 0.0)
        for i in range(t):
            u[i+1] = u[i] + C*(A.dot(u[i])) + dt*q(x[1:], i*dt, u[i], 2)
    elif bc_type == "robin" and method == "implicit":
        # create the boundary conditions
        bc= BoundaryCondition("Robin", 0.0, 0.0,[],gamma,[])
        # construct the matrix A and the vector b
        A, b = construct_A_and_b(grid, bc)
        u = np.zeros((t+1, N-1))
        u[0, :] = u0(x[1:], 0.0)
        LHS = np.eye(N-1) - C*A
        for i in range(t):
            RHS = u[i] + C*b + dt*(q(x[1:], i*dt, u[i], 2))
            u[i+1] = np.linalg.solve(LHS, RHS)
    elif bc_type == "robin" and method == "crank-nicolson":
        # create the boundary conditions
        bc= BoundaryCondition("Robin", 0.0, 0.0,[],gamma,[])
        # construct the matrix A and the vector b
        A, b = construct_A_and_b(grid, bc)
        u = np.zeros((t+1, N-1))
        u[0, :] = u0(x[1:], 0.0)
        LHS = np.eye(N-1) - 0.5*C*A
        for i in range(t):
            RHS = (np.eye(N-1) + 0.5*C*A).dot(u[i]) + 0.5*dt*(q(x[1:], i*dt, u[i], 2) + q(x[1:], (i+1)*dt, u[i+1], 2))
            u[i+1] = np.linalg.solve(LHS, RHS)
    elif bc_type != "dirichlet" and bc_type != "neumann" and bc_type != "robin":
        raise ValueError("Invalid boundary condition type. Supported values are 'dirichlet', 'neumann', 'robin'.")
    elif method != "explicit" and method != "implicit" and method != "crank-nicolson":
        raise ValueError("Invalid method. Supported values are 'explicit', 'implicit', 'crank-nicolson'.")
    else:
        raise ValueError("Invalid boundary condition type and method combination.")

    # plot the solution
    plt.figure()
    if bc_type == "dirichlet":
        plt.plot(x[1:-1], u[-1], 'o', label="Method = {}".format(method))
    else:
        plt.plot(x[1:], u[-1], 'o', label="Method = {}".format(method))
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("Solution for {} method with {} boundary condition".format(method,bc_type))
    plt.legend()
    plt.show()


# define the source term for the bratu equation (e^mu*u)
def q(x, t, u, mu):
    return (1-u)*np.exp(-x)

#def q(x, t, u, mu):
#    return np.exp(mu*u)

# define the source term for the 

# define the initial condition
def u0(x, t):
    return 0.0





if __name__ == "__main__":
    # solve the PDE
    solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type="dirichlet", method="explicit", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
    solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type="dirichlet", method="implicit", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
    solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type="dirichlet", method="crank-nicolson", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
    solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type="neumann", method="explicit", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
    solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type="neumann", method="implicit", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
    solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type="neumann", method="crank-nicolson", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
    solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type="robin", method="explicit", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
    solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type="robin", method="implicit", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
    solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type="robin", method="crank-nicolson", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)