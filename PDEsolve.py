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
        q (function): Function representing the coefficient added to the right-hand side of the PDE. Required for "explicit" and "crank-nicolson" methods. Default: None.
        alpha (float): Coefficient of the Dirichlet boundary condition at x=a.
        beta (float): Coefficient of the Dirichlet boundary condition at x=b.
        delta (float, optional): Coefficient of the Neumann boundary condition at x=b. Required for "neumann" BC. Default: None.
        gamma (float, optional): Coefficient of the Robin boundary condition at x=b. Required for "robin" BC. Default: None.

    Returns:
        u (numpy array): Solution of the PDE.
        x (numpy array): Grid points.
    """
    if N <= 1:
        raise ValueError("N must be greater than or equal to 1.")
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
    if not callable(u0):
        raise TypeError("u0 must be a function.")
    if not isinstance(N, int):
        raise TypeError("N must be an integer.")
    if not isinstance(a, (float, int)) or not isinstance(b, (float, int)) or not isinstance(D, (float, int)) or not isinstance(t_max, (float, int)) or not isinstance(dt, (float, int)):
        raise TypeError("a, b, D, t_max, and dt must be numbers.")
    if not isinstance(bc_type, str) or not isinstance(method, str):
        raise TypeError("bc_type and method must be strings.")
    if bc_type not in ["dirichlet", "neumann", "robin"]:
        raise ValueError("bc_type must be either 'dirichlet', 'neumann', or 'robin'.")
    if method not in ["explicit", "implicit", "crank-nicolson"]:
        raise ValueError("method must be either 'explicit', 'implicit', or 'crank-nicolson'.")

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
    else:
        raise ValueError("Invalid boundary condition type and method combination.")

    return u, x


# define the source term for the bratu equation (e^mu*u)
#def q(x, t, u, mu):
    #return (1-u)*np.exp(-x)

#def q(x, t, u, mu):
    #return np.exp(mu*u)

def q(x, t, u, mu):
    return 1.0

# define the source term for the 

# define the initial condition
def u0(x, t):
    return 0.0

def plot_method_forall_bctypes(bctype):
    '''
    This function plots the solution for all methods for a given boundary condition type.
    
    Args: 
        bctype (str): The boundary condition type. It can be "dirichlet", "neumann" or "robin".
        
    Returns:
        None - plots the solution for all methods for a given boundary condition type.
    '''
    if bctype not in ["dirichlet", "neumann", "robin"]:
        raise ValueError("bctype must be either 'dirichlet', 'neumann', or 'robin'.")
    if bctype == "dirichlet":
        u,x = solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type=bctype, method="explicit", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
        plt.plot(x[1:-1], u[-1], 'o', label="explicit")
        u,x = solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type=bctype, method="implicit", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
        plt.plot(x[1:-1], u[-1], '--', label="implicit")
        u,x = solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type=bctype, method="crank-nicolson", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
        plt.plot(x[1:-1], u[-1], linewidth=0.5, label="crank-nicolson")
        plt.xlabel("x")
        plt.ylabel("u")
        plt. title("Each method for {} boundary conditions".format(bctype))
        plt.legend()
        plt.show()
    else:
        u,x = solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type=bctype, method="explicit", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
        plt.plot(x[1:], u[-1], 'o',label="explicit")
        u,x = solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type=bctype, method="implicit", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
        plt.plot(x[1:], u[-1], '--', label="implicit")
        u,x = solve_pde(N=10, a=0.0, b=1.0, D=1, u0=u0, t_max=1.0, dt=0.001, bc_type=bctype, method="crank-nicolson", q=q, alpha=0.0, beta=0.0, delta=0.0, gamma=0.0)
        plt.plot(x[1:], u[-1], linewidth=0.5, label="crank-nicolson")
        plt.xlabel("x")
        plt.ylabel("u")
        plt. title("Each method for {} boundary conditions".format(bctype))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # solve the PDE
    plot_method_forall_bctypes("dirichlet")
    plot_method_forall_bctypes("neumann")
    plot_method_forall_bctypes("robin")