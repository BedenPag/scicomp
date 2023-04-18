# Use the explicit Euler method to solve the linear diffusion equation (du/dt = D d^2u/dx^2) without a source term



import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.integrate
import scipy.optimize

from BVP_uberclass import Grid, BoundaryCondition, construct_A_and_b

# define the explicit Euler method
def explicit_euler(u, f, dt):
    '''Parameters:
    u: numpy array, the solution at the previous time step
    f: function, the right-hand side of the ODE
    dt: float, the time step

    Output:
    u: numpy array, the solution at the current time step
    '''
    u = u + dt * f
    return u

# define the implicit Euler method
def implicit_euler(u, f, dt):
    '''Parameters:
    u: numpy array, the solution at the previous time step
    f: function, the right-hand side of the ODE
    dt: float, the time step

    Output:
    u: numpy array, the solution at the current time step
    '''
    u = scipy.optimize.fsolve(lambda u: u - dt * f, u)
    return u

# define the Crank-Nicolson method
def crank_nicolson(u, f, dt):
    '''Parameters:
    u: numpy array, the solution at the previous time step
    f: function, the right-hand side of the ODE
    dt: float, the time step

    Output:
    u: numpy array, the solution at the current time step
    '''
    u = scipy.optimize.fsolve(lambda u: u - dt * f - 0.5 * dt**2 * f(u - 0.5 * dt * f), u)
    return u

# define the function f(u) = D d^2u/dx^2
def f(u):
    '''Parameters:
    u: numpy array, the solution at the previous time step

    Output:
    f: numpy array, the right-hand side of the ODE
    '''
    f = D * np.gradient(np.gradient(u))
    return f

# define the initial condition
def u0(x):
    '''Parameters:
    x: numpy array, the spatial grid

    Output:
    u0: numpy array, the initial condition
    '''
    u0 = np.sin(np.pi * x)
    return u0

# define the exact solution
def u_exact(x, t):
    '''Parameters:
    x: numpy array, the spatial grid
    t: float, the time

    Output:
    u_exact: numpy array, the exact solution
    '''
    u_exact = np.exp(-np.pi**2 * D * t) * np.sin(np.pi * x)
    return u_exact

# define the function to plot the solution
def plot_solution(x, u, u_exact, t, method):
    '''Parameters:
    x: numpy array, the spatial grid
    u: numpy array, the numerical solution
    u_exact: numpy array, the exact solution
    t: float, the time
    method: string, the numerical method

    Output:
    None
    '''
    plt.plot(x, u, 'o', label = 'numerical solution')
    plt.plot(x, u_exact, 'r', label = 'exact solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('t = ' + str(t) + ', ' + method)
    plt.legend()
    plt.show()

# run the code
if __name__ == '__main__':
    # define the diffusion coefficient
    D = 1.0

    # define the spatial grid
    grid = Grid(50, 0.0, 1.0)
    dx = grid.dx
    x = grid.x

    # define the time grid
    t_min = 0.0
    t_max = 1.0
    N_t = 100
    t = np.linspace(t_min, t_max, N_t)

    # define the time step
    dt = t[1] - t[0]

    # define the initial condition
    u = u0(x)

    # define the exact solution
    u_exact = u0(x)

    # define the boundary conditions
    bc_left = BoundaryCondition('Dirichlet', 0.0)
    bc_right = BoundaryCondition('Dirichlet', 0.0)

    # define the matrix A and the vector b
    A, b = construct_A_and_b(grid, bc_left, bc_right)

    # define the right-hand side of the ODE
    f = f(u)

    # solve the ODE using the explicit Euler method
    for i in range(1, N_t):
        u = explicit_euler(u, f, dt)
        u_exact = u_exact(x, t[i])
        plot_solution(x, u, u_exact, t[i], 'explicit Euler')
    
    # solve the ODE using the implicit Euler method
    for i in range(1, N_t):
        u = implicit_euler(u, f, dt)
        u_exact = u_exact(x, t[i])
        plot_solution(x, u, u_exact, t[i], 'implicit Euler')

    # solve the ODE using the Crank-Nicolson method
    for i in range(1, N_t):
        u = crank_nicolson(u, f, dt)
        u_exact = u_exact(x, t[i])
        plot_solution(x, u, u_exact, t[i], 'Crank-Nicolson')





