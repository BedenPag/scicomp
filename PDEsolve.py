# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from BVP_uberclass import Grid

def heat_equation_solver(kappa, L, T, u_I, u_exact, mx, mt):
    """
    Solves the 1D heat equation using the forward Euler method.

    Parameters:
        - kappa (float): diffusion constant
        - L (float): length of the domain
        - T (float): final time
        - u_I (function): initial temperature distribution function u_I(x)
        - mx (int): number of grid points in space
        - mt (int): number of grid points in time

    Returns:
        - plot of the numerical solution and exact solution
    """

    # Set up the numerical environment variables using the Grid class
    x = np.linspace(0, L, mx + 1) # grid points in space
    xx = np.linspace(0, L, 250) # fine grid for plotting exact solution
    t = np.linspace(0, T, mt + 1) # grid points in time
    dx = x[1] - x[0] # grid spacing in space
    dt = t[1] - t[0] # grid spacing in time
    lamda = kappa * dt / (dx**2) # fourier number
    

    # Set up the solution variables
    u_t = np.zeros(x.size)        # u at time step t
    u_t1 = np.zeros(x.size)      # u at time step t+1

    # Set initial condition
    for i in range(0, mx+1):
        u_t[i] = u_I(x[i])

    # Solve the PDE: loop over all time points
    for j in range(0, mt):
        # Forward Euler timestep at inner mesh points
        # PDE discretized at position x[i], time t[j]
        for i in range(1, mx):
            u_t1[i] = u_t[i] + lamda * (u_t[i - 1] - 2 * u_t[i] + u_t[i + 1])

        # Boundary conditions
        u_t1[0] = 0
        u_t1[mx] = 0

        # Save u_t at time t[j+1]
        u_t[:] = u_t1[:]

    # Plot the final result and exact solution
    plt.plot(x, u_t, 'ro', label='Numerical solution')
    plt.plot(xx, u_exact(xx,T), 'b-', label='Exact solution')
    plt.xlabel('x')
    plt.ylabel('u(x,{})'.format(T))
    plt.legend(loc='upper right')
    plt.show()

# Define the problem parameters
kappa = 1.0   # diffusion constant
L = 2.0         # length of the domain
T = 0.5         # final time

# Define initial temperature distribution function
def u_I(x):
    # initial temperature distribution
    y = np.sin(pi * x / L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

# Set numerical parameters
mx = 10     # number of grid points in space
mt = 1000   # number of grid points in time

# Call the heat_equation_solver function

heat_equation_solver(kappa, L, T, u_I, u_exact, mx, mt)