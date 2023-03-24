import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate

# Define the function
def f(x, c):
    return x**3 - x + c


# Define the shooting method
def shooting(ode, x0, par):
    # Define the time interval
    t0 = 0
    tmax = 1
    deltat_max = 0.01
    # Define the step function
    def RK4(ode, t, x, delta_t, par):
        k1 = ode(x, par)
        k2 = ode(t + delta_t/2, x + delta_t/2 * k1, par)
        k3 = ode(t + delta_t/2, x + delta_t/2 * k2, par)
        k4 = ode(t + delta_t, x + delta_t * k3, par)
        return x + delta_t/6 * (k1 + 2*k2 + 2*k3 + k4)
    # Define the solver function
    def solve_to(ode, x0, t0, deltat_max, step_function, par):
        t = t0
        x = x0
        while t <= tmax:
            x = step_function(ode, t, x, deltat_max, par)
            t += deltat_max
        return x
    # Solve the ODE
    x = solve_to(ode, x0, t0, deltat_max, RK4, par)
    return x

# Define the natural parameter continuation function
def continuation(ode, x0, par0, vary_par=0, step_size=0.1, max_steps=100, discretisation=shooting, solver=scipy.optimize.fsolve):
    # Define the initial conditions
    x = x0
    par = par0
    # Define the step function
    def step_function(ode, x, par, step_size):
        par[vary_par] += step_size
        x = solver(ode, x, par)
        return x, par
    # Define the solver function
    def solve_to(ode, x, par, step_size, step_function):
        for i in range(max_steps):
            x, par = step_function(ode, x, par, step_size)
        return x, par
    # Solve the ODE
    x, par = solve_to(ode, x, par, step_size, step_function)
    return x, par

# Define the initial conditions
x0 = 0.5
par0 = 0.5

# Plot the solution for values of c between -2 and 2
c_values = np.linspace(-2, 2, 100)
x_values = [shooting(f, x0, c) for c in c_values]
plt.plot(c_values, x_values)
plt.show()

# Perform natural parameter continuation
results = continuation(f, x0, par0, vary_par=0, step_size=0.1, max_steps=100, discretisation=shooting, solver=scipy.optimize.fsolve)

# Plot the solution
plt.plot(results[0], results[1])
plt.show()









