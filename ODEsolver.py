import numpy as np
import matplotlib.pyplot as plt

# Euler step function
def euler_step(f, t, x, delta_t):
    return x + delta_t * f(t, x)

# Runge-Kutta step function
def rk4_step(f, t, x, delta_t):
    k1 = f(t, x)
    k2 = f(t + delta_t/2, x + delta_t/2 * k1)
    k3 = f(t + delta_t/2, x + delta_t/2 * k2)
    k4 = f(t + delta_t, x + delta_t * k3)
    return x + delta_t/6 * (k1 + 2*k2 + 2*k3 + k4)

# Solver function
def solve_to(f, x0, t0, deltat_max, step_function):
    t = t0
    x = x0
    t_values = [t]
    values = [x]
    while t <= 20:
        x = step_function(f, t, x, deltat_max)
        t += deltat_max
        t_values.append(t)
        values.append(x)
    return t_values, values

def compare_euler_rk4_error(f, x0, t0, delta_t_values):
    """
    Compare the error of Euler's method and Runge-Kutta (RK4) method for different step sizes.

    Args:
        f (callable): Function representing the ODE to be solved.
        x0 (float): Initial value of the dependent variable.
        t0 (float): Initial time.
        delta_t_values (array-like): Array of step sizes (time increments) to be tested.

    Returns:
        None: Plots the error vs. step size on a double logarithmic scale.
    """
    # Initialize arrays to store error values
    error_euler = []
    error_rk4 = []

    for t in delta_t_values:
        # Solve with Euler's method
        t_values_euler, values_euler = solve_to(f, x0, t0, t, euler_step)
        x_analytical = np.exp(t_values_euler)
        error_euler.append(np.abs(values_euler - x_analytical).max())

        # Solve with Runge-Kutta (RK4) method
        t_values_rk4, values_rk4 = solve_to(f, x0, t0, t, rk4_step)
        x_analytical = np.exp(t_values_rk4)
        error_rk4.append(np.abs(values_rk4 - x_analytical).max())

    # Plot the error vs. step size on a double logarithmic scale
    plt.loglog(delta_t_values, error_euler, label='Euler')
    plt.loglog(delta_t_values, error_rk4, label='Runge-Kutta')
    plt.xlabel('delta_t')
    plt.ylabel('error')
    plt.title("Error in Euler's method and RK4 method")
    plt.legend()
    plt.show()