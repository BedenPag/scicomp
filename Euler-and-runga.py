#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:17:59 2023

@author: edenpage
"""

import numpy as np
from matplotlib import pyplot as plt


# Define the function
f = lambda t, x: x # x' = x ODE
deltat_max = 0.1 # Step size
x0 = 1 # Initial condition
t0 = 0 # Initial time

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

# Solve the first order ODE using Euler's method
t_values, values = solve_to(f, x0, t0, deltat_max, euler_step)
# solve the first order ODE using Runge-Kutta's method
t_values, values = solve_to(f, x0, t0, deltat_max, rk4_step)

# Calculate the error for eulers method for different step sizes 
delta_t_values = np.logspace(-4, 0, 100)
error = []
for t in delta_t_values:
    t_values, values = solve_to(f, x0, t0, t, euler_step)
    x_analytical = np.exp(t_values)
    error.append(np.abs(values - x_analytical).max())
# plot with double logarithmic scale showing how the error depends on the size of the timestep delta_t
plt.loglog(delta_t_values, error, label = 'Euler')

# Calculate the error for runge-kutta's method for different step sizes 
delta_t_values = np.logspace(-4, 0, 100)
error = []
for t in delta_t_values:
    t_values, values = solve_to(f, x0, t0, t, rk4_step)
    x_analytical = np.exp(t_values)
    error.append(np.abs(values - x_analytical).max())
plt.loglog(delta_t_values, error, label='Runge-Kutta')
plt.xlabel('delta_t')
plt.ylabel('error')
plt.title('Error in Euler''s method and RK4 method')
plt.legend()
plt.show()

# Define the true solution for the 2nd order ODE x'' = -x
def true_sol(t):
    x = np.sin(t) + np.cos(t)
    y = np.cos(t) - np.sin(t)
    return np.array([x, y])


# Define the 2nd order ODE x'' = -x
def f1(t, x):
    x_actual = x[0]
    y = x[1]

    dydt = -x_actual
    dxdt = y
    return np.array([dxdt, dydt])

x0 = np.array([1, 1]) # Initial condition, x = 1, y = 1


# Solve for x and y using Euler's method
t_values, values_euler = solve_to(f1, x0, t0, deltat_max, euler_step)
euler_x_values = [x[0] for x in values_euler]
euler_y_values = [x[1] for x in values_euler]
# Solve for x and y using Runge-Kutta's method
t_values, values_rk4 = solve_to(f1, x0, t0, deltat_max, rk4_step)
rk4_x_values = [x[0] for x in values_rk4]
rk4_y_values = [x[1] for x in values_rk4]
# Calculate the true solution for x and y
t = np.linspace(0, 20, 100)
true = true_sol(t)
true_x = true[0]
true_y = true[1]


plt.subplot(2, 1, 1)
# Plot the solution for x using Euler's method and Runge-Kutta's method and the true solution
plt.plot(t_values, euler_x_values, label='Euler')
plt.plot(t_values, rk4_x_values, label='Runge-Kutta')
plt.plot(t, true_x, label='True x', color='black', linestyle=':')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Solution for x using Euler''s method and RK4 method with step size 0.1')
plt.legend()
plt.subplot(2, 1, 2)
# Plot the solution for y using Euler's method and Runge-Kutta's method and the true solution
plt.plot(t_values, euler_y_values, label='Euler')
plt.plot(t_values, rk4_y_values, label='Runge-Kutta')
plt.plot(t, true_y, label='True y', color='black', linestyle=':')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution for y using Euler''s method and RK4 method with step size 0.1')
plt.legend()
plt.show()


# We can see that Euler's method is not accurate for this step size while Runge-Kutta's method is much more accurate for this step size
# The true value is the same as the Runge-Kutta's method

# If we use a smaller step size, we can see that Euler's method becomes closer to the true value

deltat_max = 0.01 # Step size
# Solve for x and y using Euler's method
t_values, values_euler = solve_to(f1, x0, t0, deltat_max, euler_step)
euler_x_values = [x[0] for x in values_euler]
euler_y_values = [x[1] for x in values_euler]
# Solve for x and y using Runge-Kutta's method
t_values, values_rk4 = solve_to(f1, x0, t0, deltat_max, rk4_step)
rk4_x_values = [x[0] for x in values_rk4]
rk4_y_values = [x[1] for x in values_rk4]
# Calculate the true solution for x and y
t = np.linspace(0, 20, 100)
true = true_sol(t)
true_x = true[0]
true_y = true[1]

plt.subplot(2, 1, 1)
# Plot the solution for x using Euler's method and Runge-Kutta's method and the true solution
plt.plot(t_values, euler_x_values, label='Euler')
plt.plot(t_values, rk4_x_values, label='Runge-Kutta', linestyle='--')
plt.plot(t, true_x, label='True x', color='black', linestyle=':')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Solution for x using Euler''s method and RK4 method with a step size 0.01')
plt.legend()
plt.subplot(2, 1, 2)
# Plot the solution for y using Euler's method and Runge-Kutta's method and the true solution
plt.plot(t_values, euler_y_values, label='Euler')
plt.plot(t_values, rk4_y_values, label='Runge-Kutta', linestyle='--')
plt.plot(t, true_y, label='True y', color='black', linestyle=':')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution for y using Euler''s method and RK4 method with a step size 0.01')
plt.legend()
plt.show()


