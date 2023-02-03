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
deltat_max = 0.01 # Step size
x0 = 1 # Initial condition
t0 = 0 # Initial time

# Euler step function
def euler_step(f, t, x, delta_t):
    return x + delta_t * f(t, x)

# Runge-Kutta step function
def runge_kutta_step(f, t, x, delta_t):
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
    x_values = [x]
    while t <= 1:
        x = step_function(f, t, x, deltat_max)
        t += deltat_max
        t_values.append(t)
        x_values.append(x)
    return t_values, x_values

# Solve the ODE using Euler's method
t_values, x_values = solve_to(f, x0, t0, deltat_max, euler_step)
# solve the ODE using Runge-Kutta's method
t_values, x_values = solve_to(f, x0, t0, deltat_max, runge_kutta_step)
# plot the solutions
plt.plot(t_values, x_values, label='Euler')
plt.plot(t_values, x_values, label='Runge-Kutta')
plt.show()

# Calculate the error for eulers method for different step sizes 
delta_t_values = np.logspace(-4, 0, 100)
error_values = []
for delta_t in delta_t_values:
    t_values, x_values = solve_to(f, x0, t0, delta_t, euler_step)
    x_analytical = np.exp(t_values)
    error_values.append(np.abs(x_values - x_analytical).max())
# plot with double logarithmic scale showing how the error depends on the size of the timestep delta_t
plt.loglog(delta_t_values, error_values)
plt.xlabel('delta_t')
plt.ylabel('error')
plt.title('Error in Euler method')
plt.show()

