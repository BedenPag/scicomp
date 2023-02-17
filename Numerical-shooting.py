#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  10 11:16:50 2023

@author: edenpage
"""

#%% Import libraries

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

#%% Small b value

# Define the predator-prey function where b = 0.1 (lower limit)
def ode(t, u):
    x = u[0] # Number of prey
    y = u[1] # Number of predators
    a = 0.5
    d = 0.1
    b = 0.1
    dxdt = x*(1-x) - (a*x*y)/(d + x)
    dydt = b*y*(1 - (y/x))
    return np.array([dxdt, dydt])

# Define the time interval
deltat_max = 0.01

# Define the initial conditions
u0 = [0.5,0.5]

# Define the step function - using this instead of a scipy solver to get a better understanding of the process
def RK4(f, t, u, delta_t): 
    k1 = f(t, u)
    k2 = f(t + delta_t/2, u + delta_t/2 * k1)
    k3 = f(t + delta_t/2, u + delta_t/2 * k2)
    k4 = f(t + delta_t, u + delta_t * k3)
    return u + delta_t/6 * (k1 + 2*k2 + 2*k3 + k4)

# Define the solver function 
def solve_to(f, u0, t0, deltat_max, step_function):
    t = t0
    u = u0
    t_values = [t]
    values = [u]
    while t <= 50:
        u = step_function(f, t, u, deltat_max)
        t += deltat_max
        t_values.append(t)
        values.append(u)
    return t_values, values

# Solve the predator-prey ODE using Runge-Kutta's method
t_values, values = solve_to(ode, u0, 0, deltat_max, RK4)
x_values = [x[0] for x in values] # Extract the x values from the list of values
y_values = [x[1] for x in values] # Extract the y values from the list of values


# Plot the solution
plt.plot(t_values, x_values, label='Prey')
plt.plot(t_values, y_values, label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Predator-Prey Model using Runge-Kutta Method for a small b value')
plt.legend()
plt.show()

# limit cycle plot for small b
plt.plot(x_values, y_values)
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title('Phase-Plane Plot for the Predator-Prey Model with a small b value')
plt.show()

#%% Large b value

# Define the predator-prey function where b = 0.5 (upper limit)
def ode(t, u):
    x = u[0] # Number of prey
    y = u[1] # Number of predators
    a = 0.5
    d = 0.1
    b = 0.5
    dxdt = x*(1-x) - (a*x*y)/(d + x)
    dydt = b*y*(1 - y/x)
    return np.array([dxdt, dydt])

# Define the step function - using this instead of a scipy solver to get a better understanding of the process
def RK4(f, t, u, delta_t): 
    k1 = f(t, u)
    k2 = f(t + delta_t/2, u + delta_t/2 * k1)
    k3 = f(t + delta_t/2, u + delta_t/2 * k2)
    k4 = f(t + delta_t, u + delta_t * k3)
    return u + delta_t/6 * (k1 + 2*k2 + 2*k3 + k4)

# Define the solver function 
def solve_to(f, u0, t0, deltat_max, step_function):
    t = t0
    u = u0
    t_values = [t]
    values = [u]
    while t <= 50:
        u = step_function(f, t, u, deltat_max)
        t += deltat_max
        t_values.append(t)
        values.append(u)
    return t_values, values

# Solve the predator-prey ODE using Runge-Kutta's method
t_values, values = solve_to(ode, u0, 0, deltat_max, RK4)
x_values = [x[0] for x in values] # Extract the x values from the list of values
y_values = [x[1] for x in values] # Extract the y values from the list of values


# Plot the solution
plt.plot(t_values, x_values, label='Prey')
plt.plot(t_values, y_values, label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Predator-Prey Model using Runge-Kutta Method for a large b value')
plt.legend()
plt.show()

# limit cycle plot for large b
plt.plot(x_values, y_values)
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title('Phase-Plane Plot for the Predator-Prey Model with a large b value')
plt.show()

#%% Shooting method

# Phase condition function using the initial conditions
def phase_condition(u0, u1):
    return u0[1] - u1[1]

# Shooting function using the phase condition
def shooting(u0, phase_condition):
    u1 = u0
    u1[1] += 0.1
    while phase_condition(u0, u1) < 0:
        u1[1] += 0.1
    while phase_condition(u0, u1) > 0:
        u1[1] -= 0.01
    return u1

# Solver function using the shooting method
def solve_to_shooting(f, u0, t0, deltat_max, step_function, phase_condition):
    u1 = shooting(u0, phase_condition) # find the initial conditions using the shooting method
    t_values, values = solve_to(f, u1, t0, deltat_max, step_function) # solve the ODE using the initial conditions found
    return t_values, values

# Solve the predator-prey ODE using Shooting method and Runge-Kutta's method
t_values, values = solve_to_shooting(ode, u0, 0, deltat_max, RK4, phase_condition)
x_values = [x[0] for x in values] # Extract the x values from the list of values
y_values = [x[1] for x in values] # Extract the y values from the list of values

# Plot the limit cycle for the shooting method
plt.plot(x_values, y_values)
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title('Phase-Plane Plot with Shooting Method for a large b value')
plt.show()









# %%
