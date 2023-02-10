#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  10 11:16:50 2023

@author: edenpage
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate

# Define the predator-prey function

def ode(t, u):
    a = 0.5
    d = 0.1
    x = u[0]
    y = u[1]
    b = 0.1
    print(b)

    dxdt = x(1-x) - (a*x*y)/(d + x)
    dydt = b*y*(1 - y/x)
    return np.array([dxdt, dydt])

# Define the initial conditions
u0 = np.array([0.1, 0.1])

# Solve the system
t = np.linspace(0, 100, 1000)
u = integrate.odeint(ode, u0, t)
# Plot the solution
plt.plot(t, u[:, 0], label='x')
plt.plot(t, u[:, 1], label='y')
plt.legend()
plt.show()


