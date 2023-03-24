import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def f(x, c):
    return x**3 - x + c

def continuation(f, x0, c0, step_size, max_steps, solver):
    results = []
    for i in range(max_steps):
        # Update the parameter value
        c = c0 + i*step_size

        # Solve the equation for the current parameter value
        sol = solver(lambda x: f(x, c), x0)

        # Store the solution
        results.append((c, sol[0]))

        # Update the initial guess for the next iteration
        x0 = sol[0]

    return results

# Define the initial guess and the parameter range
x0 = 0.5
c0 = -2

# Solve the equation
solutions = continuation(f, x0, -2, 0.01, 400, fsolve)

# Plot the solution
c_values = [sol[0] for sol in solutions]
x_values = [sol[1] for sol in solutions]

plt.plot(c_values, x_values)
plt.xlabel('c')
plt.ylabel('x')
plt.show()




