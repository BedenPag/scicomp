import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

def f(x, c):
    return x**3 - x + c

def continuation(f, x0, c0, step_size, max_steps, solver):
    results = []
    for i in range(max_steps):
        # Update the parameter value
        c = c0 + i*step_size

        # Solve the equation forward from the initial condition
        sol_forward = solver(lambda x: f(x, c), x0)
        if sol_forward.success:
            results.append((c, sol_forward.x[0]))
            x0 = sol_forward.x[0]
        else:
            print('Solver failed at c =', c)
            break

        # Solve the equation backward from the initial condition
        sol_backward = solver(lambda x: f(x, c), x0-0.01) # Slightly offset initial condition
        if sol_backward.success:
            results.append((c, sol_backward.x[0]))
            x0 = sol_backward.x[0]
        else:
            print('Solver failed at c =', c)
            break

    return results

# Define the initial guess and the parameter range
x0 = 1  # Use a single initial condition
c0 = -2

# Solve the equation
solutions = continuation(f, x0, c0, 0.1, 40, root)

# Plot the solution
c_values = [sol[0] for sol in solutions]
x_values = [sol[1] for sol in solutions]

plt.plot(c_values, x_values,'o')
plt.xlabel('c')
plt.ylabel('x')
plt.show()