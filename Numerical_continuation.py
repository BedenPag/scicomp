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

        # Solve the equation for the current parameter value
        sol = solver(lambda x: f(x, c), x0)

        # Store the solution
        if sol.success:
            results.append((c, sol.x[0]))
            x0 = sol.x[0]
        else:
            print('Solver failed at c =', c)
            

    return results

# Define the initial guess and the parameter range
x0_1 = 1
x0_2 = -1
c0 = -2

# Solve the equation
solutions = continuation(f, x0_1, c0, 0.1, 40, root)
solutions += continuation(f, x0_2, c0, 0.1, 40, root)

# Plot the solution
c_values = [sol[0] for sol in solutions]
x_values = [sol[1] for sol in solutions]

plt.plot(c_values, x_values,'o')
plt.xlabel('c')
plt.ylabel('x')
plt.show()




