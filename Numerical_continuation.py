import numpy as np
from scipy.optimize import fsolve, root
import matplotlib.pyplot as plt

def f(x, c):
    return x**3 - x + c

def cubic_continuation(f, x0, c0, step_size, max_steps, solver, method):
    results = []
    for i in range(max_steps):
        # Update the parameter value
        c = c0 + i*step_size
        
        if method == 'natural':
            # Solve the equation for the current parameter value
            sol = solver(lambda x: f(x, c), x0)

            # Store the solution
            if solver.__name__ == 'root':
                if sol.success:
                    results.append((c, sol.x[0]))
                    x0 = sol.x[0]
                else:
                    print('Solver failed at c =', c)
            elif solver.__name__ == 'fsolve':
                results.append((c, sol[0]))
                x0 = sol[0]
            else:
                raise ValueError('Unknown solver')
        elif method == 'arclength':
            # Psuedo-arclength continuation
            # Define the tangent direction
            def df(x, c):
                return 3*x**2 - 1
            
            # Solve the equation for the current parameter value and tangent direction
            sol = solver(lambda x: np.array([f(x[0], c) - x[1], df(x[0], c) - x[2], x[1]**2 + x[2]**2 - 1]), np.array([x0, 0, 1]))

            # Store the solution
            if solver.__name__ == 'root':
                if sol.success:
                    results.append((c, sol.x[0]))
                    x0 = sol.x[0]
                else:
                    print('Solver failed at c =', c)
            elif solver.__name__ == 'fsolve':
                results.append((c, sol[0]))
                x0 = sol[0]
            else:
                raise ValueError('Unknown solver')
        else:
            raise ValueError('Unknown method, please use "natural" or "arclength"')

    return results

# Define the initial guess and the parameter range
x0_1 = 1
c0 = -2

# Solve the equation
natural_solutions = cubic_continuation(f, x0_1, c0, 0.1, 40, fsolve, 'natural')
psuedo_solutions = cubic_continuation(f, x0_1, c0, 0.1, 40, fsolve, 'arclength')

# Plot the solution
natural_c_values = [sol[0] for sol in natural_solutions]
natural_x_values = [sol[1] for sol in natural_solutions]
psuedo_c_values = [sol[0] for sol in psuedo_solutions]
psuedo_x_values = [sol[1] for sol in psuedo_solutions]

plt.plot(natural_c_values, natural_x_values, label='Natural continuation')
plt.plot(psuedo_c_values, psuedo_x_values, label='Psuedo-arclength continuation')
plt.xlabel('c')
plt.ylabel('x')
plt.legend()
plt.show()




