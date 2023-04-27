from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt

# Natural parameter continuation

def f(x, c):
    return x**3 - x + c

def natural_param_cont(f, x0, c0, c1):
    c = np.linspace(c0, c1, 500) # Parameter range - interestingly it doesn't plot the entire range if num of points < 101
    solutions = [x0]
    c_array = [c0]
    failed_c = []
    for i in range(len(c)-1):
        sol = root(f,solutions[-1],args=(c[i]))
        if sol.success == True:
            solutions.append(sol.x[0])
            c_array.append(c[i])
        else:
            failed_c.append(c[i])
    if len(failed_c) > 0:
        print('Solver failed when c was in the range', failed_c[0], 'to', failed_c[-1])
    return c_array[1:], solutions[1:]

# Psuedo-arclength continuation

def arclength_cont(f, x0, c0, c1):
    c = np.linspace(c0, c1, 500)
    x = [x0]
    solutions = np.array([])
    c_array = np.array([])
    for i in [0,1]: # Find the first two values of x for the first two values of c
        sol = root(f, x[-1], args=(c[i]))
        if sol.success == True:
            solutions = np.append(solutions, sol.x[0])
            c_array = np.append(c_array, c[i])
        else:
            raise ValueError('Unable to find a solution for the initial value of c')
    first_sol = np.array([solutions[0], c_array[0]])
    second_sol = np.array([solutions[1], c_array[1]])
    # Define the secant vector and the predicted next x value
    secant = second_sol - first_sol
    next_x = second_sol + secant
    # define the conditions for the root finder 
    def cond(x):
        return np.array([f(x[0], x[1]), np.dot(secant, x - next_x)])
    i = c0
    while i < c1:
        sol = root(cond, next_x)
        first_sol = second_sol
        second_sol = np.array([sol.x[0], sol.x[1]])
        # Update the secant vector and the next x
        secant = second_sol - first_sol
        next_x = second_sol + secant
        if sol.success == True:
            solutions = np.append(solutions, sol.x[0])
            c_array = np.append(c_array, sol.x[1])
            i = sol.x[1]
        else:
            i += 0.01
    return c_array, solutions

# Define the initial guess and the parameter range
x0 = 1
c0 = -2
c1 = 2

def plot_continuation(f, x0, c0, c1, method):
    if method == 'natural':
        c, x = natural_param_cont(f, x0, c0, c1)
        plt.plot(c, x, 'o')
        plt.xlabel('c')
        plt.ylabel('x')
        plt.title('Natural parameter continuation')
        plt.show()
    elif method == 'arclength':
        c, x = arclength_cont(f, x0, c0, c1)
        plt.plot(c, x, 'o')
        plt.xlabel('c')
        plt.ylabel('x')
        plt.title('Psuedo-arclength continuation')
        plt.show()

plot_continuation(f, x0, c0, c1, 'natural')
plot_continuation(f, x0, c0, c1, 'arclength')

