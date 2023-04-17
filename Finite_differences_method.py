# import libraries
import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.integrate
import scipy.optimize

from BVP_class import BVPclass

# boundary conditions
gamma_1 = 4
gamma_2 = 8

# initial conditions
a = 0
b = 20
N = 50

# define the function d^2u/dx^2 = 0
def f(x, u):

    return 0

# define the grid
grid = np.linspace(a, b, N-1)
def real_sol(x, a, b, alpha, beta):
    ((-alpha+beta)/(b-a))*(x-a) +alpha
    return ((-alpha+beta)/(b-a))*(x-a) +alpha

true_y = real_sol(grid,a,b,gamma_1,gamma_2)
plt.plot(grid, true_y)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('True solution')
plt.show()

finite = BVPclass.finitediff
solver = BVPclass(gamma_1, gamma_2, f, finite)
solver.plot(a, b, N)


