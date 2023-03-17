# import libraries
import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.integrate
import scipy.optimize

from BVP_class import BVPclass

gamma_1 = 1
gamma_2 = 2
a = 1
b = 2
N = 50
f = 


grid = np.linspace(a, b, N-1)
def real_sol(x, a, b, alpha, beta):
    ((-alpha+beta)/(b-a))*(x-a) +alpha
    return ((-alpha+beta)/(b-a))*(x-a) +alpha

true_y = real_sol(grid,a,b,gamma_1,gamma_2)
plt.plot(grid, true_y)
plt.show()

finite = BVPclass.finitediff
solver = BVPclass(gamma_1, gamma_2, f, finite)

solver.plot(a, b, N)



