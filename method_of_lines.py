# Use the explicit Euler method to solve the linear diffusion equation (du/dt = D d^2u/dx^2) without a source term



import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.integrate
import scipy.optimize

from BVP_class import BVPclass

# boundary conditions
gamma_1 = 0
gamma_2 = 0



# initial conditions
a = 0
b = 1
N = 50

u_0 = np.sin((np.pi*(x-a))/(b-a))



