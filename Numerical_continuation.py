import numpy as np
from matplotlib import pyplot as plt

# Write a code that performs natural parameter continuation

def f(x,c):
    return x**3 - x + c
        
def natural_parameter_continuation(x0, c0, c1, N):
    '''
    x0: initial value
    c0: initial parameter value
    c1: final parameter value
    N: number of steps
    '''
    x = np.zeros(N)
    c = np.linspace(c0, c1, N)
    x[0] = x0
    for i in range(N-1):
        x[i+1] = x[i] - f(x[i], c[i])/(3*x[i]**2 - 1)
    return x, c

# Plot c vs x

x, c = natural_parameter_continuation(0.5, 0, 1, 100)
plt.plot(c, x)
plt.xlabel('c')
plt.ylabel('x')
plt.show()


