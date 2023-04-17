
# import libraries
import numpy as np
from matplotlib import pyplot as plt

# Define the class
class BVPclass:
    # oink oink Sim a pig oink oink
    def __init__(self, bcleft, bcright, f, step_function):
        self.bcleft = bcleft
        self.bcright = bcright
        self.step_function = step_function
        self.f = f

    
    # Creat the grid
    def Grid(self, a, b, N):
        h = (b-a)/N
        t = np.linspace(a, b, N-1)
        return t, h
        
    # Define the finite difference method
    def finitediff(self, u, t, h):
        N = len(u)
        u_new = np.zeros(N)
        u_new[0] = self.bcleft
        u_new[-1] = self.bcright
        for i in range(1, N-1):
            u_new[i] = u[i] + h**2 * self.f(t[i], u[i])
        return u_new
    
    # Define the solve function 
    def solve(self, a, b, N):
        t, h = self.Grid(a, b, N)
        u = np.zeros(N-1)
        u[0] = self.bcleft
        u[-1] = self.bcright
        for i in range(1, N-2):
            u[i] = (self.bcleft + self.bcright)/2
        for i in range(0, N):
            u = self.step_function(self, u, t, h)
            print(u)
        return t, u
    
    # Plot function using dots for eack step where a and b are the endpoints and N is the number of points
    def plot(self, a, b, N):
        t, u = self.solve(a, b, N)
        plt.plot(t, u, 'o')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Numerical solution')
        plt.show()
