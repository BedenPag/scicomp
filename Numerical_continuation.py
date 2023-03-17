import numpy as np
from matplotlib import pyplot as plt
import scipy
def continuation(ode, x0, t0, tf, dt, n):
    """
    Continuation of the solution of an ODE
    ode: function of the form ode(x, t)
    x0: initial condition
    t0: initial time
    tf: final time
    dt: time step
    n: number of time steps
    discretisation: function of the form discretisation(ode, x0, t0, tf, dt, n)
    solver: function of the form solver(ode, x0, t0, tf, dt, n)
    """
    # Initial guess
    x = np.zeros((n+1, len(x0)))
    x[0] = x0
    t = np.linspace(t0, tf, n+1)
    # Iterate over the time steps
    for i in range(n):
        # Solve the ODE
        sol = scipy.integrate.solve_ivp(ode, [t[i], t[i+1]], x[i], method='RK45', args=(a, b, d))
        # Update the solution
        x[i+1] = sol.y[:, -1]
    return x, t

def funct(x):
    y = x[0]**3 + x[0] - x[1]
    return np.array([y])

if __name__ == "__main__":
    #%% Define the parameters
    a = 1
    b = 2
    d = 3
    #%% Define the initial conditions
    x0 = [0.5,0.5]
    t0 = 0
    tf = 1000
    dt = 0.01
    n = int((tf-t0)/dt)
    #%% Solve the ODE
    x, t = continuation(funct, x0, t0, tf, dt, n)
    
    #%%Plot the real solution
    plt.plot(t, x[:,0], label='real')
    #%% Plot the solution
    plt.plot(t, x[:,1], label='Continuation')
    plt.legend()
    plt.show()