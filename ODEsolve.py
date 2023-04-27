import numpy as np
import matplotlib.pyplot as plt

# Euler step function
def euler_step(f, t, x, delta_t):
    '''
    Take a single step using Euler's method.

    Args:
        f (callable): Function representing the ODE to be solved.
        t (float): Current time.
        x (list): Current value of the dependent variable(s).
        delta_t (float): Time increment.
    
    Returns:
        x (list): New value of the dependent variable(s).
    '''
    return x + delta_t * f(t, x)

# Runge-Kutta step function
def rk4_step(f, t, x, delta_t):
    '''
    Take a single step using the Runge-Kutta method.

    Args:
        f (callable): Function representing the ODE to be solved.
        t (float): Current time.
        x (list): Current value of the dependent variable.
        delta_t (float): Time increment.
    
    Returns:
        x (list): New value of the dependent variable.
    '''
    k1 = f(t, x)
    k2 = f(t + delta_t/2, x + delta_t/2 * k1)
    k3 = f(t + delta_t/2, x + delta_t/2 * k2)
    k4 = f(t + delta_t, x + delta_t * k3)
    return x + delta_t/6 * (k1 + 2*k2 + 2*k3 + k4)

# Solver function
def solve_to(f, x0, t0, deltat_max, step_function, true_sol=None):
    '''
    Solve the ODE using the specified step function.
    
    Args:
        f (callable): Function representing the ODE to be solved.
        x0 (list): Initial value of the dependent variable.
        t0 (float): Initial time.
        deltat_max (float): Maximum step size (time increment).
        step_function (callable): Function to be used to take a single step.
        true_sol (callable): Function representing the true solution to the ODE.

    Returns:
        t_values (list): List of time values.
        values (list): List of dependent variable(s) values.
    '''
    if f is None:
        raise ValueError('No function provided. The function requires the form solve_to(f, x0, t0, deltat_max, step_function, true_sol=None).')
    elif f.__code__.co_argcount < 2:
        raise ValueError('Invalid function provided. The function must have the form f(t, u, a, b, c) where t is the time, u is the dependent variable(s) and a, b, c are optional parameters required.')
    elif f.__code__.co_argcount > 5:
        raise ValueError('Invalid function provided. The function must have the form f(t, u, a, b, c) where t is the time, u is the dependent variable(s) and a, b, c are optional parameters required.')
    elif x0 is None:
        raise ValueError('No initial condition provided. The function requires the form solve_to(f, x0, t0, deltat_max, step_function, true_sol=None).')
    elif t0 is None:
        raise ValueError('No initial time provided. The function requires the form solve_to(f, x0, t0, deltat_max, step_function, true_sol=None).')
    elif step_function is None:
        raise ValueError('No step function provided. The function requires the form solve_to(f, x0, t0, deltat_max, step_function, true_sol=None).')
    elif step_function != euler_step and step_function != rk4_step:
        raise ValueError('Invalid step function provided. The function must be either euler_step or rk4_step.')
    t = t0
    x = x0
    values = [x]
    t_values = [t]
    while t <= 21:
        x = step_function(f, t, x, deltat_max)
        t += deltat_max
        t_values.append(t)
        values.append(x)
    
    return t_values, values

def compare_euler_rk4_error(f, x0, t0, delta_t_values):
    """
    Compare the error of Euler's method and Runge-Kutta (RK4) method for different step sizes.

    Args:
        f (callable): Function representing the ODE to be solved.
        x0 (float): Initial value of the dependent variable.
        t0 (float): Initial time.
        delta_t_values (array-like): Array of step sizes (time increments) to be tested.

    Returns:
        None: Plots the error vs. step size on a double logarithmic scale.
    """
    # Initialize arrays to store error values
    error_euler = []
    error_rk4 = []

    for t in delta_t_values:
        # Solve with Euler's method
        t_values_euler, values_euler = solve_to(f, x0, t0, t, euler_step)
        x_analytical = np.exp(t_values_euler)
        error_euler.append(np.abs(values_euler - x_analytical).max())

        # Solve with Runge-Kutta (RK4) method
        t_values_rk4, values_rk4 = solve_to(f, x0, t0, t, rk4_step)
        x_analytical = np.exp(t_values_rk4)
        error_rk4.append(np.abs(values_rk4 - x_analytical).max())

    # Plot the error vs. step size on a double logarithmic scale
    plt.loglog(delta_t_values, error_euler, label='Euler')
    plt.loglog(delta_t_values, error_rk4, label='Runge-Kutta')
    plt.xlabel('delta_t')
    plt.ylabel('error')
    plt.title("Error in Euler's method and RK4 method")
    plt.legend()
    plt.show()

def solve_euler_rk4(f, x0, t0, deltat_max, step_function, true_sol=None):
    """
    Solve the ODE using Euler's method and Runge-Kutta (RK4) method for different step sizes.

    Args:
        f (callable): Function representing the ODE to be solved.
        x0 (list): Initial value of the dependent variable.
        t0 (float): Initial time.
        delta_t_values (array-like): Array of step sizes (time increments) to be tested.
        true_sol (callable): Function representing the true solution to the ODE.
    
    Returns:
        None: Plots the solution vs. time for each step size.
    """
    t_values, values = solve_to(f, x0, t0, deltat_max, step_function, true_sol)
    # Initialize arrays to store error values
    if len(x0) == 1:
        # Plot the solution for x using the specified step function
        plt.plot(t_values, values, 'o', label=step_function.__name__)
        # Plot the true solution for x
        if true_sol is not None:
            domain = np.linspace(0, 21, 100)
            plt.plot(domain, true_sol, label='True solution')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend()
        plt.title('Solution for x using ' + step_function.__name__)
        plt.show()
    elif len(x0) == 2:
        if true_sol is not None:
            domain = np.linspace(0, 21, 100)
            true = true_sol(domain)
            true_x = true[0]
            true_y = true[1]
        # Solve for x and y
        x_values = [x[0] for x in values]
        y_values = [x[1] for x in values]
        # Calculate the true solution
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        # Plot the solution for x using the specified step function
        ax1.plot(t_values, x_values, 'o', label=step_function.__name__)
        # Plot the true solution for x
        if true_sol is not None:
            ax1.plot(domain, true_x, label='True solution')
        ax1.set_ylabel('x')
        ax1.legend()
        # Plot the solution for y using the specified step function
        ax2.plot(t_values, y_values, 'o',label=step_function.__name__)
        # Plot the true solution for y
        if true_sol is not None:
            ax2.plot(domain, true_y, label='True solution')
        ax2.set_xlabel('t')
        ax2.set_ylabel('y')
        ax2.legend()
        fig.suptitle('Solution for x and y using ' + step_function.__name__)
        plt.show()
    else:
        raise ValueError('Invalid number of dimensions. The function can only solve for 1 or 2 dimensions. Please input x0 as a list.')
