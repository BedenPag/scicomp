from ODEsolve import solve_to, rk4_step, euler_step
import numpy as np
import matplotlib.pyplot as plt

def phase_condition(u0, u1):
    # Phase condition for the one dimensional ODE
    return u0 - u1

def phase_condition2(u0, u1):
    # Phase condition for the two dimensional ODE
    return u0[1] - u1[1]

def shooting(u0, phase_condition):
    '''
    Preforms shooting on an ODEs.

    Args:   
        u0 (float): Initial value of the dependent variable.   
        phase_condition (callable): Function to be used to determine the phase condition.

    Returns:
        u1 (float): Value of the dependent variable at the end of the interval.
    '''
    u1 = u0
    u1 += 0.1
    while phase_condition(u0, u1) < 0:
        u1 += 0.1
    while phase_condition(u0, u1) > 0:
        u1 -= 0.01
    return u1


def shooting2(u0, phase_condition2):
    '''
    Preforms shooting on a system of ODEs.

    Args:
        u0 (list): List of initial values of the dependent variables.
        phase_condition (callable): Function to be used to determine the phase condition.

    Returns:
        u1 (list): List of values of the dependent variables at the end of the interval.
    '''
    u1 = u0
    u1[1] += 0.1
    while phase_condition2(u0, u1) < 0:
        u1[1] += 0.1
    while phase_condition2(u0, u1) > 0:
        u1[1] -= 0.01
    return u1

def solve_to_shooting(f, u0, t0, deltat_max, step_function):
    '''
    Solve the ODE using the shooting method.
    
    Args:
        f (callable): Function representing the ODE to be solved.
        u0 (float): Initial value of the dependent variable.
        t0 (float): Initial time.
        deltat_max (float): Maximum step size (time increment).
        step_function (callable): Function to be used to take a single step.
        phase_condition (callable): Function to be used to determine the phase condition.

    Returns:
        t_values (list): List of time values.
        values (list): List of dependent variable values.
    '''
    if not callable(f):
        raise TypeError('f must be a function')
    if not callable(step_function):
        raise TypeError('step_function must be a function')
    if not callable(phase_condition):
        raise TypeError('phase_condition must be a function')
    if deltat_max <= 0:
        raise ValueError('deltat_max must be positive')
    if t0 < 0:
        raise ValueError('t0 must be non-negative')
    if len(u0) == 2:
        u1 = shooting2(u0, phase_condition2)
    else:
        u1 = shooting(u0, phase_condition)
    t_values, values = solve_to(f, u1, t0, deltat_max, step_function)
    return t_values, values

def plot_limit_cycle(f, u0, deltat_max, method):
    '''
    Plot the limit cycle for the shooting method.
    
    Args:
        f (function): The function that defines the ODE system.
        u0 (list): The initial conditions.
        deltat_max (float): The maximum time step.
        method (string): The method to be used to solve the ODE.
        
    Returns:
        None - The plot for the limit cycle using the shooting method.
    '''
    if method.__name__ not in ['euler_step', 'rk4_step']:
        raise ValueError('method must be "euler_step" or "rk4_step"')
    if deltat_max <= 0:
        raise ValueError('deltat_max must be positive')
    if len(u0) != 2:
        raise ValueError('u0 must be a list of length 2')
    if not callable(f):
        raise TypeError('f must be a function')

    t_values, values = solve_to_shooting(f, u0, 0, deltat_max, method)
    x_values = [x[0] for x in values] # Extract the x values from the list of values
    y_values = [x[1] for x in values] # Extract the y values from the list of values

    # Plot the limit cycle for the shooting method
    plt.plot(x_values, y_values)
    plt.xlabel('Prey')
    plt.ylabel('Predator')
    plt.title('Limit Cycle for the Shooting Method')
    plt.show()

def plot_solution(f, u0, deltat_max, method):
    '''
    Plot the solution for the shooting method.
        
    Args:
        f (function): The function that defines the ODE system.
        u0 (list): The initial conditions.
        deltat_max (float): The maximum time step.
        method (string): The method to be used to solve the ODE.
            
    Returns:
        None - The plot for the solution using the shooting method.
    '''
    if len([u0]) == 2:
        t_values, values = solve_to_shooting(f, u0, 0, deltat_max, method)
        x_values = [x[0] for x in values] # Extract the x values from the list of values
        y_values = [x[1] for x in values] # Extract the y values from the list of values

        # Plot the solution for the shooting method
        plt.plot(t_values, x_values, label = 'Prey')
        plt.plot(t_values, y_values, label = 'Predator')
        plt.xlabel('Time')
        plt.title('Solution for the Shooting Method')
        plt.legend()
    elif len([u0]) == 1:
        t_values, values = solve_to_shooting(f, u0, 0, deltat_max, method)
        values = np.array(values)
        plt.plot(t_values, values, label = 'Solution')
        plt.xlabel('Time')
        plt.title('Solution for the Shooting Method')
        plt.legend()