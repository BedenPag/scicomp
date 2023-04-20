from ODEsolve import rk4_step
from Shooting_method import solve_to_shooting, phase_condition

import matplotlib.pyplot as plt

# Cotinuation function
def continuation(f, x0, par0, vary_par, step_size, max_steps, discretisation, solver):
    t0 = 0
    deltat_max = 0.01
    # Initial guess for the parameter
    u0 = [x0, vary_par]
    u0[-1] = par0
    # Solve the equation
    t_values, values = solve_to_shooting(f, u0, t0, deltat_max, solver, phase_condition)
    results = [(par0, t_values, values)]
    for i in range(max_steps):
        u0 = values[-1]
        print(u0)
        par0 = par0 + step_size
        u0[-1] = par0
        t_values, values = discretisation(f, u0, t0, deltat_max, solver, phase_condition)
        results.append((par0, t_values, values))
    return results

# test function
def cubic_ode(t, u):
    x, c = u
    return [x**3 - x + c, 0]

# Define the initial guess and the parameter range
x0 = 0.5
par0 = -2

# Solve the equation
solutions = continuation(cubic_ode, x0, par0, 0, 0.01, 400, lambda x:x, rk4_step)

# Plot the solution
par_values = [sol[0] for sol in solutions]
x_values = [sol[2][-1][0] for sol in solutions]

plt.plot(par_values, x_values)
plt.xlabel('c')
plt.ylabel('x')
plt.show()

