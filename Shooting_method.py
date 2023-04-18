from ODEsolver import solve_to, rk4_step

# Phase condition function using the initial conditions for ODE with one variables
def phase_condition(u0, u1):
    return u0 - u1

# Phase condition function using the initial conditions for ODE with two variables
def phase_condition2(u0, u1):
    return u0[1] - u1[1]

# Shooting function using the phase condition with one variable
def shooting(u0, phase_condition):
    u1 = u0
    u1 += 0.1
    while phase_condition(u0, u1) < 0:
        u1 += 0.1
    while phase_condition(u0, u1) > 0:
        u1 -= 0.01
    return u1

# Shooting function using the phase condition with two variables
def shooting2(u0, phase_condition):
    u1 = u0
    u1[1] += 0.1
    while phase_condition(u0, u1) < 0:
        u1[1] += 0.1
    while phase_condition(u0, u1) > 0:
        u1[1] -= 0.01
    return u1

# Solver function using the shooting method
def solve_to_shooting(f, u0, t0, deltat_max, step_function, phase_condition):
    if len(u0) == 2:
        u1 = shooting2(u0, phase_condition2)
    else:
        u1 = shooting(u0, phase_condition)
    t_values, values = solve_to(f, u1, t0, deltat_max, step_function)
    return t_values, values
