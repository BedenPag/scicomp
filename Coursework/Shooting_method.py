from ODEsolver import solve_to, rk4_step

# Phase condition function using the initial conditions
def phase_condition(u0, u1):
    return u0[1] - u1[1]

# Shooting function using the phase condition
def shooting(u0, phase_condition):
    u1 = u0
    u1[1] += 0.1
    while phase_condition(u0, u1) < 0:
        u1[1] += 0.1
    while phase_condition(u0, u1) > 0:
        u1[1] -= 0.01
    return u1

# Solver function using the shooting method
def solve_to_shooting(f, u0, t0, deltat_max, step_function, phase_condition):
    u1 = shooting(u0, phase_condition) # find the initial conditions using the shooting method
    t_values, values = solve_to(f, u1, t0, deltat_max, step_function) # solve the ODE using the initial conditions found
    return t_values, values