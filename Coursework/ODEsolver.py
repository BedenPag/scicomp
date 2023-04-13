# Euler step function
def euler_step(f, t, x, delta_t):
    return x + delta_t * f(t, x)

# Runge-Kutta step function
def rk4_step(f, t, x, delta_t):
    k1 = f(t, x)
    k2 = f(t + delta_t/2, x + delta_t/2 * k1)
    k3 = f(t + delta_t/2, x + delta_t/2 * k2)
    k4 = f(t + delta_t, x + delta_t * k3)
    return x + delta_t/6 * (k1 + 2*k2 + 2*k3 + k4)

# Solver function
def solve_to(f, x0, t0, deltat_max, step_function):
    t = t0
    x = x0
    t_values = [t]
    values = [x]
    while t <= 20:
        x = step_function(f, t, x, deltat_max)
        t += deltat_max
        t_values.append(t)
        values.append(x)
    return t_values, values