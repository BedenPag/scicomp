# Perform pytest on ODEsolve.py

import pytest
import numpy as np
from ODEsolve import solve_to, rk4_step, euler_step


def test_solve_to():
    '''
    Test the solve_to function.
    '''
    # Test for a scalar function
    assert np.allclose(solve_to(lambda t, x: x, 1, 0, 0.1, euler_step), 2.7182818284590455)
    with pytest.raises(TypeError):
        solve_to(lambda t, x: x, 'a', 0, 0.1, euler_step)
    with pytest.raises(TypeError):
        solve_to(lambda t, x: x, 1, 'a', 0.1, euler_step)
    with pytest.raises(TypeError):
        solve_to(lambda t, x: x, 1, 0, 'a', euler_step)
    with pytest.raises(TypeError):
        solve_to(lambda t, x: x, 1, 0, 0.1, 'a')
    
    # Test for a vector function
    def f(t,u):
        x = u[0]
        y = u[1]
        dydt = -x
        dxdt = y
        return np.array([dxdt, dydt])

    assert np.allclose(solve_to(f, np.array([1, 2]), 0, 0.1, rk4_step), np.array([[1.10517092, 1.22140276]]))
    with pytest.raises(TypeError):
        solve_to(f, 'a', 0, 0.1, rk4_step)
    with pytest.raises(TypeError):
        solve_to(f, np.array([1, 2]), 'a', 0.1, rk4_step)
    with pytest.raises(TypeError):
        solve_to(f, np.array([1, 2]), 0, 'a', rk4_step)
    with pytest.raises(TypeError):
        solve_to(f, np.array([1, 2]), 0, 0.1, 'a')
   