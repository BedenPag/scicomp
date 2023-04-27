# Preform pytest on BVPsolve.py

import pytest
import numpy as np

from BVPsolve import solve_bvp

def test_solve_bvp():
    '''
    Test the solve_bvp function.
    '''
    # Test for a scalar function
    def q(x):
        return 1
    
    assert solve_bvp(N=20, a=0, b=1, alpha=0, beta=0, q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N='a', a=0, b=1, alpha=0, beta=0, q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N=20, a='a', b=1, alpha=0, beta=0, q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N=20, a=0, b='a', alpha=0, beta=0, q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N=20, a=0, b=1, alpha='a', beta=0, q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N=20, a=0, b=1, alpha=0, beta='a', q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N=20, a=0, b=1, alpha=0, beta=0, q, "a")

    # Test for a vector function
    def q(x):
        return np.array([1, 1])
    
    assert solve_bvp(N=20, a=0, b=1, alpha=0, beta=0, q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N='a', a=0, b=1, alpha=0, beta=0, q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N=20, a='a', b=1, alpha=0, beta=0, q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N=20, a=0, b='a', alpha=0, beta=0, q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N=20, a=0, b=1, alpha='a', beta=0, q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N=20, a=0, b=1, alpha=0, beta='a', q, "dirichlet")
    with pytest.raises(TypeError):
        solve_bvp(N=20, a=0, b=1, alpha=0, beta=0, q, "a")
    