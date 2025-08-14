"""
test_utils.py

Unit tests for utility functions in utils.py
"""

import numpy as np
from src.python.finite_difference_solver import solve_burgers

def test_solve_burgers_shapes_and_ic():
    nx, nt = 5, 3
    L, T, nu = 1.0, 0.1, 0.01
    u0 = lambda x: np.sin(2 * np.pi * x / L)

    x, t, U = solve_burgers(nx=nx, nt=nt, L=L, T=T, nu=nu, u0=u0)

    # Check shapes
    assert x.shape == (nx, )
    assert t.shape == (nt + 1, )
    assert U.shape == (nt + 1, nx)

    # Initial condition preserved
    np.testing.assert_allclose(U[0], u0(x), rtol=1e-12, atol=1e-12)

    # Values should be finite numbers
    assert np.all(np.isfinite(U))

def test_solve_burgers_periodic_bc():
    nx, nt = 5, 2
    L, T, nu = 1.0, 0.05, 0.0

    # IC: a delta-like spike at one point
    u0 = np.zeros(nx)
    u0[0] = 1.0

    x, t, U = solve_burgers(nx=nx, nt=nt, L=L, T=T, nu=nu, u0=u0)

    # Check periodic wrap: after one time step, "mass" should be conserved
    total_mass_initial = np.sum(U[0])
    total_mass_final = np.sum(U[-1])
    assert np.isclose(total_mass_initial, total_mass_final, rtol=1e-12)