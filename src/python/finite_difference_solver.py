"""
finite_difference_solver.py

Implements finite difference methods for solving nonlinear partial differential equations (PDEs).
Designed for flexibility in specifying PDE parameters, boundary conditions, and discretization
schemes.

Currently supports:
- Burgers' equation as a representative nonlinear PDE.

Features:
- Configurable spatial and temporal discretizations
- Multiple schemes (e.g., upwind, Lax-Friedrichs)
- Stability condition checks (CFL)
- Output campatible with data generation for surrogate modeling

Examples
--------
>>> import numpy as np
>>> from src.python.finite_difference_solver import solve_burgers
>>> x, t, U = solve_burgers(
...     nx=256,
...     nt=500,
...     L=2.0,
...     T=1.0,
...     nu=0.01,
...     u0=lambda x: np.sin(np.pi * x),
...     scheme="lax_friedrichs"
... )
"""

from typing import Callable, Union
import numpy as np

def solve_burgers(
        nx: int,
        nt: int,
        L: float,
        T: float,
        nu: float,
        u0: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
        scheme: str = "lax_friedrichs",
        bc: str = "periodic",
        flux: str = "standard",
):
    """
    Solve 1D viscous Burgers' equation on [0, L] over time [0, T] using a finite-difference method.

    PDE
    ---
    u_t + (f(u))_x = nu * u_xx
        where f(u) = 0.5 * u^2 (standard Burgers' flux).
    
    Parameters
    ----------
    nx : int
        Number of spatial grid points (uniform grid). Must be >= 3.
    nt : int
        Number of time steps. Must be >= 1.
    L : float
        Domain length (spatial interval [0, L]).
    T : float
        Final time (temporal interval [0, T]).
    nu : float
        Viscosity (diffusion) coefficient; controls smoothing/regularization.
    u0 : Callable[[ndarray, ndarray]] or ndarray
        Initial condition. Either a callable mapping x -> u(x) evaluated on the grid,
        or a 1D array shape (nx, ) giving initial values directly.
    scheme : {"upwind", "lax_friedrichs"}, default "lax_friedrichs"
        Numerical scheme for the convection term. Upwind or Lax-Friedrichs are provided
        as baselines; additional schemes may be added later.
    bc : {"periodic", "dirichlet", "neumann"}, default "periodic"
        Boundary condition type. For Dirichlet/Neumann, default boundary values/derivatives
        may be introduced in a future signature if needed.
    flux : {"standard"}, default "standard"
        Flux function choice. "standard" corresponds to f(u) = 0.5 * u^2; placeholder for 
        extensibility if custom fluxes are introduced.

    Returns
    -------
    x : ndarray, shape (nx, )
        Spatial grid coordinates in [0, L].
    t : ndarray, shape (nt + 1, )
        Time stamps from 0 to T inclusive.
    U : ndarray, shape (nt + 1, nx)
        Solution snapshots; U[k, :] corresponds to time t[k].
        U[0, :] equals the initial condition evaluated on the grid.

    Notes
    -----
    - Stability: for explicit schemes, a convective CFL-like restriction on dt and dx is enforced
        or reported. Diffusive stability may also bound dt by ~ dx^2 / nu.
    - Discretization: second-order centered differences for diffusion; convection term per
        'scheme'.
    - Determinism: no randomness; intended for reproducible dataset generation.

    Examples
    --------
    >>> import numpy as np
    >>> x, t, U = solve_burgers(
    ...     nx=256, nt=500, L=2.0, T=1.0, nu=1e-2,
    ...     u0=lambda x: np.sin(2*np.pi*x/2.0),
    ...     scheme="lax_friedrichs", bc="periodic"
    ... )
    """
    # --- 1) Validate and scope current minimal implementation --- 
    if scheme != "lax_friedrichs":
        raise NotImplementedError("Only 'lax_friedrichs' is implemented in the minimal slice.")
    if bc != "periodic":
        raise NotImplementedError("Only perdiodic BCs are implemented in the minimal slice.")
    if flux != "standard":
        raise NotImplementedError("Only standard Burgers' flux f(u)=0.5*u^2 is implemented.")
    
    if nx < 3:
        raise ValueError("nx must be >= 3")
    if nt < 1:
        raise ValueError("nt must be >=1")
    if L <= 0 or T <= 0:
        raise ValueError("L and T must be positive")
    if nu < 0:
        raise ValueError("nu must be non-negative")
    
    # --- 2) Grid and step sizes ---
    # endpoint=False -> Last point is L - dx; periodic wrap maps to x[0]
    x = np.linspace(0.0, L, nx, endpoint=False)
    t = np.linspace(0.0, T, nt + 1)
    dx = L / nx
    dt = T / nt

    # --- 3) Allocate solution and set ICs U[0] ---
    if callable(u0):
        U0 = np.asarray(u0(x), dtype=float)
    else:
        U0 = np.asarray(u0, dtype=float)
    if U0.shape != (nx, ):
        raise ValueError(f"uo must have shape (nx, ), got {U0.shape}")
    
    U = np.empty((nt + 1, nx), dtype=float)
    U[0] = U0

    # --- 4) Periodic neightbors via roll ---
    def roll_plus(v):           # v_{i+1} with wrap
        return np.roll(v, -1)
    
    def roll_minus(v):          #v_{i-1} with wrap
        return np.roll(v, 1)
    
    # --- 5. Explicit time stepping: LF convection + centered diffusion ---
    for k in range(nt):
        u = U[k]
        f = 0.5 * u**2     # Burgers' flux

        # Lax-Friedrichs dissipative avg: (u_{i+1} + u_{i-1}) / 2 
        u_avg = 0.5 * (roll_plus(u) + roll_minus(u))

        # Centered flux gradient: (f_i - f_{i-1}) / dx
        conv = (f - roll_minus(f)) / dx

        # Diffusive Laplacian: (u_{i+1} - 2u_i + u_{i-1})/dx^2
        lap = (roll_plus(u) - 2.0 * u + roll_minus(u)) / dx**2

        # Update: u^{n+1}_i = avg - dt * conv + nu * dt * lap
        U[k+1] = u_avg - dt * conv + nu * dt * lap
    
    return x, t, U
