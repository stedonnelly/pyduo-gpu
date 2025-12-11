"""
Kinetic energy matrix construction for PyDuo

Implements the vibrational kinetic energy operator using:
- SINC-DVR (Discrete Variable Representation)
- 5-point finite difference
- Lobatto quadrature

All operations are GPU-accelerated using CuPy.
"""

import numpy as np
from typing import Optional

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = np
    HAS_CUDA = False

from .constants import PhysicalConstants
from .grid import Grid


class KineticEnergyMatrix:
    """
    Construct the vibrational kinetic energy matrix.

    The kinetic energy operator is:
    T = -hbar^2 / (2*mu) * d^2/dr^2

    In DVR or finite-difference representation, this becomes a matrix.
    """

    METHODS = ['SINC', '5POINTDIFFERENCES', 'LOBATTO']

    def __init__(self, grid: Grid, reduced_mass: float,
                 method: str = 'SINC', use_gpu: bool = True):
        """
        Initialize kinetic energy matrix builder.

        Parameters
        ----------
        grid : Grid
            Radial coordinate grid
        reduced_mass : float
            Reduced mass of the molecule in atomic mass units (amu)
        method : str
            Method for kinetic energy: 'SINC', '5POINTDIFFERENCES', 'LOBATTO'
        use_gpu : bool
            Whether to use GPU acceleration
        """
        self.grid = grid
        self.reduced_mass = reduced_mass
        self.method = method.upper()
        self.use_gpu = use_gpu and HAS_CUDA

        self.xp = cp if self.use_gpu else np

        # Pre-computed scale factor
        # hstep = grid step, scale = amass/aston
        # The Fortran code uses: sc = h12 * scale where h12 = 12 * hstep^2
        self.scale = reduced_mass / PhysicalConstants.ASTON

    def build(self) -> np.ndarray:
        """
        Build the kinetic energy matrix.

        Returns
        -------
        kinmat : ndarray
            Kinetic energy matrix of shape (ngrid, ngrid)
        """
        if self.method == 'SINC':
            return self._build_sinc_dvr()
        elif self.method == '5POINTDIFFERENCES':
            return self._build_5point_fd()
        elif self.method == 'LOBATTO':
            return self._build_lobatto()
        else:
            raise ValueError(f"Unknown kinetic energy method: {self.method}")

    def _build_sinc_dvr(self):
        """
        Build kinetic energy matrix using SINC-DVR.

        Following Fortran Duo's convention exactly (see diatom.f90 lines 13031-13056):
        - kinmat already includes a factor of 12 (to match finite differences form)
        - Diagonal: 12 * pi^2 / 3
        - Off-diagonal: 12 * 2 * (-1)^(i-j) / (i-j)^2

        The h^2 factor is NOT included - it's part of sc = 12*h^2*(amass/aston)
        which is applied when building the full Hamiltonian.

        When sc is applied to kinmat, the h^2 cancels with the denominator
        of the standard SINC-DVR formula, leaving just the mass factor.
        """
        xp = self.xp
        ngrid = self.grid.npoints

        # Create index arrays for vectorized computation
        i_idx = xp.arange(ngrid, dtype=xp.float64).reshape(-1, 1)
        j_idx = xp.arange(ngrid, dtype=xp.float64).reshape(1, -1)

        # Compute (i - j) using 1-based indexing like Fortran
        # Actually, the formula only depends on the difference, so 0 vs 1-based is the same
        diff = i_idx - j_idx

        # Avoid division by zero on diagonal
        diff_safe = xp.where(diff == 0, 1.0, diff)

        # Off-diagonal elements: +12 * 2 * (-1)^(i+j) / (i-j)^2
        # Note: (-1)^(i+j) = (-1)^(i-j) for the sign pattern
        kinmat = 12.0 * 2.0 * ((-1.0) ** diff) / (diff_safe ** 2)

        # Diagonal elements: 12 * pi^2 / 3
        kinmat = xp.where(diff == 0, 12.0 * np.pi ** 2 / 3.0, kinmat)

        return kinmat

    def _build_5point_fd(self):
        """
        Build kinetic energy matrix using 5-point finite differences.

        Following Fortran Duo convention:
        - Diagonal: d2dr(igrid) = 30*z^2 (for uniform grid z=1, so diagonal = 30)
        - Off by 1: -16*z*z = -16
        - Off by 2: +1*z*z = +1

        The factor of 12 is built in to match: (-1 + 16 - 30 + 16 - 1) / 12h^2
        But Fortran stores it without the 12h^2 denominator.
        """
        xp = self.xp
        ngrid = self.grid.npoints

        kinmat = xp.zeros((ngrid, ngrid), dtype=xp.float64)

        # For uniform grid (z = 1), the coefficients match 5-point stencil * 12
        # Diagonal: 30
        # Off by 1: -16
        # Off by 2: +1
        for i in range(ngrid):
            kinmat[i, i] = 30.0

            if i >= 1:
                kinmat[i, i-1] = -16.0
                kinmat[i-1, i] = -16.0

            if i >= 2:
                kinmat[i, i-2] = 1.0
                kinmat[i-2, i] = 1.0

        return kinmat

    def _build_lobatto(self):
        """
        Build kinetic energy matrix using Lobatto quadrature (DVR).

        This uses the Gauss-Lobatto-Legendre DVR representation.
        """
        xp = self.xp
        ngrid = self.grid.npoints
        r = self.grid.r
        w = self.grid.weights

        # Get Lobatto derivative matrix
        D = self._lobatto_derivative_matrix(r)

        # Kinetic energy matrix: T = -hbar^2/(2*mu) * D^T * W * D
        # where W is diagonal weight matrix
        W = xp.diag(w)

        # In DVR, the kinetic energy matrix is:
        kinmat = -PhysicalConstants.ASTON / self.reduced_mass * (D.T @ W @ D)

        # Scale for Hamiltonian
        h = (self.grid.rmax - self.grid.rmin) / (ngrid - 1)
        sc = 12.0 * h**2 * self.scale
        kinmat = kinmat * sc

        return kinmat

    def _lobatto_derivative_matrix(self, r):
        """
        Compute the Lobatto derivative matrix.

        D_ij = d/dr L_j(r_i)

        where L_j are the Lagrange interpolating polynomials.
        """
        xp = self.xp
        n = len(r)

        # Map r to [-1, 1]
        x = 2.0 * (r - r[0]) / (r[-1] - r[0]) - 1.0

        D = xp.zeros((n, n), dtype=xp.float64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal elements
                    if i == 0:
                        D[i, j] = -n * (n - 1) / 4.0
                    elif i == n - 1:
                        D[i, j] = n * (n - 1) / 4.0
                    else:
                        D[i, j] = 0.0
                else:
                    # Off-diagonal elements
                    # Use Legendre polynomial values
                    xi, xj = float(x[i]), float(x[j])
                    Pi = self._legendre_P(n - 1, xi)
                    Pj = self._legendre_P(n - 1, xj)
                    D[i, j] = Pi / (Pj * (xi - xj))

        # Scale derivative for [rmin, rmax]
        D = D * 2.0 / (r[-1] - r[0])

        return D

    def _legendre_P(self, n: int, x: float) -> float:
        """Compute Legendre polynomial P_n(x)."""
        if n == 0:
            return 1.0
        elif n == 1:
            return x

        P_prev = 1.0
        P_curr = x
        for k in range(1, n):
            P_next = ((2*k + 1) * x * P_curr - k * P_prev) / (k + 1)
            P_prev = P_curr
            P_curr = P_next

        return P_curr

    def build_first_derivative(self):
        """
        Build the first derivative matrix (needed for NAC terms).

        Returns
        -------
        kinmat1 : ndarray
            First derivative matrix of shape (ngrid, ngrid)
        """
        xp = self.xp
        ngrid = self.grid.npoints
        h = self.grid.step if isinstance(self.grid.step, float) else float(self.grid.step[0])

        if self.method == 'SINC':
            # SINC-DVR first derivative
            i_idx = xp.arange(ngrid, dtype=xp.float64).reshape(-1, 1)
            j_idx = xp.arange(ngrid, dtype=xp.float64).reshape(1, -1)
            diff = i_idx - j_idx
            diff_safe = xp.where(diff == 0, 1.0, diff)

            # Off-diagonal: (-1)^(i-j) / ((i-j) * h)
            kinmat1 = ((-1.0) ** diff) / (diff_safe * h)

            # Diagonal: 0
            kinmat1 = xp.where(diff == 0, 0.0, kinmat1)

        else:
            # 5-point central difference for first derivative
            kinmat1 = xp.zeros((ngrid, ngrid), dtype=xp.float64)
            coeffs = xp.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * h)

            for i in range(ngrid):
                for k, offset in enumerate([-2, -1, 0, 1, 2]):
                    j = i + offset
                    if 0 <= j < ngrid:
                        kinmat1[i, j] = coeffs[k]

        return kinmat1


def build_kinetic_matrix_gpu(ngrid: int, h: float, reduced_mass: float,
                             method: str = 'SINC'):
    """
    GPU kernel for building kinetic energy matrix.

    This is a convenience function that creates a temporary grid
    and builds the kinetic matrix in one call.

    Parameters
    ----------
    ngrid : int
        Number of grid points
    h : float
        Grid step size
    reduced_mass : float
        Reduced mass in amu
    method : str
        Method: 'SINC' or '5POINTDIFFERENCES'

    Returns
    -------
    kinmat : cupy.ndarray
        Kinetic energy matrix on GPU
    """
    if not HAS_CUDA:
        raise RuntimeError("CUDA not available. Install CuPy for GPU support.")

    scale = reduced_mass / PhysicalConstants.ASTON
    sc = 12.0 * h**2 * scale

    if method.upper() == 'SINC':
        # Create index arrays on GPU
        i_idx = cp.arange(ngrid, dtype=cp.float64).reshape(-1, 1)
        j_idx = cp.arange(ngrid, dtype=cp.float64).reshape(1, -1)
        diff = i_idx - j_idx
        diff_safe = cp.where(diff == 0, 1.0, diff)

        kinmat = 2.0 * ((-1.0) ** diff) / (diff_safe ** 2 * h ** 2)
        kinmat = cp.where(diff == 0, np.pi ** 2 / (6.0 * h ** 2), kinmat)
        kinmat = -kinmat * PhysicalConstants.ASTON / reduced_mass * sc

    else:
        kinmat = cp.zeros((ngrid, ngrid), dtype=cp.float64)
        coeffs = cp.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * h**2)

        for i in range(ngrid):
            for k, offset in enumerate([-2, -1, 0, 1, 2]):
                j = i + offset
                if 0 <= j < ngrid:
                    kinmat[i, j] = coeffs[k]

        kinmat = -kinmat * PhysicalConstants.ASTON / reduced_mass * sc

    return kinmat
