"""
Field definitions for PyDuo

Implements various potential energy curves, spin-orbit couplings,
and other Hamiltonian terms. All computations are vectorized for GPU.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = np
    HAS_CUDA = False

from .constants import PhysicalConstants


class FieldType(Enum):
    """Types of Hamiltonian field terms."""
    POTEN = 1           # Potential energy curve
    SPINORBIT = 2       # Spin-orbit coupling
    L2 = 3              # L^2 term
    LXLY = 4            # L+/L- coupling
    SPINSPIN = 5        # Spin-spin coupling
    SPINSPINO = 6       # Spin-spin-o coupling
    BOBROT = 7          # BOB rotational correction
    SPINROT = 8         # Spin-rotation coupling
    DIABATIC = 9        # Diabatic coupling
    LAMBDAOPQ = 10      # Lambda-doubling o-p-q
    LAMBDAP2Q = 11      # Lambda-doubling p2q
    LAMBDAQ = 12        # Lambda-doubling q
    NAC = 13            # Non-adiabatic coupling
    BOBVIB = 14         # BOB vibrational correction
    DIPOLE = 33         # Electric dipole moment


@dataclass
class ElectronicState:
    """Electronic state quantum numbers."""
    name: str
    istate: int
    lambda_: int          # Lambda quantum number (projection of L)
    multiplicity: int     # 2S+1
    spin: float           # S
    parity_pm: int = 0    # +/- parity for Sigma states
    parity_gu: int = 0    # g/u parity for homonuclear molecules


@dataclass
class FieldBase:
    """Base class for all Hamiltonian field terms."""

    name: str
    field_type: FieldType
    istate: int                    # Bra state index
    jstate: int                    # Ket state index
    lambda_i: int = 0              # Lambda for bra
    lambda_j: int = 0              # Lambda for ket
    spin_i: float = 0.0            # Spin for bra
    spin_j: float = 0.0            # Spin for ket
    sigma_i: float = 0.0           # Sigma for bra
    sigma_j: float = 0.0           # Sigma for ket
    multiplicity_i: int = 1        # 2S+1 for bra
    multiplicity_j: int = 1        # 2S+1 for ket

    # Values
    parameters: np.ndarray = field(default_factory=lambda: np.array([]))
    grid_values: np.ndarray = field(default_factory=lambda: np.array([]))
    matrix_elements: np.ndarray = field(default_factory=lambda: np.array([]))

    # Function type for analytic potentials
    func_type: str = "GRID"

    # Morphing support - multiply analytic function by ab initio data
    morphing: bool = False
    abinitio_field: Any = None  # Reference to AbInitioField when morphing
    factor: float = 1.0         # Scaling factor from input (real part)
    complex_f: complex = 1.0+0j # Complex factor (for imaginary factors like 'i')

    # MOLPRO <x|Lz|y> representation parameters for basis transformation
    # When non-zero, the molpro_duo transformation is applied (sqrt(2) factors)
    ix_lz_y: int = 0           # <x|Lz|y> value for bra state
    jx_lz_y: int = 0           # <x|Lz|y> value for ket state
    parity_i: int = 1          # +/- parity for bra state (+1 or -1)
    parity_j: int = 1          # +/- parity for ket state (+1 or -1)

    def evaluate(self, r, use_gpu: bool = True):
        """Evaluate the field at given r values."""
        raise NotImplementedError("Subclasses must implement evaluate()")

    def apply_morphing(self, analytic_values, r, use_gpu: bool = True):
        """
        Apply morphing by multiplying analytic values by ab initio data.

        This follows Duo's approach (diatom.f90 line 5494):
        field%gridvalue = field%gridvalue * abinitio(iabi)%gridvalue

        Parameters
        ----------
        analytic_values : array
            The analytic function values (e.g., BOBLEROY output)
        r : array
            The r grid points
        use_gpu : bool
            Whether to use GPU

        Returns
        -------
        morphed_values : array
            analytic_values * abinitio_values * factor
        """
        xp = cp if (use_gpu and HAS_CUDA) else np

        # First apply the factor
        result = analytic_values * self.factor

        # Then apply morphing if enabled
        if self.morphing and self.abinitio_field is not None:
            # Get ab initio values on the same grid
            abinitio_values = self.abinitio_field.evaluate(r, use_gpu=use_gpu)
            # Ensure type consistency between result and abinitio_values
            if use_gpu and HAS_CUDA:
                if isinstance(abinitio_values, np.ndarray):
                    abinitio_values = cp.asarray(abinitio_values)
                if isinstance(result, np.ndarray):
                    result = cp.asarray(result)
            else:
                if HAS_CUDA and hasattr(abinitio_values, 'get'):
                    abinitio_values = abinitio_values.get()
                if HAS_CUDA and hasattr(result, 'get'):
                    result = result.get()

            # Handle case where result is empty but morphing is enabled
            # In this case, use the ab initio values directly (analytic function = 1)
            if len(result) == 0 and len(abinitio_values) > 0:
                result = abinitio_values
            else:
                result = result * abinitio_values

        return result


class AbInitioField:
    """
    Ab initio field data from quantum chemistry calculations.

    Stores grid data and provides interpolation to arbitrary r points.
    Used for morphing analytic functions with ab initio reference data.
    """

    def __init__(self, name: str, field_type: FieldType,
                 istate: int, jstate: int,
                 lambda_i: int = 0, lambda_j: int = 0,
                 spin_i: float = 0.0, spin_j: float = 0.0,
                 sigma_i: float = 0.0, sigma_j: float = 0.0,
                 grid_r: Optional[np.ndarray] = None,
                 grid_values: Optional[np.ndarray] = None,
                 units_r: str = "angstrom",
                 units_v: str = "cm-1"):
        """
        Initialize ab initio field.

        Parameters
        ----------
        name : str
            Field name
        field_type : FieldType
            Type of field (SPINORBIT, LXLY, etc.)
        istate, jstate : int
            State indices
        lambda_i, lambda_j : int
            Lambda quantum numbers
        spin_i, spin_j : float
            Spin quantum numbers
        sigma_i, sigma_j : float
            Sigma quantum numbers
        grid_r : array
            R grid points from ab initio calculation
        grid_values : array
            Field values at grid points
        units_r : str
            Units for r ('angstrom' or 'bohr')
        units_v : str
            Units for values ('cm-1', 'hartree', etc.)
        """
        self.name = name
        self.field_type = field_type
        self.istate = istate
        self.jstate = jstate
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.spin_i = spin_i
        self.spin_j = spin_j
        self.sigma_i = sigma_i
        self.sigma_j = sigma_j
        self.units_r = units_r.lower()
        self.units_v = units_v.lower()

        # Convert r to Angstrom if needed
        if grid_r is not None:
            if self.units_r == "bohr":
                self.grid_r = grid_r * PhysicalConstants.BOHR  # BOHR is already in Angstroms
            else:
                self.grid_r = np.array(grid_r)
        else:
            self.grid_r = None

        self.grid_values = np.array(grid_values) if grid_values is not None else None

        # Spline coefficients (computed on first evaluation)
        self._spline_coeffs = None

    def evaluate(self, r, use_gpu: bool = True):
        """
        Evaluate ab initio field at given r points using cubic spline interpolation.

        Parameters
        ----------
        r : array
            R points to evaluate at (in Angstrom)
        use_gpu : bool
            Whether to use GPU

        Returns
        -------
        values : array
            Interpolated values
        """
        xp = cp if (use_gpu and HAS_CUDA) else np

        if self.grid_r is None or self.grid_values is None:
            return xp.zeros_like(r)

        # Convert r to numpy for scipy interpolation
        # Check for CuPy array first (has .get() method) - always check regardless of use_gpu
        if HAS_CUDA and hasattr(r, 'get'):
            r_np = r.get()
        elif isinstance(r, np.ndarray):
            r_np = r
        else:
            r_np = np.array(r)

        # Use scipy for interpolation (CPU)
        from scipy.interpolate import CubicSpline

        if self._spline_coeffs is None:
            # Build spline - don't extrapolate, we'll handle that separately
            self._spline_coeffs = CubicSpline(
                self.grid_r, self.grid_values,
                bc_type='natural', extrapolate=False
            )

        # Evaluate spline for points within range
        values = np.zeros_like(r_np)
        r_min, r_max = self.grid_r[0], self.grid_r[-1]

        # Points within the grid range - use cubic spline
        in_range = (r_np >= r_min) & (r_np <= r_max)
        if np.any(in_range):
            values[in_range] = self._spline_coeffs(r_np[in_range])

        # Points below the grid range - use linear extrapolation (like Duo)
        below_range = r_np < r_min
        if np.any(below_range):
            # Linear extrapolation using first two points
            x1, x2 = self.grid_r[0], self.grid_r[1]
            y1, y2 = self.grid_values[0], self.grid_values[1]
            slope = (y2 - y1) / (x2 - x1)
            values[below_range] = y1 + (r_np[below_range] - x1) * slope

        # Points above the grid range - use linear extrapolation
        above_range = r_np > r_max
        if np.any(above_range):
            # Linear extrapolation using last two points
            x1, x2 = self.grid_r[-2], self.grid_r[-1]
            y1, y2 = self.grid_values[-2], self.grid_values[-1]
            slope = (y2 - y1) / (x2 - x1)
            values[above_range] = y2 + (r_np[above_range] - x2) * slope

        if use_gpu and HAS_CUDA:
            return cp.asarray(values)
        return values


class PotentialField(FieldBase):
    """
    Potential energy curve.

    Supports multiple functional forms: Morse, EMO, MLR, polynomial, grid, etc.
    """

    SUPPORTED_TYPES = [
        'MORSE', 'EMO', 'MLR', 'POLYNOMIAL', 'GRID', 'BOB', 'DUNHAM',
        'SPF', 'CHEBYSHEV', 'MARQUARDT', 'MLJ'
    ]

    def __init__(self, name: str, istate: int, lambda_: int, multiplicity: int,
                 func_type: str = "MORSE", parameters: Optional[np.ndarray] = None,
                 grid_values: Optional[np.ndarray] = None,
                 grid_r: Optional[np.ndarray] = None,
                 grid_v: Optional[np.ndarray] = None):
        super().__init__(
            name=name,
            field_type=FieldType.POTEN,
            istate=istate,
            jstate=istate,
            lambda_i=lambda_,
            lambda_j=lambda_,
            spin_i=(multiplicity - 1) / 2.0,
            spin_j=(multiplicity - 1) / 2.0,
            multiplicity_i=multiplicity,
            multiplicity_j=multiplicity,
            func_type=func_type.upper(),
            parameters=parameters if parameters is not None else np.array([]),
            grid_values=grid_values if grid_values is not None else np.array([])
        )
        # Store r and V grid data separately for interpolation
        self.grid_r = grid_r
        self.grid_v = grid_v

    def evaluate(self, r, use_gpu: bool = True):
        """
        Evaluate the potential at given r values.

        Parameters
        ----------
        r : array-like
            Radial distances in Angstroms
        use_gpu : bool
            Whether to use GPU computation

        Returns
        -------
        V : array
            Potential values in cm^-1
        """
        xp = cp if (use_gpu and HAS_CUDA) else np

        if isinstance(r, np.ndarray) and use_gpu and HAS_CUDA:
            r = cp.asarray(r)
        elif not use_gpu and HAS_CUDA and isinstance(r, cp.ndarray):
            r = cp.asnumpy(r)

        func_type = self.func_type.upper()
        params = self.parameters

        if use_gpu and HAS_CUDA and isinstance(params, np.ndarray):
            params = cp.asarray(params)

        if func_type == "MORSE":
            return self._morse(r, params, xp)
        elif func_type == "EMO":
            return self._emo(r, params, xp)
        elif func_type == "MLR":
            return self._mlr(r, params, xp)
        elif func_type in ["POLYNOMIAL", "POLYNOM"]:
            return self._polynomial(r, params, xp)
        elif func_type == "BOB" or func_type == "BOBLEROY" or func_type == "SURKUS":
            return self._bob_leroy(r, params, xp)
        elif func_type == "GRID":
            return self._interpolate_grid(r, xp)
        else:
            raise ValueError(f"Unsupported potential type: {func_type}")

    def _morse(self, r, params, xp):
        """
        Morse potential: V(r) = De * (1 - exp(-a*(r-re)))^2 + Te

        Parameters:
        [0] Te - electronic energy
        [1] De - dissociation energy
        [2] re - equilibrium distance
        [3] a  - Morse parameter
        """
        Te = params[0]
        De = params[1]
        re = params[2]
        a = params[3]

        y = 1.0 - xp.exp(-a * (r - re))
        V = De * y**2 + Te

        return V

    def _emo(self, r, params, xp):
        """
        Expanded Morse Oscillator (EMO) potential.

        V(r) = De * (1 - exp(-phi(r) * (r-re)))^2 + V0

        where phi(r) = sum_k beta_k * z^k  (k = 0 to N)
        and z = (r^p - rref^p) / (r^p + rref^p)

        Duo EMO format (from functions.f90 poten_EMO):
        [0] V0   - Te (electronic energy)
        [1] RE   - re (equilibrium distance)
        [2] DE   - De (absolute dissociation energy, actual De = params[2] - V0)
        [3] RREF - reference distance (if <= 0, use re)
        [4] PL   - power p for r <= rref
        [5] PR   - power p for r > rref
        [6] NL   - N (highest power) for r <= rref
        [7] NR   - N (highest power) for r > rref
        [8:] B0, B1, B2, ... - phi coefficients (max(NL,NR)+1 coefficients)

        Note: Fortran checks size(parameters) == 8 + max(NL,NR) + 1
        """
        V0 = params[0]
        re = params[1]
        De_abs = params[2]
        De = De_abs - V0  # Actual dissociation energy relative to V0

        rref = params[3] if len(params) > 3 else re
        if rref <= 0:
            rref = re

        # Powers for left and right of rref
        pL = int(params[4]) if len(params) > 4 else 4
        pR = int(params[5]) if len(params) > 5 else 4

        # N = highest power for left (NL) and right (NR) of rref
        # These are the max k in sum_k=0^N, so we use coefficients B0..BN
        NL = int(params[6]) if len(params) > 6 else 2
        NR = int(params[7]) if len(params) > 7 else 4

        # Beta coefficients start at index 8: B0, B1, B2, ...
        beta_params = params[8:] if len(params) > 8 else xp.array([1.0])

        # For vectorized computation, we need to handle the piecewise selection
        # Fortran: if (r<=rref) then p=pL, N=NL else p=pR, N=NR

        # Create masks for left and right regions
        if hasattr(r, '__len__'):
            # Array input - need to handle piecewise
            r_arr = xp.asarray(r) if not isinstance(r, (np.ndarray, )) or (HAS_CUDA and isinstance(r, cp.ndarray)) else r

            # Initialize output
            V = xp.zeros_like(r_arr)

            # Left region: r <= rref
            mask_L = r_arr <= rref
            if xp.any(mask_L):
                r_L = r_arr[mask_L]
                p = pL
                N = NL
                rp = r_L**p
                rrefp = rref**p
                z_L = (rp - rrefp) / (rp + rrefp)

                # phi = sum_{k=0}^{N} beta_k * z^k (standard polynomial evaluation)
                phi_L = xp.zeros_like(r_L)
                z_power = xp.ones_like(r_L)
                for k in range(N + 1):
                    if k < len(beta_params):
                        phi_L = phi_L + float(beta_params[k]) * z_power
                    z_power = z_power * z_L

                y_L = 1.0 - xp.exp(-phi_L * (r_L - re))
                V[mask_L] = De * y_L**2 + V0

            # Right region: r > rref
            mask_R = ~mask_L
            if xp.any(mask_R):
                r_R = r_arr[mask_R]
                p = pR
                N = NR
                rp = r_R**p
                rrefp = rref**p
                z_R = (rp - rrefp) / (rp + rrefp)

                # phi = sum_{k=0}^{N} beta_k * z^k
                phi_R = xp.zeros_like(r_R)
                z_power = xp.ones_like(r_R)
                for k in range(N + 1):
                    if k < len(beta_params):
                        phi_R = phi_R + float(beta_params[k]) * z_power
                    z_power = z_power * z_R

                y_R = 1.0 - xp.exp(-phi_R * (r_R - re))
                V[mask_R] = De * y_R**2 + V0
        else:
            # Scalar input
            if r <= rref:
                p = pL
                N = NL
            else:
                p = pR
                N = NR

            rp = r**p
            rrefp = rref**p
            z = (rp - rrefp) / (rp + rrefp)

            # phi = sum_{k=0}^{N} beta_k * z^k
            phi = 0.0
            for k in range(N + 1):
                if k < len(beta_params):
                    phi = phi + float(beta_params[k]) * (z**k)

            y = 1.0 - xp.exp(-phi * (r - re))
            V = De * y**2 + V0

        return V

    def _mlr(self, r, params, xp):
        """
        Morse/Long-Range (MLR) potential.

        V(r) = De * (1 - uLR(r)/uLR(re) * exp(-beta(r) * yp))^2 + Te

        where yp = (r^p - re^p) / (r^p + re^p)
        """
        Te = params[0]
        De = params[1]
        re = params[2]
        p = int(params[3])
        q = int(params[4])

        # Long-range coefficients
        n_lr = int(params[5])
        lr_n = params[6:6+n_lr].astype(int) if n_lr > 0 else []
        lr_c = params[6+n_lr:6+2*n_lr] if n_lr > 0 else []

        # Beta expansion
        beta_params = params[6+2*n_lr:]

        rp = r**p
        rep = re**p
        yp = (rp - rep) / (rp + rep)

        # Long-range function uLR - vectorized
        uLR = xp.zeros_like(r)
        uLR_re = 0.0
        if n_lr > 0:
            # Convert to arrays for vectorized computation
            lr_n_arr = xp.array(lr_n, dtype=xp.float64).reshape(-1, 1)
            lr_c_arr = xp.array(lr_c, dtype=xp.float64).reshape(-1, 1)
            r_expanded = r.reshape(1, -1)

            # uLR = sum_i c_i / r^n_i (vectorized over both i and r)
            uLR = xp.sum(lr_c_arr / r_expanded**lr_n_arr, axis=0)
            uLR_re = float(xp.sum(xp.array(lr_c) / re**xp.array(lr_n)))

        if abs(uLR_re) < 1e-30:
            uLR_re = 1.0
            uLR = xp.ones_like(r)

        # Beta expansion - Horner's method (GPU-efficient)
        yq = (r**q - re**q) / (r**q + re**q)
        n_beta = len(beta_params)
        if n_beta > 0:
            beta = xp.full_like(r, float(beta_params[-1]))
            for k in range(n_beta - 2, -1, -1):
                beta = beta * yq + float(beta_params[k])
        else:
            beta = xp.ones_like(r)

        V = De * (1.0 - uLR / uLR_re * xp.exp(-beta * yp))**2 + Te

        return V

    def _polynomial(self, r, params, xp):
        """
        Polynomial potential: V(r) = sum_i a_i * (r - re)^i

        Parameters:
        [0] re - expansion center
        [1:] a_i - polynomial coefficients
        """
        re = params[0]
        coeffs = params[1:]

        dr = r - re
        # Horner's method for polynomial evaluation (GPU-efficient)
        n_coeffs = len(coeffs)
        if n_coeffs > 0:
            V = xp.full_like(r, float(coeffs[-1]))
            for k in range(n_coeffs - 2, -1, -1):
                V = V * dr + float(coeffs[k])
        else:
            V = xp.zeros_like(r)

        return V

    def _bob_leroy(self, r, params, xp):
        """
        BOB/LeRoy expansion with interpolation to infinity value.

        Fortran formula: f = (1-z)*t + z*tinf
        where z = (r^p - r0^p) / (r^p + r0^p)
        and t = sum_k beta_k * z^k

        Parameters format: [RE, P, B0, B1, ..., BINF]
        """
        re = params[0]
        p = int(params[1]) if len(params) > 1 else 1

        # Coefficients are params[2:-1], BINF is params[-1]
        if len(params) > 3:
            coeffs = params[2:-1]
            tinf = float(params[-1])
        elif len(params) > 2:
            coeffs = params[2:]
            tinf = 0.0
        else:
            coeffs = []
            tinf = 0.0

        rp = r**p
        rep = re**p
        z = (rp - rep) / (rp + rep)

        # Sum t = sum_k beta_k * z^k using Horner's method
        n_coeffs = len(coeffs)
        if n_coeffs > 0:
            t = xp.full_like(r, float(coeffs[-1]))
            for k in range(n_coeffs - 2, -1, -1):
                t = t * z + float(coeffs[k])
        else:
            t = xp.zeros_like(r)

        # Fortran formula: f = (1-z)*t + z*tinf
        f = (1.0 - z) * t + z * tinf

        return f

    def _interpolate_grid(self, r, xp):
        """
        Interpolate grid values to new r points using cubic splines.

        GPU-optimized: computes spline coefficients once on CPU, then
        evaluates entirely on GPU using vectorized operations.
        """
        # Check if we have grid data
        if self.grid_r is None or self.grid_v is None:
            if len(self.grid_values) == 0:
                raise ValueError("No grid values available for interpolation")
            # Fall back to old behavior if grid_values is set but not grid_r/grid_v
            return self.grid_values

        r_grid = self.grid_r
        v_grid = self.grid_v

        # Compute spline coefficients once (on CPU - this is O(n) and done once)
        if not hasattr(self, '_spline_coeffs') or self._spline_coeffs is None:
            self._spline_coeffs = self._compute_spline_coefficients(r_grid, v_grid)

        # Evaluate spline entirely on GPU using vectorized operations
        return self._evaluate_spline_gpu(r, r_grid, v_grid, self._spline_coeffs, xp)

    def _compute_spline_coefficients(self, x, y):
        """
        Compute natural cubic spline coefficients (done once on CPU).
        Returns second derivatives at each knot point.
        """
        n = len(x)

        # Tridiagonal system for natural cubic spline
        # Second derivatives at endpoints are zero (natural spline)
        h = np.diff(x)

        # Build tridiagonal matrix coefficients
        alpha = np.zeros(n)
        for i in range(1, n - 1):
            alpha[i] = 3.0 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

        # Solve tridiagonal system
        l = np.ones(n)
        mu = np.zeros(n)
        z = np.zeros(n)

        for i in range(1, n - 1):
            l[i] = 2.0 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

        # Back substitution
        c = np.zeros(n)
        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]

        # Compute b and d coefficients
        b = np.zeros(n - 1)
        d = np.zeros(n - 1)
        for i in range(n - 1):
            b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (c[i+1] + 2.0 * c[i]) / 3.0
            d[i] = (c[i+1] - c[i]) / (3.0 * h[i])

        return {'x': x, 'y': y, 'b': b, 'c': c[:-1], 'd': d}

    def _evaluate_spline_gpu(self, r, x_knots, y_knots, coeffs, xp):
        """
        Evaluate cubic spline entirely on GPU using vectorized operations.

        For each point r, finds the interval and evaluates:
        S(r) = y[i] + b[i]*(r-x[i]) + c[i]*(r-x[i])^2 + d[i]*(r-x[i])^3
        """
        # Move data to GPU if needed
        if xp == cp and HAS_CUDA:
            x = cp.asarray(x_knots)
            y = cp.asarray(y_knots)
            b = cp.asarray(coeffs['b'])
            c = cp.asarray(coeffs['c'])
            d = cp.asarray(coeffs['d'])
            if isinstance(r, np.ndarray):
                r = cp.asarray(r)
        else:
            x = np.asarray(x_knots)
            y = np.asarray(y_knots)
            b = np.asarray(coeffs['b'])
            c = np.asarray(coeffs['c'])
            d = np.asarray(coeffs['d'])
            if HAS_CUDA and isinstance(r, cp.ndarray):
                r = cp.asnumpy(r)

        n = len(x)

        # Find interval indices using searchsorted (GPU-accelerated in CuPy)
        # idx[i] is the index of the interval containing r[i]
        idx = xp.searchsorted(x, r, side='right') - 1

        # Clamp to valid range [0, n-2]
        idx = xp.clip(idx, 0, n - 2)

        # Compute dx = r - x[idx] for all points at once (vectorized)
        dx = r - x[idx]

        # Evaluate spline: S = y + b*dx + c*dx^2 + d*dx^3 (all vectorized)
        # Using Horner's method for numerical stability: S = y + dx*(b + dx*(c + dx*d))
        V = y[idx] + dx * (b[idx] + dx * (c[idx] + dx * d[idx]))

        return V


class SpinOrbitField(FieldBase):
    """
    Spin-orbit coupling field.

    Matrix element: <Lambda, S, Sigma | H_SO | Lambda', S', Sigma'>
    """

    def __init__(self, name: str, istate: int, jstate: int,
                 lambda_i: int, lambda_j: int,
                 spin_i: float, spin_j: float,
                 sigma_i: float, sigma_j: float,
                 func_type: str = "GRID",
                 parameters: Optional[np.ndarray] = None,
                 grid_values: Optional[np.ndarray] = None):

        super().__init__(
            name=name,
            field_type=FieldType.SPINORBIT,
            istate=istate,
            jstate=jstate,
            lambda_i=lambda_i,
            lambda_j=lambda_j,
            spin_i=spin_i,
            spin_j=spin_j,
            sigma_i=sigma_i,
            sigma_j=sigma_j,
            func_type=func_type.upper(),
            parameters=parameters if parameters is not None else np.array([]),
            grid_values=grid_values if grid_values is not None else np.array([])
        )

    def evaluate(self, r, use_gpu: bool = True):
        """
        Evaluate spin-orbit coupling at given r values.

        This includes:
        1. Computing the analytic function (BOBLEROY, polynomial, etc.)
        2. Applying the factor from input
        3. Applying morphing (multiplication by ab initio data) if enabled
        """
        xp = cp if (use_gpu and HAS_CUDA) else np

        if isinstance(r, np.ndarray) and use_gpu and HAS_CUDA:
            r = cp.asarray(r)

        params = self.parameters
        if use_gpu and HAS_CUDA and isinstance(params, np.ndarray):
            params = cp.asarray(params)

        func_type = self.func_type.upper()

        # Compute the raw analytic function value
        if func_type == "GRID":
            analytic_values = self.grid_values
        elif func_type in ["POLYNOMIAL", "POLYNOM"]:
            analytic_values = self._polynomial(r, params, xp)
        elif func_type in ["BOB", "SURKUS", "BOBLEROY"]:
            analytic_values = self._bob_leroy(r, params, xp)
        else:
            # Default: constant value
            if len(params) > 0:
                analytic_values = xp.full_like(r, params[0])
            else:
                analytic_values = xp.zeros_like(r)

        # Apply factor and morphing (multiplication by ab initio data)
        result = self.apply_morphing(analytic_values, r, use_gpu=use_gpu)

        # Apply MOLPRO <x|Lz|y> to Duo spherical basis transformation
        result = self._apply_molpro_transformation(result, xp)

        return result

    def _apply_molpro_transformation(self, values, xp):
        """
        Apply MOLPRO to Duo basis transformation for spin-orbit fields.

        This implements the full transformation from Duo's molpro_duo subroutine:
        1. Build eigenvector matrices a and b from <x|Lz|y> values
        2. Build coupling matrix according to ix_lz_y/jx_lz_y combinations
        3. Transform: f_t = a† @ coupling @ b
        4. Return values scaled by real(f_t[0,0])

        The transformation depends on:
        - ix_lz_y and jx_lz_y values (0 for Sigma, +/-1 for Pi states)
        - Parity of the states involved
        """
        # Build transformation matrices a and b (2x2 complex)
        a, b = self._build_molpro_matrices()

        # Compute transformation factor based on case
        transform_factor = self._compute_spinorbit_transform_factor(a, b)

        return values * transform_factor

    def _build_molpro_matrices(self):
        """Build the a and b transformation matrices for molpro_duo."""
        # Default: identity
        a = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], dtype=np.complex128)
        b = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], dtype=np.complex128)

        # If ix_lz_y != 0, build a from diagonalization of Lz-like matrix
        if self.ix_lz_y != 0:
            a = self._build_lz_eigenvector_matrix(self.ix_lz_y)
        elif self.parity_i == -1:
            # For Sigma- states with parity=-1, a(1,1) = i
            a[0, 0] = 1j

        # If jx_lz_y != 0, build b from diagonalization
        if self.jx_lz_y != 0:
            b = self._build_lz_eigenvector_matrix(self.jx_lz_y)
        elif self.parity_j == -1:
            b[0, 0] = 1j

        return a, b

    def _build_lz_eigenvector_matrix(self, lz_value):
        """
        Build the eigenvector matrix for the Lz-like matrix analytically.

        For the matrix [[0, i*lz], [-i*lz, 0]], the eigenvalues are +|lz| and -|lz|.
        The eigenvectors (following Duo's convention with positive eigenvalue first):
        - For lz > 0: eigenvector for +lz is [1, -i]/sqrt(2), for -lz is [1, i]/sqrt(2)
        - For lz < 0: eigenvector for +|lz| is [1, i]/sqrt(2), for -|lz| is [1, -i]/sqrt(2)

        After multiplying by i (Duo convention):
        - For lz > 0: [i, 1]/sqrt(2) and [i, -1]/sqrt(2)
        - For lz < 0: [i, -1]/sqrt(2) and [i, 1]/sqrt(2)

        Returns the matrix with columns as eigenvectors, positive eigenvalue first.
        """
        sqrt2 = np.sqrt(2.0)

        if lz_value > 0:
            # Eigenvalues: +lz, -lz (positive first after Duo convention)
            # Eigenvector for +lz: [1, -i]/sqrt(2), after *i: [i, 1]/sqrt(2)
            # Eigenvector for -lz: [1, i]/sqrt(2), after *i: [i, -1]/sqrt(2)
            result = np.array([
                [1j/sqrt2, 1j/sqrt2],
                [1.0/sqrt2, -1.0/sqrt2]
            ], dtype=np.complex128)
        else:
            # lz_value < 0: eigenvalues are -lz (positive), lz (negative)
            # Eigenvalues: +|lz|, -|lz| = -lz, lz
            # Eigenvector for -lz: [1, i]/sqrt(2), after *i: [i, -1]/sqrt(2)
            # Eigenvector for lz: [1, -i]/sqrt(2), after *i: [i, 1]/sqrt(2)
            result = np.array([
                [1j/sqrt2, 1j/sqrt2],
                [-1.0/sqrt2, 1.0/sqrt2]
            ], dtype=np.complex128)

        return result

    def _compute_spinorbit_transform_factor(self, a, b):
        """
        Compute the transformation factor for spin-orbit fields.

        This implements the coupling matrix construction and transformation
        from Duo's molpro_duo subroutine for SPINORBIT fields.
        """
        c = self.complex_f  # complex factor from input

        # Case 1: Both ix_lz_y == 0 and jx_lz_y == 0
        if self.ix_lz_y == 0 and self.jx_lz_y == 0:
            # coupling(1,1) = gridvalue * complex_f
            # f_t = a† @ coupling @ b where a, b are identity (or have i on diagonal for Sigma-)
            # f_t[0,0] = conj(a[0,0]) * c * b[0,0]
            coupling = np.zeros((2, 2), dtype=np.complex128)
            coupling[0, 0] = c
            f_t = np.conj(a.T) @ coupling @ b
            return f_t[0, 0].real

        # Case 2: ix_lz_y != 0 and jx_lz_y == 0 (bra is Pi, ket is Sigma)
        elif self.ix_lz_y != 0 and self.jx_lz_y == 0:
            coupling = np.zeros((2, 2), dtype=np.complex128)
            if self.parity_j == -1:
                # Regular case: <Pi|SO|Sigma->
                coupling[0, 0] = c
                coupling[1, 0] = -c * np.conj(a[0, 1] / a[1, 1])
            else:
                # Irregular case: <Pi|SO|Sigma+>
                coupling[1, 0] = c
                coupling[0, 0] = -c * np.conj(a[1, 1] / a[0, 1])
            f_t = np.conj(a.T) @ coupling @ b
            return f_t[0, 0].real

        # Case 3: ix_lz_y == 0 and jx_lz_y != 0 (bra is Sigma, ket is Pi)
        elif self.ix_lz_y == 0 and self.jx_lz_y != 0:
            coupling = np.zeros((2, 2), dtype=np.complex128)
            if self.parity_i == -1:
                # Regular case: <Sigma-|SO|Pi>
                coupling[0, 0] = c
                coupling[0, 1] = -c * b[0, 1] / b[1, 1]
            else:
                # Irregular case: <Sigma+|SO|Pi>
                coupling[0, 1] = c
                coupling[0, 0] = -c * b[1, 1] / b[0, 1]
            f_t = np.conj(a.T) @ coupling @ b
            return f_t[0, 0].real

        # Case 4: Both non-zero with different absolute values (Pi-Pi with different Lambda)
        elif abs(self.ix_lz_y) != abs(self.jx_lz_y):
            # <Pi_i|SO|Pi_j> where Lambda_i != Lambda_j
            coupling = np.zeros((2, 2), dtype=np.complex128)
            coupling[0, 0] = -c * b[1, 1] / b[0, 1]
            coupling[0, 1] = c
            coupling[1, 0] = c * np.conj(a[0, 1] / a[1, 1]) * b[1, 1] / b[0, 1]
            coupling[1, 1] = -c * np.conj(a[0, 1] / a[1, 1])
            f_t = np.conj(a.T) @ coupling @ b
            return f_t[0, 0].real

        # Case 5: Both non-zero with same absolute values (diagonal spin-orbit, SOZ)
        else:
            # <Pi|SOZ|Pi> same state
            coupling = np.zeros((2, 2), dtype=np.complex128)
            coupling[0, 1] = c
            coupling[1, 0] = -np.conj(a[0, 0]) * c * b[1, 1] / (np.conj(a[1, 0]) * b[0, 1])
            f_t = np.conj(a.T) @ coupling @ b
            return f_t[0, 0].real

    def _polynomial(self, r, params, xp):
        """Polynomial expansion using Horner's method (GPU-efficient)."""
        re = params[0]
        coeffs = params[1:]
        dr = r - re

        n_coeffs = len(coeffs)
        if n_coeffs > 0:
            f = xp.full_like(r, float(coeffs[-1]))
            for k in range(n_coeffs - 2, -1, -1):
                f = f * dr + float(coeffs[k])
        else:
            f = xp.zeros_like(r)
        return f

    def _bob_leroy(self, r, params, xp):
        """
        BOB/LeRoy expansion with interpolation to infinity value.

        Fortran formula: f = (1-z)*t + z*tinf
        where z = (r^p - r0^p) / (r^p + r0^p)
        and t = sum_k beta_k * z^k

        Parameters format: [RE, P, B0, B1, ..., BINF]
        """
        re = params[0]
        p = int(params[1]) if len(params) > 1 else 1

        # Coefficients are params[2:-1], BINF is params[-1]
        if len(params) > 3:
            coeffs = params[2:-1]
            tinf = float(params[-1])
        elif len(params) > 2:
            coeffs = params[2:]
            tinf = 0.0
        else:
            coeffs = []
            tinf = 0.0

        rp = r**p
        rep = re**p
        z = (rp - rep) / (rp + rep)

        # Sum t = sum_k beta_k * z^k using Horner's method
        n_coeffs = len(coeffs)
        if n_coeffs > 0:
            t = xp.full_like(r, float(coeffs[-1]))
            for k in range(n_coeffs - 2, -1, -1):
                t = t * z + float(coeffs[k])
        else:
            t = xp.zeros_like(r)

        # Fortran formula: f = (1-z)*t + z*tinf
        f = (1.0 - z) * t + z * tinf

        return f


class CouplingField(FieldBase):
    """
    Generic coupling field for various Hamiltonian terms.

    Includes: L2, Lx/Ly, spin-spin, spin-rotation, diabatic, NAC, etc.
    """

    def __init__(self, name: str, field_type: FieldType,
                 istate: int, jstate: int,
                 lambda_i: int = 0, lambda_j: int = 0,
                 spin_i: float = 0.0, spin_j: float = 0.0,
                 sigma_i: float = 0.0, sigma_j: float = 0.0,
                 func_type: str = "GRID",
                 parameters: Optional[np.ndarray] = None,
                 grid_values: Optional[np.ndarray] = None):

        super().__init__(
            name=name,
            field_type=field_type,
            istate=istate,
            jstate=jstate,
            lambda_i=lambda_i,
            lambda_j=lambda_j,
            spin_i=spin_i,
            spin_j=spin_j,
            sigma_i=sigma_i,
            sigma_j=sigma_j,
            func_type=func_type.upper(),
            parameters=parameters if parameters is not None else np.array([]),
            grid_values=grid_values if grid_values is not None else np.array([])
        )

    def evaluate(self, r, use_gpu: bool = True):
        """
        Evaluate coupling at given r values.

        This includes:
        1. Computing the analytic function (BOBLEROY, polynomial, etc.)
        2. Applying the factor from input
        3. Applying morphing (multiplication by ab initio data) if enabled
        """
        xp = cp if (use_gpu and HAS_CUDA) else np

        if isinstance(r, np.ndarray) and use_gpu and HAS_CUDA:
            r = cp.asarray(r)

        params = self.parameters
        if use_gpu and HAS_CUDA and isinstance(params, np.ndarray):
            params = cp.asarray(params)

        # Compute the raw analytic function value
        if self.func_type == "GRID":
            analytic_values = self.grid_values
        elif self.func_type in ["POLYNOMIAL", "POLYNOM"]:
            analytic_values = self._polynomial(r, params, xp)
        elif self.func_type in ["BOB", "SURKUS", "BOBLEROY"]:
            analytic_values = self._bob_leroy(r, params, xp)
        else:
            if len(params) > 0:
                analytic_values = xp.full_like(r, params[0])
            else:
                analytic_values = xp.zeros_like(r)

        # Apply factor and morphing (multiplication by ab initio data)
        result = self.apply_morphing(analytic_values, r, use_gpu=use_gpu)

        # Apply MOLPRO <x|Lz|y> to Duo spherical basis transformation
        result = self._apply_molpro_transformation(result, xp)

        return result

    def _apply_molpro_transformation(self, values, xp):
        """
        Apply MOLPRO to Duo basis transformation for coupling fields (L+, LXLY, etc.).

        This implements the full transformation from Duo's molpro_duo subroutine
        for L+ type couplings. The logic is similar to spin-orbit but with
        different coupling matrix construction.
        """
        # Build transformation matrices a and b
        a, b = self._build_molpro_matrices()

        # Compute transformation factor based on field type and ix_lz_y/jx_lz_y
        transform_factor = self._compute_lplus_transform_factor(a, b)

        return values * transform_factor

    def _build_molpro_matrices(self):
        """Build the a and b transformation matrices for molpro_duo."""
        # Default: identity
        a = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], dtype=np.complex128)
        b = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], dtype=np.complex128)

        if self.ix_lz_y != 0:
            a = self._build_lz_eigenvector_matrix(self.ix_lz_y)
        elif self.parity_i == -1:
            a[0, 0] = 1j

        if self.jx_lz_y != 0:
            b = self._build_lz_eigenvector_matrix(self.jx_lz_y)
        elif self.parity_j == -1:
            b[0, 0] = 1j

        return a, b

    def _build_lz_eigenvector_matrix(self, lz_value):
        """
        Build the eigenvector matrix for the Lz-like matrix analytically.
        Same as SpinOrbitField._build_lz_eigenvector_matrix.
        """
        sqrt2 = np.sqrt(2.0)

        if lz_value > 0:
            result = np.array([
                [1j/sqrt2, 1j/sqrt2],
                [1.0/sqrt2, -1.0/sqrt2]
            ], dtype=np.complex128)
        else:
            result = np.array([
                [1j/sqrt2, 1j/sqrt2],
                [-1.0/sqrt2, 1.0/sqrt2]
            ], dtype=np.complex128)

        return result

    def _compute_lplus_transform_factor(self, a, b):
        """
        Compute the transformation factor for L+/LX coupling fields.

        This implements the coupling matrix construction from Duo's molpro_duo
        for L+ (ABINITIO-LX, ABINITIO-L+, MAGNETIC) field types.
        """
        c = self.complex_f

        # Case 1: Both ix_lz_y == 0 and jx_lz_y == 0
        if self.ix_lz_y == 0 and self.jx_lz_y == 0:
            coupling = np.zeros((2, 2), dtype=np.complex128)
            coupling[0, 0] = c
            f_t = np.conj(a.T) @ coupling @ b
            return f_t[0, 0].real

        # Case 2: ix_lz_y != 0 and jx_lz_y == 0 (Pi to Sigma)
        elif self.ix_lz_y != 0 and self.jx_lz_y == 0:
            coupling = np.zeros((2, 2), dtype=np.complex128)
            if self.parity_j == -1:
                coupling[0, 0] = c
                coupling[1, 0] = -c * np.conj(a[0, 1] / a[1, 1])
            else:
                coupling[1, 0] = c
                coupling[0, 0] = -c * np.conj(a[1, 1] / a[0, 1])
            f_t = np.conj(a.T) @ coupling @ b
            return f_t[0, 0].real

        # Case 3: ix_lz_y == 0 and jx_lz_y != 0 (Sigma to Pi)
        elif self.ix_lz_y == 0 and self.jx_lz_y != 0:
            coupling = np.zeros((2, 2), dtype=np.complex128)
            if self.parity_i == -1:
                coupling[0, 0] = c
                coupling[0, 1] = -c * b[0, 1] / b[1, 1]
            else:
                coupling[0, 1] = c
                coupling[0, 0] = -c * b[1, 1] / b[0, 1]
            f_t = np.conj(a.T) @ coupling @ b
            return f_t[0, 0].real

        # Case 4: Both non-zero with different absolute values (Pi-Pi different Lambda)
        elif abs(self.ix_lz_y) != abs(self.jx_lz_y):
            # L+ coupling between different Pi states
            coupling = np.zeros((2, 2), dtype=np.complex128)
            coupling[0, 0] = -c * b[1, 1] / b[0, 1]
            coupling[0, 1] = c
            coupling[1, 0] = c * np.conj(a[0, 1] / a[1, 1]) * b[1, 1] / b[0, 1]
            coupling[1, 1] = -c * np.conj(a[0, 1] / a[1, 1])
            f_t = np.conj(a.T) @ coupling @ b
            return f_t[0, 0].real

        # Case 5: Same absolute values (diagonal case)
        else:
            coupling = np.zeros((2, 2), dtype=np.complex128)
            coupling[0, 1] = c
            coupling[1, 0] = -np.conj(a[0, 0]) * c * b[1, 1] / (np.conj(a[1, 0]) * b[0, 1])
            f_t = np.conj(a.T) @ coupling @ b
            return f_t[0, 0].real

    def _polynomial(self, r, params, xp):
        """Polynomial expansion using Horner's method (GPU-efficient)."""
        re = params[0]
        coeffs = params[1:]
        dr = r - re

        n_coeffs = len(coeffs)
        if n_coeffs > 0:
            f = xp.full_like(r, float(coeffs[-1]))
            for k in range(n_coeffs - 2, -1, -1):
                f = f * dr + float(coeffs[k])
        else:
            f = xp.zeros_like(r)
        return f

    def _bob_leroy(self, r, params, xp):
        """
        BOB/LeRoy expansion with interpolation to infinity value.

        Fortran formula: f = (1-z)*t + z*tinf
        where z = (r^p - r0^p) / (r^p + r0^p)
        and t = sum_k beta_k * z^k

        Parameters format: [RE, P, B0, B1, ..., BINF]
        """
        re = params[0]
        p = int(params[1]) if len(params) > 1 else 1

        # Coefficients are params[2:-1], BINF is params[-1]
        if len(params) > 3:
            coeffs = params[2:-1]
            tinf = float(params[-1])
        elif len(params) > 2:
            coeffs = params[2:]
            tinf = 0.0
        else:
            coeffs = []
            tinf = 0.0

        rp = r**p
        rep = re**p
        z = (rp - rep) / (rp + rep)

        # Sum t = sum_k beta_k * z^k using Horner's method
        n_coeffs = len(coeffs)
        if n_coeffs > 0:
            t = xp.full_like(r, float(coeffs[-1]))
            for k in range(n_coeffs - 2, -1, -1):
                t = t * z + float(coeffs[k])
        else:
            t = xp.zeros_like(r)

        # Fortran formula: f = (1-z)*t + z*tinf
        f = (1.0 - z) * t + z * tinf

        return f


class DipoleMomentField(FieldBase):
    """
    Electric dipole moment function.

    Used for transition moment calculations.
    """

    def __init__(self, name: str, istate: int, jstate: int,
                 lambda_i: int = 0, lambda_j: int = 0,
                 spin_i: float = 0.0, spin_j: float = 0.0,
                 func_type: str = "GRID",
                 parameters: Optional[np.ndarray] = None,
                 grid_values: Optional[np.ndarray] = None):

        super().__init__(
            name=name,
            field_type=FieldType.DIPOLE,
            istate=istate,
            jstate=jstate,
            lambda_i=lambda_i,
            lambda_j=lambda_j,
            spin_i=spin_i,
            spin_j=spin_j,
            func_type=func_type.upper(),
            parameters=parameters if parameters is not None else np.array([]),
            grid_values=grid_values if grid_values is not None else np.array([])
        )

    def evaluate(self, r, use_gpu: bool = True):
        """Evaluate dipole moment at given r values."""
        xp = cp if (use_gpu and HAS_CUDA) else np

        if isinstance(r, np.ndarray) and use_gpu and HAS_CUDA:
            r = cp.asarray(r)

        params = self.parameters
        if use_gpu and HAS_CUDA and isinstance(params, np.ndarray):
            params = cp.asarray(params)

        if self.func_type == "GRID":
            return self.grid_values
        elif self.func_type in ["POLYNOMIAL", "POLYNOM"]:
            re = params[0]
            coeffs = params[1:]
            dr = r - re
            # Horner's method (GPU-efficient)
            n_coeffs = len(coeffs)
            if n_coeffs > 0:
                mu = xp.full_like(r, float(coeffs[-1]))
                for k in range(n_coeffs - 2, -1, -1):
                    mu = mu * dr + float(coeffs[k])
            else:
                mu = xp.zeros_like(r)
            return mu
        elif self.func_type == "POLYNOM_DECAY":
            # mu(r) = sum_i a_i * (r-re)^i * exp(-beta*(r-re)^2)
            re = params[0]
            beta = params[1]
            coeffs = params[2:]
            dr = r - re
            # Horner's method (GPU-efficient)
            n_coeffs = len(coeffs)
            if n_coeffs > 0:
                mu = xp.full_like(r, float(coeffs[-1]))
                for k in range(n_coeffs - 2, -1, -1):
                    mu = mu * dr + float(coeffs[k])
            else:
                mu = xp.zeros_like(r)
            mu = mu * xp.exp(-beta * dr**2)
            return mu
        else:
            if len(params) > 0:
                return xp.full_like(r, params[0])
            return xp.zeros_like(r)
