"""
Grid generation module for PyDuo

Implements various grid types for the radial coordinate:
- Uniform grid
- Non-uniform grids (Chebyshev, Lobatto, etc.)
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = np
    HAS_CUDA = False


class Grid:
    """
    Radial coordinate grid for diatomic molecule calculations.

    Supports multiple grid types matching the Fortran Duo implementation.
    """

    GRID_TYPES = {
        0: 'uniform',
        1: 'stretched_1',
        2: 'stretched_2',
        3: 'stretched_3',
        4: 'stretched_4',
        5: 'chebyshev',
        6: 'lobatto'
    }

    def __init__(self, npoints: int, rmin: float, rmax: float,
                 grid_type: int = 0, alpha: float = 0.2, re: float = 1.0,
                 use_gpu: bool = True):
        """
        Initialize the radial grid.

        Parameters
        ----------
        npoints : int
            Number of grid points
        rmin : float
            Minimum radial distance (Angstroms)
        rmax : float
            Maximum radial distance (Angstroms)
        grid_type : int
            Type of grid (0=uniform, 1-4=stretched, 5=Chebyshev, 6=Lobatto)
        alpha : float
            Grid stretching parameter
        re : float
            Reference distance (equilibrium bond length)
        use_gpu : bool
            Whether to use GPU acceleration
        """
        self.npoints = npoints
        self.rmin = rmin
        self.rmax = rmax
        self.grid_type = grid_type
        self.alpha = alpha
        self.re = re
        self.use_gpu = use_gpu and HAS_CUDA

        # Select array module
        self.xp = cp if self.use_gpu else np

        # Generate the grid
        self.r = self._generate_grid()
        self.step = self._compute_step()
        self.weights = self._compute_weights()

    def _generate_grid(self):
        """Generate the grid points based on grid type."""
        xp = self.xp
        n = self.npoints

        if self.grid_type == 0:
            # Uniform grid
            r = xp.linspace(self.rmin, self.rmax, n)

        elif self.grid_type == 1:
            # Stretched grid type 1: r = rmin + (rmax-rmin)*(1-exp(-alpha*i))/(1-exp(-alpha*n))
            i = xp.arange(n, dtype=xp.float64)
            r = self.rmin + (self.rmax - self.rmin) * (
                1.0 - xp.exp(-self.alpha * i)
            ) / (1.0 - xp.exp(-self.alpha * (n - 1)))

        elif self.grid_type == 2:
            # Stretched grid type 2: based on tanh
            i = xp.arange(n, dtype=xp.float64)
            x = -1.0 + 2.0 * i / (n - 1)
            r = self.re + (self.rmax - self.rmin) * 0.5 * xp.tanh(self.alpha * x) / xp.tanh(self.alpha)

        elif self.grid_type == 3:
            # Stretched grid type 3: sinh-based
            i = xp.arange(n, dtype=xp.float64)
            x = -1.0 + 2.0 * i / (n - 1)
            r = self.re + (self.rmax - self.rmin) * 0.5 * xp.sinh(self.alpha * x) / xp.sinh(self.alpha)

        elif self.grid_type == 4:
            # Stretched grid type 4: polynomial stretching
            i = xp.arange(n, dtype=xp.float64)
            x = i / (n - 1)
            beta = (self.re - self.rmin) / (self.rmax - self.rmin)
            r = self.rmin + (self.rmax - self.rmin) * (
                beta * x + (1 - beta) * x**self.alpha
            )

        elif self.grid_type == 5:
            # Chebyshev grid
            i = xp.arange(n, dtype=xp.float64)
            theta = xp.pi * (n - 1 - i) / (n - 1)
            x = xp.cos(theta)
            r = 0.5 * (self.rmax + self.rmin) + 0.5 * (self.rmax - self.rmin) * x

        elif self.grid_type == 6:
            # Lobatto grid (Gauss-Lobatto-Legendre points)
            r = self._lobatto_points()

        else:
            raise ValueError(f"Unknown grid type: {self.grid_type}")

        return r

    def _lobatto_points(self):
        """Compute Gauss-Lobatto-Legendre points."""
        xp = self.xp
        n = self.npoints

        # Initial guess: Chebyshev points
        i = xp.arange(n, dtype=xp.float64)
        x = -xp.cos(xp.pi * i / (n - 1))

        # Newton-Raphson iteration for interior points
        P = xp.zeros((n, n + 1))

        for _ in range(10):  # Usually converges in a few iterations
            P[:, 0] = 1.0
            P[:, 1] = x

            for k in range(1, n):
                P[:, k + 1] = ((2*k + 1) * x * P[:, k] - k * P[:, k - 1]) / (k + 1)

            # Derivative of Legendre polynomial
            dP = n * (x * P[:, n] - P[:, n - 1]) / (x**2 - 1.0 + 1e-16)

            # Newton step for interior points
            dx = -P[:, n] / (dP + 1e-16)
            x[1:-1] = x[1:-1] + dx[1:-1]

            if float(xp.max(xp.abs(dx[1:-1]))) < 1e-15:
                break

        # Ensure endpoints are exactly -1 and 1
        x[0] = -1.0
        x[-1] = 1.0

        # Map to [rmin, rmax]
        r = 0.5 * (self.rmax + self.rmin) + 0.5 * (self.rmax - self.rmin) * x

        return r

    def _compute_step(self):
        """Compute grid step size (for uniform grid) or average step."""
        if self.grid_type == 0:
            return (self.rmax - self.rmin) / (self.npoints - 1)
        else:
            # For non-uniform grids, return the array of local steps
            xp = self.xp
            steps = xp.zeros(self.npoints - 1)
            steps[:] = self.r[1:] - self.r[:-1]
            return steps

    def _compute_weights(self):
        """Compute integration weights for the grid."""
        xp = self.xp
        n = self.npoints

        if self.grid_type == 0:
            # Uniform grid: trapezoidal rule weights
            w = xp.ones(n) * self.step
            w[0] = 0.5 * self.step
            w[-1] = 0.5 * self.step

        elif self.grid_type == 6:
            # Lobatto weights
            w = self._lobatto_weights()

        else:
            # General case: use trapezoidal rule with variable spacing
            w = xp.zeros(n)
            w[0] = 0.5 * (self.r[1] - self.r[0])
            w[-1] = 0.5 * (self.r[-1] - self.r[-2])
            w[1:-1] = 0.5 * (self.r[2:] - self.r[:-2])

        return w

    def _lobatto_weights(self):
        """Compute Gauss-Lobatto-Legendre weights."""
        xp = self.xp
        n = self.npoints

        # Lobatto weights on [-1, 1]
        x = 2.0 * (self.r - self.rmin) / (self.rmax - self.rmin) - 1.0

        # Compute Legendre polynomial at grid points
        P = xp.zeros(n)
        for i in range(n):
            xi = float(x[i])
            Pn_1 = 1.0
            Pn = xi
            for k in range(1, n - 1):
                Pn_2 = Pn_1
                Pn_1 = Pn
                Pn = ((2*k + 1) * xi * Pn_1 - k * Pn_2) / (k + 1)
            P[i] = Pn

        # Lobatto weights formula
        w = 2.0 / (n * (n - 1) * P**2 + 1e-16)

        # Scale to [rmin, rmax]
        w = w * 0.5 * (self.rmax - self.rmin)

        return w

    def to_gpu(self):
        """Move grid data to GPU."""
        if HAS_CUDA and not self.use_gpu:
            self.r = cp.asarray(self.r)
            if isinstance(self.step, np.ndarray):
                self.step = cp.asarray(self.step)
            self.weights = cp.asarray(self.weights)
            self.use_gpu = True
            self.xp = cp

    def to_cpu(self):
        """Move grid data to CPU."""
        if self.use_gpu:
            self.r = cp.asnumpy(self.r) if HAS_CUDA else self.r
            if isinstance(self.step, cp.ndarray if HAS_CUDA else np.ndarray):
                self.step = cp.asnumpy(self.step) if HAS_CUDA else self.step
            self.weights = cp.asnumpy(self.weights) if HAS_CUDA else self.weights
            self.use_gpu = False
            self.xp = np

    def get_numpy(self):
        """Return grid as numpy array (for compatibility)."""
        if self.use_gpu and HAS_CUDA:
            return cp.asnumpy(self.r)
        return self.r

    def __len__(self):
        return self.npoints

    def __repr__(self):
        return (f"Grid(npoints={self.npoints}, rmin={self.rmin:.4f}, "
                f"rmax={self.rmax:.4f}, type={self.GRID_TYPES.get(self.grid_type, 'unknown')}, "
                f"gpu={self.use_gpu})")
