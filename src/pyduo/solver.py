"""
Eigenvalue solvers for PyDuo

Implements GPU-accelerated eigenvalue decomposition using cuSOLVER.
"""

import numpy as np
from typing import Optional, Tuple, Literal

try:
    import cupy as cp
    from cupy.linalg import eigh as cp_eigh
    HAS_CUDA = True
except ImportError:
    cp = np
    HAS_CUDA = False


class EigenSolver:
    """
    GPU-accelerated eigenvalue solver for symmetric matrices.

    Uses cuSOLVER for CUDA operations, falling back to NumPy/SciPy
    when CUDA is not available.
    """

    def __init__(self, use_gpu: bool = True, method: str = 'syevd'):
        """
        Initialize the eigenvalue solver.

        Parameters
        ----------
        use_gpu : bool
            Whether to use GPU acceleration
        method : str
            Solver method: 'syev', 'syevd', 'syevr', 'syevx'
            - syev: Standard divide-and-conquer
            - syevd: Divide-and-conquer (usually fastest)
            - syevr: Relatively robust representations (subset selection)
            - syevx: Expert driver with more control
        """
        self.use_gpu = use_gpu and HAS_CUDA
        self.method = method.lower()
        self.xp = cp if self.use_gpu else np

    def solve(self, matrix: np.ndarray,
              nroots: Optional[int] = None,
              eigenvalues_only: bool = False,
              subset_by_index: Optional[Tuple[int, int]] = None,
              subset_by_value: Optional[Tuple[float, float]] = None
              ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Solve the symmetric eigenvalue problem.

        Parameters
        ----------
        matrix : ndarray
            Symmetric matrix to diagonalize
        nroots : int, optional
            Number of eigenvalues to compute (smallest)
        eigenvalues_only : bool
            If True, only compute eigenvalues (faster)
        subset_by_index : tuple, optional
            (il, iu) - compute eigenvalues il to iu (1-indexed)
        subset_by_value : tuple, optional
            (vl, vu) - compute eigenvalues in range [vl, vu]

        Returns
        -------
        eigenvalues : ndarray
            Eigenvalues in ascending order
        eigenvectors : ndarray or None
            Eigenvectors (column vectors) if eigenvalues_only=False
        """
        n = matrix.shape[0]

        if self.use_gpu:
            return self._solve_gpu(matrix, nroots, eigenvalues_only,
                                   subset_by_index, subset_by_value)
        else:
            return self._solve_cpu(matrix, nroots, eigenvalues_only,
                                   subset_by_index, subset_by_value)

    def _solve_gpu(self, matrix: np.ndarray,
                   nroots: Optional[int],
                   eigenvalues_only: bool,
                   subset_by_index: Optional[Tuple[int, int]],
                   subset_by_value: Optional[Tuple[float, float]]
                   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Solve using CuPy (cuSOLVER backend)."""
        # Transfer to GPU
        mat_gpu = cp.asarray(matrix, dtype=cp.float64)

        # CuPy's eigh uses cuSOLVER's syevd internally
        if eigenvalues_only:
            eigenvalues = cp.linalg.eigvalsh(mat_gpu)
            eigenvalues = cp.asnumpy(eigenvalues)
            eigenvectors = None
        else:
            eigenvalues, eigenvectors = cp.linalg.eigh(mat_gpu)
            eigenvalues = cp.asnumpy(eigenvalues)
            eigenvectors = cp.asnumpy(eigenvectors)

        # Apply subset selection if requested
        if subset_by_index is not None:
            il, iu = subset_by_index
            il = max(0, il - 1)  # Convert to 0-indexed
            iu = min(len(eigenvalues), iu)
            eigenvalues = eigenvalues[il:iu]
            if eigenvectors is not None:
                eigenvectors = eigenvectors[:, il:iu]
        elif subset_by_value is not None:
            vl, vu = subset_by_value
            mask = (eigenvalues >= vl) & (eigenvalues <= vu)
            eigenvalues = eigenvalues[mask]
            if eigenvectors is not None:
                eigenvectors = eigenvectors[:, mask]
        elif nroots is not None:
            nroots = min(nroots, len(eigenvalues))
            eigenvalues = eigenvalues[:nroots]
            if eigenvectors is not None:
                eigenvectors = eigenvectors[:, :nroots]

        return eigenvalues, eigenvectors

    def _solve_cpu(self, matrix: np.ndarray,
                   nroots: Optional[int],
                   eigenvalues_only: bool,
                   subset_by_index: Optional[Tuple[int, int]],
                   subset_by_value: Optional[Tuple[float, float]]
                   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Solve using NumPy/SciPy on CPU."""
        try:
            from scipy.linalg import eigh as scipy_eigh
            has_scipy = True
        except ImportError:
            has_scipy = False

        if has_scipy and (subset_by_index is not None or subset_by_value is not None):
            # Use scipy for subset eigenvalue problems
            if subset_by_index is not None:
                il, iu = subset_by_index
                il = max(0, il - 1)
                iu = min(matrix.shape[0], iu) - 1
                subset_by_index = (il, iu)

            kwargs = {}
            if subset_by_index is not None:
                kwargs['subset_by_index'] = subset_by_index
            if subset_by_value is not None:
                kwargs['subset_by_value'] = subset_by_value

            if eigenvalues_only:
                kwargs['eigvals_only'] = True
                eigenvalues = scipy_eigh(matrix, **kwargs)
                eigenvectors = None
            else:
                eigenvalues, eigenvectors = scipy_eigh(matrix, **kwargs)

        else:
            # Use numpy
            if eigenvalues_only:
                eigenvalues = np.linalg.eigvalsh(matrix)
                eigenvectors = None
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(matrix)

            # Apply nroots
            if nroots is not None:
                nroots = min(nroots, len(eigenvalues))
                eigenvalues = eigenvalues[:nroots]
                if eigenvectors is not None:
                    eigenvectors = eigenvectors[:, :nroots]

        return eigenvalues, eigenvectors


class BatchEigenSolver:
    """
    Solve multiple eigenvalue problems in parallel on GPU.

    Useful for solving vibrational problems for multiple states
    or J values simultaneously.
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and HAS_CUDA
        self.xp = cp if self.use_gpu else np

    def solve_batch(self, matrices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve eigenvalue problems for a batch of matrices.

        Parameters
        ----------
        matrices : ndarray
            Array of shape (batch_size, n, n) containing symmetric matrices

        Returns
        -------
        eigenvalues : ndarray
            Array of shape (batch_size, n) containing eigenvalues
        eigenvectors : ndarray
            Array of shape (batch_size, n, n) containing eigenvectors
        """
        batch_size = matrices.shape[0]
        n = matrices.shape[1]

        if self.use_gpu:
            # Transfer to GPU
            mats_gpu = cp.asarray(matrices, dtype=cp.float64)

            # CuPy doesn't have batched eigh, so we loop
            # In the future, could use cuSOLVER's batched routines
            eigenvalues = cp.zeros((batch_size, n), dtype=cp.float64)
            eigenvectors = cp.zeros((batch_size, n, n), dtype=cp.float64)

            for i in range(batch_size):
                evals, evecs = cp.linalg.eigh(mats_gpu[i])
                eigenvalues[i] = evals
                eigenvectors[i] = evecs

            eigenvalues = cp.asnumpy(eigenvalues)
            eigenvectors = cp.asnumpy(eigenvectors)

        else:
            eigenvalues = np.zeros((batch_size, n), dtype=np.float64)
            eigenvectors = np.zeros((batch_size, n, n), dtype=np.float64)

            for i in range(batch_size):
                evals, evecs = np.linalg.eigh(matrices[i])
                eigenvalues[i] = evals
                eigenvectors[i] = evecs

        return eigenvalues, eigenvectors


def solve_generalized_eigenvalue(A: np.ndarray, B: np.ndarray,
                                 use_gpu: bool = True
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenvalue problem A*x = lambda*B*x.

    Parameters
    ----------
    A : ndarray
        Left-hand side matrix (symmetric)
    B : ndarray
        Right-hand side matrix (symmetric positive definite)
    use_gpu : bool
        Whether to use GPU

    Returns
    -------
    eigenvalues : ndarray
        Generalized eigenvalues
    eigenvectors : ndarray
        Generalized eigenvectors
    """
    if use_gpu and HAS_CUDA:
        # CuPy doesn't have generalized eigh, so we do Cholesky decomposition
        A_gpu = cp.asarray(A, dtype=cp.float64)
        B_gpu = cp.asarray(B, dtype=cp.float64)

        # B = L * L^T
        L = cp.linalg.cholesky(B_gpu)
        L_inv = cp.linalg.inv(L)

        # Transform to standard eigenvalue problem: L^{-1} A L^{-T} y = lambda y
        A_transformed = L_inv @ A_gpu @ L_inv.T

        eigenvalues, y = cp.linalg.eigh(A_transformed)

        # Back-transform eigenvectors: x = L^{-T} y
        eigenvectors = cp.linalg.solve(L.T, y)

        eigenvalues = cp.asnumpy(eigenvalues)
        eigenvectors = cp.asnumpy(eigenvectors)

    else:
        try:
            from scipy.linalg import eigh as scipy_eigh
            eigenvalues, eigenvectors = scipy_eigh(A, B)
        except ImportError:
            # Manual implementation
            L = np.linalg.cholesky(B)
            L_inv = np.linalg.inv(L)
            A_transformed = L_inv @ A @ L_inv.T
            eigenvalues, y = np.linalg.eigh(A_transformed)
            eigenvectors = np.linalg.solve(L.T, y)

    return eigenvalues, eigenvectors


def three_j_symbol(j1: float, j2: float, j3: float,
                   m1: float, m2: float, m3: float) -> float:
    """
    Compute the Wigner 3j symbol.

    (j1 j2 j3)
    (m1 m2 m3)

    Used for angular momentum coupling calculations.
    """
    # Selection rules
    if abs(m1 + m2 + m3) > 1e-10:
        return 0.0
    if abs(j1 - j2) > j3 or j3 > j1 + j2:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0

    # Check that 2j's are integers
    if (2*j1) % 1 != 0 or (2*j2) % 1 != 0 or (2*j3) % 1 != 0:
        return 0.0
    if (2*m1) % 1 != 0 or (2*m2) % 1 != 0 or (2*m3) % 1 != 0:
        return 0.0

    # Use the Racah formula
    from math import factorial, sqrt

    def _factorial(n):
        if n < 0:
            return 0
        return factorial(int(n))

    # Triangle coefficient
    def _delta(a, b, c):
        return (_factorial(a + b - c) * _factorial(a - b + c) *
                _factorial(-a + b + c) / _factorial(a + b + c + 1))

    # Phase
    phase = (-1) ** int(j1 - j2 - m3)

    # Sum over t
    tmin = max(0, int(j2 - j3 - m1), int(j1 - j3 + m2))
    tmax = min(int(j1 + j2 - j3), int(j1 - m1), int(j2 + m2))

    sum_val = 0.0
    for t in range(tmin, tmax + 1):
        num = _factorial(t)
        num *= _factorial(j3 - j2 + t + m1)
        num *= _factorial(j3 - j1 + t - m2)
        num *= _factorial(j1 + j2 - j3 - t)
        num *= _factorial(j1 - t - m1)
        num *= _factorial(j2 - t + m2)

        if num != 0:
            sum_val += (-1) ** t / num

    delta_val = sqrt(_delta(j1, j2, j3))
    factor = sqrt(_factorial(j1 + m1) * _factorial(j1 - m1) *
                  _factorial(j2 + m2) * _factorial(j2 - m2) *
                  _factorial(j3 + m3) * _factorial(j3 - m3))

    return phase * delta_val * factor * sum_val


def setup_factorials_lookup(nmax: int = 200) -> np.ndarray:
    """
    Pre-compute factorial lookup table.

    Used for efficient 3j symbol calculations.
    """
    from math import lgamma

    # Use log factorials to avoid overflow
    log_factorials = np.zeros(nmax + 1)
    for i in range(1, nmax + 1):
        log_factorials[i] = log_factorials[i-1] + np.log(i)

    return log_factorials
