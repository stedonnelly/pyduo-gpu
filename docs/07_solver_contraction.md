# Eigenvalue Solver and Contraction Scheme

## Overview

Duo uses a two-step approach:
1. **Contraction**: Solve the vibrational problem to reduce basis size
2. **Diagonalization**: Solve the full rovibronic problem in the contracted basis

## Contraction Schemes

Duo supports several contraction schemes:

| Scheme | Description |
|--------|-------------|
| **VIB** | Contract by vibrational quantum number (default) |
| **OMEGA** | Contract in Omega representation (adiabatic) |
| **ROT** | Contract by rotational quantum number |

### VIB Contraction (Default)

For each electronic state:
1. Solve 1D vibrational Schrödinger equation
2. Keep first `vmax` vibrational eigenstates
3. Use these as basis for full rovibronic calculation

**Advantage**: Dramatically reduces basis size (from `ngrid` to `vmax`)

### OMEGA Contraction

At each grid point:
1. Diagonalize the electronic Hamiltonian (PEC + SO)
2. Get adiabatic Omega-states
3. Solve vibrational problem for each adiabatic state

**Advantage**: Better for strongly coupled states

## Input Specification

```
CONTRACTION
  vib           ! Contraction type
  vmax 30       ! Maximum vibrational states per electronic state
END
```

Or per-state:
```
CONTRACTION
  vib
  vibmax 1 30   ! State 1: 30 vib states
  vibmax 2 20   ! State 2: 20 vib states
END
```

## Eigenvalue Solvers

### PyDuo: `solver.py`

PyDuo uses CuPy/NumPy's eigensolvers:

```python
# pyduo/solver.py - EigenSolver class

class EigenSolver:
    """Eigenvalue solver with GPU support."""

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and HAS_CUDA
        self.xp = cp if self.use_gpu else np

    def solve(self, matrix, nroots=None, energy_range=None):
        """
        Solve eigenvalue problem.

        Parameters
        ----------
        matrix : ndarray
            Symmetric matrix to diagonalize
        nroots : int, optional
            Number of lowest eigenvalues to compute
        energy_range : tuple, optional
            (Emin, Emax) energy window

        Returns
        -------
        eigenvalues : ndarray
        eigenvectors : ndarray
        """
        xp = self.xp

        if self.use_gpu:
            # GPU solver
            eigenvalues, eigenvectors = cp.linalg.eigh(matrix)
        else:
            # CPU solver
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Sort by eigenvalue
        idx = xp.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Truncate if requested
        if nroots is not None:
            eigenvalues = eigenvalues[:nroots]
            eigenvectors = eigenvectors[:, :nroots]

        return eigenvalues, eigenvectors
```

**Location**: `pyduo/solver.py`, lines 30-100

### Fortran Duo: `lapack.f90`

Fortran Duo wraps LAPACK routines:

```fortran
! lapack.f90 - Eigenvalue solver wrappers

subroutine lapack_syev(a, d)
  ! Full diagonalization using DSYEV
  real(rk), intent(inout) :: a(:,:)  ! On output: eigenvectors
  real(rk), intent(out)   :: d(:)    ! Eigenvalues

  call dsyev('V', 'U', n, a, lda, d, work, lwork, info)
end subroutine

subroutine lapack_syevr(a, d, vrange, irange, nroots)
  ! Partial diagonalization using DSYEVR
  ! Computes only eigenvalues in range or specified count
  real(rk), intent(inout) :: a(:,:)
  real(rk), intent(out)   :: d(:)
  real(rk), intent(in)    :: vrange(2)  ! (Vmin, Vmax)
  integer,  intent(in)    :: irange(2)  ! (imin, imax)
  integer,  intent(out)   :: nroots     ! Number found

  call dsyevr('V', 'V', 'U', n, a, lda, vl, vu, il, iu, &
              abstol, nroots, d, z, ldz, isuppz, work, lwork, &
              iwork, liwork, info)
end subroutine
```

**Location**: `lapack.f90`

### cuSOLVER (GPU)

For GPU acceleration, cuSOLVER is used:

```python
# CuPy automatically uses cuSOLVER for cp.linalg.eigh
eigenvalues, eigenvectors = cp.linalg.eigh(matrix)
```

This calls `cusolverDnDsyevd` internally.

## Matrix Element Computation

After contraction, matrix elements of coupling fields are computed:

### Integration on Grid

```python
# pyduo/hamiltonian.py - compute_matrix_elements()

def compute_matrix_elements(self, field):
    """Compute <v'|f(r)|v> matrix elements."""
    ngrid = self.grid.npoints
    h = self.grid.step

    nbasis = len(self.contracted_wavefunctions[0])
    matelem = np.zeros((nbasis, nbasis))

    # Get field values on grid
    f = field.evaluate(self.grid.r)

    # Integrate
    for i in range(nbasis):
        psi_i = self.contracted_wavefunctions[:, i]
        for j in range(i + 1):
            psi_j = self.contracted_wavefunctions[:, j]

            # Simpson's rule or simple sum
            matelem[i, j] = np.sum(psi_i * f * psi_j) * h

            # Symmetrize
            matelem[j, i] = matelem[i, j]

    return matelem
```

### Fortran Implementation

```fortran
! diatom.f90 - Matrix element computation

do ilevel = 1, totalroots
  do jlevel = 1, ilevel
    matelem(ilevel, jlevel) = 0

    ! Sum over grid points
    do igrid = 1, ngrid
      matelem(ilevel, jlevel) = matelem(ilevel, jlevel) + &
          contrfunc(igrid, ilevel) * field%gridvalue(igrid) * &
          contrfunc(igrid, jlevel)
    enddo

    ! Include grid step
    matelem(ilevel, jlevel) = matelem(ilevel, jlevel) * hstep

    ! Symmetrize
    matelem(jlevel, ilevel) = matelem(ilevel, jlevel)
  enddo
enddo
```

**Location**: `diatom.f90`, lines 8850-8950

## Basis Size Considerations

### Without Contraction
- Basis size = `ngrid × nstates × (2S+1)`
- For BeH: 501 × 1 × 2 = 1002 per J value

### With VIB Contraction
- Basis size = `vmax × nstates × (2S+1)`
- For BeH: 30 × 1 × 2 = 60 per J value

**Speedup**: Factor of ~250 reduction in matrix size!

## Memory Management

For large calculations:

```fortran
! diatom.f90 - Deallocating after vibrational solve

deallocate(vibmat, vibener)
call ArrayStop('vibmat')

! Keep only contracted basis
! contrfunc has dimension (ngrid, totalroots)
! where totalroots << ngrid
```

## Convergence Checks

### Energy Window

```
CONTRACTION
  vib
  vmax 30
  enermax 10000.0  ! Only keep states below 10000 cm^-1
END
```

### Basis Set Size

Increase `vmax` until energies converge to desired precision.

## Output Format

### Eigenvalue Output

```
Eigenvalues for J =    0.5

       J      i        Energy/cm  State   v  lambda spin   sigma   omega  parity
     0.5    1        0.000000   1     0   0   0.5    0.5    0.5   +    ||X2Sigma+
     0.5    2       20.123456   1     0   0   0.5   -0.5   -0.5   -    ||X2Sigma+
     0.5    3     1986.987654   1     1   0   0.5    0.5    0.5   +    ||X2Sigma+
```

### PyDuo Implementation

```python
# pyduo/duo.py - print_eigenvalues()

def print_eigenvalues(self, J=None, relative=True):
    """Print eigenvalues in Duo format."""
    zpe = min(lvl.energy for lvl in self.energy_levels)

    for j_val in j_values:
        levels_j = [lvl for lvl in self.energy_levels
                    if abs(lvl.J - j_val) < 0.01]

        print(f"\nEigenvalues for J = {j_val:6.1f}")
        print("       J      i        Energy/cm  State   v  ...")

        for i, lvl in enumerate(levels_j, 1):
            e = lvl.energy - zpe if relative else lvl.energy
            print(f"  {j_val:6.1f} {i:4d}  {e:16.6f} ...")
```

**Location**: `pyduo/duo.py`, lines 381-427

## Summary: Complete Workflow

```
┌──────────────────────────────────────────────────────┐
│                    INPUT PARSING                      │
│  - Grid, masses, potentials, couplings               │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│               VIBRATIONAL PROBLEM                     │
│  For each electronic state:                          │
│  1. Build H_vib = T + V(r)*sc + L²(r)*f_rot         │
│  2. Diagonalize: H_vib |v⟩ = E_v |v⟩                │
│  3. Store vmax eigenstates as contracted basis       │
│                                                      │
│  Result: contracted_energies, contracted_wavefunctions│
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│           COMPUTE MATRIX ELEMENTS                     │
│  For each coupling field:                            │
│  matelem[v',v] = ∫ ψ_v'(r) f(r) ψ_v(r) dr           │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│              ROVIBRONIC PROBLEM                       │
│  For each J value:                                   │
│  1. Build basis: |state, v, Λ, Σ, Ω; J⟩            │
│  2. Construct full Hamiltonian in contracted basis   │
│  3. Add rotation, spin-orbit, couplings             │
│  4. Diagonalize                                      │
│  5. Assign quantum numbers                           │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                      OUTPUT                           │
│  - Energy levels with quantum number assignments     │
│  - Wavefunctions for intensity calculations          │
└──────────────────────────────────────────────────────┘
```
