# Kinetic Energy Matrix

## Mathematical Foundation

The vibrational kinetic energy operator in the radial Schrödinger equation is:

$$\hat{T} = -\frac{\hbar^2}{2\mu} \frac{d^2}{dr^2}$$

Where $\mu$ is the reduced mass. In matrix form on a discrete grid, this becomes the **kinetic energy matrix** $\mathbf{T}$.

## Methods

Duo supports three methods for constructing the kinetic energy matrix:

### 1. SINC-DVR (Default)

The Colbert-Miller SINC discrete variable representation uses cardinal sine functions as the basis. For a uniform grid with spacing $h$:

**Standard SINC-DVR formula:**

$$T_{ii} = \frac{\hbar^2}{2\mu} \frac{\pi^2}{3h^2}$$

$$T_{ij} = \frac{\hbar^2}{2\mu} \frac{2(-1)^{i-j}}{(i-j)^2 h^2}, \quad i \neq j$$

**Duo's convention** (pre-multiplied by $12h^2$ to match finite differences scaling):

$$\text{kinmat}_{ii} = 12 \cdot \frac{\pi^2}{3} = 4\pi^2 \approx 39.478$$

$$\text{kinmat}_{ij} = 12 \cdot \frac{2(-1)^{i-j}}{(i-j)^2}, \quad i \neq j$$

The factor of 12 is absorbed so that when multiplied by:
$$sc = 12h^2 \cdot \frac{\mu}{\text{ASTON}}$$

The $12h^2$ cancels with the $h^2$ denominator, leaving the correct kinetic energy.

### 2. Five-Point Finite Differences

The 5-point central difference formula for the second derivative is:

$$f''(x) = \frac{-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)}{12h^2} + O(h^4)$$

In matrix form (pre-multiplied by 12 to match Duo's convention):

$$\text{kinmat}_{ii} = 30$$
$$\text{kinmat}_{i,i\pm 1} = -16$$
$$\text{kinmat}_{i,i\pm 2} = +1$$

For non-uniform grids, the coefficients are modified by the Jacobian $z_j$:

$$\text{kinmat}_{ii} = 30 \cdot z_i^2 - 12h^2 \cdot f_i$$

where $f_i$ is the additional term from coordinate mapping.

### 3. Lobatto DVR

For Lobatto-based DVR, the kinetic matrix is constructed using:

$$T_{ij} = \sum_k w_k D_{ik} D_{jk}$$

where $D$ is the derivative matrix and $w_k$ are Lobatto quadrature weights.

## Code Implementation

### PyDuo: `kinetic.py`

```python
# pyduo/kinetic.py - KineticEnergyMatrix class

def _build_sinc_dvr(self):
    """
    Build kinetic energy matrix using SINC-DVR.

    Following Fortran Duo's convention exactly (see diatom.f90 lines 13031-13056):
    - kinmat already includes a factor of 12 (to match finite differences form)
    - Diagonal: 12 * pi^2 / 3
    - Off-diagonal: 12 * 2 * (-1)^(i-j) / (i-j)^2
    """
    xp = self.xp  # numpy or cupy
    ngrid = self.grid.npoints

    # Create index arrays for vectorized computation
    i_idx = xp.arange(ngrid, dtype=xp.float64).reshape(-1, 1)
    j_idx = xp.arange(ngrid, dtype=xp.float64).reshape(1, -1)
    diff = i_idx - j_idx

    # Avoid division by zero on diagonal
    diff_safe = xp.where(diff == 0, 1.0, diff)

    # Off-diagonal elements: +12 * 2 * (-1)^(i-j) / (i-j)^2
    kinmat = 12.0 * 2.0 * ((-1.0) ** diff) / (diff_safe ** 2)

    # Diagonal elements: 12 * pi^2 / 3
    kinmat = xp.where(diff == 0, 12.0 * np.pi ** 2 / 3.0, kinmat)

    return kinmat
```

**Location**: `pyduo/kinetic.py`, lines 84-120

### Fortran Duo: `diatom.f90`

```fortran
! diatom.f90 - subroutine kinetic_energy_grid_points()
! Lines 12960-13108

subroutine kinetic_energy_grid_points(ngrid, kinmat, kinmat1, LobWeights, LobDerivs)
  integer(ik), intent(in) :: ngrid
  real(rk), intent(inout) :: kinmat(ngrid, ngrid)
  ! ... optional arguments for first derivative and Lobatto

  kinmat = 0

  select case(solution_method)

  case ("5POINTDIFFERENCES")
    !$omp parallel do private(igrid) shared(kinmat) schedule(dynamic)
    do igrid = 1, ngrid
      ! Diagonal: 30*z^2 - 12*h^2*f (for uniform grid: 30)
      kinmat(igrid, igrid) = kinmat(igrid, igrid) + d2dr(igrid)

      ! Off-diagonal by 1: -16*z(i)*z(j)
      if (igrid > 1) then
        kinmat(igrid, igrid-1) = -16.0_rk * z(igrid-1) * z(igrid)
        kinmat(igrid-1, igrid) = kinmat(igrid, igrid-1)
      endif

      ! Off-diagonal by 2: +z(i)*z(j)
      if (igrid > 2) then
        kinmat(igrid, igrid-2) = z(igrid-2) * z(igrid)
        kinmat(igrid-2, igrid) = kinmat(igrid, igrid-2)
      endif
    enddo
    !$omp end parallel do

  case("SINC")
    ! Colbert-Miller sinc DVR (works only for uniform grids)
    !$omp parallel do private(igrid, jgrid) shared(kinmat) schedule(dynamic)
    do igrid = 1, ngrid
      ! Diagonal: 12 * pi^2 / 3
      kinmat(igrid, igrid) = kinmat(igrid, igrid) + (12._rk) * pi**2 / 3._rk

      ! Off-diagonal: 12 * 2 * (-1)^(i+j) / (i-j)^2
      do jgrid = igrid+1, ngrid
        kinmat(igrid, jgrid) = +(12._rk) * 2._rk * real((-1)**(igrid+jgrid), rk) &
                               / real(igrid - jgrid, rk)**2
        kinmat(jgrid, igrid) = kinmat(igrid, jgrid)
      enddo
    enddo
    !$omp end parallel do

  end select
end subroutine
```

**Location**: `diatom.f90`, lines 13021-13056 (SINC case)

## First Derivative Matrix (for NAC terms)

For non-adiabatic coupling (NAC) terms, the first derivative matrix is also needed:

**SINC-DVR first derivative:**
$$D_{ij} = \frac{(-1)^{i-j}}{(i-j) \cdot h}, \quad i \neq j$$
$$D_{ii} = 0$$

**In Duo's convention** (pre-multiplied by $12h$):
$$\text{kinmat1}_{ij} = -12 \cdot \frac{(-1)^{i-j} \cdot h}{i-j}$$

### Fortran Implementation

```fortran
! diatom.f90, lines 13043-13053
if (present(kinmat1)) then
  ! f'(0) = (-1)^(i+j)/(i-j)/h
  ! kinetic factor is 12*h^2/(h) = 12*h
  do jgrid = igrid+1, ngrid
    kinmat1(igrid, jgrid) = -(12._rk) * real((-1)**(igrid+jgrid), rk) * hstep &
                           / real(igrid - jgrid, rk)
    kinmat1(jgrid, igrid) = -kinmat1(igrid, jgrid)
  enddo
endif
```

## Applying the Kinetic Matrix

The kinetic matrix is applied to the Hamiltonian by:

1. Starting with the pre-computed `kinmat` (unscaled)
2. Adding potential energy terms (scaled by `sc`)
3. The `kinmat` implicitly becomes scaled when added to the Hamiltonian

### PyDuo

```python
# pyduo/hamiltonian.py - _build_vibrational_hamiltonian()

h = self.grid.step
sc = 12.0 * h**2 * self.reduced_mass / self.aston

# Copy kinetic matrix (unscaled but with factor of 12 built in)
vibmat = xp.copy(self.kinmat)

# Add potential and L^2 terms (scaled by sc)
for igrid in range(ngrid):
    vibmat[igrid, igrid] += V[igrid] * sc + f_l2[igrid] * f_rot[igrid]
```

### Fortran Duo

```fortran
! diatom.f90, lines 8413-8445

vibmat = kinmat  ! Copy kinetic matrix

!$omp parallel do private(igrid, f_rot, epot, f_l2, erot) shared(vibmat)
do igrid = 1, ngrid
  f_rot = b_rot / r(igrid)**2 * sc
  epot = poten(istate)%gridvalue(igrid) * sc

  ! L^2 term
  f_l2 = 0
  do iL2 = 1, Nl2
    if (L2(iL2)%istate == istate .and. L2(iL2)%jstate == istate) then
      f_l2 = f_rot * L2(iL2)%gridvalue(igrid)
      exit
    endif
  enddo

  erot = f_l2
  vibmat(igrid, igrid) = vibmat(igrid, igrid) + epot + erot
enddo
!$omp end parallel do
```

## Summary: The Factor of 12

The key insight is that Duo stores `kinmat` with a **built-in factor of 12**, so that:

| Quantity | Standard Formula | Duo's Convention |
|----------|------------------|------------------|
| Diagonal | $\pi^2/(3h^2)$ | $12\pi^2/3$ |
| Off-diag | $2(-1)^{i-j}/((i-j)^2 h^2)$ | $24(-1)^{i-j}/(i-j)^2$ |
| Scale factor | $\hbar^2/(2\mu)$ | $sc = 12h^2 \mu/\text{ASTON}$ |

When `kinmat` is used in the Hamiltonian with scaling `sc`, the $12h^2$ in `sc` combines with the factor of 12 in `kinmat` to give the correct physical kinetic energy.

## Numerical Values

For a typical calculation with:
- Grid step $h = 0.0152$ Å (from 501 points over 0.4-8.0 Å)
- Reduced mass $\mu = 1.497$ amu (BeH)

The scale factor is:
$$sc = 12 \times (0.0152)^2 \times \frac{1.497}{16.857629206} \approx 2.47 \times 10^{-4}$$

And the kinetic matrix diagonal is:
$$\text{kinmat}_{ii} = 12 \times \frac{\pi^2}{3} \approx 39.478$$

After scaling by $sc$, the kinetic contribution to the diagonal is approximately $0.00975$ cm⁻¹ per grid point.
