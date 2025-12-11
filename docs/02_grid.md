# Grid Construction

## Mathematical Foundation

The radial coordinate $r$ (internuclear distance) is discretized onto a grid of $N$ points. Duo supports several grid types, but the most common is the **uniform grid** (type 0).

### Uniform Grid (Type 0)

For a uniform grid:
$$r_j = r_{\min} + h \cdot (j-1), \quad j = 1, 2, \ldots, N$$

Where the step size is:
$$h = \frac{r_{\max} - r_{\min}}{N - 1}$$

For uniform grids, two auxiliary quantities are defined:
- $z_j = 1$ (coordinate mapping Jacobian)
- $f_j = 0$ (additional kinetic term from non-uniform mapping)

### Non-Uniform Grids

For non-uniform grids, a coordinate transformation $r = r(z)$ is used where $z$ is uniformly spaced. This allows denser sampling near the equilibrium distance. The Jacobian $\frac{dr}{dz}$ appears in the kinetic energy operator.

**Type 1**: $z = \exp(-\exp(-\alpha(r-r_e)))$

**Type 2**: $z = 1 - [1 + \exp(\alpha(r-r_e))]^{-1}$

**Type 4** (Surkus): $z = \frac{y-1}{y+1}$ where $y = (r/r_e)^\alpha$

**Type 6**: Lobatto quadrature points (for DVR methods)

## Code Implementation

### PyDuo: `grid.py`

```python
# pyduo/grid.py - Grid class

class Grid:
    def __init__(self, npoints, rmin, rmax, grid_type=0, ...):
        self.npoints = npoints
        self.rmin = rmin
        self.rmax = rmax

        # Uniform grid (type 0)
        if grid_type == 0:
            self.step = (rmax - rmin) / (npoints - 1)
            self.r = np.linspace(rmin, rmax, npoints)
            self.z = np.ones(npoints)      # Jacobian = 1
            self.add = np.zeros(npoints)   # No additional terms
```

**Key attributes:**
- `self.r` - Grid points in Ångström
- `self.step` (or `h`) - Grid spacing
- `self.z` - Coordinate mapping derivative $dz/dr$
- `self.add` - Additional kinetic term from non-uniform mapping

### Fortran Duo: `grids.f90`

```fortran
! grids.f90 - subroutine gridred()

subroutine gridred(nsub, alpha, re, mes, rmin, rmax, h, r, z, add, iverbose)
  ! nsub = grid type (0=uniform, 1,2,3,4,5,6 = various mappings)
  ! mes  = number of grid points
  ! h    = step size (in mapped coordinate z)
  ! r    = output radial coordinates
  ! z    = dz/dr (Jacobian of coordinate mapping)
  ! add  = additional kinetic energy term from mapping

  select case (nsub)
    case (0)  ! Uniform grid
      h = (rmax - rmin) / real(mes - 1, rk)
      do j = 1, mes
        r(j) = rmin + h * real(j-1, rk)
        z(j) = 1.0_rk        ! dr/dz = 1 for uniform
        add(j) = 0.0_rk      ! No additional term
      end do

    case (4)  ! Surkus variable (recommended for molecules)
      ! z = (y-1)/(y+1) where y = (r/re)^alpha
      y = (rmin/re)**alpha
      zmin = (y - 1.0_rk)/(y + 1.0_rk)
      ! ... similar for zmax
      h = (zmax - zmin) / real(mes - 1, rk)
      do j = 1, mes
        y = zmin + h * real(j-1, rk)
        r(j) = re * ((1.0_rk + y)/(1.0_rk - y))**(1.0_rk/alpha)
        t = (r(j)/re)**alpha
        z(j) = 2.0_rk * alpha * t / r(j) / (t + 1.0_rk)**2
        add(j) = (alpha**2 - 1.0_rk) / 4.0_rk / r(j)**2
      end do
  end select
end subroutine
```

**Location**: `grids.f90`, lines 1-148

## The Scaling Factor

A critical quantity is the **scaling factor** `sc` which converts between matrix elements and energy units:

$$sc = 12 \cdot h^2 \cdot \frac{\mu}{\text{ASTON}}$$

Where:
- Factor of 12 comes from the 5-point finite difference denominator
- $h^2$ is the grid spacing squared
- $\mu/\text{ASTON}$ converts to energy units

### PyDuo

```python
# pyduo/hamiltonian.py - HamiltonianBuilder._build_vibrational_hamiltonian()

h = self.grid.step
sc = 12.0 * h**2 * self.reduced_mass / self.aston
```

### Fortran Duo

```fortran
! diatom.f90, lines 4855-4858

hstep = h
h12 = 12.0_rk * h**2
sc = h12 * scale   ! where scale = amass/aston
```

**Location**: `diatom.f90`, lines 4855-4858

## Grid Weights (for Lobatto DVR)

For Lobatto-based DVR, quadrature weights $w_j$ are needed:

```fortran
! Called in diatom.f90, line 8375-8376
call LobattoAbsWeights(LobAbs, LobWeights, ngrid, grid%rmin, grid%rmax)
```

**Location**: `Lobatto.f90`

## Grid Representation of Fields

All potential curves and coupling functions are evaluated on the grid:

```fortran
! diatom.f90, line 8427
epot = poten(istate)%gridvalue(igrid) * sc
```

The potential values are stored pre-multiplied by `sc` during the field mapping phase.

## Input File Specification

```
grid
  npoints 501      ! Number of grid points
  range 0.4 8.0    ! rmin and rmax in Angstrom
  type 0           ! Grid type (0 = uniform)
end
```

## Summary Table

| Quantity | Symbol | PyDuo | Fortran | Units |
|----------|--------|-------|---------|-------|
| Grid points | $N$ | `grid.npoints` | `ngrid` | - |
| Minimum r | $r_{\min}$ | `grid.rmin` | `grid%rmin` | Å |
| Maximum r | $r_{\max}$ | `grid.rmax` | `grid%rmax` | Å |
| Step size | $h$ | `grid.step` | `hstep` | Å |
| Grid array | $r_j$ | `grid.r` | `r(:)` | Å |
| Jacobian | $z_j$ | `grid.z` | `z(:)` | - |
| Scale factor | $sc$ | computed in hamiltonian | `sc` | cm⁻¹ |
