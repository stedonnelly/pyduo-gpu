# Potential Energy and Field Evaluation

## Overview

Duo supports many analytical forms for potential energy curves (PECs) and coupling functions. The most commonly used are:

- **GRID**: Tabulated r,V pairs with spline interpolation
- **MORSE**: Standard Morse potential
- **EMO**: Expanded Morse Oscillator
- **MLR**: Morse/Long-Range
- **POLYNOMIAL**: Simple polynomial expansion

## Mathematical Forms

### 1. Morse Potential

$$V(r) = V_e + D_e \left(1 - e^{-\beta(r-r_e)}\right)^2$$

**Parameters:**
- $V_e$ = Equilibrium energy (Te)
- $r_e$ = Equilibrium bond length
- $D_e$ = Dissociation energy (relative to $V_e$)
- $\beta$ = Morse parameter

### 2. Expanded Morse Oscillator (EMO)

$$V(r) = V_e + D_e \left(1 - e^{-\phi(r)(r-r_e)}\right)^2$$

Where $\phi(r)$ is a polynomial in the Šurkus variable:

$$z = \frac{r^p - r_{\text{ref}}^p}{r^p + r_{\text{ref}}^p}$$

$$\phi(r) = \sum_{k=0}^{N} \beta_k z^k$$

**Parameters:**
- $V_e$ = Te (parameters[0])
- $r_e$ = Equilibrium distance (parameters[1])
- $D_e$ = Dissociation energy (parameters[2] - parameters[0])
- $r_{\text{ref}}$ = Reference distance (parameters[3])
- $p$ = Power (parameters[4])
- $\beta_k$ = Expansion coefficients (parameters[5:])

### 3. Morse/Long-Range (MLR)

$$V(r) = D_e \left(1 - \frac{u_{\text{LR}}(r)}{u_{\text{LR}}(r_e)} e^{-\phi(r) y_p(r)}\right)^2 + V_e$$

Where $y_p(r)$ is the Šurkus variable and $u_{\text{LR}}(r)$ describes long-range behavior.

### 4. GRID (Tabulated Values)

For tabulated potentials, values are provided as (r, V) pairs and interpolated using cubic splines.

$$V(r) = \text{CubicSpline}(\{r_i, V_i\})$$

## Code Implementation

### PyDuo: `fields.py`

```python
# pyduo/fields.py - PotentialField class

class PotentialField(FieldBase):
    """Potential energy curve for a single electronic state."""

    SUPPORTED_TYPES = [
        'MORSE', 'EMO', 'MLR', 'POLYNOMIAL', 'GRID', 'BOB', 'DUNHAM',
        'SPF', 'CHEBYSHEV', 'MARQUARDT', 'MLJ'
    ]

    def __init__(self, name, istate, lambda_, multiplicity,
                 func_type="MORSE", parameters=None,
                 grid_r=None, grid_v=None):
        # Store grid data for interpolation
        self.grid_r = grid_r  # r values
        self.grid_v = grid_v  # V values

    def evaluate(self, r, use_gpu=True):
        """Evaluate the potential at given r values."""
        xp = cp if (use_gpu and HAS_CUDA) else np

        if self.func_type == "MORSE":
            return self._evaluate_morse(r, xp)
        elif self.func_type == "GRID":
            return self._interpolate_grid(r, xp)
        # ... other types

    def _evaluate_morse(self, r, xp):
        """Standard Morse potential."""
        params = self.parameters
        Ve = params[0]   # Equilibrium energy
        re = params[1]   # Equilibrium distance
        De = params[2]   # Dissociation energy (relative to Ve)
        beta = params[3] # Morse parameter

        y = 1.0 - xp.exp(-beta * (r - re))
        V = Ve + De * y**2
        return V

    def _interpolate_grid(self, r, xp):
        """Interpolate grid values using cubic splines."""
        from scipy.interpolate import CubicSpline

        r_grid = self.grid_r
        v_grid = self.grid_v

        # Convert to numpy for interpolation
        r_np = r if isinstance(r, np.ndarray) else cp.asnumpy(r)

        cs = CubicSpline(r_grid, v_grid, extrapolate=True)
        V = cs(r_np)

        # Convert back to GPU if needed
        if xp == cp:
            V = cp.asarray(V)

        return V
```

**Location**: `pyduo/fields.py`, lines 120-330

### Fortran Duo: `functions.f90`

```fortran
! functions.f90 - poten_morse()

function poten_morse(r, parameters) result(f)
  real(rk), intent(in) :: r             ! geometry (Ang)
  real(rk), intent(in) :: parameters(:) ! potential parameters
  real(rk)             :: y, v0, r0, a0, de, f

  v0 = parameters(1)  ! Te
  r0 = parameters(2)  ! re
  de = parameters(3)  ! De (dissociation energy)
  a0 = parameters(4)  ! beta

  y = 1.0_rk - exp(-a0 * (r - r0))
  f = de * y**2 + v0

end function poten_morse
```

**Location**: `functions.f90`, lines ~520-540

```fortran
! functions.f90 - poten_EMO()

function poten_EMO(r, parameters) result(f)
  real(rk), intent(in) :: r
  real(rk), intent(in) :: parameters(:)
  real(rk)             :: y, v0, r0, de, f, rref, z, phi
  integer(ik)          :: k, N, p

  v0 = parameters(1)    ! Te
  r0 = parameters(2)    ! re
  de = parameters(3) - v0  ! De relative to minimum
  rref = parameters(4)  ! Reference r
  p = nint(parameters(5))  ! Power for Surkus variable

  ! Surkus variable
  z = (r**p - rref**p) / (r**p + rref**p)

  ! Polynomial expansion
  phi = 0
  do k = 0, N-6
    phi = phi + parameters(k+6) * z**k
  enddo

  y = 1.0_rk - exp(-phi * (r - r0))
  f = de * y**2 + v0

end function poten_EMO
```

**Location**: `functions.f90`, lines 690-730

## Grid Representation

After evaluation, all potentials are stored on the computational grid:

### PyDuo

```python
# pyduo/hamiltonian.py - Used in vibrational Hamiltonian

V = poten.evaluate(r, use_gpu=self.use_gpu)
# V is now an array of length ngrid
```

### Fortran Duo

```fortran
! diatom.f90 - Field mapping onto grid

do igrid = 1, ngrid
  field%gridvalue(igrid) = field%fanalytic_field(r(igrid), field%value)
enddo
```

**Location**: `diatom.f90`, field mapping section (~lines 4890-5000)

## Spline Interpolation

For GRID type potentials, cubic spline interpolation is used.

### PyDuo: Using SciPy

```python
from scipy.interpolate import CubicSpline

cs = CubicSpline(r_grid, v_grid, extrapolate=True)
V = cs(r_new)
```

### Fortran Duo: Custom Spline

```fortran
! grids.f90 - spline() and splint()

! Compute second derivatives for spline
call spline(xa, ya, n, yp1, ypn, y2a)

! Interpolate at new point
call splint(xa, ya, y2a, n, x, y)
```

**Location**: `grids.f90`, lines 155-240

## Spin-Orbit Coupling

Spin-orbit coupling fields follow the same pattern:

$$H_{SO} = A(r) \hat{L} \cdot \hat{S}$$

The radial function $A(r)$ can be any of the supported analytical forms.

### PyDuo

```python
# pyduo/fields.py - SpinOrbitField class

class SpinOrbitField(FieldBase):
    """Spin-orbit coupling between states."""

    def evaluate(self, r, use_gpu=True):
        # Similar to PotentialField
        return self._evaluate_function(r, xp)
```

## Other Coupling Fields

| Field Type | Physical Meaning | Matrix Element |
|------------|------------------|----------------|
| `L2` | Electronic angular momentum | $\langle L^2 \rangle$ |
| `LXLY` | $L_+$ / $L_-$ coupling | $\langle \Lambda \pm 1 | L_\pm | \Lambda \rangle$ |
| `SPIN-SPIN` | Spin-spin coupling | $\hat{S}_z^2$ terms |
| `SPIN-ROT` | Spin-rotation coupling | $\gamma(r) \hat{N} \cdot \hat{S}$ |
| `BOB-ROT` | Born-Oppenheimer breakdown (rotation) | Mass-dependent corrections |
| `NAC` | Non-adiabatic coupling | $\langle \psi_i | d/dr | \psi_j \rangle$ |

## Input File Format

### Morse Potential
```
poten 1
  name "X2Sigma+"
  lambda 0
  mult 2
  type morse
  values
    0.0      ! Te (cm^-1)
    1.3426   ! re (Angstrom)
    17722.0  ! De (cm^-1)
    1.956    ! beta (1/Angstrom)
  end
end
```

### GRID Potential
```
poten 1
  name "X2Sigma+"
  lambda 0
  mult 2
  type grid
  values
    0.60  105169.63
    0.65   77543.34
    0.70   55670.88
    ...
  end
end
```

## Units

All potential curves use:
- **Distance**: Ångström
- **Energy**: cm⁻¹

The conversion to the Hamiltonian is handled by the scaling factor `sc`.

## Summary of Function Pointer Mechanism

### Fortran

```fortran
! functions.f90 - Function pointer assignment

subroutine define_fanalytic_field(ftype, fanalytic_field)
  character(len=*), intent(in) :: ftype
  procedure(fanalytic_fieldT), pointer :: fanalytic_field

  select case(ftype)
    case("MORSE")
      fanalytic_field => poten_morse
    case("EMO")
      fanalytic_field => poten_EMO
    case("GRID")
      fanalytic_field => poten_grid  ! Uses spline interpolation
    ! ...
  end select
end subroutine
```

**Location**: `functions.f90`, lines 61-260

### PyDuo

```python
# pyduo/fields.py - Method dispatch

def evaluate(self, r, use_gpu=True):
    if self.func_type == "MORSE":
        return self._evaluate_morse(r, xp)
    elif self.func_type == "EMO":
        return self._evaluate_emo(r, xp)
    elif self.func_type == "GRID":
        return self._interpolate_grid(r, xp)
    # ...
```
