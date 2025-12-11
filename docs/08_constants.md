# Physical Constants

## Key Constants

### ASTON

The **ASTON constant** is the most important scaling factor in Duo:

$$\text{ASTON} = \frac{\hbar^2}{2 \cdot m_u \cdot \text{Å}^2 \cdot hc \cdot \text{cm}} = 16.857629206 \, \text{cm}^{-1} \cdot \text{amu} \cdot \text{Å}^2$$

This converts between:
- Kinetic energy in atomic units
- Energy in cm⁻¹ with mass in amu and length in Å

**Physical meaning**: The rotational constant $B$ for a molecule with $\mu = 1$ amu and $r = 1$ Å would be 16.857629206 cm⁻¹.

### Derivation

Starting from:
$$\hat{T} = -\frac{\hbar^2}{2\mu} \frac{d^2}{dr^2}$$

To express in cm⁻¹ with $\mu$ in amu and $r$ in Å:
$$\text{ASTON} = \frac{\hbar^2}{2} \times \frac{1}{m_u} \times \frac{1}{\text{Å}^2} \times \frac{1}{hc \cdot \text{cm}^{-1}}$$

## Code Implementation

### PyDuo: `constants.py`

```python
# pyduo/constants.py

class PhysicalConstants:
    """Physical constants for PyDuo calculations."""

    # ASTON: hbar^2 / (2 * amu * Angstrom^2 * cm^-1)
    # This is the factor that converts kinetic energy to cm^-1
    # when mass is in amu and distance in Angstrom
    ASTON = 16.857629206

    # Other constants
    BOHR = 0.52917721092       # Bohr radius in Angstrom
    HARTREE = 219474.6313705   # Hartree in cm^-1
    AMU_TO_AU = 1822.888486    # amu to electron masses

    # Speed of light
    C_CM_S = 2.99792458e10     # cm/s

    # Planck constant
    H_CM_S = 6.62607015e-34 / (1.98644568e-23)  # h in cm^-1 * s
```

**Location**: `pyduo/constants.py`

### Fortran Duo: `accuracy.f90`

```fortran
! accuracy.f90

module accuracy
  implicit none

  integer, parameter :: rk = kind(1.0d0)  ! Double precision
  integer, parameter :: ik = 4            ! Integer kind
  integer, parameter :: hik = 8           ! Large integer kind

  ! Physical constants
  real(rk), parameter :: pi = 3.14159265358979323846_rk
  real(rk), parameter :: bohr = 0.52917721092_rk    ! Angstrom
  real(rk), parameter :: aston = 16.857629206_rk    ! cm^-1 * amu * Angstrom^2
  real(rk), parameter :: hartree = 219474.6313705_rk ! cm^-1
  real(rk), parameter :: umatoau = 1822.888486_rk   ! amu to electron masses

  ! Small number threshold
  real(rk), parameter :: small_ = epsilon(1.0_rk) * 100.0_rk

end module accuracy
```

**Location**: `accuracy.f90`

## Usage in Calculations

### Scaling Factor

The main scaling factor `sc`:
$$sc = 12 \cdot h^2 \cdot \frac{\mu}{\text{ASTON}}$$

### Rotational Constant

$$B(r) = \frac{\text{ASTON}}{\mu \cdot r^2}$$

**Units**: cm⁻¹

### Vibrational Energy

Vibrational energies come out directly in cm⁻¹ when using ASTON correctly.

## Unit Conversions

| From | To | Factor |
|------|-----|--------|
| Hartree | cm⁻¹ | 219474.6313705 |
| eV | cm⁻¹ | 8065.544 |
| kJ/mol | cm⁻¹ | 83.5935 |
| kcal/mol | cm⁻¹ | 349.755 |
| Bohr | Å | 0.52917721092 |
| amu | electron mass | 1822.888486 |

## Example: BeH Molecule

For BeH with:
- $\mu = 1.497$ amu
- $r_e = 1.34$ Å

The equilibrium rotational constant is:
$$B_e = \frac{16.857629206}{1.497 \times 1.34^2} = 6.27 \, \text{cm}^{-1}$$

This matches experimental values!

## Atomic Mass Data

Duo includes atomic mass data from:

```fortran
! atomic_and_nuclear_data.f90

subroutine check_and_set_atomic_data(iverbose)
  ! Sets masses from isotope symbols like "Be-9", "H-1"

  ! Example masses (in amu):
  ! H-1:   1.00782503207
  ! H-2:   2.01410177785
  ! Be-9:  9.0121822
  ! C-12:  12.0000000
  ! O-16:  15.99491461956
end subroutine
```

**Location**: `atomic_and_nuclear_data.f90`

### PyDuo

```python
# pyduo/constants.py

ATOMIC_MASSES = {
    'H': 1.00782503207,
    'D': 2.01410177785,
    'He': 4.00260325413,
    'Li': 7.016003436,
    'Be': 9.0121822,
    'B': 11.0093054,
    'C': 12.0,
    'N': 14.003074005,
    'O': 15.99491461956,
    # ...
}
```

## Reduced Mass Calculation

$$\mu = \frac{m_1 \cdot m_2}{m_1 + m_2}$$

```python
# PyDuo
reduced_mass = (mass1 * mass2) / (mass1 + mass2)
```

```fortran
! Fortran Duo
amass = m1 * m2 / (m1 + m2)
```

## Consistency Checks

To verify your implementation produces correct results:

1. **Rotational spacing** for J→J+1:
   $$\Delta E = 2B(J+1)$$

2. **Vibrational spacing** for harmonic oscillator:
   $$\Delta E = \omega_e$$

3. **Rotational constant** at equilibrium:
   $$B_e = \frac{\text{ASTON}}{\mu \cdot r_e^2}$$

For BeH with $r_e = 1.3426$ Å and $\mu = 1.497$ amu:
$$B_e = \frac{16.857629206}{1.497 \times 1.3426^2} = 6.25 \, \text{cm}^{-1}$$

The vibrational spacing should be approximately $\omega_e \approx 2060$ cm⁻¹ for BeH.
