# PyDuo Overview

## Introduction

PyDuo is a Python/CUDA implementation of the Duo program for solving the rovibronic Schrödinger equation for diatomic molecules. This documentation explains the mathematical foundations and their correspondence to both the PyDuo code and the original Fortran Duo implementation.

## The Rovibronic Schrödinger Equation

For a diatomic molecule, the full rovibronic Hamiltonian in the rotating (body-fixed) frame is:

$$\hat{H} = \hat{T}_{\text{vib}} + \hat{T}_{\text{rot}} + \hat{V}(r) + \hat{H}_{\text{SO}} + \hat{H}_{\text{SS}} + \hat{H}_{\text{SR}} + \ldots$$

Where:
- $\hat{T}_{\text{vib}}$ = Vibrational kinetic energy operator
- $\hat{T}_{\text{rot}}$ = Rotational kinetic energy operator
- $\hat{V}(r)$ = Potential energy curve (PEC)
- $\hat{H}_{\text{SO}}$ = Spin-orbit coupling
- $\hat{H}_{\text{SS}}$ = Spin-spin coupling
- $\hat{H}_{\text{SR}}$ = Spin-rotation coupling

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  1. INPUT PARSING                                                │
│     - Read atomic masses → compute reduced mass                  │
│     - Read grid parameters (npoints, rmin, rmax)                │
│     - Read potential curves, spin-orbit, couplings              │
├─────────────────────────────────────────────────────────────────┤
│  2. GRID CONSTRUCTION                                           │
│     - Generate r-coordinate grid points                         │
│     - Compute grid step h and scaling factor sc                 │
├─────────────────────────────────────────────────────────────────┤
│  3. KINETIC ENERGY MATRIX                                       │
│     - Build kinmat using SINC-DVR or 5-point finite differences │
│     - Store unscaled matrix (scaling applied in Hamiltonian)    │
├─────────────────────────────────────────────────────────────────┤
│  4. VIBRATIONAL PROBLEM (J=0)                                   │
│     - For each electronic state:                                │
│       - Build H_vib = T + V(r)*sc + L²(r)*f_rot                │
│       - Diagonalize → vibrational eigenstates                   │
│       - Store contracted basis functions                        │
├─────────────────────────────────────────────────────────────────┤
│  5. ROVIBRONIC PROBLEM (J>0)                                    │
│     - Build full Hamiltonian in contracted basis                │
│     - Include rotation, spin-orbit, couplings                   │
│     - Diagonalize for each J value                              │
├─────────────────────────────────────────────────────────────────┤
│  6. OUTPUT                                                       │
│     - Energy levels with quantum number assignments             │
│     - Wavefunctions for intensity calculations                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Scaling Factors

The most important scaling factor in Duo is:

$$sc = 12 \cdot h^2 \cdot \frac{\mu}{\text{ASTON}}$$

Where:
- $h$ = grid step size (Ångström)
- $\mu$ = reduced mass (amu)
- ASTON = $\hbar^2 / (2 \cdot \text{amu} \cdot \text{Å}^2 \cdot \text{cm}^{-1}) = 16.857629206$

This factor converts between the dimensionless kinetic matrix and physical energy units (cm⁻¹).

## File Structure

| File | Purpose | Fortran Equivalent |
|------|---------|-------------------|
| `grid.py` | Coordinate grid construction | `grids.f90`, `gridred()` |
| `kinetic.py` | Kinetic energy matrix | `diatom.f90:kinetic_energy_grid_points()` |
| `fields.py` | Potential/coupling curves | `diatom.f90`, `functions.f90` |
| `hamiltonian.py` | Hamiltonian construction | `diatom.f90:duo_j0()` |
| `solver.py` | Eigenvalue solver | `lapack.f90` wrappers |
| `input_parser.py` | Input file parsing | `input.f90` |
| `duo.py` | Main driver | `duo.f90` |
| `constants.py` | Physical constants | `accuracy.f90` |

## Units Convention

All internal calculations use:
- **Length**: Ångström (Å)
- **Energy**: cm⁻¹ (wavenumbers)
- **Mass**: atomic mass units (amu/Dalton)
- **Angular momentum**: ℏ = 1 (dimensionless)

## References

1. Yurchenko, S.N., Lodi, L., Tennyson, J., Stolyarov, A.V. (2016). "Duo: A general program for calculating spectra of diatomic molecules", Computer Physics Communications, 202, 262-275.

2. Colbert, D.T., Miller, W.H. (1992). "A novel discrete variable representation for quantum mechanical reactive scattering via the S-matrix Kohn method", J. Chem. Phys. 96, 1982.
