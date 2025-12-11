# PyDuo Documentation

This documentation explains the mathematical foundations and code implementation of PyDuo, a Python/CUDA implementation of the Duo program for diatomic molecular spectroscopy.

## Contents

1. **[Overview](01_overview.md)** - Introduction to PyDuo and the rovibronic problem
2. **[Grid Construction](02_grid.md)** - Coordinate grids and the scaling factor
3. **[Kinetic Energy Matrix](03_kinetic_energy.md)** - SINC-DVR and finite differences
4. **[Potential Energy](04_potential_energy.md)** - PECs and field evaluation
5. **[Vibrational Hamiltonian](05_vibrational_hamiltonian.md)** - Building H_vib
6. **[Rovibronic Hamiltonian](06_rovibronic_hamiltonian.md)** - Full Hamiltonian with couplings
7. **[Solver & Contraction](07_solver_contraction.md)** - Eigensolvers and basis contraction
8. **[Physical Constants](08_constants.md)** - ASTON and unit conversions

## Quick Reference

### Key Files

| PyDuo File | Fortran File | Purpose |
|------------|--------------|---------|
| `constants.py` | `accuracy.f90` | Physical constants |
| `grid.py` | `grids.f90` | Grid construction |
| `kinetic.py` | `diatom.f90:kinetic_energy_grid_points()` | Kinetic energy matrix |
| `fields.py` | `functions.f90` | Potential curves |
| `hamiltonian.py` | `diatom.f90:duo_j0()` | Hamiltonian builder |
| `solver.py` | `lapack.f90` | Eigenvalue solver |
| `input_parser.py` | `input.f90` | Input file parser |
| `duo.py` | `duo.f90` | Main driver |

### Key Equations

**Scaling Factor:**
$$sc = 12 \cdot h^2 \cdot \frac{\mu}{\text{ASTON}}$$

**SINC-DVR Kinetic Matrix (Duo convention):**
$$T_{ii} = \frac{12\pi^2}{3}, \quad T_{ij} = \frac{24(-1)^{i-j}}{(i-j)^2}$$

**Vibrational Hamiltonian:**
$$H_{vib} = T + V(r) \cdot sc + L^2(r) \cdot f_{rot}(r)$$

**Rotational Factor:**
$$f_{rot}(r) = \frac{\text{ASTON}}{\mu \cdot r^2} \cdot sc = \frac{12h^2}{r^2}$$

### Key Fortran Line Numbers

| Section | File | Lines |
|---------|------|-------|
| Grid construction | `grids.f90` | 1-148 |
| Kinetic matrix (SINC) | `diatom.f90` | 13021-13056 |
| Scale factor setup | `diatom.f90` | 4855-4863 |
| Vibrational Hamiltonian | `diatom.f90` | 8345-8540 |
| Rovibronic Hamiltonian | `diatom.f90` | 9493-9800 |
| Potential functions | `functions.f90` | 61-900 |

## Debugging Guide

If energies are incorrect:

1. **Check `sc` calculation:**
   ```python
   sc = 12.0 * h**2 * reduced_mass / 16.857629206
   ```

2. **Verify kinetic matrix diagonal:**
   - SINC: ~39.478 (= 12π²/3)
   - 5-point FD: 30

3. **Check potential values are in cm⁻¹**

4. **Verify reduced mass is correct**

5. **Compare first few eigenvalues to Fortran Duo**

## References

1. Yurchenko, S.N., et al. (2016). "Duo: A general program for calculating spectra of diatomic molecules", Computer Physics Communications, 202, 262-275.

2. Colbert, D.T., Miller, W.H. (1992). "A novel discrete variable representation for quantum mechanical reactive scattering via the S-matrix Kohn method", J. Chem. Phys. 96, 1982.

3. Le Roy, R.J. (2017). "LEVEL: A computer program for solving the radial Schrödinger equation for bound and quasibound levels", JQSRT 186, 167-178.
