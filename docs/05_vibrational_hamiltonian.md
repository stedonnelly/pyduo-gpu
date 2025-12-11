# Vibrational Hamiltonian Construction

## Overview

The vibrational problem (J=0) is solved first to generate a contracted basis for the full rovibronic calculation. This is the most computationally intensive part of the calculation.

## Mathematical Foundation

### The Vibrational Schrödinger Equation

For a single electronic state with no rotation (J=0):

$$\left[ -\frac{\hbar^2}{2\mu} \frac{d^2}{dr^2} + V(r) + \frac{\hbar^2}{2\mu r^2} \langle L^2 \rangle \right] \psi_v(r) = E_v \psi_v(r)$$

Where:
- $\mu$ = reduced mass
- $V(r)$ = potential energy curve
- $\langle L^2 \rangle$ = expectation value of electronic angular momentum squared
- $\psi_v(r)$ = vibrational wavefunction
- $E_v$ = vibrational eigenvalue

### Matrix Form

In the grid representation, this becomes a matrix eigenvalue problem:

$$\mathbf{H}_{\text{vib}} \vec{\psi}_v = E_v \vec{\psi}_v$$

Where:

$$H_{\text{vib},ij} = T_{ij} + \delta_{ij} \left[ V(r_i) \cdot sc + L^2(r_i) \cdot f_{\text{rot}}(r_i) \right]$$

### Scaling Factors

The key scaling factors are:

$$sc = 12 h^2 \cdot \frac{\mu}{\text{ASTON}}$$

$$f_{\text{rot}}(r) = \frac{\text{ASTON}}{\mu \cdot r^2} \cdot sc = \frac{12 h^2}{r^2}$$

Note: $f_{\text{rot}}$ simplifies because ASTON cancels out.

Actually in Duo:
$$f_{\text{rot}} = b_{\text{rot}} / r^2 \cdot sc$$

Where $b_{\text{rot}} = \text{ASTON}/\mu$.

## Code Implementation

### PyDuo: `hamiltonian.py`

```python
# pyduo/hamiltonian.py - HamiltonianBuilder._build_vibrational_hamiltonian()

def _build_vibrational_hamiltonian(self, istate: int):
    """Build the vibrational Hamiltonian for a single electronic state."""

    xp = self.xp  # numpy or cupy
    ngrid = self.grid.npoints
    r = self.grid.r

    # Build kinetic energy matrix (if not already done)
    if self.kinmat is None:
        self.kinmat = self.kinetic.build()

    # Get potential for this state
    poten = None
    state = self.states[istate]
    state_id = state.istate  # Use input file's state number

    for p in self.potentials:
        if p.istate == state_id:
            poten = p
            break

    # Evaluate potential on grid
    V = poten.evaluate(r, use_gpu=self.use_gpu)

    # Get L^2 term for this state (diagonal)
    f_l2 = xp.zeros(ngrid, dtype=xp.float64)
    for l2_field in self.l2:
        if l2_field.istate == istate and l2_field.jstate == istate:
            f_l2 = l2_field.evaluate(r, use_gpu=self.use_gpu)
            break

    # Calculate scaling factors
    h = self.grid.step
    sc = 12.0 * h**2 * self.reduced_mass / self.aston

    # Copy kinetic matrix and add potential
    vibmat = xp.copy(self.kinmat)

    # Rotational factor for L^2 term
    f_rot = self.b_rot / r**2 * sc

    # Add potential and L^2 terms on diagonal
    for igrid in range(ngrid):
        vibmat[igrid, igrid] += V[igrid] * sc + f_l2[igrid] * f_rot[igrid]

    return vibmat
```

**Location**: `pyduo/hamiltonian.py`, lines 160-235

### Fortran Duo: `diatom.f90`

```fortran
! diatom.f90 - Building vibrational Hamiltonian (VIB contraction case)
! Lines 8345-8450

case ("VIB")

  ! Allocate arrays
  allocate(vibmat(ngrid, ngrid), vibener(ngrid), ...)
  allocate(kinmat(ngrid, ngrid), ...)

  ! Build kinetic energy matrix
  call kinetic_energy_grid_points(ngrid, kinmat)

  do istate = 1, Nestates

    ! Copy kinetic matrix
    vibmat = kinmat

    ! Loop over grid points
    !$omp parallel do private(igrid, f_rot, epot, f_l2, erot) shared(vibmat)
    do igrid = 1, ngrid

      ! Centrifugal factor for L^2 term
      f_rot = b_rot / r(igrid)**2 * sc

      ! Potential energy (already scaled by sc during grid mapping)
      epot = poten(istate)%gridvalue(igrid) * sc

      ! L^2 term (diagonal)
      f_l2 = 0
      do iL2 = 1, Nl2
        if (L2(iL2)%istate == istate .and. L2(iL2)%jstate == istate) then
          f_l2 = f_rot * L2(iL2)%gridvalue(igrid)
          exit
        endif
      enddo

      erot = f_l2

      ! Add to diagonal
      vibmat(igrid, igrid) = vibmat(igrid, igrid) + epot + erot

    enddo
    !$omp end parallel do

    ! Diagonalize
    call lapack_syev(vibmat, vibener)

  enddo
```

**Location**: `diatom.f90`, lines 8345-8540

## Detailed Breakdown

### Step 1: Kinetic Matrix

The kinetic matrix `kinmat` contains the pre-scaled second derivative operator:

| Method | Diagonal | Off-diagonal |
|--------|----------|--------------|
| SINC | $12\pi^2/3 \approx 39.48$ | $24(-1)^{i-j}/(i-j)^2$ |
| 5-point FD | 30 | -16 (±1), +1 (±2) |

### Step 2: Scaling Factor

```python
h = self.grid.step                        # Grid spacing (Angstrom)
sc = 12.0 * h**2 * self.reduced_mass / self.aston
```

**Example for BeH:**
- h = 0.0152 Å (501 points from 0.4 to 8.0 Å)
- μ = 1.497 amu
- ASTON = 16.857629206

$$sc = 12 \times (0.0152)^2 \times \frac{1.497}{16.857629206} = 2.47 \times 10^{-4}$$

### Step 3: Rotational Factor

```python
b_rot = self.aston / self.reduced_mass    # B_rot in cm^-1 * Angstrom^2
f_rot = b_rot / r**2 * sc                 # Scaling for L^2 term
```

For the L² term:
$$f_{\text{rot}}(r_i) = \frac{\text{ASTON}}{\mu \cdot r_i^2} \times sc = \frac{12 h^2}{r_i^2}$$

### Step 4: Build Hamiltonian

```python
vibmat = xp.copy(self.kinmat)

for igrid in range(ngrid):
    vibmat[igrid, igrid] += V[igrid] * sc + f_l2[igrid] * f_rot[igrid]
```

The final matrix has:
- **Diagonal**: kinmat[i,i] + V(r_i)·sc + L²(r_i)·f_rot(r_i)
- **Off-diagonal**: kinmat[i,j] (from kinetic energy only)

### Step 5: Diagonalize

```python
eigenvalues, eigenvectors = self.solver.solve(vibmat, nroots=nmax)
```

The eigenvalues are in cm⁻¹ (due to the sc scaling).

## Physical Interpretation

The vibrational Hamiltonian matrix element is:

$$H_{ij}^{\text{vib}} = \langle \chi_i | \hat{T} + \hat{V} + \langle L^2 \rangle \hat{B}(r) | \chi_j \rangle$$

Where $\chi_i$ are the DVR basis functions (delta functions at grid points).

In DVR, potential energy is diagonal:
$$\langle \chi_i | V(r) | \chi_j \rangle = V(r_i) \delta_{ij}$$

And the kinetic energy connects neighboring points:
$$\langle \chi_i | \hat{T} | \chi_j \rangle = T_{ij}$$

## Output: Contracted Basis

After diagonalization, we store:

1. **Eigenvalues**: $E_v$ (vibrational energies in cm⁻¹)
2. **Eigenvectors**: $\psi_v(r_i)$ (vibrational wavefunctions on grid)

```python
# pyduo/hamiltonian.py - Storing contracted basis

for v in range(nmax):
    spin = state.spin
    for imulti in range(state.multiplicity):
        sigma = -spin + imulti
        omega = state.lambda_ + sigma

        basis_state = BasisState(
            istate=istate,
            v=v,
            ivib=total_roots,
            lambda_=state.lambda_,
            sigma=sigma,
            omega=omega,
            spin=spin,
            imulti=imulti + 1,
            name=state.name
        )
        self.contracted_basis.append(basis_state)

    all_energies.append(eigenvalues[v])
    all_wavefunctions.append(eigenvectors[:, v])
    total_roots += 1
```

**Location**: `pyduo/hamiltonian.py`, lines 255-280

## Fortran Equivalent

```fortran
! diatom.f90, lines 8560-8620

! Store contracted basis
do v = 1, nroots
  totalroots = totalroots + 1

  ! Store energy
  contrenergy(totalroots) = vibener(v)

  ! Store wavefunction
  contrfunc(:, totalroots) = vibmat(:, v)

  ! Store quantum numbers
  icontrvib(totalroots)%istate = istate
  icontrvib(totalroots)%v = v - 1  ! 0-based v
  icontrvib(totalroots)%ilambda = poten(istate)%lambda
  ! ...
enddo
```

## Summary Table

| Step | PyDuo | Fortran | Purpose |
|------|-------|---------|---------|
| 1 | `kinetic.build()` | `kinetic_energy_grid_points()` | Build T matrix |
| 2 | `poten.evaluate()` | `poten%gridvalue` | Get V(r) |
| 3 | `sc = 12*h²*μ/ASTON` | `sc = h12*scale` | Scaling factor |
| 4 | `vibmat[i,i] += V*sc` | `vibmat(i,i) = vibmat(i,i) + epot` | Add potential |
| 5 | `solver.solve()` | `lapack_syev()` | Diagonalize |
| 6 | `contracted_basis` | `icontrvib, contrfunc` | Store results |

## Debugging Tips

If eigenvalues are wrong:
1. Check `sc` is computed correctly
2. Verify `kinmat` diagonal ≈ 39.48 (for SINC)
3. Check potential values V(r) are in cm⁻¹
4. Ensure reduced mass is correct
