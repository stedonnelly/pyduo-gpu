# Rovibronic Hamiltonian and Coupling Terms

## Overview

The full rovibronic Hamiltonian includes:
1. Vibrational energy (from contracted basis)
2. Rotational energy
3. Spin-orbit coupling
4. Other coupling terms (spin-spin, spin-rotation, etc.)

## Basis Set: Lambda-Sigma Representation

The rovibronic basis states are labeled by:

$$|\Lambda, S, \Sigma; v, J, \Omega\rangle$$

Where:
- $\Lambda$ = projection of electronic orbital angular momentum on molecular axis
- $S$ = total electron spin
- $\Sigma$ = projection of spin on molecular axis
- $v$ = vibrational quantum number
- $J$ = total angular momentum
- $\Omega = \Lambda + \Sigma$ = projection of total angular momentum on molecular axis

## The Full Hamiltonian

$$\hat{H} = \hat{H}_{\text{vib}} + \hat{H}_{\text{rot}} + \hat{H}_{\text{SO}} + \hat{H}_{\text{SS}} + \hat{H}_{\text{SR}} + \ldots$$

### Rotational Hamiltonian

$$\hat{H}_{\text{rot}} = B(r) \left[ \hat{J}^2 - \hat{J}_z^2 + \hat{S}^2 - \hat{S}_z^2 + \hat{L}^2 - \hat{L}_z^2 - \hat{S}^+ \hat{L}^- - \hat{S}^- \hat{L}^+ + \hat{J}^+ \hat{S}^- + \hat{J}^- \hat{S}^+ \right]$$

Where $B(r) = \hbar^2 / (2\mu r^2)$.

For diagonal elements ($\Delta\Omega = 0$):
$$\langle J\Omega | \hat{H}_{\text{rot}} | J\Omega \rangle = B(r) \left[ J(J+1) - \Omega^2 + S(S+1) - \Sigma^2 + \langle L^2 \rangle - \Lambda^2 \right]$$

### Spin-Orbit Coupling

$$\hat{H}_{\text{SO}} = A(r) \hat{L}_z \hat{S}_z = A(r) \Lambda \Sigma$$

Selection rules: $\Delta\Lambda = 0$, $\Delta\Sigma = 0$

Diagonal contribution:
$$\langle \Lambda\Sigma | \hat{H}_{\text{SO}} | \Lambda\Sigma \rangle = A(r) \Lambda \Sigma$$

### L-uncoupling (L+ terms)

The ladder operators $\hat{L}_\pm$ couple states with $\Delta\Lambda = \pm 1$:

$$\langle \Lambda \pm 1, \Sigma | \hat{H}_{L\pm} | \Lambda, \Sigma \rangle = -B(r) \left[ J(J+1) - \Omega(\Omega \pm 1) \right]^{1/2} \langle L_\pm \rangle$$

### Spin-Rotation Coupling

$$\hat{H}_{\text{SR}} = \gamma(r) \hat{N} \cdot \hat{S}$$

Where $\hat{N} = \hat{J} - \hat{S}$.

## Matrix Elements in Contracted Basis

The matrix elements are computed in the contracted vibrational basis:

$$H_{i'v',iv} = \langle v' | f(r) | v \rangle$$

Where $|v\rangle$ are the contracted vibrational wavefunctions.

### Computing Vibrational Matrix Elements

```python
# Integration over r using grid representation
matelem = sum(psi_v' * f(r) * psi_v * h)
```

For diagonal vibrational terms:
$$\langle v | f(r) | v \rangle = \sum_{i=1}^{N} |\psi_v(r_i)|^2 f(r_i) h$$

For off-diagonal:
$$\langle v' | f(r) | v \rangle = \sum_{i=1}^{N} \psi_{v'}(r_i) f(r_i) \psi_v(r_i) h$$

## Code Implementation

### PyDuo: `hamiltonian.py`

```python
# pyduo/hamiltonian.py - HamiltonianBuilder.build_rovibronic_hamiltonian()

def build_rovibronic_hamiltonian(self, J: float):
    """
    Build the full rovibronic Hamiltonian for a given J value.

    Parameters
    ----------
    J : float
        Total angular momentum quantum number
    """
    xp = self.xp
    nbasis = len(self.contracted_basis)
    ngrid = self.grid.npoints
    h = self.grid.step

    # Initialize Hamiltonian matrix
    hmat = xp.zeros((nbasis, nbasis), dtype=xp.float64)

    # Loop over basis states
    for i, bra in enumerate(self.contracted_basis):
        for j, ket in enumerate(self.contracted_basis):
            if j > i:
                continue  # Use symmetry

            # Get vibrational wavefunctions
            psi_i = self.contracted_wavefunctions[:, bra.ivib]
            psi_j = self.contracted_wavefunctions[:, ket.ivib]

            # Vibrational energy (diagonal in vib index)
            if bra.ivib == ket.ivib and bra.istate == ket.istate:
                hmat[i, j] += self.contracted_energies[bra.ivib]

            # Rotational term
            if self._same_electronic_state(bra, ket):
                hmat[i, j] += self._rotational_matrix_element(
                    J, bra, ket, psi_i, psi_j)

            # Spin-orbit coupling
            hmat[i, j] += self._spin_orbit_matrix_element(
                bra, ket, psi_i, psi_j)

            # Symmetrize
            if i != j:
                hmat[j, i] = hmat[i, j]

    return hmat
```

**Location**: `pyduo/hamiltonian.py`, lines 350-450

### Fortran Duo: `diatom.f90`

```fortran
! diatom.f90 - Building rovibronic Hamiltonian (VIB contraction)
! Lines 9493-9800

case('VIB')

  ! Loop over basis states
  !$omp parallel do private(...) schedule(dynamic)
  do i = 1, Ntotal
    do j = 1, i

      ilevel = icontr(i)%ilevel
      jlevel = icontr(j)%ilevel
      ivib = icontr(i)%ivib
      jvib = icontr(j)%ivib

      ! Get quantum numbers
      istateI = icontrvib(ilevel)%istate
      istateJ = icontrvib(jlevel)%istate
      sigmai = icontr(i)%sigma
      sigmaj = icontr(j)%sigma
      omegai = icontr(i)%omega
      omegaj = icontr(j)%omega

      ! Diagonal vibrational energy
      if (ilevel == jlevel) then
        hmat(i, j) = hmat(i, j) + contrenergy(ilevel)
      endif

      ! Rotational term (diagonal in electronic state)
      if (istateI == istateJ) then
        f_rot = J*(J+1) - omegai*omegaj
        f_s = spini*(spini+1) - sigmai*sigmaj
        f_l = ilambda*(ilambda+1)  ! or from L2 field

        ! Integrate B(r) with vibrational functions
        b_term = 0
        do igrid = 1, ngrid
          b_term = b_term + contrfunc(igrid, ilevel) * &
                   b_rot/r(igrid)**2 * contrfunc(igrid, jlevel)
        enddo
        b_term = b_term * hstep * sc

        hmat(i, j) = hmat(i, j) + f_rot * b_term
      endif

      ! Spin-orbit coupling
      if (abs(sigmai - sigmaj) < 0.1 .and. abs(omegai - omegaj) < 0.1) then
        do iso = 1, Nspinorbits
          if (spinorbit(iso)%istate == istateI .and. &
              spinorbit(iso)%jstate == istateJ) then
            ! Compute <v'|A(r)|v> matrix element
            so_term = spinorbit(iso)%matelem(ilevel, jlevel)
            hmat(i, j) = hmat(i, j) + sigmai * ilambda * so_term
          endif
        enddo
      endif

      ! Symmetrize
      hmat(j, i) = hmat(i, j)

    enddo
  enddo
  !$omp end parallel do
```

**Location**: `diatom.f90`, lines 9493-9800

## Quantum Number Selection Rules

| Coupling | $\Delta\Lambda$ | $\Delta\Sigma$ | $\Delta\Omega$ | $\Delta S$ |
|----------|-----------------|----------------|----------------|------------|
| Spin-Orbit (diagonal) | 0 | 0 | 0 | 0 |
| L-uncoupling ($L_\pm$) | ±1 | 0 | ±1 | 0 |
| S-uncoupling ($S_\pm$) | 0 | ±1 | ±1 | 0 |
| Spin-Spin | 0 | 0, ±2 | 0, ±2 | 0 |
| Spin-Rotation | 0 | ±1 | ±1 | 0 |

## Angular Momentum Factors

### 3-j Symbols

Duo uses 3-j symbols for angular momentum coupling:

$$\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}$$

```fortran
! diatom.f90 - Pre-computing 3-j symbols

threej(ispin, nDeltaS, jspin, imulti, idelta) = &
    three_j(spin_i, real(nDeltaS, rk), spin_j, sigma_i, sigma_j-sigma_i, -sigma_j)
```

### Ladder Operator Matrix Elements

For $\hat{J}_\pm$:
$$\langle JM | \hat{J}_\pm | JM' \rangle = \hbar \sqrt{J(J+1) - M'(M' \pm 1)} \delta_{M, M'\pm 1}$$

For $\hat{S}_\pm$:
$$\langle S\Sigma | \hat{S}_\pm | S\Sigma' \rangle = \hbar \sqrt{S(S+1) - \Sigma'(\Sigma' \pm 1)} \delta_{\Sigma, \Sigma'\pm 1}$$

## Symmetry

For homonuclear diatomics or in general, the Hamiltonian can be block-diagonalized by parity:

$$|J\Omega^\pm\rangle = \frac{1}{\sqrt{2}} \left( |J\Omega\rangle \pm |J,-\Omega\rangle \right)$$

The parity is:
$$p = \pm (-1)^{J-S}$$

### PyDuo

```python
# pyduo/hamiltonian.py - Parity handling

# Determine parity from sigma sign
parity = 1 if basis.sigma >= 0 else -1
```

### Fortran Duo

```fortran
! diatom.f90 - Symmetry handling

! Build transformation matrix to symmetry-adapted basis
do i = 1, Ntotal
  omega = icontr(i)%omega

  if (nint(2.0*omega) == 0) then
    ! Omega = 0: only one parity block
    ipar = mod(nint(jval), 2) + 1
  else
    ! Omega /= 0: both parities
    do ipar = 1, 2
      ! Add to symmetry block
    enddo
  endif
enddo
```

## Output: Rovibronic Energies

After diagonalization, eigenvalues are in cm⁻¹ and eigenvectors give the composition in terms of the contracted basis:

```fortran
! diatom.f90 - After diagonalization

call lapack_syev(hmat, eigenvalues)

! Assign quantum numbers based on largest eigenvector component
do i = 1, nroots
  imax = maxloc(abs(eigenvectors(:, i)), dim=1)
  ! Use quantum numbers of dominant basis state
  v_assigned = icontrvib(icontr(imax)%ilevel)%v
  state_assigned = icontrvib(icontr(imax)%ilevel)%istate
enddo
```

## Summary: Rovibronic Hamiltonian Structure

```
         Basis: |state, v, Λ, Σ, Ω; J⟩

    ┌───────────────────────────────────────────┐
    │     E_vib    │    H_rot    │    H_SO     │
    │   (diag)     │  B(r)·f(J)  │  A(r)·ΛΣ   │
H = ├───────────────────────────────────────────┤
    │              │ L± couples  │             │
    │              │ ΔΛ=±1 states│             │
    ├───────────────────────────────────────────┤
    │              │             │ S± couples  │
    │              │             │ ΔΣ=±1       │
    └───────────────────────────────────────────┘
```
