"""
Hamiltonian matrix construction for PyDuo

Builds the full rovibronic Hamiltonian matrix on GPU.
Includes all coupling terms: spin-orbit, spin-spin, L^2, etc.
"""

import numpy as np
import scipy.linalg
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = np
    HAS_CUDA = False

from .constants import PhysicalConstants
from .grid import Grid
from .fields import PotentialField, SpinOrbitField, CouplingField, FieldType, ElectronicState
from .kinetic import KineticEnergyMatrix


@dataclass
class BasisState:
    """Quantum numbers for a basis state."""
    istate: int          # Electronic state index
    v: int               # Vibrational quantum number
    ivib: int            # Index in contracted basis
    lambda_: int         # Lambda quantum number
    sigma: float         # Sigma (spin projection)
    omega: float         # Omega = Lambda + Sigma
    spin: float          # Total spin S
    imulti: int          # Multiplicity index (1 to 2S+1)
    name: str = ""       # State name
    parity_pm: int = 1   # Parity of electronic state (+1 or -1, from input)


@dataclass
class EigenState:
    """Quantum numbers and energy for an eigenstate after diagonalization."""
    energy: float        # Energy in cm^-1
    J: float             # Total angular momentum
    istate: int          # Dominant electronic state index
    v: int               # Vibrational quantum number (from assignment)
    lambda_: int         # Lambda quantum number (from largest contributor)
    sigma: float         # Sigma (from largest contributor)
    omega: float         # Omega (from largest contributor)
    spin: float          # Total spin S
    parity: str          # '+' or '-'
    name: str = ""       # State name
    coefficients: np.ndarray = None  # Eigenvector coefficients


class HamiltonianBuilder:
    """
    Build the rovibronic Hamiltonian matrix.

    The Hamiltonian includes:
    - Vibrational kinetic energy
    - Potential energy curves
    - Rotational kinetic energy
    - Spin-orbit coupling
    - Spin-spin coupling
    - L^2, Lx, Ly terms
    - Lambda-doubling
    - Non-adiabatic coupling
    - BOB corrections
    """

    def __init__(self, grid: Grid, reduced_mass: float,
                 states: List[ElectronicState],
                 potentials: List[PotentialField],
                 use_gpu: bool = True,
                 verbose: int = 0):
        """
        Initialize the Hamiltonian builder.

        Parameters
        ----------
        grid : Grid
            Radial coordinate grid
        reduced_mass : float
            Reduced mass in amu
        states : list
            List of electronic states
        potentials : list
            List of potential energy curves
        use_gpu : bool
            Whether to use GPU acceleration
        verbose : int
            Verbosity level (0=quiet, 1+=debug info)
        """
        self.grid = grid
        self.reduced_mass = reduced_mass
        self.states = states
        self.potentials = potentials
        self.use_gpu = use_gpu and HAS_CUDA
        self.verbose = verbose

        self.xp = cp if self.use_gpu else np

        # Coupling terms (to be set later)
        self.spinorbit: List[SpinOrbitField] = []
        self.l2: List[CouplingField] = []
        self.lxly: List[CouplingField] = []
        self.spinspin: List[CouplingField] = []
        self.spinspino: List[CouplingField] = []
        self.spinrot: List[CouplingField] = []
        self.bobrot: List[CouplingField] = []
        self.bobvib: List[CouplingField] = []
        self.diabatic: List[CouplingField] = []
        self.nac: List[CouplingField] = []
        self.lambdaopq: List[CouplingField] = []
        self.lambdap2q: List[CouplingField] = []
        self.lambdaq: List[CouplingField] = []

        # Pre-compute physical constants
        self.aston = PhysicalConstants.ASTON
        self.b_rot = self.aston / reduced_mass  # Rotational constant factor

        # Build kinetic energy matrix
        self.kinetic = KineticEnergyMatrix(grid, reduced_mass, use_gpu=use_gpu)
        self.kinmat = None  # Built lazily

        # Contracted basis
        self.contracted_basis: List[BasisState] = []
        self.contracted_energies: np.ndarray = np.array([])
        self.contracted_wavefunctions: np.ndarray = np.array([])

        # GPU cache attributes (will be populated during computation)
        self._r_gpu = None
        self._wf_gpu = None
        self._energies_gpu = None
        self._brot_matelem_gpu = None
        self._b_rot_r2_gpu = None
        self._so_matelem_cache = []
        self._lx_matelem_cache = []
        self._ss_matelem_cache = []

    def cleanup_gpu(self):
        """
        Clean up GPU memory by releasing cached arrays.
        Call this after completing calculations to free GPU memory.
        """
        if not self.use_gpu or not HAS_CUDA:
            return

        # Clear cached GPU arrays
        self._r_gpu = None
        self._wf_gpu = None
        self._energies_gpu = None
        self._brot_matelem_gpu = None
        self._b_rot_r2_gpu = None
        self._so_matelem_cache = []
        self._lx_matelem_cache = []
        self._ss_matelem_cache = []

        # Force garbage collection and clear GPU memory pool
        import gc
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def add_coupling(self, field: CouplingField):
        """Add a coupling field to the Hamiltonian."""
        if field.field_type == FieldType.SPINORBIT:
            self.spinorbit.append(field)
        elif field.field_type == FieldType.L2:
            self.l2.append(field)
        elif field.field_type == FieldType.LXLY:
            self.lxly.append(field)
        elif field.field_type == FieldType.SPINSPIN:
            self.spinspin.append(field)
        elif field.field_type == FieldType.SPINSPINO:
            self.spinspino.append(field)
        elif field.field_type == FieldType.SPINROT:
            self.spinrot.append(field)
        elif field.field_type == FieldType.BOBROT:
            self.bobrot.append(field)
        elif field.field_type == FieldType.BOBVIB:
            self.bobvib.append(field)
        elif field.field_type == FieldType.DIABATIC:
            self.diabatic.append(field)
        elif field.field_type == FieldType.NAC:
            self.nac.append(field)
        elif field.field_type == FieldType.LAMBDAOPQ:
            self.lambdaopq.append(field)
        elif field.field_type == FieldType.LAMBDAP2Q:
            self.lambdap2q.append(field)
        elif field.field_type == FieldType.LAMBDAQ:
            self.lambdaq.append(field)

    def build_vibrational_hamiltonian(self, istate: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build and diagonalize the vibrational Hamiltonian for a single electronic state.

        This is the J=0 problem: H_vib = T_vib + V(r)

        Parameters
        ----------
        istate : int
            Electronic state index

        Returns
        -------
        eigenvalues : ndarray
            Vibrational eigenvalues in cm^-1
        eigenvectors : ndarray
            Vibrational eigenvectors (grid points x number of states)
        """
        xp = self.xp
        ngrid = self.grid.npoints
        r = self.grid.r

        if self.use_gpu and isinstance(r, np.ndarray):
            r = cp.asarray(r)

        # Build kinetic energy matrix
        if self.kinmat is None:
            self.kinmat = self.kinetic.build()
            if self.use_gpu and isinstance(self.kinmat, np.ndarray):
                self.kinmat = cp.asarray(self.kinmat)

        # Get potential for this state
        # Look up by matching the state's istate value (which uses the input file's indexing)
        poten = None
        state = self.states[istate] if istate < len(self.states) else None
        state_id = state.istate if state else istate

        # Debug info
        if self.verbose >= 4:
            print(f"  Looking for potential: array index={istate}, state_id={state_id}")
            print(f"  Available potentials: {[(p.istate, p.name) for p in self.potentials]}")

        for p in self.potentials:
            if p.istate == state_id:
                poten = p
                break

        if poten is None:
            # Try matching by array index as fallback
            for idx, p in enumerate(self.potentials):
                if idx == istate:
                    poten = p
                    break

        if poten is None:
            raise ValueError(f"No potential found for state {istate} (looked for istate={state_id}). "
                           f"Available potentials have istates: {[p.istate for p in self.potentials]}")

        # Evaluate potential on grid
        if self.verbose >= 4:
            print(f"  Potential type: {poten.func_type}")
        V = poten.evaluate(r, use_gpu=self.use_gpu)
        if self.use_gpu and isinstance(V, np.ndarray):
            V = cp.asarray(V)

        # Debug: show potential range
        if self.verbose >= 4:
            V_np = V if isinstance(V, np.ndarray) else cp.asnumpy(V)
            print(f"  Potential range: [{V_np.min():.2f}, {V_np.max():.2f}] cm^-1")

        # Get L^2 term for this state (diagonal)
        f_l2 = xp.zeros(ngrid, dtype=xp.float64)
        for l2_field in self.l2:
            if l2_field.istate == istate and l2_field.jstate == istate:
                l2_values = l2_field.evaluate(r, use_gpu=self.use_gpu)
                if self.use_gpu and isinstance(l2_values, np.ndarray):
                    l2_values = cp.asarray(l2_values)
                f_l2 = l2_values
                break

        # Build the vibrational Hamiltonian
        # H = T + V + L^2 * B(r)
        # where B(r) = aston / (amass * r^2)
        h = self.grid.step if isinstance(self.grid.step, float) else float(self.grid.step[0])
        sc = 12.0 * h**2 * self.reduced_mass / self.aston

        if self.verbose >= 4:
            print(f"  Grid step h = {h:.6f}, sc = {sc:.6e}")
            print(f"  Kinetic matrix diagonal[0] = {float(self.kinmat[0, 0]):.2f}")

        vibmat = xp.copy(self.kinmat)

        # Add potential and L^2 terms on diagonal - VECTORIZED for GPU efficiency
        f_rot = self.b_rot / r**2 * sc
        diag_contribution = V * sc + f_l2 * f_rot
        diag_indices = xp.arange(ngrid)
        vibmat[diag_indices, diag_indices] += diag_contribution

        # Diagonalize
        if self.use_gpu:
            # Use cuSOLVER via CuPy
            eigenvalues, eigenvectors = cp.linalg.eigh(vibmat)
            eigenvalues = cp.asnumpy(eigenvalues)
            eigenvectors = cp.asnumpy(eigenvectors)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(vibmat)

        # Convert from scaled to cm^-1
        eigenvalues = eigenvalues / sc

        return eigenvalues, eigenvectors

    def solve_vibrational_problem(self, max_vib_states: Optional[Dict[int, int]] = None,
                                   max_energy: Optional[Dict[int, float]] = None):
        """
        Solve the vibrational problem for all electronic states.

        Parameters
        ----------
        max_vib_states : dict, optional
            Maximum number of vibrational states per electronic state
        max_energy : dict, optional
            Maximum vibrational energy per electronic state (cm^-1)
        """
        self.contracted_basis = []
        all_energies = []
        all_wavefunctions = []

        # First pass: compute all vibrational solutions and store nmax per state
        vib_solutions = {}  # istate -> (eigenvalues, eigenvectors, nmax)

        for istate, state in enumerate(self.states):
            eigenvalues, eigenvectors = self.build_vibrational_hamiltonian(istate)

            # Determine how many states to keep
            nmax = len(eigenvalues)
            if max_vib_states and istate in max_vib_states:
                nmax = min(nmax, max_vib_states[istate])
            if max_energy and istate in max_energy:
                emax = max_energy[istate]
                nmax = min(nmax, int(np.sum(eigenvalues - eigenvalues[0] <= emax)))

            nmax = max(1, nmax)  # At least one state
            vib_solutions[state.istate] = (eigenvalues, eigenvectors, nmax)

        # Build "quanta" list: (istate, tau, sigma) like Duo's ilevel ordering
        # This is Duo's sigma-lambda basis set ordering
        quanta_list = []
        for state in self.states:
            parity_pm = getattr(state, 'parity_pm', 0)
            if parity_pm == 0:
                parity_pm = 1

            taumax = 1 if state.lambda_ != 0 else 0
            for itau in range(taumax + 1):
                ilambda = ((-1) ** itau) * state.lambda_
                for imulti in range(state.multiplicity):
                    sigma = -state.spin + imulti
                    omega = ilambda + sigma
                    quanta_list.append({
                        'istate': state.istate,
                        'ilambda': ilambda,
                        'sigma': sigma,
                        'omega': omega,
                        'spin': state.spin,
                        'imulti': imulti + 1,
                        'name': state.name,
                        'parity_pm': parity_pm
                    })

        # Build contracted basis in Duo's order: (ilevel, ivib)
        # Outer loop: ilevel (sigma-lambda quantum numbers)
        # Inner loop: ivib (vibrational states)
        total_roots = 0
        ivib_map = {}  # (istate, v) -> ivib index for wavefunction lookup

        for quanta in quanta_list:
            istate = quanta['istate']
            eigenvalues, eigenvectors, nmax = vib_solutions[istate]

            for v in range(nmax):
                # Track vibrational wavefunction index
                key = (istate, v)
                if key not in ivib_map:
                    ivib_map[key] = total_roots
                    all_energies.append(eigenvalues[v])
                    all_wavefunctions.append(eigenvectors[:, v])
                    total_roots += 1

                vib_index = ivib_map[key]

                basis_state = BasisState(
                    istate=istate,
                    v=v,
                    ivib=vib_index,
                    lambda_=quanta['ilambda'],
                    sigma=quanta['sigma'],
                    omega=quanta['omega'],
                    spin=quanta['spin'],
                    imulti=quanta['imulti'],
                    name=quanta['name'],
                    parity_pm=quanta['parity_pm']
                )
                self.contracted_basis.append(basis_state)

        self.contracted_energies = np.array(all_energies)
        self.contracted_wavefunctions = np.array(all_wavefunctions).T  # (ngrid, nstates)

    def compute_matrix_elements(self, field: CouplingField) -> np.ndarray:
        """
        Compute matrix elements of a field in the contracted basis.

        <phi_i|f(r)|phi_j> = sum_k phi_i(r_k) * f(r_k) * phi_j(r_k) * w_k

        Parameters
        ----------
        field : CouplingField
            The field to compute matrix elements for

        Returns
        -------
        matelem : ndarray
            Matrix elements in contracted basis
        """
        xp = self.xp
        ngrid = self.grid.npoints
        r = self.grid.r
        nstates = len(self.contracted_basis)

        if self.use_gpu:
            r_gpu = cp.asarray(r) if isinstance(r, np.ndarray) else r
            wf = cp.asarray(self.contracted_wavefunctions)
        else:
            r_gpu = r
            wf = self.contracted_wavefunctions

        # Evaluate field on grid
        f_values = field.evaluate(r_gpu, use_gpu=self.use_gpu)
        if self.use_gpu and isinstance(f_values, np.ndarray):
            f_values = cp.asarray(f_values)

        # Compute matrix elements: <i|f|j> = sum_k psi_i(k) * f(k) * psi_j(k)
        # This is matrix multiplication: wf.T @ diag(f) @ wf
        f_wf = f_values.reshape(-1, 1) * wf  # Broadcasting

        if self.use_gpu:
            matelem = cp.dot(wf.T, f_wf)
            matelem = cp.asnumpy(matelem)
        else:
            matelem = np.dot(wf.T, f_wf)

        return matelem

    def compute_itau(self, state: BasisState, J: float) -> int:
        """
        Compute the parity quantum number itau for a basis state.

        This follows Duo's formula (diatom.f90, line 9681):
        itau = -ilambda + nint(spini - sigmai) + nint(Jval - omegai)

        For Sigma states with negative parity (pm=-1), itau is incremented by 1.

        Even itau -> positive parity (+, irrep=1)
        Odd itau -> negative parity (-, irrep=2)

        Parameters
        ----------
        state : BasisState
            The basis state
        J : float
            Total angular momentum quantum number

        Returns
        -------
        itau : int
            Parity quantum number
        """
        ilambda = state.lambda_
        sigma = state.sigma
        spin = state.spin
        omega = state.omega
        parity_pm = state.parity_pm

        itau = -ilambda + int(round(spin - sigma)) + int(round(J - omega))

        # For Sigma states with negative parity
        if ilambda == 0 and parity_pm == -1:
            itau += 1

        return itau

    def _precompute_gpu_data(self, J: float):
        """
        Pre-compute and cache all data needed for GPU Hamiltonian construction.

        This method is called once and caches:
        - r grid on GPU
        - Wavefunctions on GPU
        - All field evaluations on GPU
        - All vibrational matrix elements on GPU
        - Quantum number arrays on GPU

        This eliminates repeated CPU-GPU transfers in the main loop.
        """
        xp = self.xp

        # Filter basis states: only include states where |omega| <= J
        self.active_basis_indices = []
        for i, state in enumerate(self.contracted_basis):
            if abs(state.omega) <= J + 0.01:
                self.active_basis_indices.append(i)

        active_basis = [self.contracted_basis[i] for i in self.active_basis_indices]
        Ntotal = len(active_basis)

        # Cache grid on GPU
        r = self.grid.r
        if self.use_gpu and isinstance(r, np.ndarray):
            self._r_gpu = cp.asarray(r)
        else:
            self._r_gpu = r

        # Cache wavefunctions on GPU
        wf = self.contracted_wavefunctions
        if self.use_gpu and isinstance(wf, np.ndarray):
            self._wf_gpu = cp.asarray(wf)
        else:
            self._wf_gpu = wf

        # Cache contracted energies on GPU
        if self.use_gpu:
            self._energies_gpu = cp.asarray(self.contracted_energies)
        else:
            self._energies_gpu = self.contracted_energies

        # Pre-compute B(r)/r^2 values on GPU
        h = self.grid.step if isinstance(self.grid.step, float) else float(self.grid.step[0])
        self._sc = 12.0 * h**2 * self.reduced_mass / self.aston
        self._b_rot_r2_gpu = self.b_rot / self._r_gpu**2

        # Pre-compute B(r) matrix elements (done once, kept on GPU)
        b_wf = self._b_rot_r2_gpu.reshape(-1, 1) * self._wf_gpu
        self._brot_matelem_gpu = xp.dot(self._wf_gpu.T, b_wf) * self._sc

        # Extract quantum numbers into GPU arrays for vectorized operations
        self._ivib_arr = xp.array([s.ivib for s in active_basis], dtype=xp.int32)
        self._istate_arr = xp.array([s.istate for s in active_basis], dtype=xp.int32)
        self._sigma_arr = xp.array([s.sigma for s in active_basis], dtype=xp.float64)
        self._omega_arr = xp.array([s.omega for s in active_basis], dtype=xp.float64)
        self._spin_arr = xp.array([s.spin for s in active_basis], dtype=xp.float64)
        self._lambda_arr = xp.array([s.lambda_ for s in active_basis], dtype=xp.float64)

        # Pre-compute meshgrids for vectorized operations (all on GPU)
        self._i_grid, self._j_grid = xp.meshgrid(xp.arange(Ntotal), xp.arange(Ntotal), indexing='ij')

        # Pre-compute ivib meshgrids for matrix element indexing
        self._ivib_i_grid = self._ivib_arr[self._i_grid]
        self._ivib_j_grid = self._ivib_arr[self._j_grid]

        # Pre-compute selection rule arrays
        self._dlambda = self._lambda_arr[self._i_grid] - self._lambda_arr[self._j_grid]
        self._dsigma = self._sigma_arr[self._i_grid] - self._sigma_arr[self._j_grid]
        self._domega = self._omega_arr[self._i_grid] - self._omega_arr[self._j_grid]

        # Pre-compute spin-orbit field evaluations and matrix elements
        self._so_matelem_cache = []
        for so_field in self.spinorbit:
            if so_field.parameters is None or len(so_field.parameters) == 0:
                self._so_matelem_cache.append(None)
                continue
            so_values = so_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if self.use_gpu and isinstance(so_values, np.ndarray):
                so_values = cp.asarray(so_values)
            so_wf = so_values.reshape(-1, 1) * self._wf_gpu
            so_matelem = xp.dot(self._wf_gpu.T, so_wf)
            self._so_matelem_cache.append(so_matelem)

        # Pre-compute L+ field evaluations (scaled by B(r)/r^2)
        self._lx_matelem_cache = []
        for lxly_field in self.lxly:
            if lxly_field.parameters is None or len(lxly_field.parameters) == 0:
                self._lx_matelem_cache.append(None)
                continue
            lx_values = lxly_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if self.use_gpu and isinstance(lx_values, np.ndarray):
                lx_values = cp.asarray(lx_values)
            lx_scaled = lx_values * self._b_rot_r2_gpu * self._sc
            lx_wf = lx_scaled.reshape(-1, 1) * self._wf_gpu
            lx_matelem = xp.dot(self._wf_gpu.T, lx_wf)
            self._lx_matelem_cache.append(lx_matelem)

        # Pre-compute spin-spin field evaluations
        self._ss_matelem_cache = []
        for ss_field in self.spinspin:
            if ss_field.parameters is None or len(ss_field.parameters) == 0:
                self._ss_matelem_cache.append(None)
                continue
            ss_values = ss_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if self.use_gpu and isinstance(ss_values, np.ndarray):
                ss_values = cp.asarray(ss_values)
            ss_wf = ss_values.reshape(-1, 1) * self._wf_gpu
            ss_matelem = xp.dot(self._wf_gpu.T, ss_wf)
            self._ss_matelem_cache.append(ss_matelem)

        # Pre-compute BOB-rot field evaluations (scaled)
        self._bobrot_matelem_cache = []
        for bobrot_field in self.bobrot:
            if bobrot_field.parameters is None or len(bobrot_field.parameters) == 0:
                self._bobrot_matelem_cache.append(None)
                continue
            bob_values = bobrot_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if self.use_gpu and isinstance(bob_values, np.ndarray):
                bob_values = cp.asarray(bob_values)
            bob_scaled = bob_values * self._b_rot_r2_gpu * self._sc
            bob_wf = bob_scaled.reshape(-1, 1) * self._wf_gpu
            bob_matelem = xp.dot(self._wf_gpu.T, bob_wf)
            self._bobrot_matelem_cache.append(bob_matelem)

        # Pre-compute lambda-doubling field evaluations (OPQ)
        self._ldopq_matelem_cache = []
        for ld_field in self.lambdaopq:
            if ld_field.parameters is None or len(ld_field.parameters) == 0:
                self._ldopq_matelem_cache.append(None)
                continue
            ld_values = ld_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if self.use_gpu and isinstance(ld_values, np.ndarray):
                ld_values = cp.asarray(ld_values)
            ld_wf = ld_values.reshape(-1, 1) * self._wf_gpu
            ld_matelem = xp.dot(self._wf_gpu.T, ld_wf)
            self._ldopq_matelem_cache.append(ld_matelem)

        # Pre-compute lambda-doubling field evaluations (P2Q)
        self._ldp2q_matelem_cache = []
        for ld_field in self.lambdap2q:
            if ld_field.parameters is None or len(ld_field.parameters) == 0:
                self._ldp2q_matelem_cache.append(None)
                continue
            ld_values = ld_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if self.use_gpu and isinstance(ld_values, np.ndarray):
                ld_values = cp.asarray(ld_values)
            ld_wf = ld_values.reshape(-1, 1) * self._wf_gpu
            ld_matelem = xp.dot(self._wf_gpu.T, ld_wf)
            self._ldp2q_matelem_cache.append(ld_matelem)

        # Pre-compute lambda-doubling field evaluations (Q)
        self._ldq_matelem_cache = []
        for ld_field in self.lambdaq:
            if ld_field.parameters is None or len(ld_field.parameters) == 0:
                self._ldq_matelem_cache.append(None)
                continue
            ld_values = ld_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if self.use_gpu and isinstance(ld_values, np.ndarray):
                ld_values = cp.asarray(ld_values)
            ld_wf = ld_values.reshape(-1, 1) * self._wf_gpu
            ld_matelem = xp.dot(self._wf_gpu.T, ld_wf)
            self._ldq_matelem_cache.append(ld_matelem)

        # Pre-compute spin-rotation field evaluations
        self._sr_matelem_cache = []
        for sr_field in self.spinrot:
            if sr_field.parameters is None or len(sr_field.parameters) == 0:
                self._sr_matelem_cache.append(None)
                continue
            sr_values = sr_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if self.use_gpu and isinstance(sr_values, np.ndarray):
                sr_values = cp.asarray(sr_values)
            sr_wf = sr_values.reshape(-1, 1) * self._wf_gpu
            sr_matelem = xp.dot(self._wf_gpu.T, sr_wf)
            self._sr_matelem_cache.append(sr_matelem)

        # Pre-compute BOB-vib field evaluations
        self._bobvib_matelem_cache = []
        for bv_field in self.bobvib:
            if bv_field.parameters is None or len(bv_field.parameters) == 0:
                self._bobvib_matelem_cache.append(None)
                continue
            bv_values = bv_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if self.use_gpu and isinstance(bv_values, np.ndarray):
                bv_values = cp.asarray(bv_values)
            bv_wf = bv_values.reshape(-1, 1) * self._wf_gpu
            bv_matelem = xp.dot(self._wf_gpu.T, bv_wf)
            self._bobvib_matelem_cache.append(bv_matelem)

        self._gpu_data_cached = True
        self._cached_J = J

    def solve_all_J_batched(self, j_list: List[float]) -> List[EigenState]:
        """
        Solve the rovibronic problem for ALL J values with proper parity separation.

        This method:
        1. Builds the unsymmetrized Hamiltonian for each J using GPU batching
        2. Applies symmetry transformation to separate + and - parity blocks
        3. Diagonalizes each parity block separately
        4. Returns eigenstates with correct parity assignments

        Parameters
        ----------
        j_list : List[float]
            List of J values to solve

        Returns
        -------
        List[EigenState]
            All eigenstates for all J values, sorted by (J, energy)
        """
        if not self.use_gpu:
            raise ValueError("Batched solver requires GPU. Initialize with use_gpu=True")

        xp = cp
        nJ = len(j_list)
        j_arr = xp.array(j_list, dtype=xp.float64)

        # Find maximum basis size (depends on largest J)
        max_J = max(j_list)
        max_basis_indices = []
        for i, state in enumerate(self.contracted_basis):
            if abs(state.omega) <= max_J + 0.01:
                max_basis_indices.append(i)
        Nmax = len(max_basis_indices)

        if Nmax == 0:
            return []

        # Pre-compute common data on GPU
        self._precompute_batched_gpu_data(j_list, max_basis_indices)

        sc = self._sc
        active_basis = [self.contracted_basis[i] for i in max_basis_indices]

        # =====================================================
        # EXTRACT QUANTUM NUMBERS AS GPU TENSORS
        # =====================================================
        ivib_arr = xp.array([s.ivib for s in active_basis], dtype=xp.int32)
        istate_arr = xp.array([s.istate for s in active_basis], dtype=xp.int32)
        sigma_arr = xp.array([s.sigma for s in active_basis], dtype=xp.float64)
        omega_arr = xp.array([s.omega for s in active_basis], dtype=xp.float64)
        spin_arr = xp.array([s.spin for s in active_basis], dtype=xp.float64)
        lambda_arr = xp.array([s.lambda_ for s in active_basis], dtype=xp.float64)
        parity_arr = xp.array([getattr(s, 'parity_pm', 1) for s in active_basis], dtype=xp.int32)

        # Create 2D meshgrids for basis pairs (N x N)
        i_grid, j_grid = xp.meshgrid(xp.arange(Nmax), xp.arange(Nmax), indexing='ij')
        ivib_i = ivib_arr[i_grid]
        ivib_j = ivib_arr[j_grid]
        istate_i = istate_arr[i_grid]
        istate_j = istate_arr[j_grid]
        sigma_i = sigma_arr[i_grid]
        sigma_j = sigma_arr[j_grid]
        omega_i = omega_arr[i_grid]
        spin_i = spin_arr[i_grid]
        lambda_i = lambda_arr[i_grid]
        lambda_j = lambda_arr[j_grid]

        # Selection rule deltas (N x N)
        dlambda = lambda_i - lambda_j
        dsigma = sigma_i - sigma_j

        # =====================================================
        # BUILD 3D HAMILTONIAN TENSOR: (nJ, N, N) - FULLY VECTORIZED
        # NO PYTHON LOOPS OVER J VALUES!
        # =====================================================
        hmat_batch = xp.zeros((nJ, Nmax, Nmax), dtype=xp.float64)

        # Create 3D J broadcast array: (nJ, 1, 1) for broadcasting with (N, N)
        J_3d = j_arr.reshape(nJ, 1, 1)
        J_Jp1_3d = J_3d * (J_3d + 1)  # (nJ, 1, 1)

        diag_idx = xp.arange(Nmax)

        # =====================================================
        # DIAGONAL TERMS (vibrational energy only - truly diagonal)
        # =====================================================

        # Vibrational energy: (N,) -> broadcast to (nJ, N)
        vib_energies = self._energies_gpu[ivib_arr] * sc  # (N,)
        hmat_batch[:, diag_idx, diag_idx] = vib_energies[None, :]  # (nJ, N)

        # =====================================================
        # ROTATIONAL TERMS: DIAGONAL ONLY
        # Duo line 10949-10955: rotational term is only added to diagonal (i==j)
        # despite ilevel==jlevel allowing same (state, lambda, sigma) with different vib
        # =====================================================
        # Define selection masks (used here and below for S-uncoupling etc.)
        same_state = (istate_i == istate_j)  # same electronic state
        same_spin = (xp.abs(spin_i - spin_arr[j_grid]) < 0.5)  # same spin

        # Rotational energy: B[v,v] * [J(J+1) - Omega^2 - Lambda^2 + S(S+1) - Sigma^2]
        # Only on diagonal elements (same basis state)
        brot_diag = self._brot_matelem_gpu[ivib_arr, ivib_arr]  # (N,)

        # J-independent rotational factor
        j_indep_factor = (-omega_arr**2 - lambda_arr**2 +
                          spin_arr * (spin_arr + 1) - sigma_arr**2)  # (N,)
        j_indep_rot = brot_diag * j_indep_factor  # (N,)
        hmat_batch[:, diag_idx, diag_idx] += j_indep_rot[None, :]  # (nJ, N)

        # J-dependent rotational: B[v,v] * J(J+1) on diagonal only
        J_Jp1_1d = j_arr * (j_arr + 1)  # (nJ,)
        j_dep_rot = xp.outer(J_Jp1_1d, brot_diag)  # (nJ, N)
        hmat_batch[:, diag_idx, diag_idx] += j_dep_rot

        # =====================================================
        # S-UNCOUPLING (fully vectorized 3D tensor operation)
        # Selection: Delta Lambda = 0, |Delta Sigma| = 1, same electronic state, same spin
        # (Duo line 11543: istate==jstate, ilambda==jlambda, |sigmai-sigmaj|==1, spini==spinj)
        # =====================================================
        # same_state and same_spin already defined above
        s_mask = same_state & same_spin & (xp.abs(dlambda) < 0.5) & (xp.abs(xp.abs(dsigma) - 1.0) < 0.5)  # (N, N)

        if xp.any(s_mask):
            brot_full = self._brot_matelem_gpu[ivib_i, ivib_j]  # (N, N)
            # Duo line 11547: f_s = sigmaj - sigmai (note: j - i, not i - j)
            # dsigma = sigma_i - sigma_j, so f_s = -dsigma
            f_s = -dsigma  # (N, N)

            # S+/S- factor (J-independent): (N, N)
            # Duo line 11552: sqrt(spini*(spini+1) - sigmai*(sigmai + f_s))
            s_factor = xp.sqrt(xp.maximum(0, spin_i * (spin_i + 1) - sigma_i * (sigma_i + f_s)))

            # J-dependent factor: sqrt(J(J+1) - omega*(omega+f_s))
            # omega_term is (N, N), J_Jp1_3d is (nJ, 1, 1)
            # Result is (nJ, N, N) via broadcasting
            omega_term = omega_i * (omega_i + f_s)  # (N, N)
            j_factor_sq = J_Jp1_3d - omega_term[None, :, :]  # (nJ, N, N)
            j_factor = xp.sqrt(xp.maximum(0, j_factor_sq))  # (nJ, N, N)

            # Full coupling: -B * S_factor * J_factor, masked
            coupling_base = -brot_full * s_factor  # (N, N)
            coupling = coupling_base[None, :, :] * j_factor  # (nJ, N, N)
            coupling = xp.where(s_mask[None, :, :], coupling, 0.0)
            hmat_batch += coupling

        # =====================================================
        # SPIN-ORBIT COUPLING (J-independent, VECTORIZED broadcast)
        # Compute once as 2D matrix, broadcast to all J values
        # =====================================================
        from pyduo.constants import three_j

        # Accumulate all SO contributions in one 2D matrix
        so_contrib_total = xp.zeros((Nmax, Nmax), dtype=xp.float64)

        for so_idx, so_field in enumerate(self.spinorbit):
            so_matelem = self._so_matelem_cache[so_idx]
            if so_matelem is None:
                continue

            istate_so = so_field.istate
            jstate_so = so_field.jstate
            lambda_i_we = so_field.lambda_i
            lambda_j_we = so_field.lambda_j
            sigma_i_we = so_field.sigma_i
            sigma_j_we = so_field.sigma_j
            spin_i_we = so_field.spin_i
            spin_j_we = so_field.spin_j
            q_we = sigma_i_we - sigma_j_we

            three_j_ref = three_j(spin_i_we, 1.0, spin_j_we, -sigma_i_we, q_we, sigma_j_we)
            if abs(three_j_ref) < 1e-12:
                continue

            # Pre-compute SO vibrational matrix elements on GPU
            so_vib_matrix = so_matelem[ivib_i, ivib_j]  # (N, N) on GPU

            # Create selection masks for this SO field (all on GPU)
            omega_match = xp.abs(omega_i - omega_arr[j_grid]) < 0.5  # (N, N)
            spin_match = xp.abs(spin_i - spin_arr[j_grid]) < 1.01  # (N, N)

            state_match_fwd = (istate_i == istate_so) & (istate_j == jstate_so)
            state_match_rev = (istate_i == jstate_so) & (istate_j == istate_so)
            state_match = state_match_fwd | state_match_rev

            base_mask = omega_match & spin_match & state_match  # (N, N)

            # Use loop-based approach for Wigner-Eckart (complex phase logic)
            # but only iterate over pairs that pass the mask
            so_contrib = xp.zeros((Nmax, Nmax), dtype=xp.float64)

            # Get indices where mask is True (upper triangle only for symmetry)
            mask_cpu = cp.asnumpy(base_mask)
            for i in range(Nmax):
                for j in range(i, Nmax):
                    if not mask_cpu[i, j]:
                        continue

                    state_i = active_basis[i]
                    state_j = active_basis[j]
                    ivib = state_i.ivib
                    jvib = state_j.ivib
                    matelem_vib = float(so_matelem[ivib, jvib])

                    # Loop over permutations and sigma values
                    for ipermute in range(2):
                        if ipermute == 0:
                            istate_ = istate_so
                            jstate_ = jstate_so
                            ilambda_we = lambda_i_we
                            jlambda_we = lambda_j_we
                            sigmai_we = sigma_i_we
                            sigmaj_we = sigma_j_we
                            spini_ = spin_i_we
                            spinj_ = spin_j_we
                        else:
                            jstate_ = istate_so
                            istate_ = jstate_so
                            jlambda_we = lambda_i_we
                            ilambda_we = lambda_j_we
                            sigmaj_we = sigma_i_we
                            sigmai_we = sigma_j_we
                            spinj_ = spin_i_we
                            spini_ = spin_j_we

                        if int(round(spini_ - state_i.spin)) != 0 or int(round(spinj_ - state_j.spin)) != 0:
                            continue

                        if ipermute == 1:
                            if (istate_ == jstate_ and ilambda_we == jlambda_we and
                                int(round(sigmai_we - sigmaj_we)) == 0 and
                                int(round(spini_ - spinj_)) == 0):
                                continue

                        if state_i.istate != istate_ or state_j.istate != jstate_:
                            continue

                        for isigma2 in range(-int(round(2 * spini_)), int(round(2 * spini_)) + 1, 2):
                            sigmai_ = float(isigma2) * 0.5
                            sigmaj_ = sigmai_ - q_we
                            if abs(sigmaj_) > spinj_:
                                continue

                            three_j_ = three_j(spini_, 1.0, spinj_, -sigmai_, q_we, sigmaj_)
                            so_we = ((-1.0) ** int(round(sigmai_ - sigmai_we))) * three_j_ / three_j_ref * matelem_vib

                            for isigmav in range(2):
                                if isigmav == 1:
                                    if (int(round(abs(2.0 * sigmai_))) + int(round(abs(2.0 * sigmaj_))) +
                                        abs(ilambda_we) + abs(jlambda_we) == 0):
                                        continue

                                ilambda_ = ilambda_we * ((-1) ** isigmav)
                                jlambda_ = jlambda_we * ((-1) ** isigmav)
                                sigmai_t = sigmai_ * ((-1.0) ** isigmav)
                                sigmaj_t = sigmaj_ * ((-1.0) ** isigmav)
                                omegai_ = sigmai_t + float(ilambda_)
                                omegaj_ = sigmaj_t + float(jlambda_)

                                if (ilambda_ - jlambda_) != -int(round(sigmai_t - sigmaj_t)):
                                    continue
                                if abs(sigmai_t - sigmaj_t) > 1:
                                    continue
                                if abs(omegai_ - omegaj_) > 0.5:
                                    continue

                                if (int(round(sigmai_t - state_i.sigma)) != 0 or
                                    int(round(sigmaj_t - state_j.sigma)) != 0 or
                                    ilambda_ != state_i.lambda_ or jlambda_ != state_j.lambda_):
                                    continue

                                f_t = so_we * sc

                                if isigmav == 1:
                                    itau = (-ilambda_ - jlambda_ +
                                            int(round(spini_ - sigmai_t)) +
                                            int(round(spinj_ - sigmaj_t)))
                                    if ilambda_ == 0 and getattr(state_i, 'parity_pm', 1) == -1:
                                        itau += 1
                                    if jlambda_ == 0 and getattr(state_j, 'parity_pm', 1) == -1:
                                        itau += 1
                                    f_t = f_t * ((-1.0) ** itau)

                                so_contrib[i, j] += f_t
                                if i != j:
                                    so_contrib[j, i] += f_t

            so_contrib_total += so_contrib

        # VECTORIZED: Add SO to ALL J values via broadcasting (no Python loop!)
        hmat_batch += so_contrib_total[None, :, :]  # (1, N, N) broadcasts to (nJ, N, N)

        # =====================================================
        # L+ COUPLING (J-dependent for L.J, J-independent for L.S)
        # VECTORIZED: Use 3D broadcasting for J-dependent terms
        # =====================================================

        # Accumulate J-independent L.S contribution
        lx_contrib_ls_total = xp.zeros((Nmax, Nmax), dtype=xp.float64)

        # Accumulate J-dependent L.J: prefactor matrix and omega_fl matrix
        lx_j_prefactor_total = xp.zeros((Nmax, Nmax), dtype=xp.float64)
        lx_omega_fl_total = xp.zeros((Nmax, Nmax), dtype=xp.float64)  # omega_i * (omega_i + f_l)

        for lx_idx, lxly_field in enumerate(self.lxly):
            lx_matelem = self._lx_matelem_cache[lx_idx]
            if lx_matelem is None:
                continue

            istate_lx = lxly_field.istate
            jstate_lx = lxly_field.jstate
            lambda_i_lx = lxly_field.lambda_i  # Lambda from input file
            lambda_j_lx = lxly_field.lambda_j  # Lambda from input file

            # Get vibrational matrix elements
            lx_vib_matrix = lx_matelem[ivib_i, ivib_j]  # (N, N) on GPU

            # Loop over permutations and sigmav transformations (like spin-orbit)
            for ipermute in range(2):
                if ipermute == 0:
                    istate_ = istate_lx
                    jstate_ = jstate_lx
                    ilambda_lx = lambda_i_lx
                    jlambda_lx = lambda_j_lx
                else:
                    istate_ = jstate_lx
                    jstate_ = istate_lx
                    ilambda_lx = lambda_j_lx
                    jlambda_lx = lambda_i_lx

                # Skip redundant permutation for diagonal cases
                if ipermute == 1:
                    if istate_ == jstate_ and ilambda_lx == jlambda_lx:
                        continue

                # Check electronic states match
                state_match = (istate_i == istate_) & (istate_j == jstate_)
                if not xp.any(state_match):
                    continue

                # Loop over sigmav transformation
                for isigmav in range(2):
                    # sigmav=1 only if some quantum numbers are non-zero
                    if isigmav == 1 and abs(ilambda_lx) + abs(jlambda_lx) == 0:
                        continue

                    # Apply sigmav transformation to lambdas
                    ilambda_t = ilambda_lx * ((-1) ** isigmav)
                    jlambda_t = jlambda_lx * ((-1) ** isigmav)

                    # Match transformed lambdas with basis lambdas
                    lambda_match = (lambda_i == ilambda_t) & (lambda_j == jlambda_t)

                    # Combined mask for this permutation/sigmav
                    lx_base_mask = state_match & lambda_match

                    if not xp.any(lx_base_mask):
                        continue

                    # L.S selection: Delta Sigma = -Delta Lambda
                    f_s = float(ilambda_t - jlambda_t)  # +1 or -1
                    sigmaj_expected = sigma_i + f_s
                    ls_mask = lx_base_mask & (xp.abs(sigmaj_expected - sigma_j) < 0.5)

                    # L.J selection: Delta Sigma = 0
                    lj_mask = lx_base_mask & (xp.abs(sigma_i - sigma_j) < 0.5)

                    # Compute phase factor for sigmav transformation
                    # For L.S: itau includes lambda terms
                    # For L.J: itau = 0 (lambda terms removed per Fortran comment)
                    phase_ls = 1.0
                    phase_lj = 1.0
                    if isigmav == 1:
                        # L.S phase: (-1)^(ilambda-jlambda + S-sigma terms + parity terms)
                        # Following Fortran lines 11908-11915
                        itau_ls = 0  # ilambda-jlambda removed per Fortran
                        # Parity terms would need parity_pm from states (usually 1)
                        phase_ls = (-1.0) ** itau_ls

                        # L.J phase: itau = 0 per Fortran line 11980
                        itau_lj = 0
                        phase_lj = (-1.0) ** itau_lj

                    # --- L.S contribution ---
                    s_factor_sq = spin_i * (spin_i + 1) - sigma_i * (sigma_i + f_s)
                    s_factor = xp.sqrt(xp.maximum(0, s_factor_sq))
                    ls_contrib = lx_vib_matrix * s_factor * phase_ls
                    ls_contrib = xp.where(ls_mask, ls_contrib, 0.0)
                    # Add to upper triangle (will symmetrize later with full matrix)
                    lx_contrib_ls_total += ls_contrib
                    # Add transpose for lower triangle
                    lx_contrib_ls_total += ls_contrib.T
                    # Fix diagonal double-counting
                    diag_ls = xp.diag(ls_contrib)
                    lx_contrib_ls_total -= xp.diag(diag_ls)

                    # --- L.J contribution ---
                    f_l = float(jlambda_t - ilambda_t)  # Direction for J+/J-
                    omega_fl = omega_i * (omega_i + f_l)
                    lj_prefactor = -lx_vib_matrix * phase_lj
                    lj_prefactor = xp.where(lj_mask, lj_prefactor, 0.0)
                    omega_fl_masked = xp.where(lj_mask, omega_fl, 0.0)

                    # Add to totals (with symmetrization)
                    lx_j_prefactor_total += lj_prefactor
                    lx_j_prefactor_total += lj_prefactor.T
                    diag_lj = xp.diag(lj_prefactor)
                    lx_j_prefactor_total -= xp.diag(diag_lj)

                    lx_omega_fl_total += omega_fl_masked
                    lx_omega_fl_total += omega_fl_masked.T
                    diag_ofl = xp.diag(omega_fl_masked)
                    lx_omega_fl_total -= xp.diag(diag_ofl)

        # VECTORIZED: Add L.S to ALL J values via broadcasting (no Python loop!)
        hmat_batch += lx_contrib_ls_total[None, :, :]  # (1, N, N) broadcasts to (nJ, N, N)

        # VECTORIZED: Add L.J term using 3D broadcasting
        # J-factor: sqrt(J(J+1) - omega*(omega + f_l))
        # lx_j_prefactor_total is (N, N), omega_fl is (N, N)
        # J_Jp1_3d is (nJ, 1, 1)
        lj_arg = J_Jp1_3d - lx_omega_fl_total[None, :, :]  # (nJ, N, N)
        lj_factor = xp.sqrt(xp.maximum(0, lj_arg))  # (nJ, N, N)
        lj_contrib = lx_j_prefactor_total[None, :, :] * lj_factor  # (nJ, N, N)
        hmat_batch += lj_contrib

        # =====================================================
        # SPIN-SPIN COUPLING (J-independent) - FULLY VECTORIZED
        # =====================================================

        # Accumulate all spin-spin contributions
        ss_contrib_total = xp.zeros((Nmax, Nmax), dtype=xp.float64)

        for ss_idx, ss_field in enumerate(self.spinspin):
            ss_matelem = self._ss_matelem_cache[ss_idx]
            if ss_matelem is None:
                continue

            istate_ss = ss_field.istate
            jstate_ss = ss_field.jstate

            # Create vectorized selection masks (N x N)
            # Duo line 10969: spinspin(iss)%istate==istate AND spinspin(iss)%jstate==jstate AND istate==jstate
            # AND it's inside ilevel==jlevel block meaning: same electronic state, same lambda, same sigma, same spin
            state_mask_i = (istate_i == istate_ss)  # (N, N)
            state_mask_j = (istate_j == jstate_ss)  # (N, N)
            same_state = (istate_i == istate_j)  # Must be same electronic state (istate==jstate)
            state_mask = state_mask_i & state_mask_j & same_state  # (N, N)

            # Spin-spin is inside ilevel==jlevel block in Duo: same lambda, same sigma, same spin
            sigma_mask = xp.abs(sigma_i - sigma_j) < 0.5  # Delta Sigma = 0
            lambda_mask = xp.abs(lambda_i - lambda_j) < 0.5  # Delta Lambda = 0
            spin_mask = xp.abs(spin_i - spin_arr[j_grid]) < 0.5  # Same spin

            ss_mask = state_mask & sigma_mask & lambda_mask & spin_mask  # (N, N)

            # Compute spin-spin term (vectorized)
            ss_vib_matrix = ss_matelem[ivib_i, ivib_j]  # (N, N)
            ss_factor = 3.0 * sigma_i**2 - spin_i * (spin_i + 1)  # (N, N)
            ss_contrib = ss_vib_matrix * ss_factor * sc  # (N, N)
            ss_contrib = xp.where(ss_mask, ss_contrib, 0.0)

            # Symmetrize
            ss_contrib = ss_contrib + ss_contrib.T
            diag_ss = xp.diag(ss_contrib)
            ss_contrib = ss_contrib - xp.diag(diag_ss / 2)

            ss_contrib_total += ss_contrib

        # VECTORIZED: Add SS to ALL J values via broadcasting (no Python loop!)
        hmat_batch += ss_contrib_total[None, :, :]  # (1, N, N) broadcasts to (nJ, N, N)

        # =====================================================
        # SPIN-ROTATION COUPLING (J-independent) - FULLY VECTORIZED
        # Duo line 10987-10993: <S, Sigma | H_SR | S, Sigma> = f_sr * [Sigma^2 - S(S+1)] * sc
        # Selection: same electronic state, same lambda, same sigma, same spin (ilevel==jlevel)
        # =====================================================

        sr_contrib_total = xp.zeros((Nmax, Nmax), dtype=xp.float64)

        for sr_idx, sr_field in enumerate(self.spinrot):
            sr_matelem = self._sr_matelem_cache[sr_idx]
            if sr_matelem is None:
                continue

            istate_sr = sr_field.istate
            jstate_sr = sr_field.jstate

            # Duo line 10988: spinrot(isr)%istate==istate AND spinrot(isr)%jstate==jstate AND istate==jstate
            state_mask_i = (istate_i == istate_sr)
            state_mask_j = (istate_j == jstate_sr)
            same_state = (istate_i == istate_j)  # Must be same electronic state
            state_mask = state_mask_i & state_mask_j & same_state

            # Inside ilevel==jlevel block: same lambda, same sigma, same spin
            sigma_mask = xp.abs(sigma_i - sigma_j) < 0.5
            lambda_mask = xp.abs(lambda_i - lambda_j) < 0.5
            spin_mask = xp.abs(spin_i - spin_arr[j_grid]) < 0.5

            sr_mask = state_mask & sigma_mask & lambda_mask & spin_mask

            # Compute spin-rotation term (vectorized)
            sr_vib_matrix = sr_matelem[ivib_i, ivib_j]
            sr_factor = sigma_i**2 - spin_i * (spin_i + 1)  # Duo line 10990
            sr_contrib = sr_vib_matrix * sr_factor * sc
            sr_contrib = xp.where(sr_mask, sr_contrib, 0.0)

            # Symmetrize
            sr_contrib = sr_contrib + sr_contrib.T
            diag_sr = xp.diag(sr_contrib)
            sr_contrib = sr_contrib - xp.diag(diag_sr / 2)

            sr_contrib_total += sr_contrib

        # Add SR to ALL J values via broadcasting
        hmat_batch += sr_contrib_total[None, :, :]

        # =====================================================
        # BOB-VIB CORRECTION (J-independent) - FULLY VECTORIZED
        # Duo line 10997-11016: adds correction to vibrational energy
        # Selection: same electronic state (ilevel==jlevel block)
        # =====================================================

        bv_contrib_total = xp.zeros((Nmax, Nmax), dtype=xp.float64)

        for bv_idx, bobvib_field in enumerate(self.bobvib):
            bv_matelem = self._bobvib_matelem_cache[bv_idx]
            if bv_matelem is None:
                continue

            istate_bv = bobvib_field.istate
            jstate_bv = bobvib_field.jstate

            # Duo: bobvib(ibobvib)%istate==istate AND bobvib(ibobvib)%jstate==jstate AND istate==jstate
            state_mask_i = (istate_i == istate_bv)
            state_mask_j = (istate_j == jstate_bv)
            same_state = (istate_i == istate_j)
            state_mask = state_mask_i & state_mask_j & same_state

            # Inside ilevel==jlevel block
            sigma_mask = xp.abs(sigma_i - sigma_j) < 0.5
            lambda_mask = xp.abs(lambda_i - lambda_j) < 0.5
            spin_mask = xp.abs(spin_i - spin_arr[j_grid]) < 0.5

            bv_mask = state_mask & sigma_mask & lambda_mask & spin_mask

            # BOB-vib is directly added (no factor, already has energy units)
            bv_vib_matrix = bv_matelem[ivib_i, ivib_j]
            bv_contrib = xp.where(bv_mask, bv_vib_matrix, 0.0)

            # Symmetrize
            bv_contrib = bv_contrib + bv_contrib.T
            diag_bv = xp.diag(bv_contrib)
            bv_contrib = bv_contrib - xp.diag(diag_bv / 2)

            bv_contrib_total += bv_contrib

        # Add BV to ALL J values via broadcasting
        hmat_batch += bv_contrib_total[None, :, :]

        # =====================================================
        # BOB-ROT CORRECTION (J-dependent) - DIAGONAL ONLY
        # Duo line 10932-10939: adds correction to f_rot (rotational constant)
        # Like main rotational term, only applied to diagonal (i==j)
        # =====================================================

        # BOB-rot correction to rotational term
        # In Duo: f_rot = brot%matelem + bobrot%matelem, then used in J(J+1) - Omega^2 - ... terms
        # For diagonal: erot = f_rot * [J(J+1) - Omega^2 - Lambda^2 + S(S+1) - Sigma^2]

        for bob_idx, bobrot_field in enumerate(self.bobrot):
            bob_matelem = self._bobrot_matelem_cache[bob_idx]
            if bob_matelem is None:
                continue

            istate_bob = bobrot_field.istate
            jstate_bob = bobrot_field.jstate

            # BOB-rot only for same electronic state
            if istate_bob != jstate_bob:
                continue

            # State matching for diagonal elements only
            state_mask = (istate_arr == istate_bob)  # (N,) - diagonal elements in this state

            # BOB-rot adds to f_rot which is used in same way as brot
            # bob_matelem is already scaled by b_rot/r^2*sc in precomputation
            # Only diagonal vibrational elements: bob_matelem[v,v]
            bob_diag = bob_matelem[ivib_arr, ivib_arr]  # (N,)

            # J-independent part: -Omega^2 - Lambda^2 + S(S+1) - Sigma^2 (diagonal only)
            j_indep_factor = -omega_arr**2 - lambda_arr**2 + spin_arr * (spin_arr + 1) - sigma_arr**2  # (N,)
            bob_j_indep = xp.where(state_mask, bob_diag * j_indep_factor, 0.0)  # (N,)
            hmat_batch[:, diag_idx, diag_idx] += bob_j_indep[None, :]

            # J-dependent part: J(J+1) * bob_diag on diagonal only
            bob_j_dep = xp.outer(J_Jp1_1d, xp.where(state_mask, bob_diag, 0.0))  # (nJ, N)
            hmat_batch[:, diag_idx, diag_idx] += bob_j_dep

        # =====================================================
        # VECTORIZED Hermitianization (no Python loop!)
        # =====================================================
        hmat_batch = (hmat_batch + hmat_batch.transpose(0, 2, 1)) / 2.0

        # =====================================================
        # PARITY-SEPARATED EIGENVALUE DECOMPOSITION
        # =====================================================
        # For each J value:
        # 1. Extract valid basis states (|omega| <= J)
        # 2. Pair up states with opposite (lambda, sigma)
        # 3. Build symmetry transformation
        # 4. Transform to + and - parity blocks
        # 5. Diagonalize each block separately

        all_eigenstates = []

        for jidx in range(nJ):
            J = j_list[jidx]
            valid_mask = xp.abs(omega_arr) <= J + 0.01
            valid_indices_gpu = xp.where(valid_mask)[0]
            Nvalid = len(valid_indices_gpu)

            if Nvalid == 0:
                continue

            valid_indices = cp.asnumpy(valid_indices_gpu)

            # Get the valid submatrix for this J (move to CPU for symmetry transformation)
            hmat_j = cp.asnumpy(hmat_batch[jidx][xp.ix_(valid_indices_gpu, valid_indices_gpu)])

            # Build valid basis list for this J
            valid_basis = [active_basis[i] for i in valid_indices]

            # =====================================================
            # Pair up states and compute parity (itau)
            # Following Duo's diatom.f90 lines 9612-9698
            # =====================================================
            iswap = [0] * Nvalid
            levels = []

            for i in range(Nvalid):
                if iswap[i] != 0:
                    continue

                state_i = valid_basis[i]
                ilambda = state_i.lambda_
                sigmai = state_i.sigma
                spini = state_i.spin
                omegai = state_i.omega
                istate_i = state_i.istate
                ivib = state_i.ivib

                # Case 1: Lambda=0 and Sigma=0 (single state, no pair)
                if ilambda == 0 and abs(sigmai) < 0.01:
                    itau = int(round(spini + J))
                    if state_i.parity_pm == -1:
                        itau += 1
                    itau = abs(itau)
                    tau = (-1.0) ** itau

                    if itau % 2 == 1:
                        # Odd itau -> negative parity only
                        levels.append({
                            'indices': [i],
                            'tau': tau,
                            'nirr': [0, 1]
                        })
                    else:
                        # Even itau -> positive parity only
                        levels.append({
                            'indices': [i],
                            'tau': tau,
                            'nirr': [1, 0]
                        })
                    iswap[i] = i + 1
                else:
                    # Case 2: Look for the paired state with opposite lambda, sigma
                    found_pair = False
                    for j in range(Nvalid):
                        if i == j or iswap[j] != 0:
                            continue

                        state_j = valid_basis[j]
                        jlambda = state_j.lambda_
                        sigmaj = state_j.sigma
                        spinj = state_j.spin
                        jvib = state_j.ivib

                        # Check if j is the paired state
                        if (ilambda == -jlambda and
                            abs(sigmai + sigmaj) < 0.01 and
                            istate_i == state_j.istate and
                            ivib == jvib and
                            abs(spini - spinj) < 0.01):

                            # Compute itau
                            itau = -ilambda + int(round(spini - sigmai)) + int(round(J - omegai))
                            if ilambda == 0 and state_i.parity_pm == -1:
                                itau += 1
                            tau = (-1.0) ** itau

                            # Order the pair
                            if ilambda > jlambda or sigmaj < sigmai:
                                idx_pos = i
                                idx_neg = j
                            else:
                                idx_pos = j
                                idx_neg = i

                            levels.append({
                                'indices': [idx_pos, idx_neg],
                                'tau': tau,
                                'nirr': [1, 1]
                            })

                            iswap[i] = (j + 1) * ((-1) ** itau)
                            iswap[j] = (i + 1) * ((-1) ** itau)
                            found_pair = True
                            break

                    # If no pair found, treat as single state
                    if not found_pair:
                        itau = int(round(spini + J))
                        if state_i.parity_pm == -1:
                            itau += 1
                        itau = abs(itau)
                        tau = (-1.0) ** itau

                        if itau % 2 == 1:
                            levels.append({
                                'indices': [i],
                                'tau': tau,
                                'nirr': [0, 1]
                            })
                        else:
                            levels.append({
                                'indices': [i],
                                'tau': tau,
                                'nirr': [1, 0]
                            })
                        iswap[i] = i + 1

            # =====================================================
            # Build parity blocks and transform - EXACTLY matching Duo
            # Build ilevel2i[ilevel, irrep] and irec[isym] = ilevel mappings
            # =====================================================
            Nlevels = len(levels)

            # ilevel2i[ilevel][irrep] = index in valid_basis
            # Duo: ilevel2i(Nlevels, 2) where irrep=1 is positive, irrep=2 is negative
            ilevel2i = [[None, None] for _ in range(Nlevels)]

            # Nirr[ilevel][irrep] = 1 if level contributes to this irrep, 0 otherwise
            Nirr = [[0, 0] for _ in range(Nlevels)]

            # tau[ilevel] = parity phase factor
            tau_arr = [1.0] * Nlevels

            for ilevel, level in enumerate(levels):
                indices = level['indices']
                nirr = level['nirr']
                tau_arr[ilevel] = level['tau']
                Nirr[ilevel] = nirr

                if len(indices) == 1:
                    # Single state - only contributes to one parity
                    idx = indices[0]
                    if nirr[0] == 1:  # Positive parity only
                        ilevel2i[ilevel][0] = idx
                    else:  # Negative parity only
                        ilevel2i[ilevel][1] = idx
                else:
                    # Paired states - contribute to both parities
                    idx_pos, idx_neg = indices
                    ilevel2i[ilevel][0] = idx_pos  # Positive parity gets Lambda>0 component
                    ilevel2i[ilevel][1] = idx_neg  # Negative parity gets Lambda<0 component

            # Build block structures for each parity
            # irec[isym] = ilevel - maps symmetrized index back to level
            pos_levels = []
            neg_levels = []
            irec_pos = []  # irec[isym] = ilevel for positive parity
            irec_neg = []  # irec[isym] = ilevel for negative parity

            for ilevel in range(Nlevels):
                if Nirr[ilevel][0] == 1:  # Contributes to positive parity
                    isym_pos = len(pos_levels)
                    irec_pos.append(ilevel)
                    idx = ilevel2i[ilevel][0]
                    if len(levels[ilevel]['indices']) == 1:
                        pos_levels.append({'idx': idx, 'single': True, 'ilevel': ilevel})
                    else:
                        pos_levels.append({
                            'idx_pos': levels[ilevel]['indices'][0],
                            'idx_neg': levels[ilevel]['indices'][1],
                            'tau': tau_arr[ilevel],
                            'single': False,
                            'ilevel': ilevel
                        })

                if Nirr[ilevel][1] == 1:  # Contributes to negative parity
                    isym_neg = len(neg_levels)
                    irec_neg.append(ilevel)
                    idx = ilevel2i[ilevel][1]
                    if len(levels[ilevel]['indices']) == 1:
                        neg_levels.append({'idx': idx, 'single': True, 'ilevel': ilevel})
                    else:
                        neg_levels.append({
                            'idx_pos': levels[ilevel]['indices'][0],
                            'idx_neg': levels[ilevel]['indices'][1],
                            'tau': tau_arr[ilevel],
                            'single': False,
                            'ilevel': ilevel
                        })

            # =====================================================
            # Diagonalize each parity block
            # =====================================================
            for irrep, (block_levels, irec, parity_char) in enumerate([
                (pos_levels, irec_pos, '+'),
                (neg_levels, irec_neg, '-')
            ]):
                if len(block_levels) == 0:
                    continue

                Nblock = len(block_levels)
                hsym = np.zeros((Nblock, Nblock), dtype=np.float64)

                for isym_i, level_i in enumerate(block_levels):
                    for isym_j, level_j in enumerate(block_levels):
                        if level_i['single'] and level_j['single']:
                            i = level_i['idx']
                            j = level_j['idx']
                            hsym[isym_i, isym_j] = hmat_j[i, j]

                        elif level_i['single'] and not level_j['single']:
                            i = level_i['idx']
                            jp, jn = level_j['idx_pos'], level_j['idx_neg']
                            tauj = level_j['tau']
                            if irrep == 0:
                                hsym[isym_i, isym_j] = (hmat_j[i, jp] + tauj * hmat_j[i, jn]) / np.sqrt(2)
                            else:
                                hsym[isym_i, isym_j] = (hmat_j[i, jp] - tauj * hmat_j[i, jn]) / np.sqrt(2)

                        elif not level_i['single'] and level_j['single']:
                            ip, in_ = level_i['idx_pos'], level_i['idx_neg']
                            taui = level_i['tau']
                            j = level_j['idx']
                            if irrep == 0:
                                hsym[isym_i, isym_j] = (hmat_j[ip, j] + taui * hmat_j[in_, j]) / np.sqrt(2)
                            else:
                                hsym[isym_i, isym_j] = (hmat_j[ip, j] - taui * hmat_j[in_, j]) / np.sqrt(2)

                        else:
                            ip, in_ = level_i['idx_pos'], level_i['idx_neg']
                            jp, jn = level_j['idx_pos'], level_j['idx_neg']
                            taui = level_i['tau']
                            tauj = level_j['tau']

                            if irrep == 0:
                                hsym[isym_i, isym_j] = 0.5 * (
                                    hmat_j[ip, jp] + taui * hmat_j[in_, jp] +
                                    tauj * hmat_j[ip, jn] + taui * tauj * hmat_j[in_, jn]
                                )
                            else:
                                hsym[isym_i, isym_j] = 0.5 * (
                                    hmat_j[ip, jp] - taui * hmat_j[in_, jp] -
                                    tauj * hmat_j[ip, jn] + taui * tauj * hmat_j[in_, jn]
                                )

                # Symmetrize
                hsym = (hsym + hsym.T) / 2.0

                # Diagonalize using SYEV (same as Duo) for consistent eigenvector ordering
                hsym_np = cp.asnumpy(hsym) if hasattr(hsym, 'get') else hsym
                eigenvalues, eigenvectors = scipy.linalg.eigh(hsym_np, driver='ev')
                eigenvalues_np = eigenvalues / sc
                eigenvectors_np = eigenvectors

                # =====================================================
                # Assign quantum numbers - EXACTLY matching Duo
                # Duo diatom.f90 lines 10015-10092
                # =====================================================
                QNs = [None] * len(eigenvalues_np)
                small_ = 1e-12

                for iroot in range(len(eigenvalues_np)):
                    energy = eigenvalues_np[iroot]
                    coeffs = eigenvectors_np[:, iroot]
                    coeffs_sq = coeffs ** 2

                    # Duo: imaxcontr = maxloc(hsym(:,i)**2, mask=hsym(:,i)**2 >= small_)
                    # Find largest coefficient that's above threshold
                    valid_mask = coeffs_sq >= small_
                    if np.any(valid_mask):
                        masked_coeffs = np.where(valid_mask, coeffs_sq, -np.inf)
                        imaxcontr = int(np.argmax(masked_coeffs))
                    else:
                        imaxcontr = int(np.argmax(coeffs_sq))
                    maxcontr = coeffs_sq[imaxcontr]

                    # Duo: loop_check - check if this imaxcontr was already used
                    i0 = 0
                    while True:
                        found_duplicate = False
                        for ilevel_prev in range(iroot):
                            if QNs[ilevel_prev] is not None and imaxcontr == QNs[ilevel_prev]:
                                # Find next largest: mask = (>= small_) AND (< current maxcontr)
                                mask2 = valid_mask & (coeffs_sq < maxcontr)
                                if np.any(mask2):
                                    masked_coeffs2 = np.where(mask2, coeffs_sq, -np.inf)
                                    imaxcontr = int(np.argmax(masked_coeffs2))
                                    maxcontr = coeffs_sq[imaxcontr]
                                i0 += 1
                                if i0 > 10:
                                    # Give up and take overall max
                                    imaxcontr = int(np.argmax(np.where(valid_mask, coeffs_sq, -np.inf)))
                                    break
                                found_duplicate = True
                                break
                        if not found_duplicate or i0 > 10:
                            break

                    QNs[iroot] = imaxcontr

                    # Duo: mlevel = transform(irrep)%irec(imaxcontr)
                    #      mterm = ilevel2i(mlevel, irrep)
                    #      quantum numbers from icontr(mterm)
                    mlevel = irec[imaxcontr]
                    mterm = ilevel2i[mlevel][irrep]
                    basis_state = valid_basis[mterm]

                    eigenstate = EigenState(
                        energy=energy,
                        J=J,
                        istate=basis_state.istate,
                        v=basis_state.v,
                        lambda_=basis_state.lambda_,
                        sigma=basis_state.sigma,
                        omega=basis_state.omega,
                        spin=basis_state.spin,
                        parity=parity_char,
                        name=basis_state.name,
                        coefficients=coeffs
                    )
                    all_eigenstates.append(eigenstate)

        # Sort by (J, energy)
        all_eigenstates.sort(key=lambda x: (x.J, x.energy))

        return all_eigenstates

    def _precompute_batched_gpu_data(self, j_list: List[float], max_basis_indices: List[int]):
        """
        Pre-compute all data needed for batched multi-J GPU Hamiltonian construction.
        """
        xp = cp

        # Cache grid on GPU
        r = self.grid.r
        if isinstance(r, np.ndarray):
            self._r_gpu = xp.asarray(r)
        else:
            self._r_gpu = r

        # Cache wavefunctions on GPU
        wf = self.contracted_wavefunctions
        if isinstance(wf, np.ndarray):
            self._wf_gpu = xp.asarray(wf)
        else:
            self._wf_gpu = wf

        # Cache contracted energies on GPU
        self._energies_gpu = xp.asarray(self.contracted_energies)

        # Pre-compute scaling factor
        h = self.grid.step if isinstance(self.grid.step, float) else float(self.grid.step[0])
        self._sc = 12.0 * h**2 * self.reduced_mass / self.aston

        # B(r)/r^2 values on GPU
        self._b_rot_r2_gpu = self.b_rot / self._r_gpu**2

        # B(r) matrix elements
        b_wf = self._b_rot_r2_gpu.reshape(-1, 1) * self._wf_gpu
        self._brot_matelem_gpu = xp.dot(self._wf_gpu.T, b_wf) * self._sc

        # Pre-compute all field matrix elements (same as before)
        self._so_matelem_cache = []
        for so_field in self.spinorbit:
            if so_field.parameters is None or len(so_field.parameters) == 0:
                self._so_matelem_cache.append(None)
                continue
            so_values = so_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if isinstance(so_values, np.ndarray):
                so_values = xp.asarray(so_values)
            so_wf = so_values.reshape(-1, 1) * self._wf_gpu
            so_matelem = xp.dot(self._wf_gpu.T, so_wf)
            self._so_matelem_cache.append(so_matelem)

        self._lx_matelem_cache = []
        for lxly_field in self.lxly:
            if lxly_field.parameters is None or len(lxly_field.parameters) == 0:
                self._lx_matelem_cache.append(None)
                continue
            lx_values = lxly_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if isinstance(lx_values, np.ndarray):
                lx_values = xp.asarray(lx_values)
            lx_scaled = lx_values * self._b_rot_r2_gpu * self._sc
            lx_wf = lx_scaled.reshape(-1, 1) * self._wf_gpu
            lx_matelem = xp.dot(self._wf_gpu.T, lx_wf)
            self._lx_matelem_cache.append(lx_matelem)

        self._ss_matelem_cache = []
        for ss_field in self.spinspin:
            if ss_field.parameters is None or len(ss_field.parameters) == 0:
                self._ss_matelem_cache.append(None)
                continue
            ss_values = ss_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if isinstance(ss_values, np.ndarray):
                ss_values = xp.asarray(ss_values)
            ss_wf = ss_values.reshape(-1, 1) * self._wf_gpu
            ss_matelem = xp.dot(self._wf_gpu.T, ss_wf)
            self._ss_matelem_cache.append(ss_matelem)

        # Pre-compute spin-rotation field matrix elements
        self._sr_matelem_cache = []
        for sr_field in self.spinrot:
            if sr_field.parameters is None or len(sr_field.parameters) == 0:
                self._sr_matelem_cache.append(None)
                continue
            sr_values = sr_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if isinstance(sr_values, np.ndarray):
                sr_values = xp.asarray(sr_values)
            sr_wf = sr_values.reshape(-1, 1) * self._wf_gpu
            sr_matelem = xp.dot(self._wf_gpu.T, sr_wf)
            self._sr_matelem_cache.append(sr_matelem)

        # Pre-compute BOB-rot field matrix elements (scaled by b_rot/r^2*sc)
        self._bobrot_matelem_cache = []
        for bobrot_field in self.bobrot:
            if bobrot_field.parameters is None or len(bobrot_field.parameters) == 0:
                self._bobrot_matelem_cache.append(None)
                continue
            bob_values = bobrot_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if isinstance(bob_values, np.ndarray):
                bob_values = xp.asarray(bob_values)
            bob_scaled = bob_values * self._b_rot_r2_gpu * self._sc
            bob_wf = bob_scaled.reshape(-1, 1) * self._wf_gpu
            bob_matelem = xp.dot(self._wf_gpu.T, bob_wf)
            self._bobrot_matelem_cache.append(bob_matelem)

        # Pre-compute BOB-vib field matrix elements (no scaling)
        self._bobvib_matelem_cache = []
        for bv_field in self.bobvib:
            if bv_field.parameters is None or len(bv_field.parameters) == 0:
                self._bobvib_matelem_cache.append(None)
                continue
            bv_values = bv_field.evaluate(self._r_gpu, use_gpu=self.use_gpu)
            if isinstance(bv_values, np.ndarray):
                bv_values = xp.asarray(bv_values)
            bv_wf = bv_values.reshape(-1, 1) * self._wf_gpu
            bv_matelem = xp.dot(self._wf_gpu.T, bv_wf)
            self._bobvib_matelem_cache.append(bv_matelem)

    def build_rovibronic_hamiltonian(self, J: float) -> np.ndarray:
        """
        Build the full rovibronic Hamiltonian for a given J value.

        GPU-optimized version that:
        1. Pre-computes and caches all field evaluations on GPU
        2. Uses fully vectorized operations for diagonal terms
        3. Uses vectorized selection rule masks for off-diagonal terms
        4. Minimizes CPU-GPU transfers

        Parameters
        ----------
        J : float
            Total angular momentum quantum number

        Returns
        -------
        hmat : ndarray
            Rovibronic Hamiltonian matrix
        """
        xp = self.xp

        if len(self.contracted_basis) == 0:
            raise ValueError("No contracted basis states. Call solve_vibrational_problem first.")

        # Pre-compute and cache GPU data if needed
        if not hasattr(self, '_gpu_data_cached') or not self._gpu_data_cached:
            self._precompute_gpu_data(J)
        elif not hasattr(self, '_cached_J') or self._cached_J != J:
            # J changed, need to recompute active basis
            self._precompute_gpu_data(J)

        Ntotal = len(self.active_basis_indices)

        if Ntotal == 0:
            raise ValueError(f"No basis states satisfy |omega| <= J={J}")

        hmat = xp.zeros((Ntotal, Ntotal), dtype=xp.float64)

        # Use cached data from _precompute_gpu_data
        sc = self._sc
        ivib_arr = self._ivib_arr
        istate_arr = self._istate_arr
        sigma_arr = self._sigma_arr
        omega_arr = self._omega_arr
        spin_arr = self._spin_arr
        lambda_arr = self._lambda_arr
        brot_matelem_gpu = self._brot_matelem_gpu
        i_grid = self._i_grid
        j_grid = self._j_grid
        ivib_i_grid = self._ivib_i_grid
        ivib_j_grid = self._ivib_j_grid
        dlambda = self._dlambda
        dsigma = self._dsigma
        domega = self._domega

        # Keep active_basis reference for quantum number lookups in couplings
        active_basis = [self.contracted_basis[i] for i in self.active_basis_indices]

        # =====================================================
        # DIAGONAL TERMS (fully vectorized on GPU)
        # =====================================================

        # Vibrational energy
        vib_energies = self._energies_gpu[ivib_arr]
        diag_idx = xp.arange(Ntotal)
        hmat[diag_idx, diag_idx] = vib_energies * sc

        # Rotational term: B[v,v] * [J(J+1) - Omega^2 - Lambda^2 + S(S+1) - Sigma^2]
        brot_diag = brot_matelem_gpu[ivib_arr, ivib_arr]
        rot_factor = (J * (J + 1) - omega_arr**2 - lambda_arr**2 +
                      spin_arr * (spin_arr + 1) - sigma_arr**2)
        hmat[diag_idx, diag_idx] += brot_diag * rot_factor

        # =====================================================
        # S-UNCOUPLING (fully vectorized on GPU)
        # Selection rules: Delta Lambda = 0, |Delta Sigma| = 1, same electronic state, same spin
        # (Duo line 11543: istate==jstate, ilambda==jlambda, |sigmai-sigmaj|==1, spini==spinj)
        # =====================================================
        istate_i = istate_arr[i_grid]
        istate_j = istate_arr[j_grid]
        spin_i = spin_arr[i_grid]
        spin_j = spin_arr[j_grid]
        same_state = (istate_i == istate_j)  # Must be same electronic state
        same_spin = (xp.abs(spin_i - spin_j) < 0.5)  # Same spin
        s_uncoupling_mask = same_state & same_spin & (xp.abs(dlambda) < 0.5) & (xp.abs(xp.abs(dsigma) - 1.0) < 0.5)

        if xp.any(s_uncoupling_mask):
            # Get B(r) matrix elements using pre-computed ivib grids
            brot_full = brot_matelem_gpu[ivib_i_grid, ivib_j_grid]

            # Duo line 11547: f_s = sigmaj - sigmai (note: j - i, not i - j)
            # dsigma = sigma_i - sigma_j, so f_s = -dsigma
            f_s = -dsigma

            # S+/S- and J+/J- matrix elements (all on GPU)
            sigma_i = sigma_arr[i_grid]
            omega_i = omega_arr[i_grid]

            # Duo line 11552: sqrt(spini*(spini+1) - sigmai*(sigmai + f_s))
            s_factor = xp.sqrt(xp.maximum(0, spin_i * (spin_i + 1) - sigma_i * (sigma_i + f_s)))
            j_factor = xp.sqrt(xp.maximum(0, J * (J + 1) - omega_i * (omega_i + f_s)))

            # Apply S-uncoupling coupling (vectorized)
            coupling = xp.where(s_uncoupling_mask, -brot_full * s_factor * j_factor, 0.0)
            hmat += coupling

        # =====================================================
        # SPIN-ORBIT COUPLING
        # =====================================================
        # Selection rules (Lefebvre-Brion and Field, Eq. 3.4.6):
        # Delta J = 0, Delta Omega = 0, g<->u, e<->f, Sigma+<->Sigma-
        # Delta S = 0 or 1, Delta Lambda = -Delta Sigma = 0 or +/-1
        #
        # Uses Wigner-Eckart theorem to compute all matrix elements from reference value

        from pyduo.constants import three_j

        for so_idx, so_field in enumerate(self.spinorbit):
            # Get the quantum numbers from the spin-orbit field definition (Wigner-Eckart reference)
            istate_so = so_field.istate
            jstate_so = so_field.jstate
            lambda_i_we = so_field.lambda_i  # Reference lambda_i from input
            lambda_j_we = so_field.lambda_j  # Reference lambda_j from input
            sigma_i_we = so_field.sigma_i    # Reference sigma_i from input
            sigma_j_we = so_field.sigma_j    # Reference sigma_j from input
            spin_i_we = so_field.spin_i      # Reference spin_i from input
            spin_j_we = so_field.spin_j      # Reference spin_j from input

            # Use pre-computed matrix elements from cache
            so_matelem = self._so_matelem_cache[so_idx]
            if so_matelem is None:
                continue

            # Reference q from Wigner-Eckart: q = sigma_i - sigma_j
            q_we = sigma_i_we - sigma_j_we

            # Reference 3j symbol:
            # / S_i    1    S_j    \
            # \-sigma_i  q  sigma_j/
            three_j_ref = three_j(spin_i_we, 1.0, spin_j_we, -sigma_i_we, q_we, sigma_j_we)

            # Skip if reference 3j is zero (inconsistent quantum numbers)
            if abs(three_j_ref) < 1e-12:
                continue

            # Loop over basis state pairs
            for i in range(Ntotal):
                state_i = active_basis[i]
                for j in range(i, Ntotal):  # Only upper triangle
                    state_j = active_basis[j]

                    # Basis state quantum numbers
                    omegai = state_i.omega
                    omegaj = state_j.omega
                    sigmai = state_i.sigma
                    sigmaj = state_j.sigma
                    ilambda = state_i.lambda_
                    jlambda = state_j.lambda_
                    spini = state_i.spin
                    spinj = state_j.spin

                    # Basic selection rules
                    # Delta Omega = 0
                    if abs(omegai - omegaj) > 0.5:
                        continue
                    # Delta S = 0 or 1
                    if abs(spini - spinj) > 1.01:
                        continue

                    # Get vibrational indices
                    ivib = state_i.ivib
                    jvib = state_j.ivib

                    # Get the vibrational matrix element
                    matelem_vib = float(so_matelem[ivib, jvib])

                    # Loop over permutations (bra-ket swap for Hermitian operator)
                    for ipermute in range(2):
                        if ipermute == 0:
                            istate_ = istate_so
                            jstate_ = jstate_so
                            ilambda_we = lambda_i_we
                            jlambda_we = lambda_j_we
                            sigmai_we = sigma_i_we
                            sigmaj_we = sigma_j_we
                            spini_ = spin_i_we
                            spinj_ = spin_j_we
                        else:
                            # Permute: swap i <-> j
                            jstate_ = istate_so
                            istate_ = jstate_so
                            jlambda_we = lambda_i_we
                            ilambda_we = lambda_j_we
                            sigmaj_we = sigma_i_we
                            sigmai_we = sigma_j_we
                            spinj_ = spin_i_we
                            spini_ = spin_j_we

                        # Check if spins match field definition
                        if int(round(spini_ - spini)) != 0 or int(round(spinj_ - spinj)) != 0:
                            continue

                        # Skip double counting for diagonal fields
                        if ipermute == 1:
                            if (istate_ == jstate_ and ilambda_we == jlambda_we and
                                int(round(sigmai_we - sigmaj_we)) == 0 and
                                int(round(spini_ - spinj_)) == 0):
                                continue

                        # Check if electronic states match
                        if state_i.istate != istate_ or state_j.istate != jstate_:
                            continue

                        # Loop over sigma values via Wigner-Eckart theorem
                        for isigma2 in range(-int(round(2 * spini_)), int(round(2 * spini_)) + 1, 2):
                            sigmai_ = float(isigma2) * 0.5
                            sigmaj_ = sigmai_ - q_we

                            # Check sigmaj_ is valid
                            if abs(sigmaj_) > spinj_:
                                continue

                            # Compute 3j for current sigma values
                            three_j_ = three_j(spini_, 1.0, spinj_, -sigmai_, q_we, sigmaj_)

                            # SO matrix element from Wigner-Eckart
                            so_we = ((-1.0) ** int(round(sigmai_ - sigmai_we))) * three_j_ / three_j_ref * matelem_vib

                            # Loop over sigmav transformation (generates +/- lambda combinations)
                            for isigmav in range(2):
                                # sigmav=1 only needed if some quantum numbers are non-zero
                                if isigmav == 1:
                                    if (int(round(abs(2.0 * sigmai_))) + int(round(abs(2.0 * sigmaj_))) +
                                        abs(ilambda_we) + abs(jlambda_we) == 0):
                                        continue

                                # Apply sigmav transformation
                                ilambda_ = ilambda_we * ((-1) ** isigmav)
                                jlambda_ = jlambda_we * ((-1) ** isigmav)
                                sigmai_t = sigmai_ * ((-1.0) ** isigmav)
                                sigmaj_t = sigmaj_ * ((-1.0) ** isigmav)

                                omegai_ = sigmai_t + float(ilambda_)
                                omegaj_ = sigmaj_t + float(jlambda_)

                                # Check SO selection rules for transformed values
                                if (ilambda_ - jlambda_) != -int(round(sigmai_t - sigmaj_t)):
                                    continue
                                if abs(sigmai_t - sigmaj_t) > 1:
                                    continue
                                if abs(omegai_ - omegaj_) > 0.5:
                                    continue

                                # Match to basis state quantum numbers
                                if (int(round(sigmai_t - sigmai)) != 0 or
                                    int(round(sigmaj_t - sigmaj)) != 0 or
                                    ilambda_ != ilambda or jlambda_ != jlambda):
                                    continue

                                f_t = so_we * sc

                                # Phase factor for sigmav transformation
                                if isigmav == 1:
                                    itau = (-ilambda_ - jlambda_ +
                                            int(round(spini_ - sigmai_t)) +
                                            int(round(spinj_ - sigmaj_t)))
                                    # Add phase for Sigma- states
                                    if ilambda_ == 0 and getattr(state_i, 'parity_pm', 1) == -1:
                                        itau += 1
                                    if jlambda_ == 0 and getattr(state_j, 'parity_pm', 1) == -1:
                                        itau += 1
                                    f_t = f_t * ((-1.0) ** itau)

                                # Add to Hamiltonian
                                hmat[i, j] += f_t
                                if i != j:
                                    hmat[j, i] += f_t

        # =====================================================
        # L+ COUPLING (L-uncoupling, L*S and L*J terms)
        # =====================================================
        # Selection rules: Delta Lambda = +/-1, same electronic state or different
        # L*S: Delta Sigma = -Delta Lambda (spin-electronic coupling)
        # L*J: Delta Sigma = 0, Delta Omega = Delta Lambda (rotational L-uncoupling)

        for lx_idx, lxly_field in enumerate(self.lxly):
            istate_lx = lxly_field.istate
            jstate_lx = lxly_field.jstate
            lambda_i_lx = lxly_field.lambda_i
            lambda_j_lx = lxly_field.lambda_j

            # Use pre-computed matrix elements from cache (already scaled by b_rot/r^2*sc)
            lx_matelem = self._lx_matelem_cache[lx_idx]
            if lx_matelem is None:
                continue

            for i in range(Ntotal):
                state_i = active_basis[i]
                for j in range(i, Ntotal):
                    state_j = active_basis[j]

                    # Check electronic states match (both are 1-indexed)
                    if not ((state_i.istate == istate_lx and state_j.istate == jstate_lx) or
                            (state_i.istate == jstate_lx and state_j.istate == istate_lx)):
                        continue

                    ilambda = state_i.lambda_
                    jlambda = state_j.lambda_
                    sigmai = state_i.sigma
                    sigmaj = state_j.sigma
                    omegai = state_i.omega
                    omegaj = state_j.omega
                    spini = state_i.spin

                    # Selection rule: |Delta Lambda| = 1
                    if abs(ilambda - jlambda) != 1:
                        continue

                    ivib = state_i.ivib
                    jvib = state_j.ivib
                    f_grid = float(lx_matelem[ivib, jvib])

                    # L*S part (spin-electronic coupling)
                    # Selection rule: Delta Sigma = -Delta Lambda
                    f_s = float(ilambda - jlambda)  # +1 or -1
                    sigmaj_expected = sigmai + f_s

                    if abs(sigmaj_expected - sigmaj) < 0.5:
                        # S+/S- matrix element
                        # Matrix elements already include b_rot/r^2*sc (from line 621)
                        s_factor = np.sqrt(max(0, spini * (spini + 1) - sigmai * (sigmai + f_s)))
                        f_t = f_grid * s_factor
                        hmat[i, j] += f_t
                        if i != j:
                            hmat[j, i] += f_t

                    # L*J part (L-uncoupling)
                    # Selection rule: Delta Sigma = 0
                    if abs(sigmai - sigmaj) < 0.5:
                        f_l = float(jlambda - ilambda)  # Direction of L+/L-
                        # J+/J- matrix element (note the minus sign)
                        # Matrix elements already include b_rot/r^2*sc (from line 621)
                        j_factor = np.sqrt(max(0, J * (J + 1) - omegai * (omegai + f_l)))
                        f_t = -f_grid * j_factor
                        hmat[i, j] += f_t
                        if i != j:
                            hmat[j, i] += f_t

        # =====================================================
        # SPIN-SPIN COUPLING
        # =====================================================
        # Diagonal: <S, Sigma | H_SS | S, Sigma> = f_ss * [3*Sigma^2 - S(S+1)]
        # Note: f_ss = (2/3) * lambda_spectroscopic

        for ss_idx, ss_field in enumerate(self.spinspin):
            istate_ss = ss_field.istate
            jstate_ss = ss_field.jstate

            # Use pre-computed matrix elements from cache
            ss_matelem = self._ss_matelem_cache[ss_idx]
            if ss_matelem is None:
                continue

            for i in range(Ntotal):
                state_i = active_basis[i]
                # Spin-spin is diagonal in electronic state (1-indexed)
                if state_i.istate != istate_ss:
                    continue

                for j in range(i, Ntotal):
                    state_j = active_basis[j]
                    if state_j.istate != jstate_ss:
                        continue

                    sigmai = state_i.sigma
                    sigmaj = state_j.sigma
                    spini = state_i.spin

                    # Diagonal in Sigma
                    if abs(sigmai - sigmaj) > 0.5:
                        continue

                    ivib = state_i.ivib
                    jvib = state_j.ivib

                    # <S, Sigma | H_SS | S, Sigma> = f_ss * [3*Sigma^2 - S(S+1)]
                    ss_factor = 3.0 * sigmai**2 - spini * (spini + 1)
                    ss_term = float(ss_matelem[ivib, jvib]) * ss_factor * sc

                    hmat[i, j] += ss_term
                    if i != j:
                        hmat[j, i] += ss_term

        # =====================================================
        # SPIN-ROTATION COUPLING
        # =====================================================
        # <S, Sigma | H_SR | S, Sigma> = f_sr * [Sigma^2 - S(S+1)]

        for sr_idx, sr_field in enumerate(self.spinrot):
            istate_sr = sr_field.istate
            jstate_sr = sr_field.jstate

            # Use pre-computed matrix elements from cache
            sr_matelem = self._sr_matelem_cache[sr_idx]
            if sr_matelem is None:
                continue

            for i in range(Ntotal):
                state_i = active_basis[i]
                if state_i.istate != istate_sr:
                    continue

                for j in range(i, Ntotal):
                    state_j = active_basis[j]
                    if state_j.istate != jstate_sr:
                        continue

                    sigmai = state_i.sigma
                    sigmaj = state_j.sigma
                    spini = state_i.spin

                    if abs(sigmai - sigmaj) > 0.5:
                        continue

                    ivib = state_i.ivib
                    jvib = state_j.ivib

                    # <S, Sigma | H_SR | S, Sigma> = f_sr * [Sigma^2 - S(S+1)]
                    sr_factor = sigmai**2 - spini * (spini + 1)
                    sr_term = float(sr_matelem[ivib, jvib]) * sr_factor * sc

                    hmat[i, j] += sr_term
                    if i != j:
                        hmat[j, i] += sr_term

        # =====================================================
        # BOB-ROT CORRECTION (Born-Oppenheimer Breakdown rotational)
        # =====================================================
        # BOB-rot adds a correction to the rotational term B(r)
        # In Fortran: f_rot = f_rot + f_bobrot
        # where f_bobrot is the BOB-rot matrix element

        for bob_idx, bobrot_field in enumerate(self.bobrot):
            istate_bob = bobrot_field.istate
            jstate_bob = bobrot_field.jstate

            # BOB-rot is typically diagonal in electronic state
            if istate_bob != jstate_bob:
                continue

            # Use pre-computed matrix elements from cache (already scaled by b_rot/r^2*sc)
            bob_matelem = self._bobrot_matelem_cache[bob_idx]
            if bob_matelem is None:
                continue

            for i in range(Ntotal):
                state_i = active_basis[i]
                if state_i.istate != istate_bob:
                    continue

                for j in range(i, Ntotal):
                    state_j = active_basis[j]
                    if state_j.istate != jstate_bob:
                        continue

                    # BOB-rot adds to the rotational factor
                    # Same selection rules as the rotational term (diagonal in quantum numbers)
                    sigmai = state_i.sigma
                    sigmaj = state_j.sigma
                    omegai = state_i.omega
                    omegaj = state_j.omega
                    ilambda = state_i.lambda_
                    jlambda = state_j.lambda_
                    spini = state_i.spin

                    # Must be diagonal in Lambda, Sigma
                    if abs(sigmai - sigmaj) > 0.5 or ilambda != jlambda:
                        continue

                    ivib = state_i.ivib
                    jvib = state_j.ivib

                    # BOB-rot correction to rotational term
                    # Same form as B(r)*[J(J+1) - Omega^2 - ...] but with bob function
                    # Matrix elements already include b_rot/r^2*sc factor (from line 838)
                    rot_factor = (J * (J + 1) - omegai**2 - ilambda**2 +
                                  spini * (spini + 1) - sigmai**2)
                    bob_term = float(bob_matelem[ivib, jvib]) * rot_factor

                    hmat[i, j] += bob_term
                    if i != j:
                        hmat[j, i] += bob_term

        # =====================================================
        # LAMBDA-DOUBLING: o+p+q term (LAMBDAOPQ)
        # =====================================================
        # Selection rules: |Lambda| = 1 (Pi states), Lambda -> -Lambda
        # Delta Sigma = +/-2, Delta Omega = 0
        # <Sigma,Omega,Lambda|Lambda-O|Sigma+/-2,Omega,-Lambda>

        for ld_idx, ld_field in enumerate(self.lambdaopq):
            istate_ld = ld_field.istate
            jstate_ld = ld_field.jstate

            # Lambda-doubling is diagonal in electronic state
            if istate_ld != jstate_ld:
                continue

            # Use pre-computed matrix elements from cache
            ld_matelem = self._ldopq_matelem_cache[ld_idx]
            if ld_matelem is None:
                continue

            for i in range(Ntotal):
                state_i = active_basis[i]
                if state_i.istate != istate_ld:
                    continue

                for j in range(i, Ntotal):
                    state_j = active_basis[j]
                    if state_j.istate != jstate_ld:
                        continue

                    sigmai = state_i.sigma
                    sigmaj = state_j.sigma
                    omegai = state_i.omega
                    omegaj = state_j.omega
                    ilambda = state_i.lambda_
                    jlambda = state_j.lambda_
                    spini = state_i.spin
                    spinj = state_j.spin

                    # Selection rules:
                    # |Lambda| = 1 (Pi state)
                    # Lambda_i = -Lambda_j (sign flip)
                    # |Delta Sigma| = 2
                    # Delta Omega = 0
                    # Delta Spin = 0
                    if (abs(ilambda) != 1 or ilambda != -jlambda or
                        abs(round(sigmaj - sigmai)) != 2 or
                        abs(omegai - omegaj) > 0.5 or
                        abs(spini - spinj) > 0.01):
                        continue

                    # Also check: Delta Lambda = Delta Sigma (sign convention)
                    if (ilambda - jlambda) != round(sigmaj - sigmai):
                        continue

                    ivib = state_i.ivib
                    jvib = state_j.ivib

                    # f_s2 = sigmai - sigmaj (total sigma change)
                    # f_s1 = sign of sigma change
                    f_s2 = sigmai - sigmaj
                    f_s1 = 1.0 if f_s2 > 0 else -1.0

                    # S+S+ or S-S- matrix element (two successive applications)
                    f_t = (np.sqrt(spini * (spini + 1) - sigmaj * (sigmaj + f_s1)) *
                           np.sqrt(spini * (spini + 1) - (sigmaj + f_s1) * (sigmaj + f_s2)))

                    f_lo = float(ld_matelem[ivib, jvib]) * f_t * sc

                    # Factor of 0.5 from the definition
                    hmat[i, j] += f_lo * 0.5
                    if i != j:
                        hmat[j, i] += f_lo * 0.5

        # =====================================================
        # LAMBDA-DOUBLING: -(p+2q) term (LAMBDAP2Q)
        # =====================================================
        # Selection rules: |Lambda| = 1 (Pi states), Lambda -> -Lambda
        # |Delta Sigma| = 1, |Delta Omega| = 1
        # Delta Sigma = Delta Omega (same sign)
        # <Sigma,Omega,Lambda|Lambda-p|Sigma+/-1,Omega-/+1,-Lambda>

        for ld_idx, ld_field in enumerate(self.lambdap2q):
            istate_ld = ld_field.istate
            jstate_ld = ld_field.jstate

            if istate_ld != jstate_ld:
                continue

            # Use pre-computed matrix elements from cache
            ld_matelem = self._ldp2q_matelem_cache[ld_idx]
            if ld_matelem is None:
                continue

            for i in range(Ntotal):
                state_i = active_basis[i]
                if state_i.istate != istate_ld:
                    continue

                for j in range(i, Ntotal):
                    state_j = active_basis[j]
                    if state_j.istate != jstate_ld:
                        continue

                    sigmai = state_i.sigma
                    sigmaj = state_j.sigma
                    omegai = state_i.omega
                    omegaj = state_j.omega
                    ilambda = state_i.lambda_
                    jlambda = state_j.lambda_
                    spini = state_i.spin
                    spinj = state_j.spin

                    # Selection rules:
                    # |Lambda| = 1 (Pi state)
                    # Lambda_i = -Lambda_j
                    # |Delta Sigma| = 1
                    # |Delta Omega| = 1
                    # Delta Sigma = Delta Omega
                    # Delta Spin = 0
                    if (abs(ilambda) != 1 or ilambda != -jlambda or
                        abs(round(sigmaj - sigmai)) != 1 or
                        abs(round(omegai - omegaj)) != 1 or
                        round(sigmaj - sigmai) != round(omegai - omegaj) or
                        abs(spini - spinj) > 0.01):
                        continue

                    ivib = state_i.ivib
                    jvib = state_j.ivib

                    f_s = sigmai - sigmaj

                    # S+/- and J+/- matrix elements
                    f_t = (np.sqrt(spini * (spini + 1) - sigmaj * (sigmaj + f_s)) *
                           np.sqrt(J * (J + 1) - omegaj * (omegaj - f_s)))

                    f_lo = float(ld_matelem[ivib, jvib]) * f_t * sc

                    # Factor of -0.5 from the definition (note the minus sign)
                    hmat[i, j] -= f_lo * 0.5
                    if i != j:
                        hmat[j, i] -= f_lo * 0.5

        # =====================================================
        # LAMBDA-DOUBLING: q term (LAMBDAQ)
        # =====================================================
        # Selection rules: |Lambda| = 1 (Pi states), Lambda -> -Lambda
        # Delta Sigma = 0, |Delta Omega| = 2
        # <Sigma,Omega,Lambda|Lambda-q|Sigma,Omega+/-2,-Lambda>

        for ld_idx, ld_field in enumerate(self.lambdaq):
            istate_ld = ld_field.istate
            jstate_ld = ld_field.jstate

            if istate_ld != jstate_ld:
                continue

            # Use pre-computed matrix elements from cache
            ld_matelem = self._ldq_matelem_cache[ld_idx]
            if ld_matelem is None:
                continue

            for i in range(Ntotal):
                state_i = active_basis[i]
                if state_i.istate != istate_ld:
                    continue

                for j in range(i, Ntotal):
                    state_j = active_basis[j]
                    if state_j.istate != jstate_ld:
                        continue

                    sigmai = state_i.sigma
                    sigmaj = state_j.sigma
                    omegai = state_i.omega
                    omegaj = state_j.omega
                    ilambda = state_i.lambda_
                    jlambda = state_j.lambda_
                    spini = state_i.spin
                    spinj = state_j.spin

                    # Selection rules:
                    # |Lambda| = 1 (Pi state)
                    # Lambda_i = -Lambda_j
                    # Delta Sigma = 0
                    # |Delta Omega| = 2
                    # Delta Lambda = Delta Omega
                    # Delta Spin = 0
                    if (abs(ilambda) != 1 or ilambda != -jlambda or
                        abs(round(sigmai - sigmaj)) != 0 or
                        abs(round(omegai - omegaj)) != 2 or
                        (ilambda - jlambda) != round(omegai - omegaj) or
                        abs(spini - spinj) > 0.01):
                        continue

                    ivib = state_i.ivib
                    jvib = state_j.ivib

                    # f_o2 = total omega change, f_o1 = direction
                    f_o2 = omegaj - omegai
                    f_o1 = 1.0 if f_o2 > 0 else -1.0

                    # J+J+ or J-J- matrix element (two successive applications)
                    f_t = (np.sqrt(J * (J + 1) - omegaj * (omegaj - f_o1)) *
                           np.sqrt(J * (J + 1) - (omegaj - f_o1) * (omegaj - f_o2)))

                    f_lo = float(ld_matelem[ivib, jvib]) * f_t * sc

                    # Factor of 0.5 from the definition
                    hmat[i, j] += f_lo * 0.5
                    if i != j:
                        hmat[j, i] += f_lo * 0.5

        # =====================================================
        # BOB-VIB CORRECTION (Born-Oppenheimer Breakdown vibrational)
        # =====================================================
        # BOB-vib adds a correction to the vibrational energy
        # This is diagonal in electronic state and other quantum numbers

        for bv_idx, bobvib_field in enumerate(self.bobvib):
            istate_bv = bobvib_field.istate
            jstate_bv = bobvib_field.jstate

            # BOB-vib is diagonal in electronic state
            if istate_bv != jstate_bv:
                continue

            # Use pre-computed matrix elements from cache
            bv_matelem = self._bobvib_matelem_cache[bv_idx]
            if bv_matelem is None:
                continue

            for i in range(Ntotal):
                state_i = active_basis[i]
                if state_i.istate != istate_bv:
                    continue

                for j in range(i, Ntotal):
                    state_j = active_basis[j]
                    if state_j.istate != jstate_bv:
                        continue

                    # BOB-vib is diagonal in all quantum numbers for same electronic state
                    sigmai = state_i.sigma
                    sigmaj = state_j.sigma
                    omegai = state_i.omega
                    omegaj = state_j.omega
                    ilambda = state_i.lambda_
                    jlambda = state_j.lambda_

                    # Must be diagonal in Lambda, Sigma, Omega
                    if (abs(sigmai - sigmaj) > 0.5 or
                        ilambda != jlambda or
                        abs(omegai - omegaj) > 0.5):
                        continue

                    ivib = state_i.ivib
                    jvib = state_j.ivib

                    # BOB-vib is simply added to the Hamiltonian
                    bv_term = float(bv_matelem[ivib, jvib])

                    hmat[i, j] += bv_term
                    if i != j:
                        hmat[j, i] += bv_term

        # Make Hermitian (average upper and lower triangular)
        hmat = (hmat + hmat.T) / 2.0

        if self.use_gpu:
            hmat = cp.asnumpy(hmat)

        return hmat

    def _compute_coupling(self, i: int, j: int, J: float, f_rot: float, sc: float) -> float:
        """
        Compute coupling matrix elements between basis states i and j.

        This implements all the selection rules and Wigner-Eckart theorem
        applications for various coupling terms.
        """
        state_i = self.contracted_basis[i]
        state_j = self.contracted_basis[j]

        coupling = 0.0

        # Get quantum numbers
        istate = state_i.istate
        jstate = state_j.istate
        sigmai = state_i.sigma
        sigmaj = state_j.sigma
        omegai = state_i.omega
        omegaj = state_j.omega
        spini = state_i.spin
        spinj = state_j.spin
        ilambda = state_i.lambda_
        jlambda = state_j.lambda_

        # Selection rules for J+/J- coupling (S-uncoupling)
        # Delta Lambda = 0, Delta Sigma = +/-1, Delta Omega = +/-1
        if abs(ilambda - jlambda) == 0 and abs(sigmai - sigmaj) == 1:
            f_s = sigmaj - sigmai
            # Matrix element of S+ or S-
            s_factor = np.sqrt(spini * (spini + 1) - sigmai * (sigmai + f_s))
            j_factor = np.sqrt(J * (J + 1) - omegai * (omegai + f_s))
            coupling += -f_rot * s_factor * j_factor

        # Spin-orbit coupling
        for so in self.spinorbit:
            if so.istate == istate and so.jstate == jstate:
                # Check selection rules: Delta Lambda = Delta Sigma, Delta Omega = 0
                if (ilambda - jlambda) == -int(sigmai - sigmaj) and abs(omegai - omegaj) < 0.5:
                    # Would need matrix elements computed beforehand
                    # For now, placeholder
                    pass

        return coupling

    def diagonalize(self, J: float, nroots: Optional[int] = None,
                    energy_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build and diagonalize the rovibronic Hamiltonian.

        Parameters
        ----------
        J : float
            Total angular momentum
        nroots : int, optional
            Number of eigenvalues to compute (default: all)
        energy_range : tuple, optional
            (Emin, Emax) energy range to find eigenvalues

        Returns
        -------
        eigenvalues : ndarray
            Energy eigenvalues in cm^-1
        eigenvectors : ndarray
            Eigenvectors
        """
        hmat = self.build_rovibronic_hamiltonian(J)

        h = self.grid.step if isinstance(self.grid.step, float) else float(self.grid.step[0])
        sc = 12.0 * h**2 * self.reduced_mass / self.aston

        if self.use_gpu:
            hmat_gpu = cp.asarray(hmat)
            eigenvalues, eigenvectors = cp.linalg.eigh(hmat_gpu)
            eigenvalues = cp.asnumpy(eigenvalues)
            eigenvectors = cp.asnumpy(eigenvectors)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(hmat)

        # Convert from scaled units
        eigenvalues = eigenvalues / sc

        # Apply nroots or energy_range filtering
        if nroots is not None:
            eigenvalues = eigenvalues[:nroots]
            eigenvectors = eigenvectors[:, :nroots]

        if energy_range is not None:
            mask = (eigenvalues >= energy_range[0]) & (eigenvalues <= energy_range[1])
            eigenvalues = eigenvalues[mask]
            eigenvectors = eigenvectors[:, mask]

        return eigenvalues, eigenvectors

    def diagonalize_with_symmetry(self, J: float, nroots: Optional[int] = None,
                                   energy_range: Optional[Tuple[float, float]] = None) -> List[EigenState]:
        """
        Build and diagonalize the rovibronic Hamiltonian with parity symmetry separation.

        This follows Duo's approach:
        1. Build the unsymmetrized Hamiltonian
        2. Pair up states with opposite (lambda, sigma)
        3. Transform to symmetrized representation
        4. Diagonalize each parity block separately
        5. Assign quantum numbers from largest contributor

        Parameters
        ----------
        J : float
            Total angular momentum
        nroots : int, optional
            Number of eigenvalues to compute per parity block (default: all)
        energy_range : tuple, optional
            (Emin, Emax) energy range to find eigenvalues

        Returns
        -------
        eigenstates : list of EigenState
            List of eigenstates with quantum numbers and parities
        """
        hmat = self.build_rovibronic_hamiltonian(J)

        h = self.grid.step if isinstance(self.grid.step, float) else float(self.grid.step[0])
        sc = 12.0 * h**2 * self.reduced_mass / self.aston

        active_basis = [self.contracted_basis[i] for i in self.active_basis_indices]
        Ntotal = len(active_basis)

        # =====================================================
        # STEP 1: Pair up states and compute parity (itau)
        # =====================================================
        # Following Duo's diatom.f90 lines 9612-9698

        iswap = [0] * Ntotal  # Maps state i to its pair j (with sign for tau)
        levels = []  # List of (ilevel, state_indices, tau, nirr)

        for i in range(Ntotal):
            if iswap[i] != 0:
                continue

            state_i = active_basis[i]
            ilambda = state_i.lambda_
            sigmai = state_i.sigma
            spini = state_i.spin
            omegai = state_i.omega
            istate_i = state_i.istate
            ivib = state_i.v  # Use v (vibrational quantum number) to match Duo

            # Case 1: Lambda=0 and Sigma=0 (single state, no pair)
            if ilambda == 0 and abs(sigmai) < 0.5:
                itau = int(round(spini + J))
                if state_i.parity_pm == -1:
                    itau += 1
                itau = abs(itau)
                tau = (-1.0) ** itau

                # nirr indicates which irreps this level has states in
                # For single states: only one irrep
                if itau % 2 == 1:
                    # Odd itau -> negative parity only
                    levels.append({
                        'indices': [i],
                        'tau': tau,
                        'nirr': [0, 1]  # [positive, negative] - only negative
                    })
                else:
                    # Even itau -> positive parity only
                    levels.append({
                        'indices': [i],
                        'tau': tau,
                        'nirr': [1, 0]  # [positive, negative] - only positive
                    })
                iswap[i] = i + 1  # Mark as used (1-indexed to distinguish from 0)
            else:
                # Case 2: Look for the paired state with opposite lambda, sigma
                for j in range(Ntotal):
                    if i == j or iswap[j] != 0:
                        continue

                    state_j = active_basis[j]
                    jlambda = state_j.lambda_
                    sigmaj = state_j.sigma
                    spinj = state_j.spin
                    jstate_j = state_j.jstate if hasattr(state_j, 'jstate') else state_j.istate
                    jvib = state_j.v  # Use v (vibrational quantum number) to match Duo

                    # Check if j is the paired state: opposite lambda and sigma, same state/vib/spin
                    if (ilambda == -jlambda and
                        abs(sigmai + sigmaj) < 0.01 and
                        istate_i == state_j.istate and
                        ivib == jvib and
                        abs(spini - spinj) < 0.01):

                        # Compute itau
                        itau = -ilambda + int(round(spini - sigmai)) + int(round(J - omegai))
                        if ilambda == 0 and state_i.parity_pm == -1:
                            itau += 1
                        tau = (-1.0) ** itau

                        # Order the pair: positive lambda first (for positive parity)
                        # Following Fortran lines 9667-9677
                        if ilambda > jlambda or sigmaj < sigmai:
                            idx_pos = i  # Goes to positive parity block
                            idx_neg = j  # Goes to negative parity block
                        else:
                            idx_pos = j
                            idx_neg = i

                        levels.append({
                            'indices': [idx_pos, idx_neg],
                            'tau': tau,
                            'nirr': [1, 1]  # Both parities have states
                        })

                        iswap[i] = (j + 1) * ((-1) ** itau)
                        iswap[j] = (i + 1) * ((-1) ** itau)
                        break

        # =====================================================
        # STEP 2: Build symmetry transformation and blocks
        # Following Duo's exact structure with irec and ilevel2i mappings
        # =====================================================
        # Build Duo's mapping arrays:
        # - irec[irrep][isym] = ilevel (level index for symmetry position isym)
        # - ilevel2i[ilevel][irrep] = contracted basis index
        # - Nsym[irrep] = count of levels in each symmetry block

        Nlevels = len(levels)
        Nsym = [0, 0]  # [positive, negative] counts
        ilevel2i = [[0, 0] for _ in range(Nlevels)]  # ilevel -> [pos_idx, neg_idx]
        ilevel2isym = [[0, 0] for _ in range(Nlevels)]  # ilevel -> [pos_sym, neg_sym]

        # Process levels to build ilevel2i and count Nsym
        for ilevel, level in enumerate(levels):
            indices = level['indices']
            nirr = level['nirr']

            if len(indices) == 1:
                # Single state - only exists in one parity
                idx = indices[0]
                if nirr[0] == 1:  # Positive parity
                    Nsym[0] += 1
                    ilevel2isym[ilevel][0] = Nsym[0]
                    ilevel2i[ilevel][0] = idx
                else:  # Negative parity
                    Nsym[1] += 1
                    ilevel2isym[ilevel][1] = Nsym[1]
                    ilevel2i[ilevel][1] = idx
            else:
                # Paired states - exist in both parities
                idx_pos, idx_neg = indices
                Nsym[0] += 1
                Nsym[1] += 1
                ilevel2isym[ilevel][0] = Nsym[0]
                ilevel2isym[ilevel][1] = Nsym[1]
                ilevel2i[ilevel][0] = idx_pos
                ilevel2i[ilevel][1] = idx_neg

        # Build irec arrays: irec[irrep] maps isym -> ilevel
        # And build block_icontr: direct mapping from isym -> contracted basis index
        irec = [[], []]  # irec[0] = positive parity, irec[1] = negative parity
        block_icontr = [[], []]  # Direct: block_icontr[irrep][isym] = contracted basis index

        # Also build block_levels for constructing the Hamiltonian
        pos_indices = []
        neg_indices = []
        pos_levels = []
        neg_levels = []

        for ilevel, level in enumerate(levels):
            indices = level['indices']
            nirr = level['nirr']
            tau = level['tau']

            if len(indices) == 1:
                # Single state - only exists in one parity
                idx = indices[0]
                if nirr[0] == 1:  # Positive parity
                    irec[0].append(ilevel)
                    block_icontr[0].append(idx)
                    pos_indices.append(idx)
                    pos_levels.append({'idx': idx, 'single': True, 'ilevel': ilevel})
                else:  # Negative parity
                    irec[1].append(ilevel)
                    block_icontr[1].append(idx)
                    neg_indices.append(idx)
                    neg_levels.append({'idx': idx, 'single': True, 'ilevel': ilevel})
            else:
                # Paired states - exist in both parities
                idx_pos, idx_neg = indices
                # Positive parity: use ilevel2i[ilevel][0] = idx_pos
                irec[0].append(ilevel)
                block_icontr[0].append(idx_pos)
                pos_indices.append(idx_pos)
                pos_levels.append({
                    'idx_pos': idx_pos,
                    'idx_neg': idx_neg,
                    'tau': tau,
                    'single': False,
                    'ilevel': ilevel
                })
                # Negative parity: use ilevel2i[ilevel][1] = idx_neg
                irec[1].append(ilevel)
                block_icontr[1].append(idx_neg)
                neg_indices.append(idx_neg)
                neg_levels.append({
                    'idx_pos': idx_pos,
                    'idx_neg': idx_neg,
                    'tau': tau,
                    'single': False,
                    'ilevel': ilevel
                })

        # =====================================================
        # STEP 3: Transform Hamiltonian to symmetrized blocks
        # =====================================================
        eigenstates = []

        for irrep, (block_indices, block_levels, parity_char) in enumerate([
            (pos_indices, pos_levels, '+'),
            (neg_indices, neg_levels, '-')
        ]):
            if len(block_indices) == 0:
                continue

            Nblock = len(block_levels)

            # Build the transformed Hamiltonian for this parity block
            hsym = np.zeros((Nblock, Nblock), dtype=np.float64)

            for isym, level_i in enumerate(block_levels):
                for jsym, level_j in enumerate(block_levels):
                    # Compute <sym_i|H|sym_j>
                    # For single states: just use the original element
                    # For paired states: use the symmetrized transformation

                    if level_i['single'] and level_j['single']:
                        # Both single states
                        i = level_i['idx']
                        j = level_j['idx']
                        hsym[isym, jsym] = hmat[i, j]

                    elif level_i['single'] and not level_j['single']:
                        # i single, j paired
                        i = level_i['idx']
                        jp, jn = level_j['idx_pos'], level_j['idx_neg']
                        tauj = level_j['tau']
                        if irrep == 0:  # Positive parity
                            hsym[isym, jsym] = (hmat[i, jp] + tauj * hmat[i, jn]) / np.sqrt(2)
                        else:  # Negative parity
                            hsym[isym, jsym] = (hmat[i, jp] - tauj * hmat[i, jn]) / np.sqrt(2)

                    elif not level_i['single'] and level_j['single']:
                        # i paired, j single
                        ip, in_ = level_i['idx_pos'], level_i['idx_neg']
                        taui = level_i['tau']
                        j = level_j['idx']
                        if irrep == 0:  # Positive parity
                            hsym[isym, jsym] = (hmat[ip, j] + taui * hmat[in_, j]) / np.sqrt(2)
                        else:  # Negative parity
                            hsym[isym, jsym] = (hmat[ip, j] - taui * hmat[in_, j]) / np.sqrt(2)

                    else:
                        # Both paired
                        ip, in_ = level_i['idx_pos'], level_i['idx_neg']
                        jp, jn = level_j['idx_pos'], level_j['idx_neg']
                        taui = level_i['tau']
                        tauj = level_j['tau']

                        if irrep == 0:  # Positive parity: (|+> + tau|->)/sqrt(2)
                            hsym[isym, jsym] = 0.5 * (
                                hmat[ip, jp] + taui * hmat[in_, jp] +
                                tauj * hmat[ip, jn] + taui * tauj * hmat[in_, jn]
                            )
                        else:  # Negative parity: (|+> - tau|->)/sqrt(2)
                            hsym[isym, jsym] = 0.5 * (
                                hmat[ip, jp] - taui * hmat[in_, jp] -
                                tauj * hmat[ip, jn] + taui * tauj * hmat[in_, jn]
                            )

            # Symmetrize
            hsym = (hsym + hsym.T) / 2.0

            # Diagonalize using SYEV (same as Duo) instead of SYEVD
            # This ensures consistent eigenvector ordering for degenerate/near-degenerate states
            if self.use_gpu:
                # For GPU: transfer to CPU, use scipy SYEV, transfer back
                hsym_np = cp.asnumpy(hsym) if hasattr(hsym, 'get') else hsym
                eigenvalues, eigenvectors = scipy.linalg.eigh(hsym_np, driver='ev')
            else:
                eigenvalues, eigenvectors = scipy.linalg.eigh(hsym, driver='ev')

            # Convert from scaled units
            eigenvalues = eigenvalues / sc

            # =====================================================
            # STEP 4: Assign quantum numbers using EXACTLY Duo's approach
            # imaxcontr = argmax of coefficient^2 (index in symmetry block)
            # block_icontr[irrep][imaxcontr] = contracted basis index directly
            # =====================================================
            QNs = [None] * len(eigenvalues)  # Store imaxcontr for each root

            for iroot in range(len(eigenvalues)):
                energy = eigenvalues[iroot]

                # Apply energy filter
                if energy_range is not None:
                    if energy < energy_range[0] or energy > energy_range[1]:
                        continue

                coeffs = eigenvectors[:, iroot]
                coeffs_sq = coeffs ** 2

                # Find largest coefficient (Duo: maxloc)
                # When coefficients are nearly equal (within tolerance), prefer the
                # LOWEST index to match Fortran's maxloc behavior which returns
                # the first occurrence of the maximum.
                tol = 1e-6  # Relative tolerance for considering coefficients equal
                max_val = np.max(coeffs_sq)
                if max_val > 0:
                    # Find all indices within tolerance of the maximum
                    near_max_mask = coeffs_sq >= max_val * (1.0 - tol)
                    near_max_indices = np.where(near_max_mask)[0]
                    # Take the lowest index among those near the maximum
                    imaxcontr = int(near_max_indices[0])
                else:
                    imaxcontr = int(np.argmax(coeffs_sq))
                maxcontr = coeffs_sq[imaxcontr]

                # Check if already used, take second largest if so (Duo's loop_check)
                small_ = 1e-12
                i0 = 0
                while True:
                    found_duplicate = False
                    for ilevel_prev in range(iroot):
                        if QNs[ilevel_prev] is not None and imaxcontr == QNs[ilevel_prev]:
                            # Find next largest that's smaller than current max
                            mask2 = (coeffs_sq >= small_) & (coeffs_sq < maxcontr)
                            if np.any(mask2):
                                masked_coeffs2 = np.where(mask2, coeffs_sq, -np.inf)
                                max_val2 = np.max(masked_coeffs2)
                                # Apply same tolerance-based tie-breaking
                                near_max_mask2 = masked_coeffs2 >= max_val2 * (1.0 - tol)
                                near_max_indices2 = np.where(near_max_mask2)[0]
                                imaxcontr = int(near_max_indices2[0])
                                maxcontr = coeffs_sq[imaxcontr]
                            i0 += 1
                            if i0 > 10:
                                # Fallback to largest
                                imaxcontr = int(np.argmax(coeffs_sq))
                                break
                            found_duplicate = True
                            break
                    if not found_duplicate or i0 > 10:
                        break

                QNs[iroot] = imaxcontr

                # Direct mapping: block_icontr[irrep][imaxcontr] gives contracted basis index
                mterm = block_icontr[irrep][imaxcontr]
                basis_state = active_basis[mterm]

                eigenstate = EigenState(
                    energy=energy,
                    J=J,
                    istate=basis_state.istate,
                    v=basis_state.v,
                    lambda_=basis_state.lambda_,
                    sigma=basis_state.sigma,
                    omega=basis_state.omega,
                    spin=basis_state.spin,
                    parity=parity_char,
                    name=basis_state.name,
                    coefficients=coeffs
                )
                eigenstates.append(eigenstate)

        # Sort by energy
        eigenstates.sort(key=lambda x: x.energy)

        # Apply nroots limit if specified
        if nroots is not None:
            eigenstates = eigenstates[:nroots]

        return eigenstates

    def print_hamiltonian_matrix(self, J: float, threshold: float = 1e-10,
                                  output_file: Optional[str] = None) -> None:
        """
        Print non-zero matrix elements of the coupled Sigma-Lambda Hamiltonian.

        This produces output similar to Duo's format showing:
        - RV: Rotational-vibrational (diagonal) elements
        - SO: Spin-Orbit interaction
        - SS: Spin-Spin interaction
        - JS: J.S interaction (S-uncoupling)
        - LS: L.J/L.S interaction (L-uncoupling/spin-electronic)
        - BR: BOB-rot correction
        - BV: BOB-vib correction
        - LD: Lambda-doubling

        Parameters
        ----------
        J : float
            Total angular momentum quantum number
        threshold : float
            Minimum absolute value for printing (default: 1e-10)
        output_file : str, optional
            If provided, write output to this file instead of stdout
        """
        import sys

        # Build the Hamiltonian with tracking of contributions
        hmat, contributions = self._build_hamiltonian_with_tracking(J)

        h = self.grid.step if isinstance(self.grid.step, float) else float(self.grid.step[0])
        sc = 12.0 * h**2 * self.reduced_mass / self.aston

        active_basis = [self.contracted_basis[i] for i in self.active_basis_indices]
        Ntotal = len(active_basis)

        # Open output file or use stdout
        if output_file:
            f = open(output_file, 'w')
        else:
            f = sys.stdout

        # Header
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Non-zero matrix elements of the coupled Sigma-Lambda matrix\n")
        f.write("=" * 80 + "\n")
        f.write(f"J = {J:.1f}\n")
        f.write(f"Threshold for printing is {threshold:.2E} cm^-1\n")
        f.write("\n")
        f.write("Legend:\n")
        f.write("  RV == Rotational-vibrational (diagonal)\n")
        f.write("  SO == Spin-Orbit interaction\n")
        f.write("  SS == Spin-Spin interaction\n")
        f.write("  JS == J.S interaction (S-uncoupling)\n")
        f.write("  LS == L.J interaction (L-uncoupling)\n")
        f.write("  LE == L.S interaction (spin-electronic)\n")
        f.write("  BR == BOB-rot correction\n")
        f.write("  BV == BOB-vib correction\n")
        f.write("  LD == Lambda-doubling (OPQ, P2Q, Q)\n")
        f.write("\n")

        # Print basis state info
        f.write("Basis states:\n")
        f.write(f"{'idx':>4} {'State':>12} {'v':>3} {'Lam':>4} {'Sig':>5} {'Omg':>5} {'Spin':>4}\n")
        f.write("-" * 50 + "\n")
        for i, state in enumerate(active_basis):
            f.write(f"{i:4d} {state.name:>12} {state.v:3d} {state.lambda_:4d} "
                    f"{state.sigma:5.1f} {state.omega:5.1f} {state.spin:4.1f}\n")
        f.write("\n")

        # Print matrix elements row by row
        f.write("Matrix elements (cm^-1):\n")
        f.write("=" * 80 + "\n")

        for i in range(Ntotal):
            state_i = active_basis[i]
            diag_energy = hmat[i, i] / sc

            # Start row with diagonal element
            row_str = f"RV= {diag_energy:14.3f}; "

            # Collect off-diagonal contributions for this row
            off_diag_terms = []

            for j in range(Ntotal):
                if i == j:
                    continue

                element = hmat[i, j] / sc
                if abs(element) < threshold:
                    continue

                state_j = active_basis[j]

                # Determine the type of interaction from contributions
                contrib_key = (min(i, j), max(i, j))
                if contrib_key in contributions:
                    for contrib_type, value in contributions[contrib_key]:
                        value_cm = value / sc
                        if abs(value_cm) >= threshold:
                            off_diag_terms.append((contrib_type, i, j, value_cm))
                else:
                    # Generic off-diagonal
                    off_diag_terms.append(("??", i, j, element))

            # Format and print off-diagonal terms
            if off_diag_terms:
                terms_str = ""
                for term_type, ti, tj, value in off_diag_terms[:10]:  # Limit to 10 terms per line
                    terms_str += f"{term_type}({ti:2d},{tj:2d})={value:10.4f}; "
                row_str += terms_str

            f.write(row_str.rstrip("; ") + "\n")

        f.write("=" * 80 + "\n")

        if output_file:
            f.close()
            print(f"Hamiltonian matrix written to: {output_file}")

    def _build_hamiltonian_with_tracking(self, J: float) -> Tuple[np.ndarray, dict]:
        """
        Build the rovibronic Hamiltonian while tracking contributions by type.

        Returns
        -------
        hmat : np.ndarray
            The Hamiltonian matrix
        contributions : dict
            Dictionary mapping (i,j) pairs to list of (type, value) tuples
        """
        # Build the standard Hamiltonian first
        hmat = self.build_rovibronic_hamiltonian(J)

        h = self.grid.step if isinstance(self.grid.step, float) else float(self.grid.step[0])
        sc = 12.0 * h**2 * self.reduced_mass / self.aston

        active_basis = [self.contracted_basis[i] for i in self.active_basis_indices]
        Ntotal = len(active_basis)

        # Track contributions by type
        contributions = {}

        # We need to rebuild tracking each contribution type separately
        # This is a simplified version - track SO, SS, L+, etc.

        # Always use NumPy for tracking (convert from CuPy if needed)
        wf = self.contracted_wavefunctions
        r = self.grid.r
        if self.use_gpu:
            import cupy as cp
            if isinstance(wf, cp.ndarray):
                wf = cp.asnumpy(wf)
            if isinstance(r, cp.ndarray):
                r = cp.asnumpy(r)

        # =====================================================
        # Track Spin-Orbit contributions
        # =====================================================
        from pyduo.constants import three_j

        for so_idx, so_field in enumerate(self.spinorbit):
            istate_so = so_field.istate
            jstate_so = so_field.jstate

            if so_field.parameters is None or len(so_field.parameters) == 0:
                continue

            so_values = so_field.evaluate(r, use_gpu=False)
            so_wf = so_values.reshape(-1, 1) * wf
            so_matelem = np.dot(wf.T, so_wf)

            # Reference values for Wigner-Eckart
            lambda_i_we = so_field.lambda_i
            lambda_j_we = so_field.lambda_j
            sigma_i_we = so_field.sigma_i
            sigma_j_we = so_field.sigma_j
            spin_i_we = so_field.spin_i
            spin_j_we = so_field.spin_j
            q_we = sigma_i_we - sigma_j_we

            three_j_ref = three_j(spin_i_we, 1.0, spin_j_we, -sigma_i_we, q_we, sigma_j_we)
            if abs(three_j_ref) < 1e-12:
                continue

            for i in range(Ntotal):
                state_i = active_basis[i]
                for j in range(i, Ntotal):
                    state_j = active_basis[j]

                    # Check basic selection rules
                    if abs(state_i.omega - state_j.omega) > 0.5:
                        continue
                    if abs(state_i.spin - state_j.spin) > 1.01:
                        continue

                    # Check if states match the SO field (either direction)
                    match_forward = (state_i.istate == istate_so and state_j.istate == jstate_so)
                    match_reverse = (state_i.istate == jstate_so and state_j.istate == istate_so)

                    if not (match_forward or match_reverse):
                        continue

                    ivib = state_i.ivib
                    jvib = state_j.ivib
                    matelem_vib = float(so_matelem[ivib, jvib])

                    # Simplified SO contribution tracking
                    if abs(matelem_vib * sc) > 1e-12:
                        key = (min(i, j), max(i, j))
                        if key not in contributions:
                            contributions[key] = []
                        contributions[key].append((f"{so_idx+1}-SO", matelem_vib * sc))

        # =====================================================
        # Track L+ (L-uncoupling) contributions
        # =====================================================
        for lx_idx, lxly_field in enumerate(self.lxly):
            istate_lx = lxly_field.istate
            jstate_lx = lxly_field.jstate

            if lxly_field.parameters is None or len(lxly_field.parameters) == 0:
                continue

            lx_values = lxly_field.evaluate(r, use_gpu=False)
            lx_wf = lx_values.reshape(-1, 1) * wf
            lx_matelem = np.dot(wf.T, lx_wf)

            for i in range(Ntotal):
                state_i = active_basis[i]
                for j in range(i, Ntotal):
                    state_j = active_basis[j]

                    if not ((state_i.istate == istate_lx and state_j.istate == jstate_lx) or
                            (state_i.istate == jstate_lx and state_j.istate == istate_lx)):
                        continue

                    ilambda = state_i.lambda_
                    jlambda = state_j.lambda_
                    delta_lambda = abs(ilambda - jlambda)

                    if delta_lambda != 1:
                        continue

                    ivib = state_i.ivib
                    jvib = state_j.ivib
                    matelem_vib = float(lx_matelem[ivib, jvib])

                    # Determine if it's L.J (LS) or L.S (LE)
                    sigmai = state_i.sigma
                    sigmaj = state_j.sigma

                    if abs(sigmai - sigmaj) < 0.01:
                        # L.J interaction (rotational L-uncoupling)
                        contrib_type = f"{lx_idx+1}-LJ"
                    else:
                        # L.S interaction (spin-electronic)
                        contrib_type = f"{lx_idx+1}-LS"

                    if abs(matelem_vib * sc) > 1e-12:
                        key = (min(i, j), max(i, j))
                        if key not in contributions:
                            contributions[key] = []
                        contributions[key].append((contrib_type, matelem_vib * sc))

        # =====================================================
        # Track Spin-Spin contributions
        # =====================================================
        for ss_idx, ss_field in enumerate(self.spinspin):
            istate_ss = ss_field.istate
            jstate_ss = ss_field.jstate

            if ss_field.parameters is None or len(ss_field.parameters) == 0:
                continue

            ss_values = ss_field.evaluate(r, use_gpu=False)
            ss_wf = ss_values.reshape(-1, 1) * wf
            ss_matelem = np.dot(wf.T, ss_wf)

            for i in range(Ntotal):
                state_i = active_basis[i]
                if state_i.istate != istate_ss:
                    continue

                for j in range(i, Ntotal):
                    state_j = active_basis[j]
                    if state_j.istate != jstate_ss:
                        continue

                    if abs(state_i.sigma - state_j.sigma) > 0.5:
                        continue

                    ivib = state_i.ivib
                    jvib = state_j.ivib
                    matelem_vib = float(ss_matelem[ivib, jvib])

                    if abs(matelem_vib * sc) > 1e-12:
                        key = (min(i, j), max(i, j))
                        if key not in contributions:
                            contributions[key] = []
                        contributions[key].append((f"{ss_idx+1}-SS", matelem_vib * sc))

        # =====================================================
        # Track J.S (S-uncoupling) contributions
        # These come from the rotational term when Delta Sigma = +/-1
        # =====================================================
        for i in range(Ntotal):
            state_i = active_basis[i]
            for j in range(i + 1, Ntotal):
                state_j = active_basis[j]

                # Same electronic state, same lambda, Delta Sigma = +/-1
                if state_i.istate != state_j.istate:
                    continue
                if state_i.lambda_ != state_j.lambda_:
                    continue
                if state_i.ivib != state_j.ivib:
                    continue

                delta_sigma = abs(state_i.sigma - state_j.sigma)
                if abs(delta_sigma - 1.0) > 0.01:
                    continue

                # This is a J.S term
                # The value comes from the rotational S+/S- terms
                spini = state_i.spin
                sigmai = state_i.sigma
                sigmaj = state_j.sigma

                # S+ or S- matrix element
                if sigmaj > sigmai:
                    # S+ raises sigma
                    s_elem = np.sqrt(spini * (spini + 1) - sigmai * (sigmai + 1))
                else:
                    # S- lowers sigma
                    s_elem = np.sqrt(spini * (spini + 1) - sigmai * (sigmai - 1))

                # Rotational factor
                omegai = state_i.omega
                rot_factor = -np.sqrt(J * (J + 1) - omegai * (omegai + np.sign(sigmaj - sigmai)))

                js_contrib = s_elem * rot_factor * sc

                if abs(js_contrib) > 1e-12:
                    key = (min(i, j), max(i, j))
                    if key not in contributions:
                        contributions[key] = []
                    contributions[key].append(("J-S", js_contrib))

        return hmat, contributions

    def print_sigma_lambda_basis(self, J: float) -> None:
        """
        Print the Sigma-Lambda basis set for a given J value (like Duo).

        Parameters
        ----------
        J : float
            Total angular momentum quantum number
        """
        # Build ilevel mapping like Duo does
        # Duo orders: loop over states, then tau (lambda sign), then sigma
        ilevel_list = []

        for state in self.states:
            istate = state.istate
            spin = state.spin
            lambda_abs = state.lambda_
            multi = state.multiplicity

            # tau loop: 0 to taumax (taumax=1 if lambda!=0, else 0)
            taumax = 1 if lambda_abs != 0 else 0
            for itau in range(taumax + 1):
                ilambda = ((-1) ** itau) * lambda_abs

                # sigma loop: from -spin to +spin
                for imulti in range(multi):
                    sigma = -spin + imulti
                    omega = ilambda + sigma

                    # Check if omega is valid for this J
                    if abs(omega) <= J + 0.01:
                        ilevel_list.append({
                            'istate': istate,
                            'spin': spin,
                            'sigma': sigma,
                            'lambda': ilambda,
                            'omega': omega,
                            'name': state.name
                        })

        Nlambdasigmas = len(ilevel_list)

        print(f"\nDefine the quanta book-keeping")
        print(f"The total number sigma/lambda states (size of the sigma-lambda submatrix) = {Nlambdasigmas}")
        print(f"\nSigma-Lambda basis set:")
        print(f"{'i':>6} {'jrot':>8} {'state':>6} {'spin':>6} {'sigma':>8} {'lambda':>7} {'omega':>7}")

        for i, level in enumerate(ilevel_list, 1):
            print(f"{i:6d} {J:8.1f} {level['istate']:6d} {level['spin']:6.1f} "
                  f"{level['sigma']:8.1f} {level['lambda']:7d} {level['omega']:7.1f}   {level['name']}")

        print("...done!\n")

        return ilevel_list

    def print_hamiltonian_duo_format(self, J: float, threshold: float = 2.22e-16,
                                      output_file: Optional[str] = None,
                                      add_potmin_offset: float = 0.0) -> None:
        """
        Print Hamiltonian matrix elements in Duo's EXACT format.

        Matches Duo's output format precisely:
        - RV= <F15.3>; followed by coupling terms
        - Coupling format: <i3>-<TYPE><i3><i3><g12.4>

        Parameters
        ----------
        J : float
            Total angular momentum quantum number
        threshold : float
            Minimum absolute value for printing (default: 2.22e-16 to match Duo)
        output_file : str, optional
            If provided, write output to this file instead of stdout
        add_potmin_offset : float
            Offset to add to diagonal elements to match Duo's energy reference.
            Duo shifts PECs by the ground state minimum (typically ~36800 cm^-1 for CaO).
            Set this to match Duo's RV values exactly.
        """
        import sys

        # Build the sigma-lambda basis (like Duo) - suppress printing
        ilevel_list = []
        for state in self.states:
            istate = state.istate
            spin = state.spin
            lambda_abs = state.lambda_
            multi = state.multiplicity
            taumax = 1 if lambda_abs != 0 else 0
            for itau in range(taumax + 1):
                ilambda = ((-1) ** itau) * lambda_abs
                for imulti in range(multi):
                    sigma = -spin + imulti
                    omega = ilambda + sigma
                    if abs(omega) <= J + 0.01:
                        ilevel_list.append({
                            'istate': istate,
                            'lambda': ilambda,
                            'sigma': sigma,
                            'omega': omega,
                            'spin': spin,
                            'name': state.name
                        })

        # Build the Hamiltonian
        hmat = self.build_rovibronic_hamiltonian(J)
        if self.use_gpu:
            import cupy as cp
            if isinstance(hmat, cp.ndarray):
                hmat = cp.asnumpy(hmat)

        h = self.grid.step if isinstance(self.grid.step, float) else float(self.grid.step[0])
        sc = 12.0 * h**2 * self.reduced_mass / self.aston

        active_basis = [self.contracted_basis[i] for i in self.active_basis_indices]
        Ntotal = len(active_basis)

        # Build mapping from (istate, lambda, sigma) -> ilevel (1-indexed)
        ilevel_map = {}
        for i, level in enumerate(ilevel_list, 1):
            key = (level['istate'], level['lambda'], round(level['sigma'], 1))
            ilevel_map[key] = i

        # Create reverse mapping: ilevel -> list of (contracted_basis_idx, v)
        ilevel_to_indices = {}
        for i, state in enumerate(active_basis):
            key = (state.istate, state.lambda_, round(state.sigma, 1))
            if key in ilevel_map:
                ilevel = ilevel_map[key]
                if ilevel not in ilevel_to_indices:
                    ilevel_to_indices[ilevel] = []
                ilevel_to_indices[ilevel].append((i, state.v))

        Nlambdasigmas = len(ilevel_list)

        # Open output file or use stdout
        if output_file:
            f = open(output_file, 'w')
        else:
            f = sys.stdout

        # Header - EXACT Duo format
        f.write("\nNon-zero matrix elements of the coupled Sigma-Lambda matrix:\n")
        f.write(f"Threshold for printing is {threshold:10.2E} cm^-1\n")
        f.write("RV == Rotational-vibrational\n")
        f.write("SO == Spin-Orbit interaction\n")
        f.write("SS == Spin-Spin interaction\n")
        f.write("JS == J.S interaction (aka S-uncoupling)\n")
        f.write("LS == L.J interaction (aka L-uncoupling)\n")
        f.write("LS == L.S interaction (spin-electronic)\n")

        # Build printout per ilevel (like Duo)
        printout = {ilevel: "" for ilevel in range(1, Nlambdasigmas + 1)}

        # Process diagonal elements - show RV energy for v=0 of each ilevel
        for ilevel in sorted(ilevel_to_indices.keys()):
            indices = ilevel_to_indices[ilevel]
            v0_idx = None
            for idx, v in indices:
                if v == 0:
                    v0_idx = idx
                    break
            if v0_idx is None:
                v0_idx = indices[0][0]

            diag_energy = hmat[v0_idx, v0_idx] / sc + add_potmin_offset
            printout[ilevel] = f"RV={diag_energy:15.3f};"

        # Process ALL off-diagonal elements and show TOTAL Hamiltonian value
        # (This shows the actual coupling in the matrix, summed from all sources)
        for idx_i in range(Ntotal):
            for idx_j in range(idx_i + 1, Ntotal):
                if idx_i not in ilevel_to_indices or idx_j not in [x[0] for il in ilevel_to_indices.values() for x in il]:
                    continue

                value = hmat[idx_i, idx_j] / sc
                if abs(value) < threshold:
                    continue

                # Find ilevel for each index
                state_i = active_basis[idx_i]
                state_j = active_basis[idx_j]
                key_i = (state_i.istate, state_i.lambda_, round(state_i.sigma, 1))
                key_j = (state_j.istate, state_j.lambda_, round(state_j.sigma, 1))

                if key_i not in ilevel_map or key_j not in ilevel_map:
                    continue

                ilevel_i = ilevel_map[key_i]
                ilevel_j = ilevel_map[key_j]

                # Determine coupling type based on quantum number changes
                delta_lambda = abs(state_i.lambda_ - state_j.lambda_)
                delta_sigma = abs(state_i.sigma - state_j.sigma)
                same_state = state_i.istate == state_j.istate

                if same_state and delta_lambda == 0 and abs(delta_sigma - 1.0) < 0.1:
                    # S-uncoupling (J.S)
                    coup_type = "JS"
                    coup_idx = 0  # No specific field index
                elif delta_lambda == 1 and abs(delta_sigma) < 0.1:
                    # L.J (rotational L-uncoupling)
                    coup_type = "LJ"
                    coup_idx = 1
                elif delta_lambda == 1 and abs(delta_sigma - 1.0) < 0.1:
                    # L.S (spin-electronic)
                    coup_type = "LS"
                    coup_idx = 1
                elif delta_lambda == 0:
                    # Could be SO or SS
                    coup_type = "SO"
                    coup_idx = 1
                else:
                    coup_type = "??"
                    coup_idx = 0

                # Format like Duo: "  N-TYPE ilev jlev value"
                term = f"{coup_idx:3d}-{coup_type}{ilevel_i:3d}{ilevel_j:3d}{value:12.4E}"
                printout[ilevel_i] += f"  {term}"

        # Write output - truncate to 250 characters like Duo
        for ilevel in sorted(printout.keys()):
            line = printout[ilevel][:250]
            f.write(line + "\n")

        f.write(" \n")

        if output_file:
            f.close()
            print(f"Hamiltonian matrix written to: {output_file}")
