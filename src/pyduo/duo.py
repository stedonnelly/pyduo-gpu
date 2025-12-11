"""
Main PyDuo solver class

This is the main entry point for PyDuo calculations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = np
    HAS_CUDA = False

from .constants import PhysicalConstants
from .grid import Grid
from .fields import PotentialField, SpinOrbitField, CouplingField, ElectronicState
from .hamiltonian import HamiltonianBuilder, BasisState, EigenState
from .solver import EigenSolver
from .input_parser import InputParser


@dataclass
class EnergyLevel:
    """Represents a computed energy level."""
    energy: float           # Energy in cm^-1
    J: float                # Total angular momentum
    parity: int             # Parity (+1 or -1)
    state: int              # Dominant electronic state
    v: int                  # Dominant vibrational quantum number
    lambda_: int            # Lambda quantum number
    sigma: float            # Sigma quantum number
    omega: float            # Omega = Lambda + Sigma
    spin: float = 0.5       # Spin quantum number S
    name: str = ""          # State name


class DuoSolver:
    """
    Main solver class for diatomic rovibronic calculations.

    This class provides a high-level interface for:
    1. Setting up the molecular system
    2. Solving the vibrational (J=0) problem
    3. Solving the full rovibronic problem for multiple J values
    4. Computing transition intensities (future)

    Example
    -------
    >>> solver = DuoSolver()
    >>> solver.load_input("input.inp")
    >>> solver.run()
    >>> energies = solver.get_energies()
    """

    def __init__(self, use_gpu: bool = True, verbose: int = 3):
        """
        Initialize the DuoSolver.

        Parameters
        ----------
        use_gpu : bool
            Whether to use GPU acceleration
        verbose : int
            Verbosity level (0=silent, 5=debug)
        """
        self.use_gpu = use_gpu and HAS_CUDA
        self.verbose = verbose

        # Configuration
        self.grid: Optional[Grid] = None
        self.reduced_mass: float = 1.0
        self.states: List[ElectronicState] = []
        self.potentials: List[PotentialField] = []
        self.j_list: List[float] = [0.0]

        # Results
        self.vibrational_energies: Dict[int, np.ndarray] = {}
        self.vibrational_wavefunctions: Dict[int, np.ndarray] = {}
        self.rovibronic_energies: Dict[float, np.ndarray] = {}
        self.rovibronic_eigenvectors: Dict[float, np.ndarray] = {}
        self.energy_levels: List[EnergyLevel] = []
        self.zpe: float = 0.0  # Zero-point energy (ground state energy)

        # Internal objects
        self.hamiltonian: Optional[HamiltonianBuilder] = None
        self.parser: Optional[InputParser] = None

        # Timing
        self.timings: Dict[str, float] = {}

        if self.verbose >= 3 and self.use_gpu:
            print("PyDuo initialized with GPU acceleration (CUDA)")
        elif self.verbose >= 3:
            print("PyDuo initialized (CPU mode)")

    def cleanup_gpu(self):
        """
        Clean up GPU memory after calculations are complete.
        Call this to free GPU memory when done with the solver.
        """
        if self.hamiltonian is not None:
            self.hamiltonian.cleanup_gpu()

        if self.use_gpu and HAS_CUDA:
            import gc
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def load_input(self, filename: str):
        """
        Load calculation setup from a Duo-format input file.

        Parameters
        ----------
        filename : str
            Path to input file
        """
        if self.verbose >= 3:
            print(f"Reading input from: {filename}")

        self.parser = InputParser()
        self.parser.parse_file(filename)

        # Extract configuration
        self.grid = self.parser.build_grid(use_gpu=self.use_gpu)
        self.reduced_mass = self.parser.molecule.reduced_mass
        self.states = self.parser.states
        self.potentials = self.parser.potentials
        self.j_list = self.parser.job.j_list

        if self.verbose >= 3:
            print(f"  Grid: {self.grid.npoints} points, r = [{self.grid.rmin:.2f}, {self.grid.rmax:.2f}] Ang")
            print(f"  Reduced mass: {self.reduced_mass:.6f} amu")
            print(f"  Electronic states: {len(self.states)}")
            for s in self.states:
                print(f"    State {s.istate}: {s.name}, lambda={s.lambda_}, mult={s.multiplicity}")
            print(f"  Potentials: {len(self.potentials)}")
            for p in self.potentials:
                print(f"    Potential for state {p.istate}: {p.name}, type={p.func_type}")
            print(f"  J values: {self.j_list}")

    def setup_manual(self, npoints: int, rmin: float, rmax: float,
                     mass1: float, mass2: float,
                     potentials: List[PotentialField],
                     states: List[ElectronicState],
                     j_list: Optional[List[float]] = None,
                     grid_type: int = 0):
        """
        Set up calculation manually without input file.

        Parameters
        ----------
        npoints : int
            Number of grid points
        rmin, rmax : float
            Grid range in Angstroms
        mass1, mass2 : float
            Atomic masses in amu
        potentials : list
            List of PotentialField objects
        states : list
            List of ElectronicState objects
        j_list : list, optional
            List of J values to compute
        grid_type : int
            Grid type (0=uniform)
        """
        self.grid = Grid(npoints, rmin, rmax, grid_type=grid_type, use_gpu=self.use_gpu)
        self.reduced_mass = mass1 * mass2 / (mass1 + mass2)
        self.potentials = potentials
        self.states = states
        self.j_list = j_list if j_list else [0.0]

    def run(self, j_values: Optional[List[float]] = None):
        """
        Run the full rovibronic calculation.

        Parameters
        ----------
        j_values : list, optional
            Override J values from input
        """
        if self.grid is None:
            raise RuntimeError("No grid defined. Call load_input() or setup_manual() first.")

        if j_values is not None:
            self.j_list = j_values

        total_start = time.time()

        # Build Hamiltonian
        if self.verbose >= 3:
            print("\nBuilding Hamiltonian...")

        self.hamiltonian = HamiltonianBuilder(
            grid=self.grid,
            reduced_mass=self.reduced_mass,
            states=self.states,
            potentials=self.potentials,
            use_gpu=self.use_gpu,
            verbose=self.verbose
        )

        # Add coupling terms if available
        if self.parser:
            for so in self.parser.spinorbit:
                self.hamiltonian.add_coupling(so)
            for coupling in self.parser.couplings:
                self.hamiltonian.add_coupling(coupling)

        # Solve vibrational problem
        self._solve_vibrational()

        # Solve rovibronic problem
        # Use batched GPU solver if available and GPU enabled
        if self.use_gpu and len(self.j_list) > 1:
            self._solve_rovibronic_batched()
        else:
            for J in self.j_list:
                self._solve_rovibronic(J)

        # Reference all energies to the ground state (ZPE), like Duo does
        self._reference_energies_to_zpe()

        total_time = time.time() - total_start
        self.timings['total'] = total_time

        if self.verbose >= 3:
            print(f"\nTotal calculation time: {total_time:.2f} s")

    def _reference_energies_to_zpe(self):
        """
        Reference all energies to the ground state (zero-point energy).

        This matches Duo's behavior where energies are reported relative to
        the lowest computed energy level (the ground state ZPE).
        """
        if not self.energy_levels:
            return

        # Find the zero-point energy (lowest energy level)
        self.zpe = min(level.energy for level in self.energy_levels)

        if self.verbose >= 3:
            print(f"\nReferencing energies to ZPE: {self.zpe:.6f} cm^-1")

        # Subtract ZPE from all energy levels
        for level in self.energy_levels:
            level.energy -= self.zpe

        # Also update the rovibronic_energies dictionary
        for J in self.rovibronic_energies:
            self.rovibronic_energies[J] = self.rovibronic_energies[J] - self.zpe

    def _solve_rovibronic_batched(self):
        """Solve the rovibronic problem for ALL J values simultaneously using GPU batching."""
        start = time.time()

        if self.verbose >= 3:
            print(f"\nSolving rovibronic problem for {len(self.j_list)} J values (batched GPU)...")
            print(f"  J range: {min(self.j_list)} to {max(self.j_list)}")

        # Use batched solver
        eigenstates = self.hamiltonian.solve_all_J_batched(self.j_list)

        # Convert EigenState to EnergyLevel
        for es in eigenstates:
            parity_int = 1 if es.parity == '+' else -1
            level = EnergyLevel(
                energy=es.energy,
                J=es.J,
                parity=parity_int,
                state=es.istate,
                v=es.v,
                lambda_=es.lambda_,
                sigma=es.sigma,
                omega=es.omega,
                spin=es.spin,
                name=es.name
            )
            self.energy_levels.append(level)

        # Store eigenvalues by J
        for J in self.j_list:
            j_levels = [es.energy for es in eigenstates if es.J == J]
            self.rovibronic_energies[J] = np.array(j_levels)

        batch_time = time.time() - start
        self.timings['rovibronic_batched'] = batch_time

        if self.verbose >= 3:
            print(f"  Batched rovibronic solved in {batch_time:.2f} s")
            print(f"  Total levels: {len(eigenstates)}")

    def _solve_vibrational(self):
        """Solve the vibrational (J=0) problem."""
        start = time.time()

        if self.verbose >= 3:
            print("\nSolving vibrational problem...")

        # Determine max vibrational states per electronic state
        max_vib = {}
        if self.parser and self.parser.job.vibmax:
            # vibmax in parser uses 1-based state indexing, convert to 0-based
            for state_id, vmax in self.parser.job.vibmax.items():
                max_vib[state_id - 1] = vmax
        else:
            # Default: use grid size / 10 or 50
            for i, _ in enumerate(self.states):
                max_vib[i] = min(self.grid.npoints // 10, 50)

        self.hamiltonian.solve_vibrational_problem(max_vib_states=max_vib)

        # Store results
        for istate in range(len(self.states)):
            energies, wfs = self.hamiltonian.build_vibrational_hamiltonian(istate)
            self.vibrational_energies[istate] = energies
            self.vibrational_wavefunctions[istate] = wfs

        vib_time = time.time() - start
        self.timings['vibrational'] = vib_time

        if self.verbose >= 3:
            print(f"  Vibrational problem solved in {vib_time:.2f} s")
            print(f"  Total contracted basis states: {len(self.hamiltonian.contracted_basis)}")

        if self.verbose >= 4:
            print("\n  Vibrational energies (cm^-1):")
            for istate, energies in self.vibrational_energies.items():
                state_name = self.states[istate].name if istate < len(self.states) else f"State_{istate}"
                zpe = energies[0]
                print(f"    {state_name}:")
                for v, e in enumerate(energies[:10]):  # First 10 levels
                    print(f"      v={v}: {e:.4f} (rel: {e-zpe:.4f})")

    def _solve_rovibronic(self, J: float):
        """Solve the rovibronic problem for a given J value."""
        start = time.time()

        if self.verbose >= 3:
            print(f"\nSolving rovibronic problem for J={J}...")

        # Use parity-separated diagonalization
        eigenstates = self.hamiltonian.diagonalize_with_symmetry(J)

        # Extract eigenvalues for compatibility
        eigenvalues = np.array([es.energy for es in eigenstates])
        self.rovibronic_energies[J] = eigenvalues
        self.rovibronic_eigenvectors[J] = None  # Not storing full eigenvectors in symmetry mode

        # Convert EigenState to EnergyLevel
        for es in eigenstates:
            parity_int = 1 if es.parity == '+' else -1
            level = EnergyLevel(
                energy=es.energy,
                J=J,
                parity=parity_int,
                state=es.istate,
                v=es.v,
                lambda_=es.lambda_,
                sigma=es.sigma,
                omega=es.omega,
                spin=es.spin,
                name=es.name
            )
            self.energy_levels.append(level)

        J_time = time.time() - start
        self.timings[f'J={J}'] = J_time

        if self.verbose >= 3:
            print(f"  J={J} solved in {J_time:.2f} s, found {len(eigenstates)} levels")

        if self.verbose >= 4:
            zpe = eigenvalues[0] if len(eigenvalues) > 0 else 0
            print(f"    First 10 levels (cm^-1):")
            for i, es in enumerate(eigenstates[:10]):
                print(f"      {i}: {es.energy:.4f} (rel: {es.energy-zpe:.4f}) {es.parity} lambda={es.lambda_} sigma={es.sigma}")

    def _assign_quantum_numbers(self, J: float, eigenvalues: np.ndarray,
                                eigenvectors: np.ndarray):
        """Assign quantum numbers to computed energy levels."""
        # Use active_basis_indices for the filtered basis used in diagonalization
        if hasattr(self.hamiltonian, 'active_basis_indices'):
            active_indices = self.hamiltonian.active_basis_indices
            nbasis = len(active_indices)
        else:
            active_indices = list(range(len(self.hamiltonian.contracted_basis)))
            nbasis = len(self.hamiltonian.contracted_basis)

        for i, energy in enumerate(eigenvalues):
            # Find dominant basis state from eigenvector
            if eigenvectors is not None and i < eigenvectors.shape[1]:
                coefs = eigenvectors[:, i]

                # The eigenvector dimension should match the active (filtered) basis
                if len(coefs) == nbasis:
                    idx_max = np.argmax(np.abs(coefs))
                    # Map back to the full contracted_basis using active_indices
                    basis = self.hamiltonian.contracted_basis[active_indices[idx_max]]

                    # Determine parity from sigma sign
                    parity = 1 if basis.sigma >= 0 else -1

                    level = EnergyLevel(
                        energy=energy,
                        J=J,
                        parity=parity,
                        state=basis.istate,
                        v=basis.v,
                        lambda_=basis.lambda_,
                        sigma=basis.sigma,
                        omega=basis.omega,
                        spin=basis.spin,
                        name=basis.name
                    )
                    self.energy_levels.append(level)
                else:
                    # Eigenvector dimension mismatch - assign based on energy ordering
                    # For a single state, v increases with energy
                    v_approx = i // 2  # Two spin components per v
                    sigma = 0.5 if (i % 2 == 0) else -0.5
                    if self.states:
                        state = self.states[0]
                        level = EnergyLevel(
                            energy=energy,
                            J=J,
                            parity=1 if sigma >= 0 else -1,
                            state=0,
                            v=v_approx,
                            lambda_=state.lambda_,
                            sigma=sigma,
                            omega=state.lambda_ + sigma,
                            spin=state.spin,
                            name=state.name
                        )
                    else:
                        level = EnergyLevel(energy=energy, J=J, parity=1,
                                           state=0, v=v_approx, lambda_=0,
                                           sigma=sigma, omega=sigma, spin=0.5)
                    self.energy_levels.append(level)
            else:
                self.energy_levels.append(EnergyLevel(energy=energy, J=J, parity=1,
                                                      state=0, v=0, lambda_=0,
                                                      sigma=0, omega=0, spin=0.5))

    def get_energies(self, J: Optional[float] = None,
                     relative: bool = True) -> np.ndarray:
        """
        Get computed energy levels.

        Parameters
        ----------
        J : float, optional
            Get energies for specific J value. If None, return all.
        relative : bool
            If True, return energies relative to ground state.

        Returns
        -------
        energies : ndarray
            Energy values in cm^-1
        """
        if J is not None:
            if J not in self.rovibronic_energies:
                raise ValueError(f"No results for J={J}")
            energies = self.rovibronic_energies[J]
        else:
            energies = np.concatenate(list(self.rovibronic_energies.values()))

        if relative and len(energies) > 0:
            energies = energies - np.min(energies)

        return energies

    def get_levels(self, J: Optional[float] = None) -> List[EnergyLevel]:
        """
        Get computed energy levels with quantum number assignments.

        Parameters
        ----------
        J : float, optional
            Filter by J value

        Returns
        -------
        levels : list
            List of EnergyLevel objects
        """
        if J is not None:
            return [lvl for lvl in self.energy_levels if abs(lvl.J - J) < 0.1]
        return self.energy_levels

    def write_states_file(self, filename: str, J: Optional[float] = None):
        """
        Write energy levels to a .states file (ExoMol format).

        Parameters
        ----------
        filename : str
            Output filename
        J : float, optional
            Filter by J value
        """
        levels = self.get_levels(J)

        with open(filename, 'w') as f:
            f.write("# PyDuo output\n")
            f.write("# i  J  parity  E(cm-1)  State  v  Lambda  Sigma  Omega\n")

            for i, lvl in enumerate(levels):
                f.write(f"{i+1:6d} {lvl.J:6.1f} {lvl.parity:+2d} {lvl.energy:16.6f} "
                        f"{lvl.state:3d} {lvl.v:3d} {lvl.lambda_:3d} "
                        f"{lvl.sigma:6.1f} {lvl.omega:6.1f}\n")

        if self.verbose >= 3:
            print(f"Wrote {len(levels)} levels to {filename}")

    def print_eigenvalues(self, J: Optional[float] = None, relative: bool = True):
        """
        Print eigenvalues in Duo format.

        Parameters
        ----------
        J : float, optional
            Print only for specific J value
        relative : bool
            If True, print energies relative to ZPE
        """
        # Find ZPE
        zpe = min(lvl.energy for lvl in self.energy_levels) if self.energy_levels else 0.0

        j_values = [J] if J is not None else sorted(set(lvl.J for lvl in self.energy_levels))

        for j_val in j_values:
            levels_j = [lvl for lvl in self.energy_levels if abs(lvl.J - j_val) < 0.01]

            # Sort by energy
            levels_j.sort(key=lambda x: x.energy)

            # Separate by parity (parity > 0 = '+', parity < 0 = '-')
            levels_pos = [lvl for lvl in levels_j if lvl.parity > 0]
            levels_neg = [lvl for lvl in levels_j if lvl.parity < 0]

            print(f"\nEigenvalues for J = {j_val:6.1f}")
            print()
            print("       J      i        Energy/cm  State   v  lambda spin   sigma   omega  parity")

            for i, lvl in enumerate(levels_pos[:30], 1):
                e = lvl.energy - zpe if relative else lvl.energy
                parity_str = '+'
                state_name = lvl.name.replace("'", "") if lvl.name else "State"
                print(f"  {j_val:6.1f} {i:4d}  {e:16.6f}   {lvl.state}  {lvl.v:4d}   {lvl.lambda_}  "
                      f"{lvl.spin:4.1f}  {lvl.sigma:6.1f}  {lvl.omega:6.1f}   {parity_str}    ||{state_name}")

            if levels_neg:
                print()
                print("       J      i        Energy/cm  State   v  lambda spin   sigma   omega  parity")
                for i, lvl in enumerate(levels_neg[:30], 1):
                    e = lvl.energy - zpe if relative else lvl.energy
                    parity_str = '-'
                    state_name = lvl.name.replace("'", "") if lvl.name else "State"
                    print(f"  {j_val:6.1f} {i:4d}  {e:16.6f}   {lvl.state}  {lvl.v:4d}   {lvl.lambda_}  "
                          f"{lvl.spin:4.1f}  {lvl.sigma:6.1f}  {lvl.omega:6.1f}   {parity_str}    ||{state_name}")

    def print_summary(self):
        """Print a summary of the calculation."""
        print("\n" + "="*60)
        print("PyDuo Calculation Summary")
        print("="*60)

        print(f"\nGrid: {self.grid.npoints} points, "
              f"r = [{self.grid.rmin:.3f}, {self.grid.rmax:.3f}] Ang")
        print(f"Reduced mass: {self.reduced_mass:.6f} amu")
        print(f"GPU acceleration: {'enabled' if self.use_gpu else 'disabled'}")

        print(f"\nElectronic states: {len(self.states)}")
        for state in self.states:
            print(f"  {state.name}: Lambda={state.lambda_}, "
                  f"Mult={state.multiplicity}, Spin={state.spin}")

        print(f"\nJ values computed: {self.j_list}")
        print(f"Total energy levels: {len(self.energy_levels)}")

        if self.energy_levels:
            min_e = min(lvl.energy for lvl in self.energy_levels)
            max_e = max(lvl.energy for lvl in self.energy_levels)
            print(f"Energy range: {min_e:.4f} to {max_e:.4f} cm^-1 (absolute)")
            print(f"Energy range: 0.0000 to {max_e - min_e:.4f} cm^-1 (relative to ZPE)")

        print("\nTiming:")
        for key, value in self.timings.items():
            print(f"  {key}: {value:.2f} s")

    def print_functions(self, output_file: Optional[str] = None):
        """
        Print evaluated functions (potentials, spin-orbit, L+, etc.) vs r.

        Similar to Duo's check_and_print_field subroutine.

        Parameters
        ----------
        output_file : str, optional
            If provided, write to file instead of stdout
        """
        if self.grid is None:
            print("No grid defined. Load input first.")
            return

        r = self.grid.r
        ngrid = len(r)

        lines = []

        # Print potentials
        lines.append("\n" + "="*70)
        lines.append("Potential Energy Curves (cm^-1)")
        lines.append("="*70)

        # Header
        header = f"{'r (Ang)':>12}"
        for pot in self.potentials:
            header += f"  {pot.name[:18]:>18}"
        lines.append(header)

        # Evaluate and print
        pot_values = []
        for pot in self.potentials:
            V = pot.evaluate(r, use_gpu=False)
            if V is not None and len(V) > 0:
                pot_values.append(V)
            else:
                pot_values.append(np.zeros(ngrid))

        for i in range(ngrid):
            line = f"{r[i]:12.6f}"
            for V in pot_values:
                line += f"  {V[i]:18.8f}"
            lines.append(line)

        # Print spin-orbit couplings if available
        if hasattr(self, 'parser') and self.parser and len(self.parser.spinorbit) > 0:
            lines.append("\n" + "="*70)
            lines.append("Spin-Orbit Couplings (cm^-1)")
            lines.append("="*70)

            header = f"{'r (Ang)':>12}"
            for so in self.parser.spinorbit:
                name = f"SO({so.istate},{so.jstate})"
                header += f"  {name:>18}"
            lines.append(header)

            so_values = []
            for so in self.parser.spinorbit:
                V = so.evaluate(r, use_gpu=False)
                if V is not None and len(V) > 0:
                    so_values.append(V)
                else:
                    so_values.append(np.zeros(ngrid))

            for i in range(ngrid):
                line = f"{r[i]:12.6f}"
                for V in so_values:
                    line += f"  {V[i]:18.8f}"
                lines.append(line)

        # Print L+ couplings if available
        if hasattr(self, 'parser') and self.parser:
            lx_couplings = [c for c in self.parser.couplings
                          if c.field_type.name in ('LXLY', 'LX')]
            if lx_couplings:
                lines.append("\n" + "="*70)
                lines.append("L+ (Lx/Ly) Couplings")
                lines.append("="*70)

                header = f"{'r (Ang)':>12}"
                for lx in lx_couplings:
                    name = f"Lx({lx.istate},{lx.jstate})"
                    header += f"  {name:>18}"
                lines.append(header)

                lx_values = []
                for lx in lx_couplings:
                    V = lx.evaluate(r, use_gpu=False)
                    if V is not None and len(V) > 0:
                        lx_values.append(V)
                    else:
                        lx_values.append(np.zeros(ngrid))

                for i in range(ngrid):
                    line = f"{r[i]:12.6f}"
                    for V in lx_values:
                        line += f"  {V[i]:18.8f}"
                    lines.append(line)

        # Print spin-spin couplings if available
        if hasattr(self, 'parser') and self.parser:
            ss_couplings = [c for c in self.parser.couplings
                          if c.field_type.name == 'SPINSPIN']
            if ss_couplings:
                lines.append("\n" + "="*70)
                lines.append("Spin-Spin Couplings (cm^-1)")
                lines.append("="*70)

                header = f"{'r (Ang)':>12}"
                for ss in ss_couplings:
                    name = f"SS({ss.istate},{ss.jstate})"
                    header += f"  {name:>18}"
                lines.append(header)

                ss_values = []
                for ss in ss_couplings:
                    V = ss.evaluate(r, use_gpu=False)
                    if V is not None and len(V) > 0:
                        ss_values.append(V)
                    else:
                        ss_values.append(np.zeros(ngrid))

                for i in range(ngrid):
                    line = f"{r[i]:12.6f}"
                    for V in ss_values:
                        line += f"  {V[i]:18.8f}"
                    lines.append(line)

        # Print BOB-rot couplings if available
        if hasattr(self, 'parser') and self.parser:
            bobrot_couplings = [c for c in self.parser.couplings
                               if c.field_type.name == 'BOBROT']
            if bobrot_couplings:
                lines.append("\n" + "="*70)
                lines.append("BOB-Rot (Mass-dependent rotational) Couplings")
                lines.append("="*70)

                header = f"{'r (Ang)':>12}"
                for br in bobrot_couplings:
                    name = f"BR({br.istate},{br.jstate})"
                    header += f"  {name:>18}"
                lines.append(header)

                br_values = []
                for br in bobrot_couplings:
                    V = br.evaluate(r, use_gpu=False)
                    if V is not None and len(V) > 0:
                        br_values.append(V)
                    else:
                        br_values.append(np.zeros(ngrid))

                for i in range(ngrid):
                    line = f"{r[i]:12.6f}"
                    for V in br_values:
                        line += f"  {V[i]:18.8f}"
                    lines.append(line)

        # Print BOB-vib couplings if available
        if hasattr(self, 'parser') and self.parser:
            bobvib_couplings = [c for c in self.parser.couplings
                               if c.field_type.name == 'BOBVIB']
            if bobvib_couplings:
                lines.append("\n" + "="*70)
                lines.append("BOB-Vib (Mass-dependent vibrational) Couplings")
                lines.append("="*70)

                header = f"{'r (Ang)':>12}"
                for bv in bobvib_couplings:
                    name = f"BV({bv.istate},{bv.jstate})"
                    header += f"  {name:>18}"
                lines.append(header)

                bv_values = []
                for bv in bobvib_couplings:
                    V = bv.evaluate(r, use_gpu=False)
                    if V is not None and len(V) > 0:
                        bv_values.append(V)
                    else:
                        bv_values.append(np.zeros(ngrid))

                for i in range(ngrid):
                    line = f"{r[i]:12.6f}"
                    for V in bv_values:
                        line += f"  {V[i]:18.8f}"
                    lines.append(line)

        # Output
        output = "\n".join(lines)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(output)
            print(f"Functions written to {output_file}")
        else:
            print(output)

    def print_basis_states(self):
        """Print the contracted basis states with quantum numbers."""
        if not hasattr(self, 'hamiltonian') or self.hamiltonian is None:
            print("No Hamiltonian built yet. Run calculation first.")
            return

        print("\n" + "="*80)
        print("Contracted Basis States")
        print("="*80)
        print(f"{'#':>4} {'istate':>7} {'v':>4} {'lambda':>7} {'sigma':>7} {'omega':>7} {'spin':>6} {'name':>15}")
        print("-"*80)

        for i, state in enumerate(self.hamiltonian.contracted_basis):
            print(f"{i:4d} {state.istate:7d} {state.v:4d} {state.lambda_:7d} "
                  f"{state.sigma:7.1f} {state.omega:7.1f} {state.spin:6.1f} {state.name:>15}")

    def print_hamiltonian(self, J: float = 1.0, threshold: float = 1e-6,
                          output_file: Optional[str] = None):
        """
        Print the Hamiltonian matrix elements for a given J value.

        This produces output similar to Duo's format showing contributions from:
        - RV: Rotational-vibrational (diagonal)
        - SO: Spin-Orbit interaction
        - SS: Spin-Spin interaction
        - JS: J.S interaction (S-uncoupling)
        - LJ: L.J interaction (L-uncoupling)
        - LS: L.S interaction (spin-electronic)
        - BR: BOB-rot correction
        - BV: BOB-vib correction

        Parameters
        ----------
        J : float
            Total angular momentum quantum number (default: 1.0)
        threshold : float
            Minimum absolute value for printing in cm^-1 (default: 1e-6)
        output_file : str, optional
            If provided, write output to this file instead of stdout
        """
        if not hasattr(self, 'hamiltonian') or self.hamiltonian is None:
            print("No Hamiltonian built yet. Run calculation first.")
            return

        self.hamiltonian.print_hamiltonian_matrix(J, threshold=threshold,
                                                   output_file=output_file)

    def print_sigma_lambda_basis(self, J: float = 1.0):
        """
        Print the Sigma-Lambda basis set for a given J value (like Duo).

        Parameters
        ----------
        J : float
            Total angular momentum quantum number (default: 1.0)
        """
        if not hasattr(self, 'hamiltonian') or self.hamiltonian is None:
            print("No Hamiltonian built yet. Run calculation first.")
            return

        return self.hamiltonian.print_sigma_lambda_basis(J)

    def print_hamiltonian_duo_format(self, J: float = 0.0, threshold: float = 1e-6,
                                      output_file: Optional[str] = None):
        """
        Print the Hamiltonian matrix in Duo's exact format.

        Groups output by sigma-lambda level (ilevel), showing one line per level
        with all vibrational coupling contributions, matching Duo's output.

        Parameters
        ----------
        J : float
            Total angular momentum quantum number (default: 0.0)
        threshold : float
            Minimum absolute value for printing in cm^-1 (default: 1e-6)
        output_file : str, optional
            If provided, write output to this file instead of stdout
        add_potmin_offset : float
            Offset to add to diagonal elements to match Duo's energy reference.
            Duo shifts PECs by the ground state minimum (~36800 cm^-1 for CaO).
        """
        if not hasattr(self, 'hamiltonian') or self.hamiltonian is None:
            print("No Hamiltonian built yet. Run calculation first.")
            return

        self.hamiltonian.print_hamiltonian_duo_format(J, threshold=threshold,
                                                       output_file=output_file)


def run_from_input(filename: str, use_gpu: bool = True, verbose: int = 3):
    """
    Convenience function to run a calculation from an input file.

    Parameters
    ----------
    filename : str
        Path to input file
    use_gpu : bool
        Whether to use GPU
    verbose : int
        Verbosity level

    Returns
    -------
    solver : DuoSolver
        The solver object with results
    """
    solver = DuoSolver(use_gpu=use_gpu, verbose=verbose)
    solver.load_input(filename)
    solver.run()
    solver.print_summary()
    return solver


def main():
    """Command-line entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pyduo <input_file>")
        print("\nPyDuo - GPU-accelerated diatomic rovibronic calculations")
        print("\nOptions:")
        print("  --cpu     Force CPU mode (no GPU)")
        print("  --quiet   Reduce output")
        print("  --debug   Maximum output")
        sys.exit(1)

    filename = sys.argv[1]
    use_gpu = '--cpu' not in sys.argv

    if '--quiet' in sys.argv:
        verbose = 1
    elif '--debug' in sys.argv:
        verbose = 5
    else:
        verbose = 3

    try:
        solver = run_from_input(filename, use_gpu=use_gpu, verbose=verbose)

        # Write output
        output_base = filename.rsplit('.', 1)[0]
        solver.write_states_file(f"{output_base}.states")

    except Exception as e:
        print(f"Error: {e}")
        if verbose >= 5:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
