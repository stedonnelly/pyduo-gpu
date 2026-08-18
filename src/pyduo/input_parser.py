"""
Input file parser for PyDuo

Parses Duo-format input files to configure calculations.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from .constants import PhysicalConstants
from .grid import Grid
from .fields import (PotentialField, SpinOrbitField, CouplingField,
                     DipoleMomentField, FieldType, ElectronicState, AbInitioField)


@dataclass
class JobConfig:
    """Configuration for the calculation."""
    molecule: str = "H2"
    contraction: str = "VIB"  # VIB or OMEGA
    symmetry: str = "CS(M)"
    diagonalizer: str = "SYEV"
    nJ: int = 1
    j_list: List[float] = field(default_factory=lambda: [0.0])
    vibmax: Dict[int, int] = field(default_factory=dict)
    vibenermax: Dict[int, float] = field(default_factory=dict)
    nroots: List[int] = field(default_factory=lambda: [10000, 10000, 10000, 10000])
    upper_ener: float = 1e9
    zpe: float = 0.0
    shift_to_zpe: bool = True
    intensity: bool = False
    fitting: bool = False
    # Duo diatom.f90:302 (default .true.) / keywords at 703-709
    assign_v_by_count: bool = True


@dataclass
class GridConfig:
    """Grid configuration."""
    npoints: int = 1000
    rmin: float = 0.5
    rmax: float = 5.0
    grid_type: int = 0
    alpha: float = 0.2
    re: float = 1.0


@dataclass
class MoleculeConfig:
    """Molecule configuration."""
    mass1: float = 1.0  # Mass of atom 1 in amu
    mass2: float = 1.0  # Mass of atom 2 in amu
    symbol1: str = "H"
    symbol2: str = "H"

    @property
    def reduced_mass(self) -> float:
        return self.mass1 * self.mass2 / (self.mass1 + self.mass2)


class InputParser:
    """
    Parse Duo-format input files.

    The input file format uses keyword-based sections:
    - GRID: radial grid definition
    - POTEN: potential energy curves
    - SPIN-ORBIT: spin-orbit coupling
    - DIPOLE: electric dipole moment
    - etc.
    """

    # Atomic masses (amu)
    ATOMIC_MASSES = {
        'H': 1.00782503207, 'D': 2.01410177785, 'T': 3.0160492777,
        'He': 4.00260325415, 'Li': 7.01600455, 'Be': 9.0121822,
        'B': 11.0093054, 'C': 12.0, 'N': 14.0030740048,
        'O': 15.99491461956, 'F': 18.99840322, 'Ne': 19.9924401754,
        'Na': 22.9897692809, 'Mg': 23.9850417, 'Al': 26.98153863,
        'Si': 27.9769265325, 'P': 30.97376163, 'S': 31.97207100,
        'Cl': 34.96885268, 'Ar': 39.9623831225, 'K': 38.96370668,
        'Ca': 39.96259098, 'Sc': 44.9559119, 'Ti': 47.9479463,
        'V': 50.9439595, 'Cr': 51.9405075, 'Mn': 54.9380451,
        'Fe': 55.9349375, 'Co': 58.9331950, 'Ni': 57.9353429,
        'Cu': 62.9295975, 'Zn': 63.9291422, 'Ga': 68.9255736,
        'Ge': 73.9211778, 'As': 74.9215965, 'Se': 79.9165213,
        'Br': 78.9183371, 'Kr': 83.911507, 'Rb': 84.911789738,
        'Sr': 87.9056121, 'Y': 88.9058483, 'Zr': 89.9047044,
    }

    def __init__(self):
        self.job = JobConfig()
        self.grid_config = GridConfig()
        self.molecule = MoleculeConfig()
        self.states: List[ElectronicState] = []
        self.potentials: List[PotentialField] = []
        self.spinorbit: List[SpinOrbitField] = []
        self.couplings: List[CouplingField] = []
        self.dipoles: List[DipoleMomentField] = []
        # Ab initio data for morphing
        self.abinitio: List[AbInitioField] = []

    def parse_file(self, filename: str):
        """Parse a Duo input file."""
        with open(filename, 'r') as f:
            content = f.read()

        self.parse_string(content)

    def parse_string(self, content: str):
        """Parse Duo input from a string."""
        # Remove comments
        lines = []
        for line in content.split('\n'):
            # Remove comments starting with ! or #
            if '!' in line:
                line = line[:line.index('!')]
            if '#' in line:
                line = line[:line.index('#')]
            lines.append(line.strip())

        content = '\n'.join(lines)

        # Parse sections
        self._parse_atoms(content)
        self._parse_grid(content)
        self._parse_jrot(content)
        # Global v-assignment mode keywords (Duo diatom.f90:703-709);
        # default is ASSIGN_V_BY_COUNT (diatom.f90:302)
        if re.search(r'(?i)\bassign_v_by_contribution\b', content):
            self.job.assign_v_by_count = False
        elif re.search(r'(?i)\bassign_v_by_count\b', content):
            self.job.assign_v_by_count = True
        self._parse_abinitio(content)  # Parse ab initio data first
        self._parse_poten(content)
        self._parse_spinorbit(content)
        self._parse_dipole(content)
        self._parse_couplings(content)
        self._parse_contraction(content)
        self._link_morphing_fields()  # Link fields to ab initio data
        self._set_field_parities()  # Set parity_i/parity_j on coupling fields

    def _parse_atoms(self, content: str):
        """Parse atom definitions."""
        # Look for ATOMS or atoms pattern
        pattern = r'(?i)atoms\s+(\S+)\s+(\S+)'
        match = re.search(pattern, content)
        if match:
            atom1 = match.group(1).strip()
            atom2 = match.group(2).strip()

            self.molecule.symbol1 = atom1
            self.molecule.symbol2 = atom2

            # Get masses
            self.molecule.mass1 = self._get_mass(atom1)
            self.molecule.mass2 = self._get_mass(atom2)

        # Look for explicit masses
        pattern = r'(?i)masses?\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, content)
        if match:
            self.molecule.mass1 = float(match.group(1))
            self.molecule.mass2 = float(match.group(2))

    def _get_mass(self, atom: str) -> float:
        """Get atomic mass from symbol."""
        # Check for isotope notation like C-12
        if '-' in atom:
            parts = atom.split('-')
            symbol = parts[0]
            mass_number = int(parts[1])
            # Approximate isotope mass
            return float(mass_number)

        symbol = atom.capitalize()
        return self.ATOMIC_MASSES.get(symbol, 1.0)

    def _parse_grid(self, content: str):
        """Parse grid section."""
        # Find GRID section
        pattern = r'(?i)grid\s*[\n,]'
        match = re.search(pattern, content)
        if not match:
            return

        start = match.end()

        # Find next keyword or end
        end = self._find_section_end(content, start)
        grid_content = content[start:end]

        # Parse grid parameters
        for pattern, attr in [
            (r'(?i)npoints\s+(\d+)', 'npoints'),
            (r'(?i)range\s+([\d.]+)\s*,?\s*([\d.]+)', ('rmin', 'rmax')),
            (r'(?i)type\s+(\d+)', 'grid_type'),
            (r'(?i)alpha\s+([\d.]+)', 'alpha'),
            (r'(?i)re\s+([\d.]+)', 're'),
        ]:
            match = re.search(pattern, grid_content)
            if match:
                if isinstance(attr, tuple):
                    setattr(self.grid_config, attr[0], float(match.group(1)))
                    setattr(self.grid_config, attr[1], float(match.group(2)))
                elif attr == 'npoints' or attr == 'grid_type':
                    setattr(self.grid_config, attr, int(match.group(1)))
                else:
                    setattr(self.grid_config, attr, float(match.group(1)))

    def _parse_jrot(self, content: str):
        """Parse J quantum number list."""
        # Handle both "jrot 0.5 - 2.5" and "jrot 0, 1, 2"
        pattern = r'(?i)jrot\s+([\d.]+)\s*-\s*([\d.]+)'
        match = re.search(pattern, content)
        if match:
            # Range notation with possible half-integers
            j_min = float(match.group(1))
            j_max = float(match.group(2))
            # Generate J values (integer or half-integer depending on j_min)
            j_list = []
            j = j_min
            while j <= j_max + 0.01:  # Small epsilon for float comparison
                j_list.append(j)
                j += 1.0
            self.job.j_list = j_list
            self.job.nJ = len(self.job.j_list)
            return

        # Try comma/space-separated list
        pattern = r'(?i)jrot\s+([\d.,\s]+)'
        match = re.search(pattern, content)
        if match:
            j_str = match.group(1).strip()
            # Comma-separated or space-separated list
            j_list = []
            for item in re.split(r'[,\s]+', j_str):
                item = item.strip()
                if item:
                    try:
                        j_list.append(float(item))
                    except ValueError:
                        pass
            if j_list:
                self.job.j_list = j_list
                self.job.nJ = len(self.job.j_list)

    def _parse_poten(self, content: str):
        """Parse POTEN sections (but not ABINITIO POTEN which are reference data)."""
        # Find all POTEN sections at start of line (not preceded by ABINITIO)
        # Match "poten N" at beginning of line only
        pattern = r'(?im)^\s*poten\s+(\d+)'
        for match in re.finditer(pattern, content):
            # Check if this is an ABINITIO POTEN by looking at the preceding context
            start_pos = match.start()
            # Look back up to 20 chars to see if "abinitio" precedes this
            lookback = content[max(0, start_pos - 20):start_pos].lower()
            if 'abinitio' in lookback:
                continue  # Skip abinitio poten sections

            istate = int(match.group(1))
            start = match.end()
            end = self._find_section_end(content, start)
            poten_content = content[start:end]

            self._parse_single_poten(istate, poten_content)

    def _parse_single_poten(self, istate: int, content: str):
        """Parse a single potential section."""
        # Parse state quantum numbers
        name = f"State_{istate}"
        lambda_ = 0
        multiplicity = 1
        symmetry = ""
        parity_pm = 0  # +1 for +, -1 for -, 0 for unspecified

        # Look for name
        match = re.search(r'(?i)name\s+"?([^"\n]+)"?', content)
        if match:
            name = match.group(1).strip()

        # Look for symmetry (+/-)
        match = re.search(r'(?i)symmetry\s+([+-])', content)
        if match:
            parity_pm = 1 if match.group(1) == '+' else -1

        # Look for lambda (from state name like Sigma, Pi, Delta)
        if 'Sigma' in name or 'SIGMA' in name:
            lambda_ = 0
        elif 'Pi' in name or 'PI' in name:
            lambda_ = 1
        elif 'Delta' in name or 'DELTA' in name:
            lambda_ = 2
        elif 'Phi' in name or 'PHI' in name:
            lambda_ = 3

        # Look for explicit lambda
        match = re.search(r'(?i)lambda\s+(\d+)', content)
        if match:
            lambda_ = int(match.group(1))

        # Look for multiplicity (from name like 1Sigma, 3Pi)
        match = re.search(r'^(\d)', name)
        if match:
            multiplicity = int(match.group(1))

        # Look for explicit mult
        match = re.search(r'(?i)mult(?:iplicity)?\s+(\d+)', content)
        if match:
            multiplicity = int(match.group(1))

        # Parse potential type and parameters
        # Use the LAST type specification (some inputs have "type grid" followed by "type EMO")
        # Also filter out numeric types like "type 0" which refer to grid type, not potential type
        func_type = "MORSE"
        valid_pot_types = ['MORSE', 'EMO', 'MLR', 'GRID', 'POLYNOMIAL', 'DUNHAM', 'SPF',
                          'CHEBYSHEV', 'BOBLEROY', 'MARQUARDT', 'MLJ', 'COSH', 'SIGMOID']
        for match in re.finditer(r'(?i)type\s+(\S+)', content):
            pot_type = match.group(1).upper()
            # Skip numeric types (grid type specification) and keep potential function types
            if pot_type in valid_pot_types:
                func_type = pot_type

        # Parse values - handle both parameter lists and r,V grid pairs
        values = []
        r_grid = None
        v_grid = None

        match = re.search(r'(?i)values', content)
        if match:
            values_start = match.end()
            values_end = self._find_keyword_end(content, values_start)
            values_content = content[values_start:values_end]

            # Use appropriate parser based on potential type
            if func_type == "EMO":
                # EMO uses named parameters (V0, RE, DE, RREF, PL, PR, NL, NR, B0, B1, ...)
                values = self._extract_emo_parameters(values_content)
            elif func_type == "GRID":
                # GRID type: values are r,V pairs (two columns)
                raw_values = self._extract_numbers(values_content)
                if len(raw_values) >= 2:
                    r_grid = np.array(raw_values[0::2])  # Every other value starting at 0
                    v_grid = np.array(raw_values[1::2])  # Every other value starting at 1
                    values = []  # Clear parameters since we use grid
            else:
                # Other types: extract all numbers
                values = self._extract_numbers(values_content)

        # Create electronic state
        spin = (multiplicity - 1) / 2.0
        state = ElectronicState(
            name=name,
            istate=istate,
            lambda_=lambda_,
            multiplicity=multiplicity,
            spin=spin,
            parity_pm=parity_pm
        )
        self.states.append(state)

        # Create potential field
        poten = PotentialField(
            name=name,
            istate=istate,
            lambda_=lambda_,
            multiplicity=multiplicity,
            func_type=func_type,
            parameters=np.array(values) if values else None,
            grid_r=r_grid,
            grid_v=v_grid
        )
        self.potentials.append(poten)

    def _parse_spinorbit(self, content: str):
        """Parse SPIN-ORBIT sections (including spin-orbit-x variant)."""
        pattern = r'(?i)spin-orbit(?:-x)?\s+(\d+)\s+(\d+)'
        for match in re.finditer(pattern, content):
            # Skip abinitio spin-orbit sections
            start_pos = match.start()
            lookback = content[max(0, start_pos - 20):start_pos].lower()
            if 'abinitio' in lookback:
                continue

            # Skip matches inside comments (parentheses)
            # Check if we're inside parentheses by counting ( and ) before this position
            prefix = content[:start_pos]
            open_parens = prefix.count('(') - prefix.count(')')
            if open_parens > 0:
                continue

            istate = int(match.group(1))
            jstate = int(match.group(2))
            start = match.end()
            end = self._find_section_end(content, start)
            so_content = content[start:end]

            self._parse_single_spinorbit(istate, jstate, so_content)

    def _parse_single_spinorbit(self, istate: int, jstate: int, content: str):
        """Parse a single spin-orbit section."""
        # Get quantum numbers
        sigma_i = 0.0
        sigma_j = 0.0
        lambda_i = 0
        lambda_j = 0

        match = re.search(r'(?i)sigma\s+([\d.+-]+)\s+([\d.+-]+)', content)
        if match:
            sigma_i = float(match.group(1))
            sigma_j = float(match.group(2))

        match = re.search(r'(?i)lambda\s+([\d+-]+)\s+([\d+-]+)', content)
        if match:
            lambda_i = int(match.group(1))
            lambda_j = int(match.group(2))

        # Parse type and values
        func_type = "GRID"
        match = re.search(r'(?i)type\s+(\S+)', content)
        if match:
            func_type = match.group(1).upper()

        # Check for morphing keyword
        morphing = bool(re.search(r'(?i)\bmorphing\b|\bmorph\b', content))

        # Parse factor (1, i, -i, sqrt(2), etc.)
        factor = 1.0
        complex_f = complex(1.0, 0.0)
        factor_match = re.search(r'(?i)factor\s+([\S]+)', content)
        if factor_match:
            factor_str = factor_match.group(1).lower()
            factor, complex_f = self._parse_factor(factor_str)

        # Parse <x|Lz|y> for MOLPRO representation
        # Format: <x|Lz|y>  ix_lz_y  jx_lz_y
        # Values are 0, 1, -1, i, -i
        ix_lz_y = 0
        jx_lz_y = 0
        xlzy_match = re.search(r'(?i)<x\|Lz\|y>\s+(\S+)\s+(\S+)', content)
        if xlzy_match:
            ix_lz_y = self._parse_xlzy_value(xlzy_match.group(1))
            jx_lz_y = self._parse_xlzy_value(xlzy_match.group(2))

        values = []
        match = re.search(r'(?i)values', content)
        if match:
            values_start = match.end()
            values_end = self._find_keyword_end(content, values_start)
            values_content = content[values_start:values_end]

            # Use appropriate extraction method based on type
            if func_type in ['BOBLEROY', 'BOB', 'SURKUS']:
                values = self._extract_bobleroy_parameters(values_content)
            else:
                values = self._extract_numbers(values_content)

        # Get spins from states
        spin_i = 0.0
        spin_j = 0.0
        for s in self.states:
            if s.istate == istate:
                spin_i = s.spin
            if s.istate == jstate:
                spin_j = s.spin

        so = SpinOrbitField(
            name=f"SO_{istate}_{jstate}",
            istate=istate,
            jstate=jstate,
            lambda_i=lambda_i,
            lambda_j=lambda_j,
            spin_i=spin_i,
            spin_j=spin_j,
            sigma_i=sigma_i,
            sigma_j=sigma_j,
            func_type=func_type,
            parameters=np.array(values) if values else None
        )
        so.morphing = morphing
        so.factor = factor
        so.complex_f = complex_f
        so.ix_lz_y = ix_lz_y
        so.jx_lz_y = jx_lz_y
        self.spinorbit.append(so)

    def _parse_factor(self, factor_str: str) -> tuple:
        """
        Parse factor string to (real_factor, complex_f) tuple.

        Returns:
            tuple: (factor: float, complex_f: complex)
                - factor: real scaling factor applied to gridvalue
                - complex_f: complex factor for molpro_duo transformation

        Supported formats:
        - 1, -1, 0 -> (value, value+0j)
        - i, -i -> (1.0, 0+1j or 0-1j)
        - sqrt(2), -sqrt(2) -> (sqrt(2), sqrt(2)+0j)
        - i*sqrt(2), -i*sqrt(2) -> (1.0, 0+sqrt(2)j or 0-sqrt(2)j)

        Duo stores these separately:
        - field%factor: real scaling multiplied into gridvalue
        - field%complex_f: complex factor used in molpro_duo transformation
        """
        factor_str = factor_str.strip().lower()

        # Handle pure imaginary cases
        if factor_str == 'i':
            return (1.0, complex(0, 1))
        elif factor_str == '-i':
            return (1.0, complex(0, -1))

        # Handle i*something or -i*something
        if factor_str.startswith('i*'):
            rest = factor_str[2:]
            real_val = self._parse_real_factor(rest)
            return (1.0, complex(0, real_val))
        elif factor_str.startswith('-i*'):
            rest = factor_str[3:]
            real_val = self._parse_real_factor(rest)
            return (1.0, complex(0, -real_val))

        # Pure real factor
        real_val = self._parse_real_factor(factor_str)
        return (real_val, complex(real_val, 0))

    def _parse_real_factor(self, factor_str: str) -> float:
        """Parse a real-valued factor string."""
        factor_str = factor_str.strip().lower()

        # Handle sqrt expressions
        sqrt_match = re.search(r'(-?)sqrt\(([\d.]+)\)', factor_str)
        if sqrt_match:
            sign = -1.0 if sqrt_match.group(1) == '-' else 1.0
            value = float(sqrt_match.group(2))
            return sign * np.sqrt(value)

        # Try to parse as float
        try:
            return float(factor_str)
        except ValueError:
            return 1.0

    def _parse_xlzy_value(self, value_str: str) -> int:
        """
        Parse <x|Lz|y> value (ix_lz_y or jx_lz_y).

        The value represents the imaginary part of the <x|Lz|y> matrix element:
        - 0: Lambda = 0 (Sigma state)
        - i or 1: Lambda != 0, positive imaginary
        - -i or -1: Lambda != 0, negative imaginary

        Returns integer: 0, 1, or -1
        """
        value_str = value_str.strip().lower()

        if value_str == '0':
            return 0
        elif value_str in ['i', '1', '+i', '+1']:
            return 1
        elif value_str in ['-i', '-1']:
            return -1
        else:
            # Try to parse as integer
            try:
                val = int(float(value_str))
                if val == 0:
                    return 0
                return 1 if val > 0 else -1
            except ValueError:
                return 0

    def _parse_dipole(self, content: str):
        """Parse DIPOLE sections."""
        pattern = r'(?i)dipole\s+(\d+)\s+(\d+)'
        for match in re.finditer(pattern, content):
            istate = int(match.group(1))
            jstate = int(match.group(2))
            start = match.end()
            end = self._find_section_end(content, start)
            dip_content = content[start:end]

            self._parse_single_dipole(istate, jstate, dip_content)

    def _parse_single_dipole(self, istate: int, jstate: int, content: str):
        """Parse a single dipole section."""
        func_type = "GRID"
        match = re.search(r'(?i)type\s+(\S+)', content)
        if match:
            func_type = match.group(1).upper()

        values = []
        match = re.search(r'(?i)values', content)
        if match:
            values_start = match.end()
            values_end = self._find_keyword_end(content, values_start)
            values = self._extract_numbers(content[values_start:values_end])

        dipole = DipoleMomentField(
            name=f"DM_{istate}_{jstate}",
            istate=istate,
            jstate=jstate,
            func_type=func_type,
            parameters=np.array(values) if values else None
        )
        self.dipoles.append(dipole)

    def _parse_couplings(self, content: str):
        """Parse various coupling sections."""
        coupling_types = [
            ('L2', FieldType.L2),
            (r'L\+', FieldType.LXLY),
            ('Lx', FieldType.LXLY),  # Lx is same as L+ (L-uncoupling)
            ('LXLY', FieldType.LXLY),
            ('SPIN-SPIN', FieldType.SPINSPIN),
            ('SPINSPIN', FieldType.SPINSPIN),  # Alias without hyphen
            ('SPIN-ROT', FieldType.SPINROT),
            ('SPINROT', FieldType.SPINROT),  # Alias without hyphen
            ('BOB-ROT', FieldType.BOBROT),
            ('BOBROT', FieldType.BOBROT),  # Alias without hyphen
            ('BOB-VIB', FieldType.BOBVIB),
            ('BOBVIB', FieldType.BOBVIB),  # Alias without hyphen
            ('DIABATIC', FieldType.DIABATIC),
            ('NAC', FieldType.NAC),
            ('LAMBDA-OPQ', FieldType.LAMBDAOPQ),
            ('LAMBDA-P2Q', FieldType.LAMBDAP2Q),
            ('LAMBDA-Q', FieldType.LAMBDAQ),
        ]

        for keyword, field_type in coupling_types:
            pattern = rf'(?i){keyword}\s+(\d+)\s+(\d+)'
            for match in re.finditer(pattern, content):
                # Skip abinitio coupling sections
                start_pos = match.start()
                lookback = content[max(0, start_pos - 20):start_pos].lower()
                if 'abinitio' in lookback:
                    continue

                # Skip matches inside comments (parentheses)
                prefix = content[:start_pos]
                open_parens = prefix.count('(') - prefix.count(')')
                if open_parens > 0:
                    continue

                istate = int(match.group(1))
                jstate = int(match.group(2))
                start = match.end()
                end = self._find_section_end(content, start)
                coupling_content = content[start:end]

                self._parse_single_coupling(istate, jstate, field_type, keyword, coupling_content)

    def _parse_single_coupling(self, istate: int, jstate: int, field_type: FieldType,
                               name: str, content: str):
        """Parse a single coupling section."""
        # Quantum numbers (lambda i j / sigma i j / spin i j lines), exactly as
        # in _parse_single_spinorbit. Duo couplings declare these per section;
        # fall back to the electronic-state values when a line is absent.
        # Without this, every coupling kept the dataclass defaults
        # lambda_i = lambda_j = 0 and the batched L+ selection rule
        # (hamiltonian.py lambda_match) never matched a real |dLambda|=1
        # coupling -- Lx fields were silently dropped.  [audit H2]
        lambda_i = None
        lambda_j = None
        match = re.search(r'(?i)lambda\s+([\d+-]+)\s+([\d+-]+)', content)
        if match:
            lambda_i = int(match.group(1))
            lambda_j = int(match.group(2))

        sigma_i = 0.0
        sigma_j = 0.0
        match = re.search(r'(?i)sigma\s+([\d.+-]+)\s+([\d.+-]+)', content)
        if match:
            sigma_i = float(match.group(1))
            sigma_j = float(match.group(2))

        spin_i = None
        spin_j = None
        match = re.search(r'(?i)spin\s+([\d.+-]+)\s+([\d.+-]+)', content)
        if match:
            spin_i = float(match.group(1))
            spin_j = float(match.group(2))

        # Fall back to the electronic states' lambda/spin when not declared
        for s in self.states:
            if s.istate == istate:
                if lambda_i is None:
                    lambda_i = s.lambda_
                if spin_i is None:
                    spin_i = s.spin
            if s.istate == jstate:
                if lambda_j is None:
                    lambda_j = s.lambda_
                if spin_j is None:
                    spin_j = s.spin
        lambda_i = 0 if lambda_i is None else lambda_i
        lambda_j = 0 if lambda_j is None else lambda_j
        spin_i = 0.0 if spin_i is None else spin_i
        spin_j = 0.0 if spin_j is None else spin_j

        func_type = "GRID"
        match = re.search(r'(?i)type\s+(\S+)', content)
        if match:
            func_type = match.group(1).upper()

        # Check for morphing keyword
        morphing = bool(re.search(r'(?i)\bmorphing\b|\bmorph\b', content))

        # Parse factor
        factor = 1.0
        complex_f = complex(1.0, 0.0)
        factor_match = re.search(r'(?i)factor\s+([\S]+)', content)
        if factor_match:
            factor_str = factor_match.group(1).lower()
            factor, complex_f = self._parse_factor(factor_str)

        # Parse <x|Lz|y> for MOLPRO representation
        ix_lz_y = 0
        jx_lz_y = 0
        xlzy_match = re.search(r'(?i)<x\|Lz\|y>\s+(\S+)\s+(\S+)', content)
        if xlzy_match:
            ix_lz_y = self._parse_xlzy_value(xlzy_match.group(1))
            jx_lz_y = self._parse_xlzy_value(xlzy_match.group(2))

        values = []
        match = re.search(r'(?i)values', content)
        if match:
            values_start = match.end()
            values_end = self._find_keyword_end(content, values_start)
            values_content = content[values_start:values_end]

            # Use appropriate extraction method based on type
            if func_type in ['BOBLEROY', 'BOB', 'SURKUS']:
                values = self._extract_bobleroy_parameters(values_content)
            else:
                values = self._extract_numbers(values_content)

        coupling = CouplingField(
            name=f"{name}_{istate}_{jstate}",
            field_type=field_type,
            istate=istate,
            jstate=jstate,
            lambda_i=lambda_i,
            lambda_j=lambda_j,
            sigma_i=sigma_i,
            sigma_j=sigma_j,
            spin_i=spin_i,
            spin_j=spin_j,
            func_type=func_type,
            parameters=np.array(values) if values else None
        )
        # An LXLY/L+ coupling must connect |dLambda| = 1 states; a mis-parsed
        # section would otherwise silently generate no (or fake) couplings.
        if field_type == FieldType.LXLY and abs(lambda_i - lambda_j) != 1:
            import warnings
            warnings.warn(
                f"LXLY coupling {istate}-{jstate}: |lambda_i - lambda_j| != 1 "
                f"(got {lambda_i}, {lambda_j}) -- check the 'lambda i j' line "
                f"in the input section")
        coupling.morphing = morphing
        coupling.factor = factor
        coupling.complex_f = complex_f
        coupling.ix_lz_y = ix_lz_y
        coupling.jx_lz_y = jx_lz_y
        self.couplings.append(coupling)

    def _parse_contraction(self, content: str):
        """Parse contraction settings."""
        # Find contraction section
        pattern = r'(?i)contraction'
        match = re.search(pattern, content)
        if not match:
            return

        start = match.end()
        end = self._find_section_end(content, start)
        contraction_content = content[start:end]

        # Parse contraction type
        type_match = re.search(r'(?i)(vib|omega)', contraction_content)
        if type_match:
            self.job.contraction = type_match.group(1).upper()

        # Parse VIBMAX per state
        vibmax_pattern = r'(?i)vibmax\s+([\d\s,]+)'
        match = re.search(vibmax_pattern, contraction_content)
        if match:
            values = self._extract_numbers(match.group(1))
            for i, v in enumerate(values):
                self.job.vibmax[i + 1] = int(v)

        # Parse VMAX (can be single value or multiple values like "vmax 80 50 50 50 50")
        # Duo behavior: each value fills forward to all subsequent states
        # So "vmax 80 50" with 5 states -> [80, 50, 50, 50, 50]
        vmax_pattern = r'(?i)vmax\s+([\d\s]+)'
        match = re.search(vmax_pattern, contraction_content)
        if match:
            values = self._extract_numbers(match.group(1))
            nstates = len(self.states)
            last_vmax = int(values[0]) if values else 50  # Default
            for i in range(nstates):
                if i < len(values):
                    last_vmax = int(values[i])
                self.job.vibmax[i + 1] = last_vmax  # Fill forward

        # Parse ENERMAX (energy cutoff for vibrational states per electronic state)
        # Same fill-forward behavior as vmax
        # Format: "enermax 50000.0" or "enermax 50000.0 40000.0 40000.0 ..."
        enermax_pattern = r'(?i)enermax\s+([\d\.\s\+\-eE]+)'
        match = re.search(enermax_pattern, contraction_content)
        if match:
            values = self._extract_numbers(match.group(1))
            nstates = len(self.states)
            last_enermax = float(values[0]) if values else 1e9  # Default (very large)
            for i in range(nstates):
                if i < len(values):
                    last_enermax = float(values[i])
                self.job.vibenermax[i + 1] = last_enermax  # Fill forward

    def _find_section_end(self, content: str, start: int) -> int:
        """Find the end of a section (next keyword or END)."""
        # Keywords that start new sections - must be at start of line (after optional whitespace)
        keywords = [
            'POTEN', 'ABINITIO', 'SPIN-ORBIT', 'SPIN-ORBIT-X', 'DIPOLE', 'L2', 'L\\+', 'Lx', 'LXLY', 'SPIN-SPIN',
            'SPIN-ROT', 'BOB-ROT', 'BOB-VIB', 'DIABATIC', 'NAC', 'JROT',
            'LAMBDA-OPQ', 'LAMBDA-P2Q', 'LAMBDA-Q',
            'CONTRACTION', 'FITTING', 'INTENSITY', 'FREQUENCY', 'CHECK_POINT', 'ASSIGN'
        ]
        # 'GRID' and 'END' need special handling - GRID is also a type value

        next_pos = len(content)

        # Look for section-starting keywords (at start of line)
        for kw in keywords:
            # Match keyword at start of line (after optional whitespace)
            pattern = rf'(?im)^\s*{kw}\b'
            for match in re.finditer(pattern, content[start:]):
                pos = start + match.start()
                if pos < next_pos:
                    next_pos = pos
                break

        # Look for END at start of line (signals end of current section)
        end_pattern = r'(?im)^\s*END\b'
        for match in re.finditer(end_pattern, content[start:]):
            pos = start + match.start()
            if pos < next_pos:
                next_pos = pos
            break

        # Look for GRID section (not "type grid") - GRID followed by newline or npoints
        grid_pattern = r'(?im)^\s*GRID\s*[\n,]'
        for match in re.finditer(grid_pattern, content[start:]):
            pos = start + match.start()
            if pos < next_pos:
                next_pos = pos
            break

        return next_pos

    def _find_keyword_end(self, content: str, start: int) -> int:
        """Find end of values block (next keyword)."""
        keywords = ['END', 'TYPE', 'NAME', 'MULT', 'LAMBDA', 'SIGMA', 'SPIN',
                    'POTEN', 'SPIN-ORBIT', 'DIPOLE', 'FITTING']

        next_pos = len(content)
        for kw in keywords:
            pattern = rf'(?i)\b{kw}\b'
            for match in re.finditer(pattern, content[start:]):
                pos = start + match.start()
                if pos < next_pos:
                    next_pos = pos
                break

        return next_pos

    def _extract_numbers(self, content: str) -> List[float]:
        """Extract all numbers from a string."""
        # Match integers and floats, including scientific notation
        pattern = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?'
        matches = re.findall(pattern, content)
        return [float(m.replace('D', 'E').replace('d', 'e')) for m in matches]

    def _extract_emo_parameters(self, content: str) -> List[float]:
        """
        Extract EMO potential parameters from named format.

        EMO format:
        V0    value  [fit]
        RE    value  [fit]
        DE    value  [link ...]
        RREF  value
        PL    value
        PR    value
        NL    value
        NR    value
        B0    value  [fit]
        B1    value  [fit]
        ...

        Returns parameters in order: [V0, RE, DE, RREF, PL, PR, NL, NR, B0, B1, ...]
        """
        params = {}

        # Parse named parameters - each line has: NAME value [optional stuff]
        # We need to extract the FIRST number after each parameter name
        param_names = ['V0', 'RE', 'DE', 'RREF', 'PL', 'PR', 'NL', 'NR']
        beta_pattern = r'B(\d+)'

        lines = content.strip().split('\n')
        beta_coeffs = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for named parameters
            for name in param_names:
                pattern = rf'(?i)^\s*{name}\s+([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)'
                match = re.match(pattern, line)
                if match:
                    params[name.upper()] = float(match.group(1).replace('D', 'E').replace('d', 'e'))
                    break

            # Check for beta coefficients (B0, B1, B2, ...)
            match = re.match(rf'(?i)^\s*B(\d+)\s+([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)', line)
            if match:
                idx = int(match.group(1))
                val = float(match.group(2).replace('D', 'E').replace('d', 'e'))
                beta_coeffs[idx] = val

        # Build parameter array in expected order
        result = []

        # Required parameters (with defaults if not found)
        result.append(params.get('V0', 0.0))
        result.append(params.get('RE', 1.5))
        result.append(params.get('DE', 50000.0))
        result.append(params.get('RREF', -1.0))
        result.append(params.get('PL', 4.0))
        result.append(params.get('PR', 4.0))
        result.append(params.get('NL', 2.0))
        result.append(params.get('NR', 4.0))

        # Beta coefficients in order
        if beta_coeffs:
            max_beta = max(beta_coeffs.keys())
            for i in range(max_beta + 1):
                result.append(beta_coeffs.get(i, 0.0))

        return result

    def _extract_bobleroy_parameters(self, content: str) -> List[float]:
        """
        Extract BOBLEROY parameters from named format.

        BOBLEROY format (for spin-orbit and other couplings):
        RE      value  [fit]
        RREF    value
        P       value
        NT      value
        B0      value  [fit]
        B1      value  [fit]
        ...
        BINF    value

        Returns parameters in order: [RE, P, B0, B1, ..., Binf]
        The format for evaluation is: RE, p-power, B0, B1, ...
        """
        params = {}

        lines = content.strip().split('\n')
        beta_coeffs = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for RE
            match = re.match(r'(?i)^\s*RE\s+([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)', line)
            if match:
                params['RE'] = float(match.group(1).replace('D', 'E').replace('d', 'e'))
                continue

            # Check for RREF
            match = re.match(r'(?i)^\s*RREF\s+([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)', line)
            if match:
                params['RREF'] = float(match.group(1).replace('D', 'E').replace('d', 'e'))
                continue

            # Check for P (power)
            match = re.match(r'(?i)^\s*P\s+([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)', line)
            if match:
                params['P'] = float(match.group(1).replace('D', 'E').replace('d', 'e'))
                continue

            # Check for NT (number of terms - not used but parsed)
            match = re.match(r'(?i)^\s*NT\s+([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)', line)
            if match:
                params['NT'] = float(match.group(1).replace('D', 'E').replace('d', 'e'))
                continue

            # Check for BINF (value at infinity)
            match = re.match(r'(?i)^\s*BINF\s+([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)', line)
            if match:
                params['BINF'] = float(match.group(1).replace('D', 'E').replace('d', 'e'))
                continue

            # Check for beta coefficients (B0, B1, B2, ...)
            match = re.match(r'(?i)^\s*B(\d+)\s+([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)', line)
            if match:
                idx = int(match.group(1))
                val = float(match.group(2).replace('D', 'E').replace('d', 'e'))
                beta_coeffs[idx] = val

        # Build parameter array in expected order for BOB_LEROY evaluation
        # Format: [RE, P, B0, B1, B2, ..., BINF]
        # Fortran format: r0=params(1), rref=params(2), p=params(3), coeffs start at params(5), tinf at end
        result = []

        result.append(params.get('RE', 1.8))  # Reference distance (r0 in Fortran)
        result.append(params.get('P', 2.0))    # Power in z = (r^p - r0^p)/(r^p + r0^p)

        # Beta coefficients in order
        if beta_coeffs:
            max_beta = max(beta_coeffs.keys())
            for i in range(max_beta + 1):
                result.append(beta_coeffs.get(i, 0.0))

        # BINF (value at infinity) - this is required for the (1-z)*t + z*tinf formula
        result.append(params.get('BINF', 0.0))

        return result

    def build_grid(self, use_gpu: bool = True) -> Grid:
        """Build the grid from parsed configuration."""
        return Grid(
            npoints=self.grid_config.npoints,
            rmin=self.grid_config.rmin,
            rmax=self.grid_config.rmax,
            grid_type=self.grid_config.grid_type,
            alpha=self.grid_config.alpha,
            re=self.grid_config.re,
            use_gpu=use_gpu
        )

    def _parse_abinitio(self, content: str):
        """
        Parse ABINITIO sections for spin-orbit and other couplings.

        These sections provide ab initio data for morphing analytic functions.
        """
        # Parse abinitio spin-orbit sections
        pattern = r'(?i)abinitio\s+spin-orbit(?:-x)?\s+(\d+)\s+(\d+)'
        for match in re.finditer(pattern, content):
            istate = int(match.group(1))
            jstate = int(match.group(2))
            start = match.end()
            end = self._find_section_end(content, start)
            ab_content = content[start:end]

            self._parse_single_abinitio_spinorbit(istate, jstate, ab_content)

        # Parse abinitio Lx (L+) sections
        pattern = r'(?i)abinitio\s+Lx\s+(\d+)\s+(\d+)'
        for match in re.finditer(pattern, content):
            istate = int(match.group(1))
            jstate = int(match.group(2))
            start = match.end()
            end = self._find_section_end(content, start)
            ab_content = content[start:end]

            self._parse_single_abinitio_coupling(istate, jstate, FieldType.LXLY, ab_content)

        # Parse abinitio spin-spin sections
        pattern = r'(?i)abinitio\s+spin-?spin\s+(\d+)\s+(\d+)'
        for match in re.finditer(pattern, content):
            istate = int(match.group(1))
            jstate = int(match.group(2))
            start = match.end()
            end = self._find_section_end(content, start)
            ab_content = content[start:end]

            self._parse_single_abinitio_coupling(istate, jstate, FieldType.SPINSPIN, ab_content)

        # Parse abinitio bob-rot / bobrot sections
        pattern = r'(?i)abinitio\s+bob-?rot\s+(\d+)\s+(\d+)'
        for match in re.finditer(pattern, content):
            istate = int(match.group(1))
            jstate = int(match.group(2))
            start = match.end()
            end = self._find_section_end(content, start)
            ab_content = content[start:end]

            self._parse_single_abinitio_coupling(istate, jstate, FieldType.BOBROT, ab_content)

        # Parse abinitio bob-vib / bobvib sections
        pattern = r'(?i)abinitio\s+bob-?vib\s+(\d+)\s+(\d+)'
        for match in re.finditer(pattern, content):
            istate = int(match.group(1))
            jstate = int(match.group(2))
            start = match.end()
            end = self._find_section_end(content, start)
            ab_content = content[start:end]

            self._parse_single_abinitio_coupling(istate, jstate, FieldType.BOBVIB, ab_content)

    def _parse_single_abinitio_spinorbit(self, istate: int, jstate: int, content: str):
        """Parse a single ABINITIO SPIN-ORBIT section."""
        # Extract name
        name_match = re.search(r'(?i)name\s+["\']?([^"\'\n]+)["\']?', content)
        name = name_match.group(1).strip() if name_match else f"abinitio_SO_{istate}_{jstate}"

        # Extract quantum numbers
        spin_i, spin_j = 0.0, 0.0
        lambda_i, lambda_j = 0, 0
        sigma_i, sigma_j = 0.0, 0.0

        spin_match = re.search(r'(?i)spin\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)', content)
        if spin_match:
            spin_i = float(spin_match.group(1))
            spin_j = float(spin_match.group(2))

        lambda_match = re.search(r'(?i)lambda\s+([-+]?\d+)\s+([-+]?\d+)', content)
        if lambda_match:
            lambda_i = int(lambda_match.group(1))
            lambda_j = int(lambda_match.group(2))

        sigma_match = re.search(r'(?i)sigma\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)', content)
        if sigma_match:
            sigma_i = float(sigma_match.group(1))
            sigma_j = float(sigma_match.group(2))

        # Parse <x|Lz|y> for MOLPRO representation
        ix_lz_y = 0
        jx_lz_y = 0
        xlzy_match = re.search(r'(?i)<x\|Lz\|y>\s+(\S+)\s+(\S+)', content)
        if xlzy_match:
            ix_lz_y = self._parse_xlzy_value(xlzy_match.group(1))
            jx_lz_y = self._parse_xlzy_value(xlzy_match.group(2))

        # Extract units (default: bohr for r, cm-1 for values)
        units_r = "bohr"
        units_v = "cm-1"
        units_match = re.search(r'(?i)units\s+(\S+)\s+(\S+)', content)
        if units_match:
            units_r = units_match.group(1).lower()
            units_v = units_match.group(2).lower()

        # Extract grid values
        r_values = []
        v_values = []
        values_match = re.search(r'(?i)values\s*\n([\s\S]*?)(?:end|$)', content)
        if values_match:
            values_content = values_match.group(1)
            for line in values_content.strip().split('\n'):
                line = line.strip()
                if not line or line.lower() == 'end':
                    break
                numbers = self._extract_numbers(line)
                if len(numbers) >= 2:
                    r_values.append(numbers[0])
                    v_values.append(numbers[1])

        if r_values and v_values:
            ab_field = AbInitioField(
                name=name,
                field_type=FieldType.SPINORBIT,
                istate=istate,
                jstate=jstate,
                lambda_i=lambda_i,
                lambda_j=lambda_j,
                spin_i=spin_i,
                spin_j=spin_j,
                sigma_i=sigma_i,
                sigma_j=sigma_j,
                grid_r=np.array(r_values),
                grid_values=np.array(v_values),
                units_r=units_r,
                units_v=units_v
            )
            # Store <x|Lz|y> values for MOLPRO representation
            ab_field.ix_lz_y = ix_lz_y
            ab_field.jx_lz_y = jx_lz_y
            self.abinitio.append(ab_field)

    def _parse_single_abinitio_coupling(self, istate: int, jstate: int,
                                         field_type: FieldType, content: str):
        """Parse a single ABINITIO coupling section (Lx, etc.)."""
        name_match = re.search(r'(?i)name\s+["\']?([^"\'\n]+)["\']?', content)
        name = name_match.group(1).strip() if name_match else f"abinitio_{field_type.name}_{istate}_{jstate}"

        # Extract quantum numbers
        spin_i, spin_j = 0.0, 0.0
        lambda_i, lambda_j = 0, 0

        spin_match = re.search(r'(?i)spin\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)', content)
        if spin_match:
            spin_i = float(spin_match.group(1))
            spin_j = float(spin_match.group(2))

        lambda_match = re.search(r'(?i)lambda\s+([-+]?\d+)\s+([-+]?\d+)', content)
        if lambda_match:
            lambda_i = int(lambda_match.group(1))
            lambda_j = int(lambda_match.group(2))

        # Parse <x|Lz|y> for MOLPRO representation
        ix_lz_y = 0
        jx_lz_y = 0
        xlzy_match = re.search(r'(?i)<x\|Lz\|y>\s+(\S+)\s+(\S+)', content)
        if xlzy_match:
            ix_lz_y = self._parse_xlzy_value(xlzy_match.group(1))
            jx_lz_y = self._parse_xlzy_value(xlzy_match.group(2))

        # Extract units
        units_r = "bohr"
        units_v = "cm-1"
        units_match = re.search(r'(?i)units\s+(\S+)\s+(\S+)', content)
        if units_match:
            units_r = units_match.group(1).lower()
            units_v = units_match.group(2).lower()

        # Extract grid values
        r_values = []
        v_values = []
        values_match = re.search(r'(?i)values\s*\n([\s\S]*?)(?:end|$)', content)
        if values_match:
            values_content = values_match.group(1)
            for line in values_content.strip().split('\n'):
                line = line.strip()
                if not line or line.lower() == 'end':
                    break
                numbers = self._extract_numbers(line)
                if len(numbers) >= 2:
                    r_values.append(numbers[0])
                    v_values.append(numbers[1])

        if r_values and v_values:
            ab_field = AbInitioField(
                name=name,
                field_type=field_type,
                istate=istate,
                jstate=jstate,
                lambda_i=lambda_i,
                lambda_j=lambda_j,
                spin_i=spin_i,
                spin_j=spin_j,
                grid_r=np.array(r_values),
                grid_values=np.array(v_values),
                units_r=units_r,
                units_v=units_v
            )
            # Store <x|Lz|y> values for MOLPRO representation
            ab_field.ix_lz_y = ix_lz_y
            ab_field.jx_lz_y = jx_lz_y
            self.abinitio.append(ab_field)

    def _link_morphing_fields(self):
        """
        Link analytic fields to their ab initio counterparts for morphing.

        This matches fields by:
        1. Field type (SPINORBIT, LXLY, etc.)
        2. State indices (istate, jstate)

        When a match is found, sets the field's morphing flag and abinitio_field reference.
        """
        # Link spin-orbit fields
        for so_field in self.spinorbit:
            if so_field.morphing:
                # Find matching ab initio field
                for ab_field in self.abinitio:
                    if (ab_field.field_type == FieldType.SPINORBIT and
                        ab_field.istate == so_field.istate and
                        ab_field.jstate == so_field.jstate):
                        so_field.abinitio_field = ab_field
                        break

        # Link other coupling fields
        for coupling in self.couplings:
            if coupling.morphing:
                for ab_field in self.abinitio:
                    if (ab_field.field_type == coupling.field_type and
                        ab_field.istate == coupling.istate and
                        ab_field.jstate == coupling.jstate):
                        coupling.abinitio_field = ab_field
                        break

    def _set_field_parities(self):
        """
        Set parity_i and parity_j on spin-orbit and coupling fields
        based on the electronic states' parity_pm values.

        This is needed for the molpro_duo transformation to work correctly.
        """
        # Build a lookup from istate to parity_pm
        state_parity = {}
        for state in self.states:
            state_parity[state.istate] = state.parity_pm

        # Set parities on spin-orbit fields
        for so in self.spinorbit:
            if so.istate in state_parity:
                so.parity_i = state_parity[so.istate]
            if so.jstate in state_parity:
                so.parity_j = state_parity[so.jstate]

        # Set parities on coupling fields (LXLY, L+, etc.)
        for coupling in self.couplings:
            if coupling.istate in state_parity:
                coupling.parity_i = state_parity[coupling.istate]
            if coupling.jstate in state_parity:
                coupling.parity_j = state_parity[coupling.jstate]
