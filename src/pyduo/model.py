"""Build a PyDuo model (and its fitting problem) from Python, no hand-written
input file.

Why generate TEXT rather than field objects directly: the fit-parameter
registry (fitting.FittingProblem.build_registry) binds parameters by re-scanning
the INPUT TEXT for `fit`-marked lines inside poten / spin-orbit-x / Lx sections,
and output.render_fitted_input writes the refined model back into that same
text. A model that exists only as live objects therefore has no fittable
parameters and no restart file. ModelBuilder closes the loop: it emits a valid
Duo-format input, so DuoSolver.load_input, build_registry and the whole output
set work unchanged.

    b = ModelBuilder('toy')
    b.set_masses(39.9625906, 15.99491463)
    b.set_grid(301, 1.2, 3.5, re=1.82)
    b.set_jrot('0.0 - 5.0')
    b.add_state(1, 'X1Sigma+', lambda_=0, spin=0.0, symmetry='+')
    b.add_emo_potential(1, V0=0.0, RE=1.822, DE=48359.45,
                        B=[1.37, -0.19, -0.30, -0.05], fit=True)
    solver, path = b.build(use_gpu=True)

Everything the emitted text does is dictated by input_parser.InputParser (the
ground truth), and the layout quirks it depends on are flagged in the writers
below.

Observations travel as CSV, one row per datum, so a line list can be edited in
a spreadsheet and a synthetic data set can be exported from a solve:

    write_observations_csv(solver.energy_levels, 'obs.csv')
    obs = load_observations_csv('obs.csv')
    prob = build_problem(b, 'obs.csv')
"""
import csv
import os
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .fitting import Observation

__all__ = ['ModelBuilder', 'load_observations_csv', 'write_observations_csv',
           'build_problem']


# ------------------------------------------------------------------ numbers ---
def _fmt(x) -> str:
    """Parameter value in the E-notation Duo inputs use, 12 significant figures
    -- identical to output._fmt_param, so a builder-written input and the
    model_final.inp rendered from it are byte-comparable."""
    return f'{float(x):.11E}'


def _pline(name: str, value, fit: bool = False) -> str:
    """One named parameter line of a `values` block.

    fitting._FIT_LINE_RE needs `NAME<ws>NUMBER<ws>fit` with the marker as the
    LAST field, and input_parser matches the name at the start of the line, so
    the name must not be indented past column 0 by anything but blanks."""
    return f'{name:<8s} {_fmt(value):>22s}' + ('  fit' if fit else '')


# ------------------------------------------------------------- fit markers ----
# Canonical order a list-of-bools `fit=` argument is matched against. PL/PR/NL/
# NR are absent on purpose: EMO truncates them to int, so the objective is
# piecewise-constant in them and build_registry refuses to register them.
def _emo_fit_names(nb: int) -> List[str]:
    return ['V0', 'RE', 'DE'] + [f'B{k}' for k in range(nb)]


def _bob_fit_names(nb: int) -> List[str]:
    return ['RE', 'P'] + [f'B{k}' for k in range(nb)] + ['BINF']


def _resolve_fit(fit, order: Sequence[str], default_true: Sequence[str]) -> set:
    """`fit` -> the set of parameter names to mark.

    None / False : nothing
    True         : `default_true` (the B terms -- the usual refinement handle)
    list of bool : positional against `order`
    iterable str : those names (case-insensitive), validated against `order`
    """
    if fit is None or fit is False:
        return set()
    if fit is True:
        return set(default_true)
    fit = list(fit)
    if fit and all(isinstance(f, (bool, np.bool_)) for f in fit):
        if len(fit) > len(order):
            raise ValueError(f'fit has {len(fit)} flags but only '
                             f'{len(order)} fittable parameters {list(order)}')
        return {n for n, f in zip(order, fit) if f}
    out = set()
    up = {n.upper() for n in order}
    for f in fit:
        n = str(f).upper()
        if n not in up:
            raise ValueError(f'fit name {f!r} is not a fittable parameter; '
                             f'expected one of {list(order)}')
        out.add(n)
    return out


# ------------------------------------------------------------- the builder ----
_COUPLING_KEYWORD = {
    # user kind -> (section keyword, abinitio keyword)
    # 'bob-rot' keeps its hyphen: fitting._SECTION_RE knows 'bob-rot' only,
    # so the unhyphenated spelling input_parser also accepts would silently
    # lose its fit markers.
    'spin-orbit-x': ('spin-orbit-x', 'abinitio spin-orbit-x'),
    'spin-orbit': ('spin-orbit', 'abinitio spin-orbit'),
    'lx': ('Lx', 'abinitio Lx'),
    'l+': ('Lx', 'abinitio Lx'),
    'lxly': ('Lx', 'abinitio Lx'),
    'spin-spin': ('spin-spin', 'abinitio spin-spin'),
    'bob-rot': ('bob-rot', 'abinitio bob-rot'),
    'bobrot': ('bob-rot', 'abinitio bob-rot'),
}


class ModelBuilder:
    """Assemble a Duo-format input in Python, then solve or fit it.

    Nothing is validated against physics here beyond what would produce an
    input the parser silently mis-reads; the solver remains the judge.
    """

    def __init__(self, name: str = 'model', workdir: Optional[str] = None,
                 title: Optional[str] = None):
        self.name = name
        self._workdir = workdir
        self.title = title or f'PyDuo model {name} (generated by ModelBuilder)'
        self.mass1: Optional[float] = None
        self.mass2: Optional[float] = None
        self.atoms: Optional[Tuple[str, str]] = None
        self.grid = dict(npoints=301, rmin=1.0, rmax=4.0, grid_type=0, re=None)
        self.jrot = '0.0'
        self.contraction = dict(kind='vib', vmax=[30])
        self.states: Dict[int, dict] = {}
        self.potentials: Dict[int, dict] = {}
        self.couplings: List[dict] = []
        self.observations: List[Observation] = []

    # ------------------------------------------------------------ workdir ----
    @property
    def workdir(self) -> str:
        """Directory the default input path lives in (a temp dir until set)."""
        if self._workdir is None:
            self._workdir = tempfile.mkdtemp(prefix=f'pyduo_{self.name}_')
        os.makedirs(self._workdir, exist_ok=True)
        return self._workdir

    @workdir.setter
    def workdir(self, d: str) -> None:
        self._workdir = d

    # ---------------------------------------------------------- global set ---
    def set_masses(self, m1: float, m2: float) -> 'ModelBuilder':
        """Nuclear masses in amu (written as `masses m1 m2`)."""
        self.mass1, self.mass2 = float(m1), float(m2)
        return self

    def set_atoms(self, symbol1: str, symbol2: str) -> 'ModelBuilder':
        """Atom symbols; the parser looks the masses up in its own table.
        set_masses(), when also given, wins -- the parser reads it last."""
        self.atoms = (str(symbol1), str(symbol2))
        return self

    def set_grid(self, npoints: int, rmin: float, rmax: float,
                 grid_type: int = 0, re: Optional[float] = None) -> 'ModelBuilder':
        self.grid = dict(npoints=int(npoints), rmin=float(rmin),
                         rmax=float(rmax), grid_type=int(grid_type),
                         re=None if re is None else float(re))
        return self

    def set_jrot(self, spec) -> 'ModelBuilder':
        """J values: a Duo spec string ('0.0 - 5.0', '0.5 1.5 2.5') or a list.

        A list is written space-separated. Either way the NEXT emitted line
        starts with a letter: input_parser's list form grabs `[\\d.,\\s]+`
        greedily and would swallow a following numeric line.
        """
        if isinstance(spec, str):
            self.jrot = spec.strip()
        else:
            self.jrot = ' '.join(f'{float(j):g}' for j in spec)
        return self

    def set_contraction(self, vmax, kind: str = 'vib') -> 'ModelBuilder':
        """Vibrational basis size per state (a scalar fills forward, as in Duo)."""
        if np.isscalar(vmax):
            vmax = [int(vmax)]
        self.contraction = dict(kind=str(kind), vmax=[int(v) for v in vmax])
        return self

    # --------------------------------------------------------------- states --
    def add_state(self, istate: int, name: str, lambda_: int, spin: float = 0.0,
                  symmetry: Optional[str] = None,
                  mult: Optional[int] = None) -> 'ModelBuilder':
        """Declare electronic state `istate`.

        `lambda` and `mult` are always written explicitly: input_parser would
        otherwise guess Lambda from the name text ('Sigma'/'Pi'/...) and the
        multiplicity from a leading digit, which silently mislabels a state
        named anything else. `symmetry` ('+'/'-') is what sets parity_pm, which
        the MOLPRO->Duo coupling transformation needs; give it for Sigma states.
        """
        if mult is None:
            mult = int(round(2 * float(spin) + 1))
        if symmetry is not None and str(symmetry) not in ('+', '-'):
            raise ValueError(f"symmetry must be '+', '-' or None, got {symmetry!r}")
        self.states[int(istate)] = dict(
            name=str(name), lambda_=int(lambda_), mult=int(mult),
            spin=(mult - 1) / 2.0, symmetry=symmetry)
        return self

    def add_emo_potential(self, istate: int, V0: float, RE: float, DE: float,
                          RREF: float = -1.0, PL: int = 4, PR: int = 4,
                          NL: int = 2, NR: Optional[int] = None,
                          B: Optional[Sequence[float]] = None,
                          fit=None) -> 'ModelBuilder':
        """Expanded Morse Oscillator potential for state `istate`.

        Parameter order follows Duo's poten_EMO packing
        [V0, RE, DE, RREF, PL, PR, NL, NR, B0, B1, ...]; NR defaults to
        len(B)-1 so the right-hand expansion uses every coefficient given
        (EMO silently drops beta_k for k > NR).

        fit : None | True (all B terms) | list of bools over
              ['V0','RE','DE','B0',...] | list of parameter names.
        """
        if int(istate) not in self.states:
            raise ValueError(f'add_state({istate}, ...) must come first')
        B = [0.0] if B is None else [float(b) for b in B]
        if NR is None:
            NR = len(B) - 1
        order = _emo_fit_names(len(B))
        marks = _resolve_fit(fit, order, [f'B{k}' for k in range(len(B))])
        self.potentials[int(istate)] = dict(
            V0=float(V0), RE=float(RE), DE=float(DE), RREF=float(RREF),
            PL=int(PL), PR=int(PR), NL=int(NL), NR=int(NR), B=B, fit=marks)
        return self

    def add_coupling(self, kind: str, istate: int, jstate: int, *,
                     lambda_i: Optional[int] = None,
                     lambda_j: Optional[int] = None,
                     spin_i: Optional[float] = None,
                     spin_j: Optional[float] = None,
                     sigma_i: Optional[float] = None,
                     sigma_j: Optional[float] = None,
                     factor=None, morphing: bool = True,
                     RE: float = 1.8, P: float = 2.0, NT: Optional[int] = None,
                     RREF: float = -1.0,
                     B: Optional[Sequence[float]] = None,
                     BINF: Optional[float] = None,
                     xlzy: Optional[Tuple] = None,
                     abinitio: Optional[Tuple] = None,
                     units_r: str = 'angstrom', units_v: str = 'cm-1',
                     name: Optional[str] = None,
                     fit=None) -> 'ModelBuilder':
        """A BOBLEROY-form coupling between states `istate` and `jstate`.

        kind : 'spin-orbit-x', 'lx' ('l+'/'lxly'), 'spin-spin', 'bob-rot'.
        abinitio : (r_array, v_array) written as an `abinitio <kind> i j` grid
            section. `morphing=True` multiplies the analytic BOBLEROY curve by
            the splined grid, exactly as Duo does -- and PyDuo WARNS and leaves
            the field unmorphed if the grid is missing, so pass one or set
            morphing=False.
        factor : '1', 'i', '-i', 'sqrt(2)', 'i*sqrt(2)', ... (kept as text; the
            parser splits it into a real factor and a complex phase).
        xlzy : the (<x|Lz|y>_i, <x|Lz|y>_j) pair, e.g. (0, 'i').
        fit : as add_emo_potential, over ['RE','P','B0',...,'BINF'].
        """
        k = str(kind).lower()
        if k not in _COUPLING_KEYWORD:
            raise ValueError(f'unsupported coupling kind {kind!r}; '
                             f'expected one of {sorted(_COUPLING_KEYWORD)}')
        istate, jstate = int(istate), int(jstate)
        for s in (istate, jstate):
            if s not in self.states:
                raise ValueError(f'add_state({s}, ...) must come first')
        si, sj = self.states[istate], self.states[jstate]
        B = [1.0] if B is None else [float(b) for b in B]
        order = _bob_fit_names(len(B))
        marks = _resolve_fit(fit, order, [f'B{k_}' for k_ in range(len(B))])
        if abinitio is not None:
            ra, va = (np.asarray(a, dtype=float).ravel() for a in abinitio)
            if ra.size != va.size or ra.size < 2:
                raise ValueError('abinitio=(r, v) must be two equal-length '
                                 'arrays of at least 2 points')
            abinitio = (ra, va)
        self.couplings.append(dict(
            kind=k, keyword=_COUPLING_KEYWORD[k][0],
            ab_keyword=_COUPLING_KEYWORD[k][1],
            istate=istate, jstate=jstate,
            lambda_i=si['lambda_'] if lambda_i is None else int(lambda_i),
            lambda_j=sj['lambda_'] if lambda_j is None else int(lambda_j),
            spin_i=si['spin'] if spin_i is None else float(spin_i),
            spin_j=sj['spin'] if spin_j is None else float(spin_j),
            sigma_i=0.0 if sigma_i is None else float(sigma_i),
            sigma_j=0.0 if sigma_j is None else float(sigma_j),
            factor=None if factor is None else str(factor),
            morphing=bool(morphing), RE=float(RE), P=float(P),
            RREF=float(RREF), NT=len(B) - 1 if NT is None else int(NT),
            B=B, BINF=1.0 if BINF is None else float(BINF),
            xlzy=xlzy, abinitio=abinitio, units_r=str(units_r),
            units_v=str(units_v),
            name=name or f'{_COUPLING_KEYWORD[k][0]}_{istate}_{jstate}',
            fit=marks))
        return self

    # -------------------------------------------------------- observations ---
    def set_observations(self, obs) -> 'ModelBuilder':
        """Attach observations so to_input() also writes a FITTING block.

        The block is what makes the input self-contained for Duo itself or for
        a FittingProblem built WITHOUT observations=; build_problem() passes the
        list in directly and never needs it.
        """
        self.observations = list(obs)
        return self

    # ---------------------------------------------------------- the writer ---
    def to_input(self) -> str:
        """The complete Duo-format input text."""
        if self.mass1 is None and self.atoms is None:
            raise ValueError('set_masses() or set_atoms() first')
        if not self.potentials:
            raise ValueError('a model needs at least one potential')
        missing = sorted(set(self.states) - set(self.potentials))
        if missing:
            raise ValueError(f'states {missing} have no potential')

        L = [f'! {self.title}', '']
        if self.atoms is not None:
            L.append(f'atoms {self.atoms[0]} {self.atoms[1]}')
        if self.mass1 is not None:
            # plain decimals: the parser's mass regex is `[\d.]+`, so an
            # E-notation mass would be truncated at the 'e'
            L.append(f'masses {self.mass1:.11f} {self.mass2:.11f}')
        L += [f'nstates {len(self.states)}',
              # jrot is followed by a letter-initial line on purpose: the
              # parser's list form is greedy over digits/whitespace.
              f'jrot {self.jrot}',
              '']
        g = self.grid
        L += ['grid', f'  npoints  {g["npoints"]}',
              f'  range  {g["rmin"]:.6f}, {g["rmax"]:.6f}',
              f'  type {g["grid_type"]}']
        if g['re'] is not None:
            L.append(f'  re {g["re"]:.6f}')
        L += ['end', '']
        c = self.contraction
        L += ['CONTRACTION', f'  {c["kind"]}',
              '  vmax ' + ' '.join(str(v) for v in c['vmax']), 'END', '']

        # potentials first, then couplings, then the abinitio grids they morph
        # against, then FITTING -- input_parser's _find_section_end treats
        # ABINITIO and FITTING as section starts, so this ordering keeps every
        # section's body intact.
        for istate in sorted(self.potentials):
            L += self._poten_lines(istate) + ['']
        for cp in self.couplings:
            L += self._coupling_lines(cp) + ['']
        for cp in self.couplings:
            if cp['abinitio'] is not None:
                L += self._abinitio_lines(cp) + ['']
        if self.observations:
            L += self._fitting_lines() + ['']
        return '\n'.join(L) + '\n'

    def _poten_lines(self, istate: int) -> List[str]:
        st, p = self.states[istate], self.potentials[istate]
        # the header line carries NOTHING else: fitting._SECTION_RE anchors on
        # '^poten N$' and would not see a section with a trailing comment
        out = [f'poten {istate}', f'name "{st["name"]}"']
        if st['symmetry'] is not None:
            out.append(f'symmetry {st["symmetry"]}')
        out += [f'lambda {st["lambda_"]}', f'mult   {st["mult"]}',
                'type   EMO', 'values']
        f = p['fit']
        out += [_pline('V0', p['V0'], 'V0' in f),
                _pline('RE', p['RE'], 'RE' in f),
                _pline('DE', p['DE'], 'DE' in f),
                _pline('RREF', p['RREF']),
                _pline('PL', p['PL']), _pline('PR', p['PR']),
                _pline('NL', p['NL']), _pline('NR', p['NR'])]
        out += [_pline(f'B{k}', b, f'B{k}' in f) for k, b in enumerate(p['B'])]
        out.append('end')
        return out

    def _coupling_lines(self, cp: dict) -> List[str]:
        out = [f'{cp["keyword"]} {cp["istate"]} {cp["jstate"]}',
               f'name "{cp["name"]}"',
               f'lambda {cp["lambda_i"]} {cp["lambda_j"]}',
               f'spin   {cp["spin_i"]:.1f} {cp["spin_j"]:.1f}',
               f'sigma  {cp["sigma_i"]:.1f} {cp["sigma_j"]:.1f}']
        if cp['factor'] is not None:
            out.append(f'factor {cp["factor"]}')
        if cp['xlzy'] is not None:
            out.append(f'<x|Lz|y> {cp["xlzy"][0]} {cp["xlzy"][1]}')
        out.append('type  BOBLEROY')
        if cp['morphing']:
            out.append('morphing')
        # 'values' must come AFTER every keyword line: _find_keyword_end ends
        # the values block at the next NAME/TYPE/LAMBDA/SIGMA/SPIN token
        # anywhere downstream, not just at a line start.
        out.append('values')
        f = cp['fit']
        out += [_pline('RE', cp['RE'], 'RE' in f),
                _pline('RREF', cp['RREF']),
                _pline('P', cp['P'], 'P' in f),
                _pline('NT', cp['NT'])]
        out += [_pline(f'B{k}', b, f'B{k}' in f) for k, b in enumerate(cp['B'])]
        out += [_pline('BINF', cp['BINF'], 'BINF' in f), 'end']
        return out

    def _abinitio_lines(self, cp: dict) -> List[str]:
        r, v = cp['abinitio']
        out = [f'{cp["ab_keyword"]} {cp["istate"]} {cp["jstate"]}',
               f'name "abinitio {cp["name"]}"',
               f'lambda {cp["lambda_i"]} {cp["lambda_j"]}',
               f'spin   {cp["spin_i"]:.1f} {cp["spin_j"]:.1f}',
               f'sigma  {cp["sigma_i"]:.1f} {cp["sigma_j"]:.1f}',
               'type   grid']
        if cp['factor'] is not None:
            out.append(f'factor {cp["factor"]}')
        if cp['xlzy'] is not None:
            out.append(f'<x|Lz|y> {cp["xlzy"][0]} {cp["xlzy"][1]}')
        # two tokens: the parser's `units` regex demands r-units AND v-units
        out += [f'units {cp["units_r"]} {cp["units_v"]}',
                f'N {len(r)}', 'values']
        out += [f'  {ri:14.8f} {vi:20.10f}' for ri, vi in zip(r, v)]
        out.append('end')
        return out

    def _fitting_lines(self) -> List[str]:
        """A minimal FITTING block holding the attached observations.

        Written in whichever of Duo's two layouts the data needs: FREQUENCY
        (17 columns) if any observation is a transition, otherwise ENERGIES
        (10 columns). No source tags -- a trailing word is harmless to
        FittingProblem but not to Duo's fixed-format reader.
        """
        obs = self.observations
        out = ['FITTING', 'itmax   10', 'fit_factor  1e4',
               f'output   {self.name}_fit']
        if any(o.kind == 'freq' for o in obs):
            out.append('frequency')
            for o in obs:
                out.append(
                    f'{o.J:7.1f} {_p(o.parity):>2s} {o.N:5d} '
                    f'{o.J2:7.1f} {_p(o.parity2):>2s} {o.N2:5d} '
                    f'{o.energy:16.6f} '
                    f'{o.state:4d} {o.v:4d} {o.lambda_:4d} '
                    f'{o.sigma:6.1f} {o.omega:6.1f} '
                    f'{o.state2:4d} {o.v2:4d} {o.lambda2:4d} '
                    f'{o.sigma2:6.1f} {o.omega2:6.1f} {o.weight:8.3f}')
        else:
            out.append('energies')
            for o in obs:
                out.append(
                    f'{o.J:7.1f} {_p(o.parity):>2s} {o.N:5d} '
                    f'{o.energy:16.6f} {o.state:4d} {o.v:4d} '
                    f'{o.lambda_:4d} {o.sigma:6.1f} {o.omega:6.1f} '
                    f'{o.weight:8.3f}')
        out.append('end')
        return out

    # ----------------------------------------------------------- solve/save --
    def write(self, path: Optional[str] = None) -> str:
        """Write the input file; returns its path."""
        if path is None:
            path = os.path.join(self.workdir, f'{self.name}.inp')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_input())
        return path

    def build(self, use_gpu: bool = True, path: Optional[str] = None,
              verbose: int = 0):
        """write() + DuoSolver.load_input(). Returns (solver, path).

        The file survives the call on purpose: it is the registry's and
        model_final.inp's source text.
        """
        from .duo import DuoSolver
        path = self.write(path)
        solver = DuoSolver(use_gpu=use_gpu, verbose=verbose)
        solver.load_input(path)
        return solver, path


def _p(parity) -> str:
    return '+' if parity > 0 else '-'


# ------------------------------------------------------------ observations ----
_CSV_COLUMNS = ['kind', 'J', 'parity', 'state', 'v', 'lambda', 'sigma',
                'omega', 'J2', 'parity2', 'state2', 'v2', 'lambda2', 'sigma2',
                'omega2', 'value', 'weight']
_REQ_ENERGY = ('kind', 'j', 'parity', 'state', 'v', 'lambda', 'sigma', 'omega',
               'value')
_REQ_FREQ = _REQ_ENERGY + ('j2', 'parity2', 'state2', 'v2', 'lambda2',
                           'sigma2', 'omega2')
_PARITY = {'+': +1, '-': -1, 'e': +1, 'f': -1, '+1': +1, '-1': -1, '1': +1}


def _parity_of(text: str, row: int, col: str) -> int:
    key = str(text).strip().lower()
    if key not in _PARITY:
        raise ValueError(f'row {row}: {col}={text!r} is not a parity '
                         f"(expected '+' or '-')")
    return _PARITY[key]


def _num(text, row: int, col: str, cast=float, default=None):
    s = '' if text is None else str(text).strip()
    if s == '':
        if default is None:
            raise ValueError(f'row {row}: column {col!r} is required '
                             f'and empty')
        return default
    try:
        return cast(float(s))
    except ValueError:
        raise ValueError(f'row {row}: column {col!r} = {text!r} is not a number')


def load_observations_csv(path: str) -> List[Observation]:
    """Read observations from a CSV file into fitting.Observation objects.

    Header (exact names, case- and order-insensitive):
        kind,J,parity,state,v,lambda,sigma,omega,
        J2,parity2,state2,v2,lambda2,sigma2,omega2,value,weight

    `kind` is 'energy' or 'freq'. `value` is the level energy (cm-1, referenced
    to the ZPE, as PyDuo reports levels) or the transition wavenumber. The *2
    columns describe the LOWER level of a frequency row and may be empty -- or
    absent entirely -- for an all-energy file. `weight` defaults to 1.

    Every failure names the offending data row (1 = first row after the
    header), because a line list is usually hand-edited.
    """
    with open(path, newline='') as f:
        rdr = csv.reader(f)
        rows = [r for r in rdr if any(str(c).strip() for c in r)]
    if not rows:
        raise ValueError(f'{path}: file is empty')
    head = [str(h).strip().lower() for h in rows[0]]
    idx = {h: i for i, h in enumerate(head)}
    missing = [c for c in _REQ_ENERGY if c not in idx]
    if missing:
        raise ValueError(f'{path}: header is missing required column(s) '
                         f'{missing}; expected {_CSV_COLUMNS}')

    def cell(r, col):
        i = idx.get(col)
        return '' if i is None or i >= len(r) else r[i]

    obs: List[Observation] = []
    for n, r in enumerate(rows[1:], start=1):
        try:
            kind = str(cell(r, 'kind')).strip().lower()
            if kind in ('e', 'energy'):
                kind = 'energy'
            elif kind in ('f', 'freq', 'frequency'):
                kind = 'freq'
            else:
                raise ValueError(f"row {n}: unknown kind {cell(r, 'kind')!r} "
                                 f"(expected 'energy' or 'freq')")
            if kind == 'freq':
                absent = [c for c in _REQ_FREQ if c not in idx]
                if absent:
                    raise ValueError(f'row {n}: kind=freq needs the lower-level '
                                     f'columns {absent}')
            o = Observation(
                kind=kind,
                J=_num(cell(r, 'j'), n, 'J'),
                parity=_parity_of(cell(r, 'parity'), n, 'parity'),
                N=0,
                energy=_num(cell(r, 'value'), n, 'value'),
                state=_num(cell(r, 'state'), n, 'state', int),
                v=_num(cell(r, 'v'), n, 'v', int),
                lambda_=_num(cell(r, 'lambda'), n, 'lambda', int),
                sigma=_num(cell(r, 'sigma'), n, 'sigma'),
                omega=_num(cell(r, 'omega'), n, 'omega'),
                weight=_num(cell(r, 'weight'), n, 'weight', float, 1.0))
            if kind == 'freq':
                o.J2 = _num(cell(r, 'j2'), n, 'J2')
                o.parity2 = _parity_of(cell(r, 'parity2'), n, 'parity2')
                o.state2 = _num(cell(r, 'state2'), n, 'state2', int)
                o.v2 = _num(cell(r, 'v2'), n, 'v2', int)
                o.lambda2 = _num(cell(r, 'lambda2'), n, 'lambda2', int)
                o.sigma2 = _num(cell(r, 'sigma2'), n, 'sigma2')
                o.omega2 = _num(cell(r, 'omega2'), n, 'omega2')
            obs.append(o)
        except ValueError as e:
            raise ValueError(f'{path}: {e}') from None
    if not obs:
        raise ValueError(f'{path}: no data rows after the header')
    return obs


def write_observations_csv(items, path: str, weight: float = 1.0,
                           kind: str = 'energy') -> str:
    """Write Observations -- or a solver's EnergyLevel list -- as an
    observations CSV (the inverse of load_observations_csv).

    Passing solver.energy_levels turns a solve into a synthetic data set: the
    quantum numbers are the ones the matcher will key on, so a fit against the
    file reproduces the model that generated it.
    """
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(_CSV_COLUMNS)
        for it in items:
            if getattr(it, 'kind', None) in ('energy', 'freq'):
                o = it
                low = ([f'{o.J2:g}', _p(o.parity2), o.state2, o.v2, o.lambda2,
                        f'{o.sigma2:g}', f'{o.omega2:g}']
                       if o.kind == 'freq' else [''] * 7)
                w.writerow([o.kind, f'{o.J:g}', _p(o.parity), o.state, o.v,
                            o.lambda_, f'{o.sigma:g}', f'{o.omega:g}'] + low +
                           [f'{o.energy:.8f}', f'{o.weight:g}'])
            else:                              # an EnergyLevel from a solve
                w.writerow([kind, f'{it.J:g}', _p(it.parity), it.state, it.v,
                            it.lambda_, f'{it.sigma:g}', f'{it.omega:g}'] +
                           [''] * 7 +
                           [f'{it.energy:.8f}', f'{weight:g}'])
    return path


# ---------------------------------------------------------------- problem -----
def build_problem(builder: ModelBuilder,
                  obs: Union[str, Sequence[Observation]],
                  use_gpu: bool = True, verbose: int = 1,
                  path: Optional[str] = None):
    """ModelBuilder + observations -> a ready FittingProblem.

    `obs` is a CSV path or a list of Observation. The input file the builder
    writes is the registry's source text, so the returned problem's parameters
    are exactly the `fit`-marked entries of the model.
    """
    from .fitting import FittingProblem
    solver, inp = builder.build(use_gpu=use_gpu, path=path)
    obs_list = load_observations_csv(obs) if isinstance(obs, (str, os.PathLike)) \
        else list(obs)
    prob = FittingProblem(inp, solver=solver, verbose=verbose,
                          observations=obs_list)
    prob.build_registry(solver.parser)
    return prob
