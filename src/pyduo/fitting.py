"""Least-squares refinement of PyDuo model parameters against observed levels.

Design goals vs Duo's refinement.f90 (see tracking/99 fitting notes):
- obs <-> calc matching by QUANTUM NUMBERS using PyDuo's Duo-exact labels
  (+ purity), not by list index. Duo's default THRESH_LOCK never fires, so
  its fits silently trust the N column of the input; ours matches physics.
- residual weighting is non-destructive: original weights are immutable;
  robust losses (soft_l1 / huber via scipy.least_squares) replace Duo's
  destructive Watson reweighting and permanent THRESH_OBS-CALC exclusion.
- the solver is the GPU-batched PyDuo (all J in one shot), so one objective
  evaluation is a single ~seconds-scale run instead of Duo's per-J loop.

Fit parameters are the `fit`-marked named entries of the input's values
blocks (EMO potentials, BOBLEROY couplings), located by re-scanning the
section text and mapped to parameter-array indices via the same packing the
parser uses:
    EMO      : [V0, RE, DE, RREF, PL, PR, NL, NR, B0, B1, ...]
    BOBLEROY : [RE, P, B0..Bn, BINF]     (RREF and NT are parsed but unused)
"""
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import re
import time
import numpy as np

__all__ = ['Observation', 'FitParameter', 'FittingProblem', 'FitResult']


class FitResult:
    """Wraps a scipy least_squares result with the statistics a spectroscopist
    actually needs: parameter uncertainties, correlations, and an SVD
    conditioning report naming the sloppy parameter COMBINATIONS the data do
    not determine (the honest answer to 'why did SO13:B0 drift?')."""

    def __init__(self, scipy_result, params):
        self.raw = scipy_result
        self.params = params
        self.labels = [p.label for p in params]

    # -- covariance from J^T J at the solution (Gauss-Newton approximation) --
    def _covariance(self):
        J = self.raw.jac
        m, n = J.shape
        dof = max(m - n, 1)
        s2 = 2.0 * self.raw.cost / dof          # cost = 0.5 * sum(rho(f^2))
        JTJ = J.T @ J
        # pseudo-inverse via SVD for singular/sloppy problems
        U, s, Vt = np.linalg.svd(JTJ)
        s_inv = np.where(s > s[0] * 1e-12, 1.0 / np.maximum(s, 1e-300), 0.0)
        return (Vt.T * s_inv) @ U.T * s2

    def uncertainties(self):
        """1-sigma parameter uncertainties. Returns {label: (value, sigma)}."""
        cov = self._covariance()
        sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return {lab: (float(x), float(s))
                for lab, x, s in zip(self.labels, self.raw.x, sig)}

    def correlation(self):
        """Parameter correlation matrix (labels, matrix)."""
        cov = self._covariance()
        d = np.sqrt(np.maximum(np.diag(cov), 1e-300))
        return self.labels, cov / np.outer(d, d)

    def conditioning(self, n_worst: int = 3, n_terms: int = 4):
        """SVD of the (column-scaled) Jacobian: condition number and the
        sloppiest parameter combinations, each written as a linear combo of
        parameter labels. Large-coefficient members of a sloppy direction can
        drift together without changing the fit -- constrain or fix one."""
        J = self.raw.jac
        scale = np.linalg.norm(J, axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        U, s, Vt = np.linalg.svd(J / scale, full_matrices=False)
        out = {'condition_number': float(s[0] / max(s[-1], 1e-300)),
               'singular_values': s.tolist(), 'sloppy': []}
        for k in range(1, min(n_worst, len(s)) + 1):
            v = Vt[-k]
            idx = np.argsort(-np.abs(v))[:n_terms]
            combo = ' '.join(f'{v[i]:+.2f}*{self.labels[i]}' for i in idx)
            out['sloppy'].append({'sv': float(s[-k]),
                                  'sv_rel': float(s[-k] / s[0]),
                                  'combo': combo})
        return out

    def plot_correlation(self, annotate: bool = None):
        """Parameter-correlation heatmap (diverging blue/neutral/red).
        Cells near +-1 mark parameter pairs the data cannot separate."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        labels, C = self.correlation()
        n = len(labels)
        cmap = LinearSegmentedColormap.from_list(
            'pyduo_div', ['#2a78d6', '#f2f1ee', '#e34948'])
        fig, ax = plt.subplots(figsize=(0.42 * n + 2.4, 0.42 * n + 1.8),
                               dpi=110)
        im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n), labels, fontsize=7)
        if annotate is None:
            annotate = n <= 14
        if annotate:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f'{C[i, j]:+.2f}'.replace('+0.', '+.')
                            .replace('-0.', '-.'), ha='center', va='center',
                            fontsize=6, color='#0b0b0b' if abs(C[i, j]) < 0.6
                            else '#ffffff')
        fig.colorbar(im, ax=ax, shrink=0.8, label='correlation')
        ax.set_title('parameter correlations', fontsize=10, loc='left')
        fig.tight_layout()
        return fig

    def pretty(self) -> str:
        lines = [f'cost={self.raw.cost:.6g}  nfev={self.raw.nfev}  '
                 f'status={self.raw.status}']
        unc = self.uncertainties()
        cond = self.conditioning()
        lines.append(f'condition number (scaled J): {cond["condition_number"]:.3g}')
        lines.append(f'{"parameter":16s} {"value":>16s} {"1-sigma":>12s} {"rel":>9s}')
        for lab, (x, sg) in unc.items():
            rel = sg / abs(x) if abs(x) > 1e-300 else float('inf')
            lines.append(f'{lab:16s} {x:16.8g} {sg:12.3g} {rel:9.2g}')
        lines.append('sloppiest directions (data-undetermined combos):')
        for d in cond['sloppy']:
            lines.append(f'  sv/sv_max={d["sv_rel"]:.2e}: {d["combo"]}')
        return '\n'.join(lines)


@dataclass
class Observation:
    """One observed datum: an energy level, or a transition FREQUENCY.

    kind='energy': (J, parity, QNs) level at `energy` (cm-1 rel. to ZPE).
    kind='freq'  : `energy` holds the transition wavenumber; the UPPER level
        is (J, parity, QNs), the LOWER level is the *2 fields. Frequencies
        are the physically superior fit target: they are the actually
        measured quantities and are immune to ZPE/absolute-calibration error.
    """
    J: float
    parity: int          # +1 / -1
    N: int               # Duo's index column (informational only)
    energy: float        # level energy, or transition frequency (kind='freq')
    state: int
    v: int
    lambda_: int
    sigma: float
    omega: float
    weight: float
    kind: str = 'energy'
    # lower level (kind='freq' only)
    J2: float = 0.0
    parity2: int = +1
    N2: int = 0
    state2: int = 0
    v2: int = 0
    lambda2: int = 0
    sigma2: float = 0.0
    omega2: float = 0.0
    matched: bool = False
    calc: Optional[float] = None     # calc level energy / calc frequency
    purity: float = 0.0


@dataclass
class FitParameter:
    field_obj: object     # the live PotentialField / SpinOrbitField / CouplingField
    index: int            # position in field_obj.parameters
    name: str             # e.g. 'B0'
    label: str            # human id, e.g. 'poten1:RE'
    value0: float = 0.0

    def get(self) -> float:
        return float(self.field_obj.parameters[self.index])

    def set(self, x: float) -> None:
        self.field_obj.parameters[self.index] = float(x)


def _emo_index(name: str) -> Optional[int]:
    order = ['V0', 'RE', 'DE', 'RREF', 'PL', 'PR', 'NL', 'NR']
    name = name.upper()
    if name in order:
        return order.index(name)
    m = re.fullmatch(r'B(\d+)', name)
    if m:
        return 8 + int(m.group(1))
    return None


def _bobleroy_index(name: str, nparams: int) -> Optional[int]:
    name = name.upper()
    if name == 'RE':
        return 0
    if name == 'P':
        return 1
    if name == 'BINF':
        return nparams - 1
    m = re.fullmatch(r'B(\d+)', name)
    if m:
        return 2 + int(m.group(1))
    return None


_SECTION_RE = re.compile(
    r'(?im)^[ \t]*(poten[ \t]+\d+|spin-orbit(?:-x)?[ \t]+\d+[ \t]+\d+|'
    r'lx[ \t]+\d+[ \t]+\d+|l\+[ \t]+\d+[ \t]+\d+|lxly[ \t]+\d+[ \t]+\d+|'
    r'spin-spin[ \t]+\d+[ \t]+\d+|bob-rot[ \t]+\d+[ \t]+\d+)[ \t]*$'
    r'(.*?)(?=^[ \t]*(?:end)[ \t]*$)', re.S | re.M)

_FIT_LINE_RE = re.compile(
    r'(?im)^[ \t]*([A-Z]+\d*|BINF)[ \t]+([-+]?[\d.]+(?:[eEdD][-+]?\d+)?)'
    r'[ \t]+fit\b')


class FittingProblem:
    """Owns the observations, the fit-parameter registry and the objective."""

    def __init__(self, input_file: str, solver=None, verbose: int = 1):
        self.input_file = input_file
        self.verbose = verbose
        self.solver = solver          # a DuoSolver, already load_input()-ed
        self.obs: List[Observation] = []
        self.params: List[FitParameter] = []
        self.j_filter: Optional[float] = None
        self.n_unmatched = 0
        self.eval_count = 0
        self.history: List[dict] = []
        self.snapshots: List[dict] = []   # reporting data (pyduo.reporting)
        with open(input_file) as f:
            raw = f.read()
        # strip parenthesis comments the way the level parser does not:
        # keep raw for section scanning; obs rows are matched line-wise
        self._raw = raw
        self._parse_observations(raw)

    # ---------------- observations ----------------
    def _parse_observations(self, raw: str) -> None:
        m_en = re.search(r'(?im)^[ \t]*energies\b.*$', raw)
        m_fr = re.search(r'(?im)^[ \t]*frequenc(?:y|ies)\b.*$', raw)
        if m_fr is not None:
            self._parse_freq_block(raw[m_fr.end():])
        elif m_en is not None:
            self._parse_energy_block(raw[m_en.end():])
        else:
            raise ValueError('no ENERGIES or FREQUENCY block in FITTING section')
        if self.verbose:
            kinds = {o.kind for o in self.obs}
            print(f'parsed {len(self.obs)} observations ({"/".join(sorted(kinds))})')

    @staticmethod
    def _block_body(block: str) -> str:
        endm = re.search(r'(?im)^[ \t]*end[ \t]*$', block)
        return block[:endm.start()] if endm else block

    def _parse_energy_block(self, block: str) -> None:
        for line in self._block_body(block).splitlines():
            line = line.split('(')[0].strip()
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                self.obs.append(Observation(
                    J=float(parts[0]),
                    parity=+1 if parts[1] in ('+', 'e') else -1,
                    N=int(parts[2]),
                    energy=float(parts[3]), state=int(parts[4]),
                    v=int(parts[5]), lambda_=int(parts[6]),
                    sigma=float(parts[7]), omega=float(parts[8]),
                    weight=float(parts[9])))
            except (ValueError, IndexError):
                continue

    def _parse_freq_block(self, block: str) -> None:
        """Duo frequency rows (17 columns + optional source tag):
        J' p' N'  J" p" N"  freq  st' v' lam' sig' om'  st" v" lam" sig" om"  wt ...
        (primed = upper level, double-primed = lower level)."""
        for line in self._block_body(block).splitlines():
            line = line.split('(')[0].strip()
            parts = line.split()
            if len(parts) < 18:
                continue
            try:
                self.obs.append(Observation(
                    kind='freq',
                    J=float(parts[0]),
                    parity=+1 if parts[1] in ('+', 'e') else -1,
                    N=int(parts[2]),
                    J2=float(parts[3]),
                    parity2=+1 if parts[4] in ('+', 'e') else -1,
                    N2=int(parts[5]),
                    energy=float(parts[6]),
                    state=int(parts[7]), v=int(parts[8]),
                    lambda_=int(parts[9]), sigma=float(parts[10]),
                    omega=float(parts[11]),
                    state2=int(parts[12]), v2=int(parts[13]),
                    lambda2=int(parts[14]), sigma2=float(parts[15]),
                    omega2=float(parts[16]),
                    weight=float(parts[17])))
            except (ValueError, IndexError):
                continue

    # ---------------- fit-parameter registry ----------------
    def build_registry(self, parser) -> None:
        """Scan the input for `fit`-marked named parameters and bind them to
        the LIVE field objects held by `parser` (shared with the solver)."""
        pot_by_id = {p.istate: p for p in parser.potentials}
        so_list = list(parser.spinorbit)
        cp_list = list(parser.couplings)
        so_i = 0
        cp_i = {}

        for m in _SECTION_RE.finditer(self._raw):
            header = m.group(1).lower().split()
            body = m.group(2)
            names = [(fm.group(1).upper(), float(fm.group(2)
                      .replace('D', 'E').replace('d', 'e')))
                     for fm in _FIT_LINE_RE.finditer(body)]
            if not names:
                if header[0] == 'poten':
                    pass
                elif header[0].startswith('spin-orbit'):
                    so_i += 1
                continue

            if header[0] == 'poten':
                fld = pot_by_id.get(int(header[1]))
                label = f'poten{header[1]}'
                idx_of = _emo_index
            elif header[0].startswith('spin-orbit'):
                fld = so_list[so_i] if so_i < len(so_list) else None
                so_i += 1
                label = f'SO{fld.istate}{fld.jstate}' if fld else 'SO?'
                idx_of = lambda n, np_=None: _bobleroy_index(
                    n, len(fld.parameters))
            else:
                key = (header[0], header[1], header[2])
                k = cp_i.get(key, 0)
                cands = [c for c in cp_list
                         if c.istate == int(header[1]) and c.jstate == int(header[2])]
                fld = cands[k] if k < len(cands) else None
                cp_i[key] = k + 1
                label = f'{header[0]}{header[1]}{header[2]}'
                idx_of = lambda n, np_=None: _bobleroy_index(
                    n, len(fld.parameters))
            if fld is None or fld.parameters is None:
                continue
            for name, val in names:
                idx = idx_of(name)
                if idx is None or idx >= len(fld.parameters):
                    continue
                cur = float(fld.parameters[idx])
                if abs(cur - val) > 1e-6 * max(1.0, abs(val)):
                    # mapping sanity check failed -- skip loudly
                    print(f'WARN: {label}:{name} maps to index {idx} but '
                          f'value differs (section {val} vs array {cur}) -- skipped')
                    continue
                self.params.append(FitParameter(
                    field_obj=fld, index=idx, name=name,
                    label=f'{label}:{name}', value0=cur))
        if self.verbose:
            print(f'fit registry: {len(self.params)} parameters '
                  f'({", ".join(p.label for p in self.params[:12])}'
                  f'{" ..." if len(self.params) > 12 else ""})')

    # ---------------- objective ----------------
    def _obs_in_scope(self, o) -> bool:
        jmax = max(o.J, o.J2) if o.kind == 'freq' else o.J
        return self.j_filter is None or jmax <= self.j_filter

    def _run_solver(self):
        s = self.solver
        js = set()
        for o in self.obs:
            if self._obs_in_scope(o):
                js.add(o.J)
                if o.kind == 'freq':
                    js.add(o.J2)
        js = sorted(js)
        s.j_list = list(js)
        s.energy_levels = []
        s.capture_blocks = True     # [phase F] keep P + eigvecs per block

        # [warm start] if only COUPLING parameters moved since the previous
        # evaluation, the contracted vibrational basis (which depends on the
        # potentials only) is still valid: skip builder construction and the
        # vibrational solve, go straight to the rovibronic solve (which
        # re-runs the matelem precompute with the updated field parameters).
        cur_pot = tuple(p.get() for p in self.params
                        if p.label.startswith('poten'))
        warm = (getattr(self, '_warm_ok', False) and
                s.hamiltonian is not None and
                cur_pot == getattr(self, '_last_pot', None) and
                list(getattr(s.hamiltonian, '_last_solved_js', [])) == list(js))
        if warm:
            s._solve_rovibronic_batched()
            s._reference_energies_to_zpe()
        else:
            s.run()
            s.hamiltonian._last_solved_js = list(js)
            self._warm_ok = True
        self._last_pot = cur_pot
        # snapshot geometry the HF Jacobian needs
        hb = s.hamiltonian
        self._last_jlist = list(js)
        maxJ = max(js)
        self._last_maxidx = [i for i, bs in enumerate(hb.contracted_basis)
                             if abs(bs.omega) <= maxJ + 0.01]
        self._active_basis = [hb.contracted_basis[i] for i in self._last_maxidx]
        return s.energy_levels

    def match(self, levels) -> np.ndarray:
        """Match obs to calc by unsigned quantum numbers; returns residuals."""
        table = {}
        for l in levels:
            key = (l.J, l.parity, l.state, l.v, abs(l.lambda_),
                   int(round(2 * abs(l.sigma))), int(round(2 * abs(l.omega))))
            if key not in table:
                table[key] = l
        res = []
        self.n_unmatched = 0
        for o in self.obs:
            if not self._obs_in_scope(o):
                continue
            key = (o.J, o.parity, o.state, o.v, abs(o.lambda_),
                   int(round(2 * abs(o.sigma))), int(round(2 * abs(o.omega))))
            l = table.get(key)
            if o.kind == 'freq':
                key2 = (o.J2, o.parity2, o.state2, o.v2, abs(o.lambda2),
                        int(round(2 * abs(o.sigma2))),
                        int(round(2 * abs(o.omega2))))
                l2 = table.get(key2)
                if l is None or l2 is None:
                    o.matched = False
                    self.n_unmatched += 1
                    continue
                o.matched = True
                o.calc = l.energy - l2.energy          # calc frequency
                o.purity = min(l.purity, l2.purity)
                o._iroot = getattr(l, 'block_iroot', -1)
                o._iroot2 = getattr(l2, 'block_iroot', -1)
                res.append((o.calc - o.energy) * np.sqrt(o.weight))
                continue
            if l is None:
                o.matched = False
                self.n_unmatched += 1
                continue
            o.matched = True
            o.calc = l.energy
            o.purity = l.purity
            o._iroot = getattr(l, 'block_iroot', -1)
            res.append((o.calc - o.energy) * np.sqrt(o.weight))
        return np.array(res)

    def residuals(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        if x is not None:
            for p, xv in zip(self.params, x):
                p.set(xv)
        t0 = time.time()
        levels = self._run_solver()
        r = self.match(levels)
        self.eval_count += 1
        matched = [o for o in self.obs if o.matched]
        rms = float(np.sqrt(np.mean([(o.calc - o.energy) ** 2 for o in matched]))) \
            if matched else float('nan')
        wrms = float(np.sqrt(np.mean(r ** 2))) if len(r) else float('nan')
        self.history.append({'eval': self.eval_count, 'rms': rms,
                             'wrms': wrms, 'unmatched': self.n_unmatched,
                             't': time.time() - t0})
        self._jac_base_key = tuple(p.get() for p in self.params)

        # [viz] snapshot for reporting: kept in memory (pyduo.reporting.
        # FitReport / prob.report) and optionally streamed to monitor_path so
        # a report can be rendered at ANY time, fit-in-progress included.
        rec = {'eval': self.eval_count, 'rms': rms, 'wrms': wrms,
               'unmatched': self.n_unmatched,
               'params': {p.label: p.get() for p in self.params},
               'obs': [{'kind': o.kind, 'J': o.J, 'p': o.parity,
                        'st': o.state, 'v': o.v, 'obs': o.energy,
                        'calc': o.calc, 'w': o.weight,
                        'pur': round(o.purity, 4)}
                       for o in matched]}
        # memory-bounded: per-obs data kept only on the FIRST and LATEST
        # snapshots; the trajectory needs just the summaries + params
        if len(self.snapshots) > 1:
            self.snapshots[-1] = {k: v for k, v in self.snapshots[-1].items()
                                  if k != 'obs'}
        self.snapshots.append(rec)
        if getattr(self, 'monitor_path', None):
            import json
            with open(self.monitor_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        if self.verbose:
            print(f'  eval {self.eval_count:3d}: RMS={rms:10.4f}  '
                  f'wRMS={wrms:10.4f}  matched={len(matched)}  '
                  f'unmatched={self.n_unmatched}  ({time.time() - t0:.1f} s)')
        return r

    # ---------------- Hellmann-Feynman Jacobian ----------------
    # dE_i/dp = <psi_i| dH/dp |psi_i> with the contracted basis frozen at the
    # current point: NO eigensolves. H is LINEAR in each coupling field's
    # matrix-element cache, so dH/dp for a coupling parameter is assembled
    # with a doctored cache set (only that field's curve-derivative, all
    # other terms zeroed); for a potential parameter dH is the <v|dV/dp|v'>
    # block over the same-(state, Lambda, Sigma) mask directly.

    _COUPLING_CACHES = ['_so_matelem_cache', '_lx_matelem_cache',
                        '_ss_matelem_cache', '_sr_matelem_cache',
                        '_bobrot_matelem_cache', '_bobvib_matelem_cache',
                        '_ldopq_matelem_cache', '_ldp2q_matelem_cache',
                        '_ldq_matelem_cache']
    _FIELD_TO_CACHE = {'spinorbit': '_so_matelem_cache',
                       'lxly': '_lx_matelem_cache',
                       'spinspin': '_ss_matelem_cache',
                       'spinrot': '_sr_matelem_cache',
                       'bobrot': '_bobrot_matelem_cache',
                       'bobvib': '_bobvib_matelem_cache',
                       'lambdaopq': '_ldopq_matelem_cache',
                       'lambdap2q': '_ldp2q_matelem_cache',
                       'lambdaq': '_ldq_matelem_cache'}

    def _field_slot(self, hb, fld):
        """Locate (builder list name, index) of a live field object."""
        for lname in self._FIELD_TO_CACHE:
            lst = getattr(hb, lname)
            for k, f in enumerate(lst):
                if f is fld:
                    return lname, k
        return None, None

    def _dcache_for_param(self, hb, p, rel_step=1e-6):
        """(cache_name, index, dM) for a coupling parameter: the field's
        matelem recomputed with dcurve/dp via the EXISTING precompute recipes
        (run twice with p +- delta; all scalings included for free)."""
        lname, k = self._field_slot(hb, p.field_obj)
        if lname is None:
            return None
        cname = self._FIELD_TO_CACHE[lname]
        x0 = p.get()
        delta = rel_step * max(1.0, abs(x0))
        caches = {}
        for sgn in (+1, -1):
            p.set(x0 + sgn * delta)
            p.field_obj._molpro_factor_cache = None  # params may shift nothing here, but stay safe
            hb._precompute_batched_gpu_data(self._last_jlist, self._last_maxidx)
            caches[sgn] = getattr(hb, cname)[k]
        p.set(x0)
        p.field_obj._molpro_factor_cache = None
        hb._precompute_batched_gpu_data(self._last_jlist, self._last_maxidx)
        dM = (caches[+1] - caches[-1]) / (2.0 * delta)
        return cname, k, dM

    def _dh_for_poten(self, hb, p, rel_step=1e-6):
        """dH tensor contribution of a potential parameter with frozen basis:
        <v| dV/dp |v'> over the same-(state, Lambda, Sigma) mask (J-indep)."""
        import cupy as cp
        x0 = p.get()
        delta = rel_step * max(1.0, abs(x0))
        r = hb._r_gpu
        p.set(x0 + delta)
        vp = p.field_obj.evaluate(r, use_gpu=True)
        p.set(x0 - delta)
        vm = p.field_obj.evaluate(r, use_gpu=True)
        p.set(x0)
        # dE_cm = <psi| dV_cm * sc |psi> / sc = <psi| dV_cm |psi> -- the sc
        # scaling cancels against the eigenvalue de-scaling, so dM is built
        # directly in cm-1.  [validated: cosine 1.000000 vs central FD]
        dV = (cp.asarray(vp) - cp.asarray(vm)) / (2.0 * delta)
        wf = hb._wf_gpu
        dM = wf.T @ (dV.reshape(-1, 1) * wf)          # (nvib, nvib)
        return dM, p.field_obj.istate

    def hf_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the weighted residual vector at x, no eigensolves."""
        import cupy as cp
        for p, xv in zip(self.params, x):
            p.set(xv)
        # base solve at x (captures blocks + level map); reuse if x unchanged
        key = tuple(float(v) for v in x)
        if getattr(self, '_jac_base_key', None) != key:
            self.residuals()      # runs solver with capture_blocks
            self._jac_base_key = key
        hb = self.solver.hamiltonian
        blocks = hb._last_blocks
        j_list = self._last_jlist

        # map matched obs -> (J, parity_char, iroot) row order of residuals
        rows = []
        for o in self.obs:
            if not self._obs_in_scope(o):
                continue
            if o.matched:
                rows.append(o)

        nres = len(rows)
        J_mat = np.zeros((nres, len(self.params)))

        for ip, p in enumerate(self.params):
            if p.label.startswith('poten'):
                dM, istate_p = self._dh_for_poten(hb, p)
                dM_np = cp.asnumpy(dM)
                dE = {}
                for b in blocks:
                    vb = [self._active_basis[i] for i in b['valid_indices']]
                    idx = [k for k, bs in enumerate(vb) if bs.istate == istate_p]
                    if not idx:
                        dE[(b['J'], b['parity'])] = np.zeros(b['V'].shape[1])
                        continue
                    ivibs = np.array([vb[k].ivib for k in idx])
                    lam = np.array([vb[k].lambda_ for k in idx])
                    sig = np.array([vb[k].sigma for k in idx])
                    sub = dM_np[np.ix_(ivibs, ivibs)]
                    same = ((lam[:, None] == lam[None, :]) &
                            (np.abs(sig[:, None] - sig[None, :]) < 0.01))
                    sub = np.where(same, sub, 0.0)
                    Pm = b['P'][idx, :]               # (nidx, Nblock)
                    Hs = Pm.T @ sub @ Pm              # (Nblock, Nblock)
                    V = b['V']
                    dE[(b['J'], b['parity'])] = np.einsum(
                        'ki,kl,li->i', V, Hs, V, optimize=True)
            else:
                out = self._dcache_for_param(hb, p)
                if out is None:
                    continue
                cname, k, dM = out
                # assemble dH with ONLY this field's derivative live
                saved = {}
                try:
                    for cn in self._COUPLING_CACHES:
                        cur = getattr(hb, cn, None)
                        saved[cn] = cur
                        if cur is not None:
                            setattr(hb, cn, [None] * len(cur))
                    lst = [None] * len(saved[cname])
                    lst[k] = dM
                    setattr(hb, cname, lst)
                    saved['_energies_gpu'] = hb._energies_gpu
                    saved['_brot_matelem_gpu'] = hb._brot_matelem_gpu
                    hb._energies_gpu = cp.zeros_like(hb._energies_gpu)
                    hb._brot_matelem_gpu = cp.zeros_like(hb._brot_matelem_gpu)
                    hb._assembly_only = True
                    import warnings as _w
                    with _w.catch_warnings():
                        _w.simplefilter('ignore')
                        dH = hb.solve_all_J_batched(j_list)
                finally:
                    hb._assembly_only = False
                    for cn, val in saved.items():
                        setattr(hb, cn, val)
                # assembled terms are in Duo's scaled units (each term carries
                # *sc); eigenvalues are de-scaled by /sc -> divide dE by sc
                inv_sc = 1.0 / hb._sc
                dE = {}
                for b in blocks:
                    sub = dH[b['jidx']][cp.ix_(
                        cp.asarray(b['valid_indices']),
                        cp.asarray(b['valid_indices']))]
                    Pg = cp.asarray(b['P'])
                    Hs = cp.asnumpy(Pg.T @ sub @ Pg) * inv_sc
                    V = b['V']
                    dE[(b['J'], b['parity'])] = np.einsum(
                        'ki,kl,li->i', V, Hs, V, optimize=True)

            for ir, o in enumerate(rows):
                d = dE.get((o.J, '+' if o.parity > 0 else '-'))
                val = d[o._iroot] if (d is not None and
                                      0 <= o._iroot < len(d)) else 0.0
                if o.kind == 'freq':
                    d2 = dE.get((o.J2, '+' if o.parity2 > 0 else '-'))
                    val2 = d2[o._iroot2] if (d2 is not None and
                                             0 <= o._iroot2 < len(d2)) else 0.0
                    val = val - val2       # d(freq)/dp = dE_up/dp - dE_low/dp
                J_mat[ir, ip] = val * np.sqrt(o.weight)
        return J_mat

    # ---------------- reporting ----------------
    @property
    def report(self):
        """Notebook-friendly FitReport bound to this problem's snapshots."""
        from pyduo.reporting import FitReport
        return FitReport.from_problem(self)

    # ---------------- driver ----------------
    def fit(self, loss: str = 'soft_l1', f_scale: float = 1.0,
            max_nfev: Optional[int] = None, diff_step: float = 1e-4,
            select: Optional[List[str]] = None, use_hf_jac: bool = False,
            bounds: Optional[dict] = None) -> 'FitResult':
        """Trust-region least squares over the registered fit parameters.

        bounds : optional {label_substring: (lo, hi)} -- physical constraints
            (e.g. keep RE positive, cap a coupling strength). Unlisted
            parameters are unbounded.
        Returns a FitResult (uncertainties / correlation / conditioning).
        """
        from scipy.optimize import least_squares
        params = self.params if select is None else \
            [p for p in self.params if any(s in p.label for s in select)]
        if not params:
            raise ValueError('no fit parameters selected')
        saved_all = self.params
        self.params = params
        x0 = np.array([p.get() for p in params])
        lo = np.full(len(params), -np.inf)
        hi = np.full(len(params), np.inf)
        if bounds:
            for k, p in enumerate(params):
                for key, (a, b) in bounds.items():
                    if key in p.label:
                        lo[k], hi[k] = a, b
            x0 = np.clip(x0, lo, hi)
        if self.verbose:
            print(f'fitting {len(params)} parameters, loss={loss}, '
                  f'jac={"hellmann-feynman" if use_hf_jac else "2-point FD"}'
                  f'{", bounded" if bounds else ""}')
        try:
            kwargs = dict(method='trf', loss=loss, f_scale=f_scale,
                          x_scale='jac', max_nfev=max_nfev,
                          bounds=(lo, hi))
            if use_hf_jac:
                kwargs['jac'] = self.hf_jacobian
            else:
                kwargs['diff_step'] = diff_step
            result = least_squares(self.residuals, x0, **kwargs)
        finally:
            self.params = saved_all
        for p, xv in zip(params, result.x):
            p.set(xv)
        return FitResult(result, params)
