"""Duo-like output files for a PyDuo run (plain text / CSV, numpy only).

Duo leaves behind an .out log plus the refined .inp; a PyDuo run should leave
the same audit trail without the user writing any plotting glue:

    energies.dat    the computed level list, Duo's (J, parity, n) ordering
    curves.csv      every live curve on the solver's own r grid
    obs_calc.csv    one row per in-scope observation: obs, calc, residual
    model_final.inp the ORIGINAL input with the fitted numbers substituted
    fit_stats.txt   FitResult.pretty() (uncertainties + conditioning)

Design notes:
- model_final.inp is a textual SUBSTITUTION, not a re-serialisation: comments,
  ab initio blocks, the FITTING section and the `fit` markers all survive, so
  the file is a drop-in restart (or a Duo input) rather than a lossy dump. The
  parameter lines are located with the SAME regexes and the SAME section walk
  FittingProblem.build_registry uses, so exactly the registered parameters are
  rewritten.
- curves are evaluated on the CPU (use_gpu=False) so the writers work whether
  or not the solve ran on the GPU; a coupling's cached MOLPRO factor is reset
  first, as in the Hellmann-Feynman path, because the factor depends on the
  field's (possibly refitted) state.
"""
import os
import warnings
from typing import List, Optional

import numpy as np

from .fitting import _SECTION_RE, _FIT_LINE_RE

__all__ = ['write_energies', 'write_curves', 'write_obs_calc',
           'render_fitted_input', 'write_model_input', 'write_fit_stats']


# ---------------------------------------------------------------- helpers ----
def _parity_char(p) -> str:
    return '+' if p > 0 else '-'


def _g(x) -> str:
    """Compact CSV number (10 significant figures)."""
    return f'{float(x):.10g}'


def _f6(x) -> str:
    return '' if x is None else f'{float(x):.6f}'


def _fmt_param(x: float) -> str:
    """Fit-parameter value for an input file: 12 significant figures in the
    E-notation Duo inputs already use (round-trips to ~1e-12 relative)."""
    return f'{float(x):.11E}'


# field-type -> column prefix; the (i, j) state pair is appended, giving the
# names a Duo user already types: SO13, lx12, bobrot55, ss33
_TYPE_PREFIX = {
    'POTEN': 'poten', 'SPINORBIT': 'SO', 'L2': 'L2', 'LXLY': 'lx',
    'SPINSPIN': 'ss', 'SPINSPINO': 'sso', 'SPINROT': 'sr',
    'BOBROT': 'bobrot', 'BOBVIB': 'bobvib', 'DIABATIC': 'diab',
    'NAC': 'nac', 'LAMBDAOPQ': 'lopq', 'LAMBDAP2Q': 'lp2q',
    'LAMBDAQ': 'lq', 'DIPOLE': 'dip',
}


def _field_column(fld) -> str:
    ft = getattr(fld, 'field_type', None)
    tname = str(getattr(ft, 'name', ft)).upper()
    prefix = _TYPE_PREFIX.get(tname, tname.lower())
    return f'{prefix}{fld.istate}{fld.jstate}'


def _label_prefix(header: List[str]) -> str:
    """Registry label prefix implied by a section header -- build_registry's
    naming, reproduced from the header words alone."""
    if header[0] == 'poten':
        return f'poten{header[1]}'
    if header[0].startswith('spin-orbit'):
        return f'SO{header[1]}{header[2]}'
    return f'{header[0]}{header[1]}{header[2]}'


# --------------------------------------------------------------- energies ----
def write_energies(levels, path: str, zpe: Optional[float] = None) -> str:
    """Duo-style level table sorted by (J, parity [+ first], energy).

    `n` is the running index within each (J, parity) block -- Duo's N column,
    which is what an ENERGIES fitting block quotes.  `zpe`, when given, is
    subtracted from every energy (the solver already references its levels to
    the ZPE, so leave it None for solver output).
    """
    rows = sorted(levels, key=lambda l: (l.J, -l.parity, l.energy))
    shift = 0.0 if zpe is None else float(zpe)
    lines = []
    if zpe is not None:
        lines.append(f'# energies shifted by zpe = {shift:.8f} cm-1')
    lines.append(f'#{"J":>6s} {"p":>2s} {"n":>5s} {"energy_cm1":>16s} '
                 f'{"state":>5s} {"v":>4s} {"lambda":>7s} {"sigma":>6s} '
                 f'{"omega":>6s} {"purity":>8s} {"band":>5s} {"v_nodes":>7s} '
                 f'name')
    n = 0
    key = None
    for l in rows:
        k = (l.J, l.parity)
        if k != key:
            key, n = k, 0
        n += 1
        vn = -1 if getattr(l, 'v_nodes', None) is None else int(l.v_nodes)
        lines.append(f'{l.J:7.1f} {_parity_char(l.parity):>2s} {n:5d} '
                     f'{l.energy - shift:16.6f} {l.state:5d} {l.v:4d} '
                     f'{l.lambda_:7d} {l.sigma:6.1f} {l.omega:6.1f} '
                     f'{l.purity:8.4f} {l.band_id:5d} {vn:7d} {l.name}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


# ----------------------------------------------------------------- curves ----
def write_curves(solver, path: str) -> str:
    """Every potential and every live coupling curve on solver.grid.r (CSV).

    Columns: r_angstrom, poten<istate>..., then one column per coupling field
    (SO13, lx12, bobrot55, ...). Evaluated on the CPU so a GPU solve needs no
    special handling; fields that cannot be evaluated on the solver's grid are
    skipped with a warning rather than aborting the whole file.
    """
    grid = solver.grid
    r = grid.get_numpy() if hasattr(grid, 'get_numpy') else np.asarray(grid.r)
    r = np.asarray(r, dtype=float)
    parser = solver.parser

    names, cols, used = [], [], {}

    def _add(name, vals):
        k = used.get(name, 0) + 1
        used[name] = k
        names.append(name if k == 1 else f'{name}_{k}')
        cols.append(vals)

    for fld in (getattr(parser, 'potentials', None) or []):
        try:
            v = np.asarray(fld.evaluate(r, use_gpu=False), dtype=float)
        except Exception as e:                      # never lose the whole file
            warnings.warn(f'write_curves: poten{fld.istate} not evaluated ({e})')
            continue
        if v.size != r.size:
            warnings.warn(f'write_curves: poten{fld.istate} returned '
                          f'{v.size} values for {r.size} grid points -- skipped')
            continue
        _add(f'poten{fld.istate}', v)

    cpl = list(getattr(parser, 'spinorbit', None) or []) + \
        list(getattr(parser, 'couplings', None) or [])
    for fld in cpl:
        name = _field_column(fld)
        try:
            fld._molpro_factor_cache = None   # factor depends on live lambdas
            v = np.asarray(fld.evaluate(r, use_gpu=False), dtype=float)
        except Exception as e:
            warnings.warn(f'write_curves: {name} not evaluated ({e})')
            continue
        if v.ndim == 0:
            v = np.full(r.shape, float(v))
        if v.size != r.size:
            warnings.warn(f'write_curves: {name} returned {v.size} values '
                          f'for {r.size} grid points -- skipped')
            continue
        _add(name, v)

    with open(path, 'w') as f:
        f.write(','.join(['r_angstrom'] + names) + '\n')
        for i in range(r.size):
            f.write(','.join([_g(r[i])] + [_g(c[i]) for c in cols]) + '\n')
    return path


# --------------------------------------------------------------- obs/calc ----
_OBS_HEADER = ('kind,J,parity,state,v,lambda,sigma,omega,'
               'J2,parity2,state2,v2,lambda2,sigma2,omega2,'
               'obs,calc,resid,weight,purity,matched')


def write_obs_calc(prob, path: str) -> str:
    """One CSV row per IN-SCOPE observation (prob.j_filter applied).

    The *2 columns hold the lower level of a frequency row and are blank for
    energy rows; `resid` is calc-obs and is blank when the observation could
    not be matched (an unmatched row carries a penalty residual internally,
    which is a fit device, not a measured discrepancy).
    """
    lines = [_OBS_HEADER]
    for o in prob.obs:
        if not prob._obs_in_scope(o):
            continue
        if o.kind == 'freq':
            low = [f'{o.J2:g}', _parity_char(o.parity2), str(o.state2),
                   str(o.v2), str(o.lambda2), f'{o.sigma2:g}', f'{o.omega2:g}']
        else:
            low = [''] * 7
        resid = _f6(o.calc - o.energy) if (o.matched and o.calc is not None) else ''
        lines.append(','.join(
            [o.kind, f'{o.J:g}', _parity_char(o.parity), str(o.state),
             str(o.v), str(o.lambda_), f'{o.sigma:g}', f'{o.omega:g}'] + low +
            [_f6(o.energy), _f6(o.calc), resid, _g(o.weight),
             f'{o.purity:.4f}', '1' if o.matched else '0']))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


# ---------------------------------------------------- fitted input file ------
def render_fitted_input(raw_text: str, params) -> str:
    """The original input text with every registry parameter's number replaced
    by its CURRENT value. Everything else -- layout, comments, `fit` markers,
    ab initio blocks, the FITTING section -- is byte-identical.

    Parameter lines are found by walking the sections with FittingProblem's own
    _SECTION_RE / _FIT_LINE_RE and matching (label prefix, parameter name), so
    a line build_registry declined to register (PL/PR/NL/NR, a name that maps
    nowhere) is left exactly as it was.
    """
    pending = {}
    for p in params:
        pending.setdefault((p.label.rsplit(':', 1)[0], p.name.upper()),
                           []).append(p)

    edits = []          # (start, end, replacement) in raw_text coordinates
    for m in _SECTION_RE.finditer(raw_text):
        header = m.group(1).lower().split()
        prefix = _label_prefix(header)
        off = m.start(2)
        for fm in _FIT_LINE_RE.finditer(m.group(2)):
            key = (prefix, fm.group(1).upper())
            queue = pending.get(key)
            if not queue:
                continue                     # not a registered fit parameter
            p = queue.pop(0)
            edits.append((off + fm.start(2), off + fm.end(2),
                          _fmt_param(p.get())))

    out, pos = [], 0
    for start, end, text in sorted(edits):
        out.append(raw_text[pos:start])
        out.append(text)
        pos = end
    out.append(raw_text[pos:])
    return ''.join(out)


def write_model_input(prob, path: str) -> str:
    """render_fitted_input(prob._raw, prob.params) written to `path`."""
    with open(path, 'w') as f:
        f.write(render_fitted_input(prob._raw, prob.params))
    return path


# ------------------------------------------------------------- fit stats -----
def write_fit_stats(fit_result, path: str) -> str:
    """FitResult.pretty() -- parameter values, 1-sigma uncertainties and the
    SVD conditioning report."""
    if fit_result is None:
        text = '# no fit result (no fit has been run)'
    else:
        try:
            text = fit_result.pretty()
        except Exception as e:
            text = f'# fit statistics unavailable: {e!r}'
    with open(path, 'w') as f:
        f.write(text.rstrip('\n') + '\n')
    return path
