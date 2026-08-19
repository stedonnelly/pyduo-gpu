"""Observation <-> calculated-level assignment for population-batched objectives.

Three independent pieces:

`frozen_match(prob)`
    Harvests the assignment a FittingProblem is CURRENTLY carrying (after a
    successful `prob.residuals()`) into a compact, parameter-free description:
    per in-scope observation, which (J, parity) block and which eigen-root of
    that block supplies the upper level (and, for a frequency observation, the
    lower level). A whole population of trial models can then be scored without
    re-running quantum-number assignment per member: every member reads the
    SAME roots. This is the population analogue of what fitting.py's
    Hellmann-Feynman Jacobian already does (`o._iroot` lookups against a base
    point) and it inherits the same caveat: the assignment is only valid inside
    the neighbourhood of the centre where no level crossing has reordered the
    block. Re-harvest (a fresh centre) when the population has drifted.

`monotone_match(E_calc, E_obs)`
    Optimal 1-D assignment of sorted observed ENERGIES into sorted calculated
    energies. In one dimension with a convex cost the optimal assignment is
    order-preserving, so the O(n*m) non-crossing DP below returns exactly the
    Hungarian optimum -- see `hungarian_match` for the reference used by the
    tests. Vectorised over the population axis (the DP recursion is a scan over
    the CALC axis, with the OBS axis and the batch axis both vectorised).

    ENERGY-MODE ONLY. A frequency observation's residual is a DIFFERENCE of two
    levels living in two different (J, parity) blocks; monotonicity within one
    block is not the right optimality structure for it, and re-assigning the two
    levels independently can silently pair an upper level of one band with the
    lower level of another. Frequency rows always keep the frozen roots.

`hungarian_match(E_calc, E_obs)`
    scipy.optimize.linear_sum_assignment reference (O(n^3), host only).
"""
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp
except ImportError:                        # pragma: no cover
    cp = None

__all__ = ['FrozenMatch', 'frozen_match', 'monotone_match', 'hungarian_match']

# residual-row slots
SLOT_UPPER = 0        # the level of an energy obs, or the upper level of a freq
SLOT_LOWER = 1        # the lower level of a freq obs


def _parity_char(p) -> str:
    return '+' if p > 0 else '-'


@dataclass
class FrozenMatch:
    """Parameter-free description of one residual vector.

    Row order is the FIXED-LENGTH convention of `FittingProblem.match`: one row
    per in-scope observation, in `prob.obs` order. Rows whose observation was
    unmatched at the centre carry a constant `penalty * sqrt(weight)` for every
    population member (never re-tried per member: the whole point of freezing).
    """
    keys: List[Tuple[float, str]]              # blocks referenced, sorted
    obs_index: np.ndarray                      # (nrow,) index into prob.obs
    is_freq: np.ndarray                        # (nrow,) bool
    matched: np.ndarray                        # (nrow,) bool
    sqrt_w: np.ndarray                         # (nrow,) sqrt(weight)
    target: np.ndarray                         # (nrow,) observed value, cm-1
    kb_up: np.ndarray                          # (nrow,) index into keys, -1
    ir_up: np.ndarray                          # (nrow,) eigen-root index, -1
    kb_lo: np.ndarray                          # (nrow,) freq only, else -1
    ir_lo: np.ndarray
    penalty: float
    block_roots: Dict[Tuple[float, str], np.ndarray]   # unique roots per block
    block_refs: Dict[Tuple[float, str], Tuple[np.ndarray, np.ndarray,
                                              np.ndarray]]
    zpe_key: Optional[Tuple[float, str]] = None
    zpe_iroot: int = -1
    j_filter: Optional[float] = None

    @property
    def nrow(self) -> int:
        return len(self.obs_index)

    @property
    def blocks(self) -> Dict[Tuple[float, str], Tuple[np.ndarray, np.ndarray,
                                                      np.ndarray]]:
        """{(J, parity): (row_indices, iroots, slots)} -- every residual row
        that reads this block, with the root it reads and whether that root is
        the row's upper (0) or lower (1) level."""
        return self.block_refs

    def energy_rows(self, key) -> np.ndarray:
        """Rows of `key` that are ENERGY-mode observations (the rows
        `monotone_match` may re-assign)."""
        rows, _iroots, slots = self.block_refs[key]
        keep = (~self.is_freq[rows]) & (slots == SLOT_UPPER)
        return rows[keep]

    def summary(self) -> str:
        return (f'FrozenMatch: {self.nrow} rows '
                f'({int(self.matched.sum())} matched, '
                f'{int((~self.matched).sum())} penalty), '
                f'{len(self.keys)} blocks, '
                f'{"freq" if self.is_freq.any() else "energy"}'
                f'{"+energy" if (self.is_freq.any() and (~self.is_freq).any()) else ""}'
                f', max root {max((int(r.max()) for r in self.block_roots.values() if r.size), default=-1)}')


def frozen_match(prob) -> FrozenMatch:
    """Harvest the current obs<->root assignment of a FittingProblem.

    Requires a completed `prob.residuals()` at the point to be used as the
    population centre (that is what sets `o.matched` / `o._iroot` /
    `o._iroot2`). Raises if the last evaluation was a guarded (penalty-surface)
    one, because such an evaluation carries no roots at all.
    """
    if getattr(prob, '_guarded', True):
        raise RuntimeError('frozen_match needs a successful (non-guarded) '
                           'prob.residuals() at the centre')
    rows = [(i, o) for i, o in enumerate(prob.obs) if prob._obs_in_scope(o)]
    if not rows:
        raise ValueError('no observations in scope')

    keyset = set()
    for _i, o in rows:
        if not o.matched:
            continue
        keyset.add((float(o.J), _parity_char(o.parity)))
        if o.kind == 'freq':
            keyset.add((float(o.J2), _parity_char(o.parity2)))

    # the ZPE reference of an energy-mode residual is itself a level: freeze
    # its (block, root) exactly like fitting.hf_jacobian's `zloc`
    zpe_key, zpe_iroot = None, -1
    levels = getattr(prob, '_last_levels', None) or []
    if levels:
        zl = min(levels, key=lambda l: l.energy)
        zpe_key = (float(zl.J), _parity_char(zl.parity))
        zpe_iroot = int(getattr(zl, 'block_iroot', -1))
        keyset.add(zpe_key)

    keys = sorted(keyset)
    kidx = {k: i for i, k in enumerate(keys)}

    n = len(rows)
    obs_index = np.array([i for i, _o in rows], dtype=np.int64)
    is_freq = np.array([o.kind == 'freq' for _i, o in rows], dtype=bool)
    matched = np.array([bool(o.matched) for _i, o in rows], dtype=bool)
    sqrt_w = np.array([np.sqrt(o.weight) for _i, o in rows], dtype=np.float64)
    target = np.array([float(o.energy) for _i, o in rows], dtype=np.float64)
    kb_up = np.full(n, -1, dtype=np.int64)
    ir_up = np.full(n, -1, dtype=np.int64)
    kb_lo = np.full(n, -1, dtype=np.int64)
    ir_lo = np.full(n, -1, dtype=np.int64)

    for r, (_i, o) in enumerate(rows):
        if not o.matched:
            continue
        ir = int(getattr(o, '_iroot', -1))
        if ir < 0:
            matched[r] = False           # matched by QN but no root captured
            continue
        kb_up[r] = kidx[(float(o.J), _parity_char(o.parity))]
        ir_up[r] = ir
        if o.kind == 'freq':
            ir2 = int(getattr(o, '_iroot2', -1))
            if ir2 < 0:
                matched[r] = False
                kb_up[r] = ir_up[r] = -1
                continue
            kb_lo[r] = kidx[(float(o.J2), _parity_char(o.parity2))]
            ir_lo[r] = ir2

    block_refs = {}
    block_roots = {}
    for k, ki in kidx.items():
        r_up = np.nonzero((kb_up == ki) & matched)[0]
        r_lo = np.nonzero((kb_lo == ki) & matched)[0]
        rr = np.concatenate([r_up, r_lo])
        ii = np.concatenate([ir_up[r_up], ir_lo[r_lo]])
        ss = np.concatenate([np.full(len(r_up), SLOT_UPPER, dtype=np.int64),
                             np.full(len(r_lo), SLOT_LOWER, dtype=np.int64)])
        block_refs[k] = (rr, ii, ss)
        roots = np.unique(ii) if ii.size else np.zeros(0, dtype=np.int64)
        if k == zpe_key and zpe_iroot >= 0:
            roots = np.unique(np.append(roots, zpe_iroot))
        block_roots[k] = roots.astype(np.int64)

    return FrozenMatch(
        keys=keys, obs_index=obs_index, is_freq=is_freq, matched=matched,
        sqrt_w=sqrt_w, target=target, kb_up=kb_up, ir_up=ir_up,
        kb_lo=kb_lo, ir_lo=ir_lo,
        penalty=float(prob.unmatched_penalty),
        block_roots=block_roots, block_refs=block_refs,
        zpe_key=zpe_key, zpe_iroot=zpe_iroot,
        j_filter=prob.j_filter)


# =====================================================================
# 1-D optimal assignment
# =====================================================================

def monotone_match(E_calc, E_obs, return_cost: bool = False):
    """Optimal order-preserving injection of `E_obs` into `E_calc`.

    E_calc : (n,) or (B, n) ASCENDING calculated energies (eigh output order).
    E_obs  : (m,) observed energies, any order (sorted internally).

    Returns `idx` (always a HOST int64 array, even for device input) of shape
    (m,) or (B, m): `idx[..., i]` is the index into
    E_calc assigned to `E_obs[i]` (in the caller's E_obs order), minimising
    sum_i (E_calc[idx[i]] - E_obs[i])^2 over all strictly increasing maps.
    Because the cost is convex in one dimension the optimal assignment cannot
    cross, so this equals the unrestricted (Hungarian) optimum -- gated against
    `hungarian_match` in tools/gate_population.py.

    ENERGY-MODE ONLY (see module docstring). The DP is always run in float64.
    """
    xp = cp if (cp is not None and isinstance(E_calc, cp.ndarray)) else np
    Ec = xp.asarray(E_calc, dtype=xp.float64)
    single = (Ec.ndim == 1)
    if single:
        Ec = Ec[None, :]
    B, n = Ec.shape
    Eo = xp.asarray(E_obs, dtype=xp.float64).ravel()
    m = int(Eo.size)
    if m == 0:
        out = xp.zeros((B, 0), dtype=xp.int64)
        return (out[0] if single else out, 0.0) if return_cost else \
            (out[0] if single else out)
    if m > n:
        raise ValueError(f'monotone_match: {m} observations cannot be injected '
                         f'into {n} calculated levels')
    order = xp.argsort(Eo)
    Eo_s = Eo[order]

    # f[i][j] = min cost of assigning obs 0..i inside calc 0..j
    #         = min(f[i][j-1], f[i-1][j-1] + (Ec[j]-Eo[i])^2)
    # g holds column j-1 with g[0] == f[-1][*] == 0 (no obs assigned yet).
    g = xp.full((B, m + 1), xp.inf, dtype=xp.float64)
    g[:, 0] = 0.0
    take = xp.zeros((B, m, n), dtype=bool)
    for j in range(n):
        d2 = (Ec[:, j][:, None] - Eo_s[None, :]) ** 2      # (B, m)
        cand = g[:, :m] + d2                               # f[i-1][j-1] + d2
        keep = g[:, 1:]                                    # f[i][j-1]
        take[:, :, j] = cand < keep
        g[:, 1:] = xp.minimum(keep, cand)

    take_h = cp.asnumpy(take) if xp is cp else take
    assign = np.full((B, m), -1, dtype=np.int64)
    ii = np.full(B, m - 1, dtype=np.int64)
    jj = np.full(B, n - 1, dtype=np.int64)
    ar = np.arange(B)
    for _step in range(n):
        act = ii >= 0
        if not act.any():
            break
        ba = ar[act]
        t = take_h[ba, ii[act], jj[act]]
        hit = ba[t]
        if hit.size:
            assign[hit, ii[hit]] = jj[hit]
            ii[hit] -= 1
        jj[act] -= 1
    if (assign < 0).any():
        raise RuntimeError('monotone_match: backtrack failed (non-finite DP)')

    order_h = cp.asnumpy(order) if xp is cp else order
    out = np.empty_like(assign)
    out[:, order_h] = assign                    # back to caller's obs order
    cost = g[:, m]
    if xp is cp:
        cost = cp.asnumpy(cost)
    if single:
        out, cost = out[0], float(cost[0])
    return (out, cost) if return_cost else out


def hungarian_match(E_calc, E_obs, return_cost: bool = False):
    """Reference assignment via scipy.optimize.linear_sum_assignment (host,
    single member). Same contract as `monotone_match`; used by the gates to
    prove the DP attains the unrestricted optimum."""
    from scipy.optimize import linear_sum_assignment
    Ec = np.asarray(E_calc, dtype=np.float64).ravel()
    Eo = np.asarray(E_obs, dtype=np.float64).ravel()
    cost = (Ec[None, :] - Eo[:, None]) ** 2
    ri, ci = linear_sum_assignment(cost)
    out = np.full(Eo.size, -1, dtype=np.int64)
    out[ri] = ci
    return (out, float(cost[ri, ci].sum())) if return_cost else out
