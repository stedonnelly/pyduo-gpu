"""Structure-factor decomposition of the rovibronic Hamiltonian.

Every term of the assembled Hamiltonian factorises exactly as

    H(J)[(a,v),(b,w)] = C_t(J)[a,b] * M_t[ivib(a,v), ivib(b,w)]

where a,b are LEVELS -- unique (istate, Lambda, Sigma) combinations, 13 for
CaO -- C_t depends only on angular quantum numbers + J (never on any fit
parameter, never on v), and M_t is the term's contracted vibrational
matrix-element matrix (depends only on parameters). The parity projection P
never mixes v or state, so the symmetrised block keeps the same form with
Cs_t = Pi^T C_t Pi at symmetrised-level ("group") granularity.

The C_t factors are NOT re-derived from the physics: they are MEASURED by
running the existing, audited assembly (`solve_all_J_batched` with
`_assembly_only`) on a reduced basis of TWO vibrational functions per level
with unit matrix elements. Two functions (not one) because the SS/SR/BOB-vib
terms carry a different coefficient on v-diagonal vs v-off-diagonal elements
(the symmetrise-then-halve pattern in the assembly), so each (a,b) needs an
independent C_eq (v==w, same state) and C_neq (v!=w) measurement. The parity
pairing (Pi, group order, tau signs) is captured from a one-function-per-
level run of the FULL path with `capture_blocks`.

Everything here is parameter-independent: a StructureFactors object is built
once per (level structure, j_list, field angular signature) and reused across
every objective evaluation, Jacobian pass and population member. A module-
level cache keyed by that signature makes rebuilds free across the fresh
HamiltonianBuilder created by each cold fit evaluation.
"""
import copy
import warnings
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp
except ImportError:      # pragma: no cover - GPU-only module in practice
    cp = np

__all__ = ['StructureFactors', 'TermSpec', 'BlockSpec', 'SFUnsupported',
           'build_structure_factors', 'get_structure_factors']


class SFUnsupported(RuntimeError):
    """Basis/field layout the structure-factor decomposition cannot
    represent -- callers fall back to the legacy assembly."""


@dataclass
class LevelInfo:
    """One (istate, Lambda, Sigma) level of the contracted basis."""
    lev: int             # level id, order of first appearance
    istate: int
    lambda_: int
    sigma: float
    omega: float
    spin: float
    imulti: int
    name: str
    parity_pm: int
    nvib: int            # vibrational functions carried by this level's state
    offset: int          # full contracted-basis index of (lev, v=0)
    state_rank: int      # order of first appearance of istate


@dataclass
class TermSpec:
    """One factorised Hamiltonian term (a field, or the B(r) rotational set)."""
    name: str            # 'brot', 'so:2', 'lx:0', ...
    cache_attr: str      # builder cache list, or '_brot_matelem_gpu'
    field_index: int     # index within the cache list (-1 for brot)
    C_eq: 'cp.ndarray'   # (nJ, nlev, nlev) fp64, device
    C_neq: 'cp.ndarray'  # (nJ, nlev, nlev)


@dataclass
class BlockSpec:
    """Everything parameter-independent about one (J, parity) block."""
    J: float
    parity: str
    jidx: int
    n: int                        # block dimension (rows)
    group_of_row: np.ndarray      # (n,) int32
    ivib_of_row: np.ndarray       # (n,) int32 -- full contracted ivib
    # parity back-projection (P has <=2 nonzeros per column):
    prim_pos_local: np.ndarray    # (n,) local (valid-subset) primitive index
    prim_neg_local: np.ndarray    # (n,) partner index or -1
    coef_pos: np.ndarray          # (n,)
    coef_neg: np.ndarray          # (n,)
    isym_to_basis: List[int]      # per row: local index of the irrep member
    valid_indices: np.ndarray     # full-basis indices of the valid subset
    # per-term symmetrised structure factors, device:
    Cs_eq: 'cp.ndarray'           # (nterms, ngroups, ngroups)
    Cs_neq: 'cp.ndarray'
    has_eq_corr: np.ndarray       # (nterms,) bool -- any Cs_eq != Cs_neq
    # device copies of the row maps for assembly:
    group_of_row_gpu: 'cp.ndarray' = None
    ivib_of_row_gpu: 'cp.ndarray' = None
    eqmask_gpu: 'cp.ndarray' = None     # (n, n) bool: ivib_i == ivib_j


class StructureFactors:
    def __init__(self, levels: List[LevelInfo], terms: List[TermSpec],
                 blocks: Dict[Tuple[float, str], BlockSpec],
                 j_list: List[float], sc: float, signature: tuple):
        self.levels = levels
        self.terms = terms
        self.blocks = blocks
        self.j_list = list(j_list)
        self.sc = sc
        self.signature = signature

    def term_index(self, cache_attr: str, field_index: int) -> Optional[int]:
        for t_i, t in enumerate(self.terms):
            if t.cache_attr == cache_attr and t.field_index == field_index:
                return t_i
        return None

    # ------------------------------------------------------------------
    def gather_ms(self, hb) -> List['cp.ndarray']:
        """Current (parameter-dependent) M_t per term from the builder's
        live caches, in term order. Shapes (nvib, nvib), device."""
        ms = []
        for t in self.terms:
            if t.cache_attr == '_brot_matelem_gpu':
                ms.append(hb._brot_matelem_gpu)
            else:
                ms.append(getattr(hb, t.cache_attr)[t.field_index])
        return ms

    def assemble(self, key: Tuple[float, str], ms: List['cp.ndarray'],
                 energies: 'cp.ndarray', sc: float,
                 dtype=None, out: Optional['cp.ndarray'] = None,
                 term_subset: Optional[List[int]] = None,
                 include_vib: bool = True) -> 'cp.ndarray':
        """Assemble the symmetrised (n, n) block -- or (B, n, n) when the ms
        entries carry a leading population axis -- in Duo scaled units,
        ready for eigh.

        ms       : per-term M matrices, (nvib, nvib) or (B, nvib, nvib)
        energies : contracted vibrational energies (nvib,), cm-1
        term_subset : term indices to include (default all); with
            include_vib=False and a single term this assembles dH/dp pieces.
        """
        xp = cp
        bs = self.blocks[key]
        n = bs.n
        if bs.group_of_row_gpu is None:
            bs.group_of_row_gpu = xp.asarray(bs.group_of_row)
            bs.ivib_of_row_gpu = xp.asarray(bs.ivib_of_row)
            bs.eqmask_gpu = (bs.ivib_of_row_gpu[:, None] ==
                             bs.ivib_of_row_gpu[None, :])
        g = bs.group_of_row_gpu
        iv = bs.ivib_of_row_gpu

        # detect population axis
        t_ids = range(len(self.terms)) if term_subset is None else term_subset
        batched = any(ms[t] is not None and ms[t].ndim == 3 for t in t_ids)
        if dtype is None:
            dtype = xp.float64
        shape = (n, n) if not batched else \
            (max(ms[t].shape[0] for t in t_ids
                 if ms[t] is not None and ms[t].ndim == 3), n, n)
        if out is None:
            out = xp.zeros(shape, dtype=dtype)
        else:
            out[...] = 0.0

        for t_i in t_ids:
            M = ms[t_i]
            if M is None:
                continue
            Cn = bs.Cs_neq[t_i][g[:, None], g[None, :]]        # (n, n) fp64
            if M.ndim == 3:
                Mg = M[:, iv[:, None], iv[None, :]]            # (B, n, n)
                out += (Cn[None] * Mg).astype(dtype, copy=False)
            else:
                Mg = M[iv[:, None], iv[None, :]]
                out += (Cn * Mg).astype(dtype, copy=False)
            if bs.has_eq_corr[t_i]:
                Cd = (bs.Cs_eq[t_i] - bs.Cs_neq[t_i])[g[:, None], g[None, :]]
                corr = xp.where(bs.eqmask_gpu, Cd, 0.0)
                if M.ndim == 3:
                    out += (corr[None] * Mg).astype(dtype, copy=False)
                else:
                    out += (corr * Mg).astype(dtype, copy=False)

        if include_vib:
            diag = (energies[iv] * sc).astype(dtype, copy=False)
            if out.ndim == 3:
                idx = xp.arange(n)
                out[:, idx, idx] += diag[None, :]
            else:
                idx = xp.arange(n)
                out[idx, idx] += diag
        return out

    def project_to_primitive(self, key: Tuple[float, str],
                             V: np.ndarray) -> np.ndarray:
        """prim_coeffs = P @ V without materialising P (<=2 nnz per column)."""
        bs = self.blocks[key]
        nvalid = len(bs.valid_indices)
        prim = np.zeros((nvalid, V.shape[1]), dtype=V.dtype)
        np.add.at(prim, bs.prim_pos_local, bs.coef_pos[:, None] * V)
        has_neg = bs.prim_neg_local >= 0
        if has_neg.any():
            np.add.at(prim, bs.prim_neg_local[has_neg],
                      bs.coef_neg[has_neg, None] * V[has_neg])
        return prim

    def build_P(self, key: Tuple[float, str]) -> np.ndarray:
        """Materialise the legacy (Nvalid, Nblock) parity projection matrix
        (validation / capture compatibility)."""
        bs = self.blocks[key]
        P = np.zeros((len(bs.valid_indices), bs.n))
        rows = np.arange(bs.n)
        P[bs.prim_pos_local, rows] = bs.coef_pos
        has_neg = bs.prim_neg_local >= 0
        P[bs.prim_neg_local[has_neg], rows[has_neg]] = bs.coef_neg[has_neg]
        return P


# =====================================================================
# construction
# =====================================================================

_TERM_CACHE_LISTS = [
    ('spinorbit', '_so_matelem_cache', 'so'),
    ('lxly', '_lx_matelem_cache', 'lx'),
    ('spinspin', '_ss_matelem_cache', 'ss'),
    ('spinrot', '_sr_matelem_cache', 'sr'),
    ('bobrot', '_bobrot_matelem_cache', 'bobrot'),
    ('bobvib', '_bobvib_matelem_cache', 'bobvib'),
    ('lambdaopq', '_ldopq_matelem_cache', 'ldopq'),
    ('lambdap2q', '_ldp2q_matelem_cache', 'ldp2q'),
    ('lambdaq', '_ldq_matelem_cache', 'ldq'),
]

_SF_CACHE: Dict[tuple, StructureFactors] = {}


def _level_table(hb) -> List[LevelInfo]:
    levels = []
    seen = {}
    state_rank = {}
    offset = 0
    counts = {}
    # first pass: nvib per level (= per state) and order
    order = []
    for bs in hb.contracted_basis:
        key = (bs.istate, bs.lambda_, int(round(2 * bs.sigma)))
        if key not in seen:
            seen[key] = len(order)
            order.append((key, bs))
            counts[key] = 0
        counts[key] += 1
    for key, bs in order:
        if bs.istate not in state_rank:
            state_rank[bs.istate] = len(state_rank)
        levels.append(LevelInfo(
            lev=len(levels), istate=bs.istate, lambda_=bs.lambda_,
            sigma=bs.sigma, omega=bs.omega, spin=bs.spin, imulti=bs.imulti,
            name=bs.name, parity_pm=getattr(bs, 'parity_pm', 1),
            nvib=counts[key], offset=offset,
            state_rank=state_rank[bs.istate]))
        offset += counts[key]
    return levels


def _signature(hb, levels: List[LevelInfo], j_list) -> tuple:
    lev_sig = tuple((l.istate, l.lambda_, int(round(2 * l.sigma)),
                     int(round(2 * l.spin)), l.parity_pm, l.nvib)
                    for l in levels)
    term_sig = []
    for list_name, cache_attr, tag in _TERM_CACHE_LISTS:
        for k, f in enumerate(getattr(hb, list_name)):
            term_sig.append((
                tag, k, f.istate, f.jstate,
                int(getattr(f, 'lambda_i', 0) or 0),
                int(getattr(f, 'lambda_j', 0) or 0),
                int(round(2 * (getattr(f, 'sigma_i', 0.0) or 0.0))),
                int(round(2 * (getattr(f, 'sigma_j', 0.0) or 0.0))),
                int(round(2 * (getattr(f, 'spin_i', 0.0) or 0.0))),
                int(round(2 * (getattr(f, 'spin_j', 0.0) or 0.0))),
                getattr(f, 'parameters', None) is not None
                and len(f.parameters) > 0))
    return (lev_sig, tuple(term_sig), tuple(float(j) for j in j_list))


def _reduced_basis(hb, levels: List[LevelInfo], nfun: int):
    """Reduced contracted basis: `nfun` vib functions per level, preserving
    level order and the shared-ivib structure ((istate, v) -> ivib)."""
    from pyduo.hamiltonian import BasisState
    red = []
    for l in levels:
        for v in range(nfun):
            red.append(BasisState(
                istate=l.istate, v=v, ivib=nfun * l.state_rank + v,
                lambda_=l.lambda_, sigma=l.sigma, omega=l.omega,
                spin=l.spin, imulti=l.imulti, name=l.name,
                parity_pm=l.parity_pm))
    nstates = max(l.state_rank for l in levels) + 1
    return red, nfun * nstates


def _probe_copy(hb, red_basis, nvib_red):
    """Shallow builder copy with the reduced basis installed and all
    parameter-dependent caches replaced (never touches the real builder)."""
    pc = copy.copy(hb)
    pc.contracted_basis = red_basis
    # distinct fake energies (never used for C, keep diagonalisations sane)
    pc.contracted_energies = np.arange(1.0, nvib_red + 1.0) * 10.0
    pc._energies_gpu = cp.zeros(nvib_red, dtype=cp.float64)
    pc._brot_matelem_gpu = cp.zeros((nvib_red, nvib_red), dtype=cp.float64)
    for _, cache_attr, _tag in _TERM_CACHE_LISTS:
        src = getattr(hb, cache_attr, None)
        if src is None:
            setattr(pc, cache_attr, None)
        else:
            setattr(pc, cache_attr, [None] * len(src))
    pc.capture_blocks = False
    pc.track_bands = False
    pc._assembly_only = True
    pc.use_structure_factors = False   # probes must run the LEGACY assembly
    return pc


def build_structure_factors(hb, j_list: List[float],
                            verbose: int = 0) -> StructureFactors:
    """Measure C_t(J) and the block structure from the existing assembly.

    Requires the builder to have a solved contracted basis AND a completed
    precompute (so `_sc` and the field lambda transforms are final).
    """
    if getattr(hb, '_sc', None) is None:
        raise RuntimeError('build_structure_factors needs a completed '
                           'precompute (call solve/_precompute first)')
    levels = _level_table(hb)
    for l in levels:
        if l.nvib < 2:
            raise SFUnsupported(f'level {l.lev} (state {l.istate}) has nvib='
                                f'{l.nvib} < 2: structure-factor probe needs '
                                f'two vib functions per level')
    sig = _signature(hb, levels, j_list)
    sc = hb._sc
    nlev = len(levels)
    j_list = [float(j) for j in j_list]
    nJ = len(j_list)

    # ---------------- term list ----------------
    term_defs = [('brot', '_brot_matelem_gpu', -1)]
    for list_name, cache_attr, tag in _TERM_CACHE_LISTS:
        cache = getattr(hb, cache_attr, None)
        flds = getattr(hb, list_name)
        for k, f in enumerate(flds):
            live = (getattr(f, 'parameters', None) is not None
                    and len(f.parameters) > 0)
            if live:
                term_defs.append((f'{tag}:{k}', cache_attr, k))

    # ---------------- C probes (2 functions per level) ----------------
    red2, nvib2 = _reduced_basis(hb, levels, 2)
    r0 = 2 * np.arange(nlev)

    # active-level bookkeeping (assembly drops |omega| > maxJ)
    max_J = max(j_list)
    active = [l.lev for l in levels if abs(l.omega) <= max_J + 0.01]
    rank_of_lev = {lev: r for r, lev in enumerate(active)}

    terms: List[TermSpec] = []
    for name, cache_attr, k in term_defs:
        pc = _probe_copy(hb, red2, nvib2)
        ones = cp.ones((nvib2, nvib2), dtype=cp.float64)
        if cache_attr == '_brot_matelem_gpu':
            pc._brot_matelem_gpu = ones
        else:
            lst = list(getattr(pc, cache_attr))
            lst[k] = ones
            setattr(pc, cache_attr, lst)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Hred = pc.solve_all_J_batched(j_list)     # (nJ, 2*nact, 2*nact)
        C_eq = cp.zeros((nJ, nlev, nlev), dtype=cp.float64)
        C_neq = cp.zeros((nJ, nlev, nlev), dtype=cp.float64)
        act = cp.asarray(np.array(active, dtype=np.int64))
        ar0 = cp.asarray(2 * np.arange(len(active), dtype=np.int64))
        # (a, v0),(b, v0) -> eq ; (a, v0),(b, v1) -> neq
        C_eq[:, act[:, None], act[None, :]] = Hred[:, ar0[:, None], ar0[None, :]]
        C_neq[:, act[:, None], act[None, :]] = Hred[:, ar0[:, None], (ar0 + 1)[None, :]]
        # consistency: (v1, v1) must reproduce eq, (v1, v0) must reproduce neq
        d1 = float(cp.abs(Hred[:, (ar0 + 1)[:, None], (ar0 + 1)[None, :]] -
                          Hred[:, ar0[:, None], ar0[None, :]]).max())
        d2 = float(cp.abs(Hred[:, (ar0 + 1)[:, None], ar0[None, :]] -
                          Hred[:, ar0[:, None], (ar0 + 1)[None, :]]).max())
        scale = max(float(cp.abs(Hred).max()), 1e-30)
        if max(d1, d2) > 1e-10 * scale:
            raise SFUnsupported(
                f'structure-factor probe inconsistency for term {name}: '
                f'v-structure is not (eq, neq)-representable '
                f'(d1={d1:.3e}, d2={d2:.3e}, scale={scale:.3e})')
        terms.append(TermSpec(name=name, cache_attr=cache_attr,
                              field_index=k, C_eq=C_eq, C_neq=C_neq))
        del pc

    # ---------------- Pi / block capture (1 function per level) --------
    red1, nvib1 = _reduced_basis(hb, levels, 1)
    pc1 = _probe_copy(hb, red1, nvib1)
    pc1._assembly_only = False
    pc1.capture_blocks = True
    # give the full path benign data: tiny distinct diagonal, no couplings
    pc1._energies_gpu = cp.asarray(np.arange(1.0, nvib1 + 1.0))
    pc1.contracted_wavefunctions = np.ones((hb.grid.npoints, nvib1))
    from pyduo.assignment import AssignmentConfig
    pc1.assign_cfg = AssignmentConfig(node_check='off', canonicalize=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pc1.solve_all_J_batched(j_list)
    lev_blocks = pc1._last_blocks

    # stack per-term C for fast per-block slicing
    C_eq_all = cp.stack([t.C_eq for t in terms])     # (nT, nJ, nlev, nlev)
    C_neq_all = cp.stack([t.C_neq for t in terms])
    nT = len(terms)

    blocks: Dict[Tuple[float, str], BlockSpec] = {}
    for b in lev_blocks:
        J, parity, jidx = b['J'], b['parity'], b['jidx']
        # valid_indices index into the ACTIVE red1 basis == active-level ranks
        valid_ranks = list(b['valid_indices'])
        valid_levs = [active[r] for r in valid_ranks]
        Pi = np.asarray(b['P'])                       # (nlev_valid, ngroups)
        ngroups = Pi.shape[1]
        isym_lev_local = list(b['isym_to_basis'])     # per group: local lev idx

        # per-group members from Pi columns
        g_state = np.zeros(ngroups, dtype=np.int64)
        g_nvib = np.zeros(ngroups, dtype=np.int64)
        g_pos_levloc = np.zeros(ngroups, dtype=np.int64)
        g_neg_levloc = np.full(ngroups, -1, dtype=np.int64)
        g_cpos = np.zeros(ngroups)
        g_cneg = np.zeros(ngroups)
        for gi in range(ngroups):
            nz = np.nonzero(np.abs(Pi[:, gi]) > 1e-14)[0]
            memb = isym_lev_local[gi]
            g_pos_levloc[gi] = memb
            g_cpos[gi] = Pi[memb, gi]
            others = [z for z in nz if z != memb]
            if others:
                g_neg_levloc[gi] = others[0]
                g_cneg[gi] = Pi[others[0], gi]
            lv = levels[valid_levs[memb]]
            g_state[gi] = lv.istate
            g_nvib[gi] = lv.nvib

        # sanity: paired members must share (istate, nvib)
        for gi in range(ngroups):
            if g_neg_levloc[gi] >= 0:
                lo = levels[valid_levs[g_neg_levloc[gi]]]
                if lo.istate != g_state[gi] or lo.nvib != g_nvib[gi]:
                    raise RuntimeError('parity pair mixes states -- '
                                       'factorisation invalid')

        # local (valid-subset) primitive offsets, full-basis valid indices
        loc_off = {}
        acc = 0
        vfull = []
        for ll in valid_levs:
            loc_off[ll] = acc
            acc += levels[ll].nvib
            vfull.extend(range(levels[ll].offset,
                               levels[ll].offset + levels[ll].nvib))
        nvalid_prim = acc

        # expand rows: group-major, v-minor (== legacy symlevel order)
        n = int(g_nvib.sum())
        group_of_row = np.zeros(n, dtype=np.int32)
        ivib_of_row = np.zeros(n, dtype=np.int32)
        prim_pos = np.zeros(n, dtype=np.int64)
        prim_neg = np.full(n, -1, dtype=np.int64)
        cpos = np.zeros(n)
        cneg = np.zeros(n)
        isym_to_basis = []
        # full contracted ivib for (istate, v): from any level of that state
        ivib_of_state_v = {}
        for bs_ in hb.contracted_basis:
            ivib_of_state_v.setdefault((bs_.istate, bs_.v), bs_.ivib)
        row = 0
        for gi in range(ngroups):
            lp = valid_levs[g_pos_levloc[gi]]
            ln = valid_levs[g_neg_levloc[gi]] if g_neg_levloc[gi] >= 0 else -1
            for v in range(g_nvib[gi]):
                group_of_row[row] = gi
                ivib_of_row[row] = ivib_of_state_v[(g_state[gi], v)]
                prim_pos[row] = loc_off[lp] + v
                cpos[row] = g_cpos[gi]
                if ln >= 0:
                    prim_neg[row] = loc_off[ln] + v
                    cneg[row] = g_cneg[gi]
                isym_to_basis.append(prim_pos[row])
                row += 1

        # symmetrised structure factors: Cs = Pi^T C Pi (level -> group)
        levs = cp.asarray(np.array(valid_levs, dtype=np.int64))
        Pi_gpu = cp.asarray(Pi)
        Cv_eq = C_eq_all[:, jidx][:, levs[:, None], levs[None, :]]
        Cv_neq = C_neq_all[:, jidx][:, levs[:, None], levs[None, :]]
        Cs_eq = cp.einsum('ls,tlm,mu->tsu', Pi_gpu, Cv_eq, Pi_gpu)
        Cs_neq = cp.einsum('ls,tlm,mu->tsu', Pi_gpu, Cv_neq, Pi_gpu)
        has_eq = cp.asnumpy(cp.abs(Cs_eq - Cs_neq).max(axis=(1, 2)) > 1e-14)

        blocks[(J, parity)] = BlockSpec(
            J=J, parity=parity, jidx=jidx, n=n,
            group_of_row=group_of_row, ivib_of_row=ivib_of_row,
            prim_pos_local=prim_pos, prim_neg_local=prim_neg,
            coef_pos=cpos, coef_neg=cneg,
            isym_to_basis=isym_to_basis,
            valid_indices=np.array(vfull, dtype=np.int64),
            Cs_eq=Cs_eq, Cs_neq=Cs_neq, has_eq_corr=has_eq)

    sf = StructureFactors(levels, terms, blocks, j_list, sc, sig)
    if verbose:
        print(f'structure factors: {nlev} levels, {nT} terms, '
              f'{len(blocks)} blocks')
    return sf


def get_structure_factors(hb, j_list: List[float],
                          verbose: int = 0) -> StructureFactors:
    """Cached build: C_t and the block structure depend only on the level
    structure + field angular data + j_list, never on parameter values, so
    they survive across builders (each cold objective evaluation constructs
    a fresh HamiltonianBuilder)."""
    levels = _level_table(hb)
    sig = _signature(hb, levels, j_list)
    sf = _SF_CACHE.get(sig)
    if sf is None:
        sf = build_structure_factors(hb, j_list, verbose=verbose)
        _SF_CACHE.clear()      # keep at most one (they're moderately sized)
        _SF_CACHE[sig] = sf
    return sf
