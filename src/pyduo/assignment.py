"""Quantum-number assignment for symmetrized rovibronic eigenstates.

One implementation shared by both solver paths (solve_all_J_batched and
diagonalize_with_symmetry), replacing two divergent inline copies.

Methods
-------
greedy_duo : exact transcription of Duo's assignment (diatom.f90:10016-10092):
    largest |c|^2 (first-index-wins, masked by `small`), a duplicate-avoidance
    walk to the next-largest coefficient (<= 10 retries, then give up and
    accept a duplicate), labels inherited from the winning basis function.
hungarian : globally optimal one-to-one matching on the doubly stochastic
    C = evecs**2 with a -log cost (scipy.optimize.linear_sum_assignment).
    Provably identical to greedy wherever every root has max c^2 > 0.5 with
    distinct argmaxes; fixes the greedy failures at avoided crossings
    (duplicate labels, labels carried by ~0 weight, exact-tie unreachability).

v modes
-------
count (Duo's DEFAULT, diatom.f90:302, 10066-10083): v = rank of the root
    within its (istate, Lambda, Sigma) stack in ascending energy, per block.
    Guarantees a contiguous, duplicate-free ladder -- what the Duo reference
    outputs were produced with.
contribution : v of the winning basis function (Duo's ASSIGN_V_BY_CONTRIBUTION;
    pyduo's previous behaviour).

Every root also carries confidence diagnostics (purity, margin, flags) --
the quantity Duo throws away.
"""
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Sequence

import numpy as np

__all__ = ['AssignmentConfig', 'RootLabel', 'assign_block',
           'canonicalize_degenerate', 'count_radial_nodes', 'nodal_v_check',
           'track_bands']

# Duo: small_ = epsilon(1.0_rk) (accuracy.f90:29)
_DUO_SMALL = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class AssignmentConfig:
    method: str = 'greedy_duo'      # 'greedy_duo' | 'hungarian'
    v_mode: str = 'count'           # 'count' (Duo default) | 'contribution'
    small: float = _DUO_SMALL       # coefficient^2 floor (Duo's small_)
    max_dedup_attempts: int = 10    # Duo's `if (i0 > 10)` give-up
    low_purity: float = 0.5         # flag threshold on accepted c^2
    ambiguous_margin: float = 0.1   # flag threshold on p1 - p2
    canonicalize: bool = True       # gauge-fix degenerate clusters pre-assignment
    node_check: str = 'flagged'     # 'off' | 'flagged' | 'all' -- nodal v validation
    node_gate: float = 1e-3         # relative amplitude gate for node counting


@dataclass
class RootLabel:
    iroot: int                # index within the block, ascending energy
    isym: int                 # accepted symmetrized-basis row
    ibasis: int               # index into the block's basis list
    v: int                    # vibrational quantum number (per v_mode)
    v_contribution: int       # v of the winning basis function (always kept)
    purity: float             # c^2 of the accepted isym
    p1: float                 # largest c^2
    p2: float                 # second-largest c^2
    flags: List[str] = dc_field(default_factory=list)
    v_nodes: Optional[int] = None   # nodal count of the channel function


def canonicalize_degenerate(eigenvalues: np.ndarray,
                            eigenvectors: np.ndarray,
                            rel_tol: float = 1e-8) -> np.ndarray:
    """Make eigenvectors within (near-)degenerate clusters deterministic.

    Within a degenerate subspace LAPACK's eigenvectors are arbitrary up to an
    orthogonal rotation, so ANY assignment rule applied to them is arbitrary
    and depends on the eigh driver ('ev' vs 'evd' vs cuSOLVER). Fix at the
    source: for each cluster, pick a deterministic row set (column-pivoted QR)
    and Loewdin/polar-rotate the cluster onto it, making the selected
    submatrix symmetric positive definite -- a driver-independent canonical
    gauge. Non-degenerate columns only get a sign convention (largest-|.|
    component positive, matching LAPACK's usual output).

    Returns a modified COPY of eigenvectors.
    """
    import scipy.linalg as sla

    w = np.asarray(eigenvalues, dtype=float)
    V = np.array(eigenvectors, dtype=float, copy=True)
    n = len(w)
    scale = np.maximum(1.0, np.abs(w))

    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(w[j] - w[j - 1]) < rel_tol * scale[j]:
            j += 1
        k = j - i
        if k > 1:
            Uc = V[:, i:j]                                # (N, k)
            _, _, piv = sla.qr(Uc.T, pivoting=True)       # deterministic rows
            R = np.sort(piv[:k])
            # polar decomposition of the selected kxk block: Uc @ Qh makes
            # the block symmetric positive definite (Loewdin orthogonalization
            # of the gauge)
            Q, _ = sla.polar(Uc[R, :])
            V[:, i:j] = Uc @ Q.conj().T
        else:
            col = V[:, i]
            m = int(np.argmax(np.abs(col)))
            if col[m] < 0:
                V[:, i] = -col
        i = j
    return V


def count_radial_nodes(psi: np.ndarray, rel_gate: float = 1e-3) -> int:
    """Count radial nodes of a channel wavefunction on the grid.

    The nodal theorem is exact for an uncoupled 1-D channel and a strong
    heuristic for the dominant channel of a coupled state. Robustness rules
    (they matter more than the counting): sinc-DVR tails oscillate at ~1e-14
    and would produce hundreds of fake nodes, so only samples with
    |psi| > rel_gate * max|psi| participate, and counting is confined to the
    span between the first and last surviving sample.
    """
    amax = float(np.max(np.abs(psi)))
    if amax <= 0.0:
        return 0
    keep = np.abs(psi) > rel_gate * amax
    idx = np.where(keep)[0]
    if len(idx) < 2:
        return 0
    sg = np.sign(psi[idx])
    return int(np.count_nonzero(np.diff(sg) != 0))


def nodal_v_check(labels: List[RootLabel],
                  prim_coeffs: np.ndarray,
                  basis: Sequence,
                  wavefunctions: np.ndarray,
                  cfg: Optional[AssignmentConfig] = None) -> None:
    """Validate v labels by counting nodes of the reconstructed channel
    function -- the check Duo drafted and commented out (diatom.f90:10610).

    This is the only v information that does NOT come from the coefficient
    matrix: it reaches back into the grid-space shape the contraction
    compressed away, so it can decide cases (e.g. a3Pi v=1 Omega=1 vs v=2
    Omega=0) that no |c|^2-based rule can.

    Parameters
    ----------
    labels : from assign_block (modified in place: v_nodes / flags)
    prim_coeffs : (Nprim, Nroots) block eigenvectors in the PRIMITIVE basis
        (= P @ block_eigenvectors, the same P as the hsym projection)
    basis : the block's primitive basis (BasisState-like, .ivib per function)
    wavefunctions : (ngrid, nvib) contracted radial functions, column = ivib
    cfg.node_check : 'off' | 'flagged' (default) | 'all'
    """
    cfg = cfg or AssignmentConfig()
    if cfg.node_check == 'off':
        return
    trigger = {'LOW_PURITY', 'AMBIGUOUS', 'LADDER_BREAK', 'GAVE_UP_DUPLICATE',
               'MOVED_BY_MATCHING', 'DISPLACED'}

    # channel groups: (istate, lambda, 2*sigma) -> primitive indices
    groups = {}
    for k, bs in enumerate(basis):
        groups.setdefault(
            (bs.istate, int(round(bs.lambda_)), int(round(2 * bs.sigma))),
            []).append(k)

    for lab in labels:
        if cfg.node_check == 'flagged' and not (set(lab.flags) & trigger):
            continue
        bs = basis[lab.ibasis]
        key = (bs.istate, int(round(bs.lambda_)), int(round(2 * bs.sigma)))
        idx_g = groups[key]
        ivibs = [basis[k].ivib for k in idx_g]
        psi = wavefunctions[:, ivibs] @ prim_coeffs[idx_g, lab.iroot]
        lab.v_nodes = count_radial_nodes(psi, cfg.node_gate)
        if lab.v_nodes != lab.v:
            lab.flags.append(f'V_DISAGREE(count={lab.v},nodes={lab.v_nodes})')


def track_bands(blocks, min_overlap_sq: float = 0.5):
    """Cross-J band tracking: follow eigenstates smoothly from J to J+1.

    Cures Duo failure 5.7 (every (J, parity) block labelled from scratch, so
    labels permute freely between adjacent J -- broken rotational
    progressions through avoided crossings). Bands are DIABATIC labels and
    are exported as ADDITIVE metadata (band_id / band_overlap); the per-J
    adiabatic labels are never overwritten.

    Two essentials (tracking/06):
    - match e/f, not +/-: itau flips per unit J, so for integer J the e
      family carries total parity (-1)^J -- the '+' block at even J
      continues into the '-' block at odd J;
    - overlap in the PRIMITIVE basis scattered into the fixed global basis
      index space (the symmetrized basis is J-dependent through tau and the
      |Omega| <= J filter).

    Parameters
    ----------
    blocks : list of dicts with keys
        J (float), parity ('+'/'-'), global_coeffs ((Nglobal, Nroots) --
        primitive coefficients scattered to the global basis), states
        (per-root objects that receive .band_id / .band_overlap)
    min_overlap_sq : continuation threshold; below it a BAND_BREAK starts a
        new band.

    Returns the number of bands created.
    """
    from scipy.optimize import linear_sum_assignment

    by_key = {(b['J'], b['parity']): b for b in blocks}
    js = sorted({b['J'] for b in blocks})
    next_band = 0

    for ef in (+1, -1):
        prev = None
        for J in js:
            pchar = '+' if ef * ((-1) ** int(round(J))) > 0 else '-'
            blk = by_key.get((J, pchar))
            if blk is None:
                prev = None
                continue
            n = len(blk['states'])
            band_of = [-1] * n
            if prev is not None:
                S = prev['global_coeffs'].T @ blk['global_coeffs']
                S2 = S ** 2
                rows, cols = linear_sum_assignment(-S2)
                for r, c in zip(rows, cols):
                    if S2[r, c] > min_overlap_sq:
                        band_of[c] = prev['band_of'][r]
                        blk['states'][c].band_overlap = float(S2[r, c])
            for k in range(n):
                if band_of[k] < 0:
                    band_of[k] = next_band
                    next_band += 1
                    blk['states'][k].band_overlap = 1.0
                blk['states'][k].band_id = band_of[k]
            blk['band_of'] = band_of
            prev = blk
    return next_band


def _greedy_duo(C: np.ndarray, cfg: AssignmentConfig):
    """Duo's assignment walk. C is (Nsym, Nroots) = eigenvectors**2.

    Returns (isym_per_root, gave_up_flags, displaced_flags).
    """
    nsym, nroots = C.shape
    used = [None] * nroots  # QNs
    gave_up = [False] * nroots
    displaced = [False] * nroots

    for iroot in range(nroots):
        w = C[:, iroot]
        valid = w >= cfg.small
        if np.any(valid):
            imax = int(np.argmax(np.where(valid, w, -np.inf)))
        else:
            imax = int(np.argmax(w))
        maxcontr = w[imax]
        argmax0 = imax

        i0 = 0
        while True:
            dup = False
            for prev in range(iroot):
                if used[prev] is not None and imax == used[prev]:
                    mask2 = valid & (w < maxcontr)
                    if np.any(mask2):
                        imax = int(np.argmax(np.where(mask2, w, -np.inf)))
                        maxcontr = w[imax]
                    i0 += 1
                    if i0 > cfg.max_dedup_attempts:
                        imax = int(np.argmax(np.where(valid, w, -np.inf))) \
                            if np.any(valid) else int(np.argmax(w))
                        gave_up[iroot] = True
                        break
                    dup = True
                    break
            if not dup or i0 > cfg.max_dedup_attempts:
                break

        used[iroot] = imax
        displaced[iroot] = (imax != argmax0) and not gave_up[iroot]
    return used, gave_up, displaced


def _hungarian(C: np.ndarray):
    """Globally optimal bijective assignment on -log(c^2).

    -log cost maximises the joint likelihood prod(c^2) and is
    near-lexicographic: moving a 0.95-pure root onto a 1e-6 component costs
    ~13.8 nats, which no realistic rearrangement recovers -- confident
    (Duo-correct) assignments are protected, only genuine collisions move.
    Rectangular (Nroots <= Nsym) is handled natively.
    """
    from scipy.optimize import linear_sum_assignment
    cost = -np.log(np.maximum(C, 1e-300))     # rows = isym, cols = roots
    sym_idx, root_idx = linear_sum_assignment(cost)
    out = np.empty(C.shape[1], dtype=int)
    out[root_idx] = sym_idx
    return out


def assign_block(eigenvalues: np.ndarray,
                 eigenvectors: np.ndarray,
                 isym_to_basis: Sequence[int],
                 basis: Sequence,
                 cfg: Optional[AssignmentConfig] = None) -> List[RootLabel]:
    """Assign quantum-number labels to one (J, parity) block.

    Parameters
    ----------
    eigenvalues : (Nroots,) ascending (cm-1; only the ORDER is used)
    eigenvectors : (Nsym, Nroots), columns = eigenvectors
    isym_to_basis : (Nsym,) map from symmetrized row to an index into `basis`
        (the collapsed irec + ilevel2i[...][irrep] de-reference)
    basis : sequence of BasisState-like objects (.istate, .v, .lambda_,
        .sigma, .omega, .spin, .name)
    cfg : AssignmentConfig

    Returns list[RootLabel], one per root, in root (energy) order.
    Process ALL roots of the block here -- apply any nroots / energy-range
    filtering AFTERWARDS, so it cannot corrupt the dedup walk or the v ladder.
    """
    cfg = cfg or AssignmentConfig()
    C = np.asarray(eigenvectors) ** 2
    nroots = C.shape[1]

    if cfg.method == 'hungarian':
        isyms = _hungarian(C)
        gave_up = [False] * nroots
        argmaxes = np.argmax(C, axis=0)
        displaced = [int(isyms[k]) != int(argmaxes[k]) for k in range(nroots)]
    elif cfg.method == 'greedy_duo':
        isyms, gave_up, displaced = _greedy_duo(C, cfg)
    else:
        raise ValueError(f"unknown assignment method: {cfg.method}")

    labels: List[RootLabel] = []
    for iroot in range(nroots):
        w = C[:, iroot]
        isym = int(isyms[iroot])
        ibasis = int(isym_to_basis[isym])
        bs = basis[ibasis]

        # confidence
        order2 = np.partition(w, -2)[-2:] if len(w) >= 2 else np.array([0.0, w[0]])
        p1 = float(order2[-1])
        p2 = float(order2[0]) if len(w) >= 2 else 0.0
        purity = float(w[isym])

        flags = []
        if purity < cfg.low_purity:
            flags.append('LOW_PURITY')
        if (p1 - p2) < cfg.ambiguous_margin:
            flags.append('AMBIGUOUS')
        if displaced[iroot]:
            flags.append('DISPLACED' if cfg.method == 'greedy_duo'
                         else 'MOVED_BY_MATCHING')
        if gave_up[iroot]:
            flags.append('GAVE_UP_DUPLICATE')

        labels.append(RootLabel(
            iroot=iroot, isym=isym, ibasis=ibasis,
            v=bs.v, v_contribution=bs.v,
            purity=purity, p1=p1, p2=p2, flags=flags))

    # --- v assignment ---
    if cfg.v_mode == 'count':
        # Duo 10066-10083: v = 1 + v of the nearest lower-energy root with
        # the same (istate, Lambda, Sigma); 0 if none. Roots are already in
        # ascending energy order, so a per-stack counter is equivalent.
        counter = {}
        for lab in labels:
            bs = basis[lab.ibasis]
            key = (bs.istate, int(round(bs.lambda_)), int(round(2 * bs.sigma)))
            lab.v = counter.get(key, -1) + 1
            counter[key] = lab.v
        # visibility for Duo failure mode 5.4 (an intruder silently shifts
        # the whole ladder): a confident root whose contribution-v disagrees
        # with its count-v marks a ladder anomaly
        for lab in labels:
            if lab.v != lab.v_contribution and lab.purity > 0.8:
                lab.flags.append('LADDER_BREAK')
    elif cfg.v_mode != 'contribution':
        raise ValueError(f"unknown v_mode: {cfg.v_mode}")

    return labels
