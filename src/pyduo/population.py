"""Population-batched objective: score B trial parameter vectors per call.

The serial objective (`FittingProblem.residuals`) rebuilds everything for one
parameter vector: 501-point vibrational eigenproblems per electronic state, a
matrix-element precompute, one symmetrised rovibronic diagonalisation per
(J, parity), quantum-number assignment, then obs<->calc matching. For a
population method (CMA-ES, differential evolution, a genetic fit) that is B
serial solves per generation and the GPU idles on Python.

This module keeps the pieces that do NOT depend on the parameters -- the
structure factors C_t(J) (pyduo.assembly), the block layout, and the obs<->root
assignment (pyduo.matching.frozen_match) -- fixed at a CENTRE point, and turns
one generation into a handful of large batched GPU operations:

  1. curves      : per member, evaluate each potential and each live coupling
                   curve on the grid (serial Python, ~14 `.evaluate` calls per
                   member; the only non-batched stage). Evaluated on the HOST:
                   the field expressions are ~20 elementwise ops over 501
                   points, which CuPy turns into ~20 kernel launches plus a
                   parameter-array upload -- measured 3.1 ms per EMO potential
                   on the GPU against 0.05 ms in NumPy (65x), i.e. 1.45 s vs
                   0.05 s per 64-member generation. Host and device evaluation
                   agree to ~3e-16 relative; the centre curves are evaluated
                   through the SAME path, so V_b - V_0 is exactly zero at the
                   centre either way.
  2. vib stage   : per electronic state, the vibrational problem restricted to
                   the span of the centre's first n_kept + nvib_pad radial
                   eigenvectors W0:
                       h_b = diag(E0*sc) + W0^T diag((V_b - V0)*sc) W0
                   (`vib_mode='twostage'`, one BATCHED eigh per state at
                   n_big ~ 100 -- where batched Jacobi is genuinely faster than
                   a loop). `'exact'` runs the full 501-point problem per member
                   (reference), `'frozen'` reuses the centre's radial functions.
  3. matelems    : M_t,b = W_b^T diag(f_t,b) W_b in the FULL contracted layout,
                   one batched GEMM pair per term.
  4. assembly    : StructureFactors.assemble on the (B, nvib, nvib) matelems ->
                   (B, n, n) symmetrised block, per-member vibrational
                   diagonal added afterwards.
  5. eigh        : LOOPED 2-D cuSOLVER syevd per member (at n >= 300 a loop
                   beats syevjBatched in both speed and accuracy), fp32 by
                   default with an fp64 Rayleigh-quotient refinement of the
                   handful of roots the residual vector actually reads.

Nothing here re-derives physics: every term's angular structure comes from the
audited assembly via the measured C_t factors, and every matrix element uses
the same recipe as `HamiltonianBuilder._precompute_batched_gpu_data` (verified
element-wise against the builder's live caches in `set_center`).

Accuracy contract (tools/gate_population.py):
  * at the centre the two-stage vib basis is EXACT (dV = 0), so fp64 residuals
    reproduce the serial residual vector to ~1e-11 cm-1;
  * away from the centre the truncated span costs < 5e-3 cm-1 for realistic
    single-parameter moves and < 0.05 cm-1 for a 5% shift of every potential
    expansion coefficient;
  * fp32 + Rayleigh refinement costs < 5e-2 cm-1 on top of that.
"""
import time
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp
except ImportError:                        # pragma: no cover - GPU module
    cp = None

from pyduo.matching import frozen_match, monotone_match, SLOT_UPPER, SLOT_LOWER

__all__ = ['PopulationSolver', 'PopulationResult']


# cache attribute -> (builder field-list name, multiply curve by b_rot/r^2*sc)
# EXACTLY mirrors HamiltonianBuilder._precompute_batched_gpu_data.
_CACHE_TO_FIELDS = {
    '_so_matelem_cache':     ('spinorbit', False),
    '_lx_matelem_cache':     ('lxly',      True),
    '_ss_matelem_cache':     ('spinspin',  False),
    '_sr_matelem_cache':     ('spinrot',   False),
    '_bobrot_matelem_cache': ('bobrot',    True),
    '_bobvib_matelem_cache': ('bobvib',    False),
    '_ldopq_matelem_cache':  ('lambdaopq', False),
    '_ldp2q_matelem_cache':  ('lambdap2q', False),
    '_ldq_matelem_cache':    ('lambdaq',   False),
}


@dataclass
class _StateSpec:
    istate: int
    arr_index: int
    n_kept: int
    n_big: int
    ivib: np.ndarray
    ivib_gpu: 'cp.ndarray'
    poten: object
    varies: bool
    W0: 'cp.ndarray'          # (ngrid, n_big) fp64 -- centre radial span
    E0: 'cp.ndarray'          # (n_big,) fp64 cm-1
    V0h: np.ndarray           # (ngrid,) fp64 cm-1 -- centre potential (host)
    V0: 'cp.ndarray'          # the same, on device
    f_l2: 'cp.ndarray'        # (ngrid,) fp64 -- L^2 curve (never a fit param)


@dataclass
class _TermSpec:
    name: str
    field_obj: object         # None for the pure B(r) rotational term
    scaled: bool              # curve *= b_rot/r^2 * sc
    varies: bool
    curve0h: np.ndarray       # (ngrid,) fp64 centre curve, recipe applied
    curve0: 'cp.ndarray' = None       # the same, on device


@dataclass
class PopulationResult:
    """Per-member spectra of one generation.

    energies : {(J, parity): (B, n) float64 array, cm-1, ASCENDING}. With
        `refine=True` the entries at the roots the frozen matching reads are
        fp64 Rayleigh-refined; the remaining entries are raw working-precision
        eigenvalues.
    zpe      : (B,) cm-1, the frozen ZPE level per member (energy-mode only;
        frequency residuals cancel it exactly).
    guarded  : (B,) validity-violation magnitude per member (0 = healthy);
        mirrors FittingProblem._validity_violation, and such members get the
        sloped penalty surface instead of a spectrum.
    """
    energies: Dict[Tuple[float, str], np.ndarray]
    zpe: np.ndarray
    guarded: np.ndarray
    B: int
    timings: Dict[str, float] = dc_field(default_factory=dict)
    peak_bytes: int = 0


class PopulationSolver:
    """Batched objective over a population of parameter vectors.

    prob must be a FittingProblem that has completed at least one successful
    (non-guarded) `residuals()` at the CENTRE parameters: that guarantees the
    solver holds a HamiltonianBuilder with structure factors, retained
    vibrational solutions and live matrix-element caches, and that every
    observation carries the eigen-root it was matched to.
    """

    def __init__(self, prob, dtype='float32', vib_mode='twostage',
                 nvib_pad: int = 40, refine: bool = True,
                 block_chunk: int = 0, eig: str = 'loop',
                 curve_device: str = 'host', profile: bool = False,
                 verbose: int = 0):
        if cp is None:
            raise RuntimeError('PopulationSolver requires CuPy')
        if curve_device not in ('host', 'device'):
            raise ValueError("curve_device must be 'host' or 'device'")
        self.curve_device = curve_device
        self.prob = prob
        self.dtype = cp.dtype(dtype)
        if self.dtype not in (cp.dtype('float32'), cp.dtype('float64')):
            raise ValueError("dtype must be 'float32' or 'float64'")
        if vib_mode not in ('twostage', 'exact', 'frozen'):
            raise ValueError("vib_mode must be 'twostage', 'exact' or 'frozen'")
        if eig not in ('loop', 'batch'):
            raise ValueError("eig must be 'loop' or 'batch'")
        self.vib_mode = vib_mode
        self.nvib_pad = int(nvib_pad)
        # refinement only means anything below fp64: in fp64 the eigenvalues
        # already ARE the Rayleigh quotients to working precision
        self.refine = bool(refine) and self.dtype == cp.dtype('float32')
        self.block_chunk = int(block_chunk)
        self.eig = eig
        self.profile = bool(profile)
        self._sub: Dict[str, float] = {}
        self.verbose = int(verbose)

        solver = prob.solver
        hb = getattr(solver, 'hamiltonian', None)
        if hb is None:
            raise RuntimeError('prob.solver has no hamiltonian -- run '
                               'prob.residuals() at the centre first')
        if getattr(hb, '_sf', None) is None:
            raise RuntimeError('builder carries no structure factors -- the '
                               'centre evaluation must have used the '
                               'structure-factor solve path')
        if not getattr(hb, '_vib_solutions', None):
            raise RuntimeError('builder carries no retained vibrational '
                               'solutions (_vib_solutions)')
        if getattr(prob, '_guarded', True):
            raise RuntimeError('the last prob.residuals() was guarded; the '
                               'population centre must be a valid model')
        if not any(getattr(o, '_iroot', -1) >= 0 for o in prob.obs
                   if prob._obs_in_scope(o)):
            raise RuntimeError('no in-scope observation carries an eigen-root '
                               '-- run prob.residuals() at the centre first')
        self.hb = hb
        self.sf = hb._sf
        self._buf: Dict[tuple, 'cp.ndarray'] = {}
        self.set_center()

    # ------------------------------------------------------------------
    # centre snapshot
    # ------------------------------------------------------------------
    def set_center(self, check: bool = True, check_tol: float = 1e-9) -> None:
        """Snapshot everything member-independent at the current parameters."""
        prob, hb, sf = self.prob, self.hb, self.sf
        self.x0 = np.array([p.get() for p in prob.params], dtype=np.float64)
        self.labels = [p.label for p in prob.params]
        self.r = hb._r_gpu
        self.r_host = cp.asnumpy(hb._r_gpu)
        self.ngrid = int(hb.grid.npoints)
        self.sc = float(hb._sc)
        self.nvib = int(len(hb.contracted_energies))
        # b_rot/r^2*sc: pure IEEE mul/div, so host and device agree bitwise
        self._brot_curve_h = hb.b_rot / self.r_host ** 2 * self.sc
        self._brot_curve = cp.asarray(self._brot_curve_h)
        fit_fields = {id(p.field_obj) for p in prob.params}

        # ---- electronic states -------------------------------------
        vmaps: Dict[int, Dict[int, int]] = {}
        for bs in hb.contracted_basis:
            vmaps.setdefault(bs.istate, {})[bs.v] = bs.ivib
        self.states: List[_StateSpec] = []
        for arr_index, st in enumerate(hb.states):
            istate = st.istate
            vmap = vmaps[istate]
            n_kept = len(vmap)
            ivib = np.array([vmap[v] for v in range(n_kept)], dtype=np.int64)
            ev, evec, nmax = hb._vib_solutions[istate]
            if nmax != n_kept:
                raise RuntimeError(f'state {istate}: contracted basis carries '
                                   f'{n_kept} functions but _vib_solutions '
                                   f'reports nmax={nmax}')
            n_big = int(min(n_kept + self.nvib_pad, evec.shape[1]))
            poten = None
            for p in hb.potentials:
                if p.istate == istate:
                    poten = p
                    break
            if poten is None:
                raise RuntimeError(f'no potential for state {istate}')
            f_l2 = cp.zeros(self.ngrid, dtype=cp.float64)
            for l2f in getattr(hb, 'l2', []):
                if l2f.istate == istate and l2f.jstate == istate:
                    f_l2 = cp.asarray(l2f.evaluate(self.r, use_gpu=True),
                                      dtype=cp.float64)
                    break
            V0h = self._eval_field(poten, False)
            spec = _StateSpec(
                istate=istate, arr_index=arr_index, n_kept=n_kept,
                n_big=n_big, ivib=ivib, ivib_gpu=cp.asarray(ivib),
                poten=poten, varies=id(poten) in fit_fields,
                W0=cp.asarray(np.ascontiguousarray(evec[:, :n_big]),
                              dtype=cp.float64),
                E0=cp.asarray(ev[:n_big], dtype=cp.float64),
                V0h=V0h, V0=cp.asarray(V0h), f_l2=f_l2)
            self.states.append(spec)
        self.nstates = len(self.states)

        # ---- terms (order MUST equal sf.terms) ----------------------
        self.terms: List[_TermSpec] = []
        for t in sf.terms:
            if t.cache_attr == '_brot_matelem_gpu':
                self.terms.append(_TermSpec(
                    name=t.name, field_obj=None, scaled=False, varies=False,
                    curve0h=self._brot_curve_h, curve0=self._brot_curve))
                continue
            lname, scaled = _CACHE_TO_FIELDS[t.cache_attr]
            fld = getattr(hb, lname)[t.field_index]
            c0h = self._eval_field(fld, scaled)
            self.terms.append(_TermSpec(
                name=t.name, field_obj=fld, scaled=scaled,
                varies=id(fld) in fit_fields,
                curve0h=c0h, curve0=cp.asarray(c0h)))
        self.nterms = len(self.terms)

        # ---- centre consistency check ------------------------------
        # W0's kept columns ARE the contracted wavefunctions, and the matelem
        # recipe reproduces the builder's live caches. Catches a wrong scaling
        # or a state/term ordering slip immediately instead of as a mystery
        # 0.3 cm-1 bias 200 lines later.
        if check:
            Wall0 = self._pack_center_W()
            wf = cp.asarray(hb.contracted_wavefunctions, dtype=cp.float64)
            dw = float(cp.abs(Wall0 - wf).max())
            if dw > 0:
                raise RuntimeError(f'centre radial span mismatch: max|dW| = '
                                   f'{dw:.3e} (expected exactly 0)')
            ms_ref = sf.gather_ms(hb)
            worst, worst_name = 0.0, ''
            for t_i, t in enumerate(self.terms):
                M = Wall0.T @ (t.curve0.reshape(-1, 1) * Wall0)
                ref = ms_ref[t_i]
                scale = max(float(cp.abs(ref).max()), 1e-300)
                d = float(cp.abs(M - ref).max()) / scale
                if d > worst:
                    worst, worst_name = d, t.name
            if worst > check_tol:
                raise RuntimeError(
                    f'matelem recipe mismatch for term {worst_name}: '
                    f'max|dM|/||M|| = {worst:.3e} > {check_tol:.1e}')
            self.center_matelem_error = worst
            if self.verbose:
                print(f'  set_center: matelem recipe max rel err {worst:.2e}')
            del Wall0, wf

        # ---- frozen assignment --------------------------------------
        self.frozen = frozen_match(prob)
        for k in self.frozen.keys:
            if k not in sf.blocks:
                raise RuntimeError(f'frozen matching references block {k} '
                                   f'which the structure factors do not carry '
                                   f'(j_list mismatch)')
        self.keys = list(self.frozen.keys)
        self._root_gpu = {k: cp.asarray(v) for k, v in
                          self.frozen.block_roots.items() if v.size}
        if self.verbose:
            print(f'  set_center: {self.frozen.summary()}')

    def _eval_field(self, fld, scaled: bool) -> np.ndarray:
        """One field curve on the grid, host fp64, precompute recipe applied.

        The single place a curve is ever produced -- centre and member alike --
        so `curve_device` can never make the centre inconsistent with the
        members (which would break the exactness of dV = 0 at the centre).
        """
        if self.curve_device == 'host':
            v = fld.evaluate(self.r_host, use_gpu=False)
            v = np.asarray(v, dtype=np.float64)
        else:
            v = cp.asnumpy(cp.asarray(fld.evaluate(self.r, use_gpu=True),
                                      dtype=cp.float64))
        if scaled:
            v = v * self._brot_curve_h
        return v

    def _pack_center_W(self) -> 'cp.ndarray':
        W = cp.zeros((self.ngrid, self.nvib), dtype=cp.float64)
        for st in self.states:
            W[:, st.ivib_gpu] = st.W0[:, :st.n_kept]
        return W

    # ------------------------------------------------------------------
    # stage 1: per-member curves (the only serial-Python stage)
    # ------------------------------------------------------------------
    def _member_curves(self, X: np.ndarray):
        prob = self.prob
        B = X.shape[0]
        V = np.empty((self.nstates, B, self.ngrid), dtype=np.float64)
        F = np.empty((self.nterms, B, self.ngrid), dtype=np.float64)
        for si, st in enumerate(self.states):
            if not st.varies:
                V[si] = st.V0h[None]
        for ti, t in enumerate(self.terms):
            if not t.varies:
                F[ti] = t.curve0h[None]
        guard = np.zeros(B, dtype=np.float64)
        try:
            for b in range(B):
                for p, xv in zip(prob.params, X[b]):
                    p.set(xv)
                bad = float(prob._validity_violation())
                if bad > 0.0:
                    guard[b] = bad
                    self._fill_center(V, F, b)
                    continue
                for si, st in enumerate(self.states):
                    if st.varies:
                        V[si, b] = self._eval_field(st.poten, False)
                for ti, t in enumerate(self.terms):
                    if t.varies:
                        F[ti, b] = self._eval_field(t.field_obj, t.scaled)
        finally:
            for p, xv in zip(prob.params, self.x0):
                p.set(xv)
        # a member whose curves blew up (overflowing EMO exponent, NaN) gets
        # the centre curves + a penalty flag: never NaN-poison the batch
        ok = np.isfinite(V).all(axis=(0, 2)) & np.isfinite(F).all(axis=(0, 2))
        for b in np.nonzero(~ok)[0]:
            guard[b] = max(guard[b], 1.0)
            self._fill_center(V, F, int(b))
        return cp.asarray(V), cp.asarray(F), guard

    def _fill_center(self, V, F, b: int) -> None:
        for si, st in enumerate(self.states):
            if st.varies:
                V[si, b] = st.V0h
        for ti, t in enumerate(self.terms):
            if t.varies:
                F[ti, b] = t.curve0h

    # ------------------------------------------------------------------
    # stage 2: vibrational problem per member
    # ------------------------------------------------------------------
    def _vib_stage(self, V: 'cp.ndarray'):
        """-> Wall (B, ngrid, nvib) fp64 member radial functions in contracted
        ivib order, Eb (B, nvib) fp64 contracted energies in cm-1."""
        B = V.shape[1]
        sc = self.sc
        Wall = cp.zeros((B, self.ngrid, self.nvib), dtype=cp.float64)
        Eb = cp.empty((B, self.nvib), dtype=cp.float64)
        f32 = self.dtype == cp.dtype('float32')
        if self.vib_mode == 'exact':
            kin = cp.asarray(self.hb.kinmat, dtype=cp.float64)
            f_rot = self.hb.b_rot / self.r ** 2 * sc
            gidx = cp.arange(self.ngrid)
        for si, st in enumerate(self.states):
            nk, nb = st.n_kept, st.n_big
            if self.vib_mode == 'frozen':
                Wall[:, :, st.ivib_gpu] = st.W0[:, :nk][None]
                Eb[:, st.ivib_gpu] = st.E0[:nk][None]
                continue
            if self.vib_mode == 'exact':
                Wb = cp.empty((B, self.ngrid, nk), dtype=cp.float64)
                Est = cp.empty((B, nk), dtype=cp.float64)
                for b in range(B):
                    H = kin.copy()
                    H[gidx, gidx] += V[si, b] * sc + st.f_l2 * f_rot
                    w, u = cp.linalg.eigh(H)
                    Est[b] = w[:nk] / sc
                    Wb[b] = u[:, :nk]
            else:                                   # 'twostage'
                dV = (V[si] - st.V0[None]) * sc                  # (B, ngrid)
                T = st.W0[None] * dV[:, :, None]                 # (B,ngrid,nb)
                h = cp.matmul(st.W0.T[None], T)                  # (B, nb, nb)
                del T
                ar = cp.arange(nb)
                h[:, ar, ar] += st.E0 * sc
                h = (h + h.transpose(0, 2, 1)) * 0.5
                if f32:
                    w32, u32 = cp.linalg.eigh(h.astype(cp.float32))
                    U = u32[:, :, :nk].astype(cp.float64)
                    del u32
                    if self.refine:
                        t2 = cp.matmul(h, U)
                        Est = ((U * t2).sum(axis=1) /
                               (U * U).sum(axis=1)) / sc
                        del t2
                    else:
                        Est = w32[:, :nk].astype(cp.float64) / sc
                else:
                    w, u = cp.linalg.eigh(h)
                    U = u[:, :, :nk]
                    Est = w[:, :nk] / sc
                Wb = cp.matmul(st.W0[None], U)                   # (B,ngrid,nk)
                del h, U
            Wall[:, :, st.ivib_gpu] = Wb
            Eb[:, st.ivib_gpu] = Est
            del Wb, Est
        return Wall, Eb

    # ------------------------------------------------------------------
    # stage 3: contracted matrix elements per term
    # ------------------------------------------------------------------
    def _matelems(self, Wall: 'cp.ndarray', F: 'cp.ndarray'):
        """M_t = W_b^T diag(f_t,b) W_b, (B, nvib, nvib) fp64, in sf.terms order.

        The full nvib x nvib layout (rather than the field's own state pair) is
        what StructureFactors.assemble indexes; entries outside a term's state
        pair are multiplied by a structure factor of exactly zero. Kept in fp64
        even when the block eigh runs in fp32: the GEMM is ~0.15 s/generation
        and cheap accuracy is worth having.
        """
        B = Wall.shape[0]
        WT = Wall.transpose(0, 2, 1)
        T = cp.empty_like(Wall)
        ms = []
        for ti in range(self.nterms):
            cp.multiply(Wall, F[ti][:, :, None], out=T)
            ms.append(cp.matmul(WT, T))
        del T
        return ms

    # ------------------------------------------------------------------
    # stage 4+5: assemble + diagonalise one block
    # ------------------------------------------------------------------
    def _block_energies(self, key, ms, Eb, B: int) -> np.ndarray:
        sf, sc = self.sf, self.sc
        bs = sf.blocks[key]
        n = bs.n
        iv = cp.asarray(bs.ivib_of_row.astype(np.int64))
        idx = cp.arange(n)
        roots = self._root_gpu.get(key)
        need_vec = self.refine and roots is not None
        chunk = self.block_chunk
        if chunk <= 0:
            chunk = max(1, min(B, int(2.0e9 // max(1, n * n * 8 * 4))))
        out = np.empty((B, n), dtype=np.float64)
        for lo in range(0, B, chunk):
            hi = min(B, lo + chunk)
            nb = hi - lo
            msc = [m[lo:hi] for m in ms]
            tk = self._tick()
            H64 = sf.assemble(key, msc, None, sc, dtype=cp.float64,
                              include_vib=False,
                              out=self._buffer((nb, n, n), cp.float64))
            H64[:, idx, idx] += Eb[lo:hi][:, iv] * sc
            tk = self._tock('assemble', tk)
            # NOT symmetrised: the SF-assembled block is symmetric to ~1e-13
            # relative (tools/gate_sf.py), cuSOLVER reads a single triangle
            # (exactly what _solve_all_J_sf feeds it), and the Rayleigh
            # quotient v^T H v is blind to the antisymmetric part.
            H = H64.astype(cp.float32) if self.dtype == cp.dtype('float32') \
                else H64
            tk = self._tock('cast', tk)
            w = cp.empty((nb, n), dtype=cp.float64)
            if need_vec:
                Vr = cp.empty((nb, n, int(roots.size)), dtype=cp.float64)
                if self.eig == 'batch':
                    wb, vb = cp.linalg.eigh(H)
                    w[...] = wb.astype(cp.float64)
                    Vr[...] = vb[:, :, roots]
                    del wb, vb
                else:
                    for b in range(nb):
                        wb, vb = cp.linalg.eigh(H[b])
                        w[b] = wb.astype(cp.float64)
                        Vr[b] = vb[:, roots]
                tk = self._tock('eigh', tk)
                w[:, roots] = self._rayleigh(H64, Vr)
                del Vr
                tk = self._tock('refine', tk)
            elif self.eig == 'batch':
                w[...] = cp.linalg.eigvalsh(H).astype(cp.float64)
                tk = self._tock('eigh', tk)
            else:
                for b in range(nb):
                    w[b] = cp.linalg.eigvalsh(H[b]).astype(cp.float64)
                tk = self._tock('eigh', tk)
            out[lo:hi] = cp.asnumpy(w / sc)
            self._tock('gather', tk)
            del H, w
            if self.dtype == cp.dtype('float32'):
                del H64
        return out

    def _tick(self):
        if not self.profile:
            return None
        cp.cuda.Stream.null.synchronize()
        return time.time()

    def _tock(self, name: str, t0):
        if not self.profile:
            return None
        cp.cuda.Stream.null.synchronize()
        now = time.time()
        self._sub[name] = self._sub.get(name, 0.0) + now - t0
        return now

    @staticmethod
    def _rayleigh(H64, Vr):
        """fp64 Rayleigh quotients lam_r = v_r^T H v_r / v_r^T v_r of the
        working-precision eigenvectors, batched over members.

        H64 : (B, n, n) fp64 assembled block, Vr : (B, n, k) fp64 selected
        eigenvectors. Second order in the eigenvector error, so it recovers
        most of the fp32 loss for the few roots the residual vector reads, at
        one (n,n)x(n,k) GEMM per member. Also blind to any antisymmetric part
        of H (v^T A v = 0), so the un-symmetrised assembled block is fine."""
        t = cp.matmul(H64, Vr)
        return (Vr * t).sum(axis=1) / (Vr * Vr).sum(axis=1)

    _BUF_MAX = 4

    def _buffer(self, shape, dtype):
        """Reusable (B, n, n) assembly target. Bounded: with many distinct
        block sizes (a full J list) an unbounded cache would pin gigabytes."""
        key = (shape, cp.dtype(dtype))
        buf = self._buf.get(key)
        if buf is None:
            if len(self._buf) >= self._BUF_MAX:
                self._buf.clear()
            buf = cp.empty(shape, dtype=dtype)
            self._buf[key] = buf
        return buf

    def free_buffers(self) -> None:
        self._buf.clear()

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------
    def solve(self, X, keys=None, mem_limit: float = 8.0e9
              ) -> PopulationResult:
        """Spectra of B members. X : (B, nparams) in `prob.params` order.

        `keys` restricts the blocks solved (default: every block the frozen
        matching reads). Dropping the ZPE block from `keys` leaves
        `PopulationResult.zpe` at zero, which only matters in energy mode.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != len(self.prob.params):
            raise ValueError(f'X has {X.shape[1]} columns but the problem has '
                             f'{len(self.prob.params)} fit parameters')
        B = X.shape[0]
        keys = list(self.keys) if keys is None else list(keys)
        tm = {}
        t0 = time.time()
        V, F, guard = self._member_curves(X)
        cp.cuda.Stream.null.synchronize()
        tm['curves'] = time.time() - t0

        t0 = time.time()
        Wall, Eb = self._vib_stage(V)
        cp.cuda.Stream.null.synchronize()
        tm['vib'] = time.time() - t0
        del V

        t0 = time.time()
        ms = self._matelems(Wall, F)
        cp.cuda.Stream.null.synchronize()
        tm['matelem'] = time.time() - t0
        del Wall, F

        t0 = time.time()
        energies = {}
        peak = 0
        self._sub = {}
        for key in keys:
            energies[key] = self._block_energies(key, ms, Eb, B)
            used = cp.get_default_memory_pool().used_bytes()
            peak = max(peak, used)
            if used > mem_limit:
                raise MemoryError(f'GPU pool at {used/1e9:.2f} GB after block '
                                  f'{key} (limit {mem_limit/1e9:.1f} GB) -- '
                                  f'lower block_chunk or B')
        cp.cuda.Stream.null.synchronize()
        tm['assemble_eigh'] = time.time() - t0
        tm.update({f'  {k}': v for k, v in self._sub.items()})
        del ms, Eb

        fz = self.frozen
        if fz.zpe_key is not None and fz.zpe_key in energies:
            zpe = energies[fz.zpe_key][:, fz.zpe_iroot].copy()
        else:
            zpe = np.zeros(B)
        tm['total'] = sum(v for k, v in tm.items()
                          if k != 'total' and not k.startswith(' '))
        return PopulationResult(energies=energies, zpe=zpe, guarded=guard,
                                B=B, timings=tm, peak_bytes=peak)

    # ------------------------------------------------------------------
    def residuals(self, X, match_mode: str = 'frozen') -> np.ndarray:
        """Weighted residual matrix (B, n_obs_in_scope).

        Row order and weighting are identical to FittingProblem.residuals:
        (calc - obs) * sqrt(weight), with `unmatched_penalty * sqrt(weight)` on
        rows whose observation was unmatched at the centre.

        match_mode='monotone' re-assigns ENERGY-mode rows per member by the
        optimal 1-D monotone assignment within their block (see
        pyduo.matching.monotone_match); frequency rows always keep the frozen
        roots.
        """
        res = X if isinstance(X, PopulationResult) else self.solve(X)
        fz = self.frozen
        B, nrow = res.B, fz.nrow
        calc = np.zeros((B, nrow), dtype=np.float64)
        # a frequency row needs BOTH of its levels: count contributions rather
        # than flag them, so a partial `keys` can never fabricate a residual
        nhave = np.zeros(nrow, dtype=np.int64)

        for key in self.keys:
            E = res.energies.get(key)
            if E is None:
                continue
            rows, iroots, slots = fz.block_refs[key]
            if rows.size == 0:
                continue
            up = slots == SLOT_UPPER
            if up.any():
                calc[:, rows[up]] += E[:, iroots[up]]
                nhave[rows[up]] += 1
            lo = slots == SLOT_LOWER
            if lo.any():
                calc[:, rows[lo]] -= E[:, iroots[lo]]
                nhave[rows[lo]] += 1

        # energy-mode rows are ZPE-referenced; frequency rows cancel it
        en = fz.matched & (~fz.is_freq)
        if en.any():
            calc[:, en] -= res.zpe[:, None]

        if match_mode == 'monotone':
            self._monotone_energy_rows(res, calc)
        elif match_mode != 'frozen':
            raise ValueError("match_mode must be 'frozen' or 'monotone'")

        R = (calc - fz.target[None, :]) * fz.sqrt_w[None, :]
        bad = ~(fz.matched & (nhave >= 1 + fz.is_freq))
        if bad.any():
            R[:, bad] = fz.penalty * fz.sqrt_w[None, bad]
        gm = res.guarded > 0.0
        if gm.any():
            pen = fz.penalty * (1.0 + res.guarded[gm])[:, None]
            R[gm, :] = pen * fz.sqrt_w[None, :]
        return R

    def _monotone_energy_rows(self, res: PopulationResult,
                              calc: np.ndarray) -> None:
        fz = self.frozen
        for key in self.keys:
            rows = fz.energy_rows(key)
            if rows.size == 0:
                continue
            E = res.energies.get(key)
            if E is None:
                continue
            Esh = E - res.zpe[:, None]
            idx = monotone_match(Esh, fz.target[rows])
            calc[:, rows] = np.take_along_axis(Esh, idx, axis=1)

    # ------------------------------------------------------------------
    def wrms(self, X, match_mode: str = 'frozen') -> np.ndarray:
        """Weighted RMS per member -- the scalar a population optimiser ranks.
        X : (B, nparams) or a PopulationResult."""
        return self.wrms_of(self.residuals(X, match_mode=match_mode))

    @staticmethod
    def wrms_of(R: np.ndarray) -> np.ndarray:
        return np.sqrt((R ** 2).mean(axis=1))
