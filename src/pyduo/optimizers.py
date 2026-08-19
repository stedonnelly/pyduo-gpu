"""Optimizers for cold-start / global fitting.

- geodesic_lm : Levenberg-Marquardt with geodesic acceleration (Transtrum &
  Sethna), proper step rejection, delayed-gratification damping and robust
  IRLS losses. The evidence base: multistart + geodesic LM solved 59/59
  cold-start NLS benchmark problems vs 27/59 for single-start LM.
- multistart_fit : Sobol multistart driver with cheap screening, top-q full
  solves and basin clustering.
- cma_search : ask/tell CMA-ES wrapper (pycma) over a BATCH cost function --
  one generation = one population-batched GPU call.
- fit_coldstart : the staged-release x J-curriculum scheduler (simultaneous
  in parameters within each stage, staged in data and expansion order).

All drivers work on a FittingProblem and return scipy-compatible results so
FitResult statistics (uncertainties / conditioning) apply unchanged.
"""
import time
from dataclasses import dataclass, field as dc_field
from typing import Callable, List, Optional, Sequence

import numpy as np

__all__ = ['geodesic_lm', 'multistart_fit', 'cma_search', 'fit_coldstart',
           'GLMResult']


# =====================================================================
# geodesic-accelerated Levenberg-Marquardt
# =====================================================================

@dataclass
class GLMResult:
    """scipy.optimize.OptimizeResult-compatible namespace."""
    x: np.ndarray = None
    cost: float = np.inf
    fun: np.ndarray = None
    jac: np.ndarray = None
    nfev: int = 0
    njev: int = 0
    status: int = 0
    message: str = ''
    success: bool = False
    history: list = dc_field(default_factory=list)


def _robust_weights(r, loss, f_scale):
    """IRLS row weights sqrt(rho'(z)), z = (r/f_scale)^2, and the true
    robust cost 0.5 * f_scale^2 * sum rho(z)."""
    if loss == 'linear':
        return np.ones_like(r), 0.5 * float(r @ r)
    z = (r / f_scale) ** 2
    if loss == 'soft_l1':
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
        w = (1.0 + z) ** -0.25
    elif loss == 'huber':
        rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(np.maximum(z, 1e-300)) - 1.0)
        w = np.where(z <= 1.0, 1.0, z ** -0.25)
    elif loss == 'cauchy':
        rho = np.log1p(z)
        w = (1.0 + z) ** -0.5
    else:
        raise ValueError(f'unknown loss {loss!r}')
    return w, 0.5 * f_scale ** 2 * float(rho.sum())


def geodesic_lm(fun: Callable, jac: Callable, x0: np.ndarray,
                bounds=None, loss: str = 'linear', f_scale: float = 1.0,
                max_nfev: Optional[int] = None,
                ftol: float = 1e-9, xtol: float = 1e-9, gtol: float = 1e-9,
                lam0: float = 1e-3, lam_down: float = 3.0, lam_up: float = 2.0,
                accel: bool = True, alpha: float = 0.75, h_accel: float = 0.1,
                max_reject: int = 25, verbose: int = 0) -> GLMResult:
    """LM with geodesic acceleration and honest step control.

    Differences from a naive LM (and from Duo's refinement.f90):
    - a step that increases the robust cost is REJECTED and re-solved at the
      same point with larger lambda (Duo keeps divergent steps);
    - delayed gratification: lambda /= lam_down on accept, *= lam_up on
      reject (Transtrum's recommendation for sloppy models);
    - geodesic acceleration: the second directional derivative along the LM
      step (ONE extra residual evaluation) cancels parameter-effects
      curvature so the iterate follows curved sloppy canyons; gated by
      2|d2|/|d1| <= alpha (0.75 default; 0.1 for hard cases), falling back
      to the plain LM step when the gate fails;
    - Marquardt scaling (diag JtJ, running max) so mixed-unit parameters
      (cm-1 / Angstrom / dimensionless morphs) are damped commensurately.

    fun(x) -> raw residual vector; jac(x) -> raw Jacobian. Robust losses are
    applied internally via IRLS row weights, frozen within an iteration.
    bounds: (lo, hi) arrays; trial steps are projected (clipped).
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    lo = -np.inf * np.ones(n) if bounds is None else np.asarray(bounds[0], float)
    hi = +np.inf * np.ones(n) if bounds is None else np.asarray(bounds[1], float)
    x = np.clip(x, lo, hi)
    if max_nfev is None:
        max_nfev = 100 * n

    res = GLMResult()
    r = np.asarray(fun(x), dtype=float)
    res.nfev = 1
    w, cost = _robust_weights(r, loss, f_scale)
    J = np.asarray(jac(x), dtype=float)
    res.njev = 1
    dtd = None
    lam = lam0
    status, message = 0, 'max_nfev reached'
    res.history.append({'cost': cost, 'lam': lam, 'accepted': True})

    while res.nfev < max_nfev:
        Jw = J * w[:, None]
        rw = r * w
        g = Jw.T @ rw
        if np.max(np.abs(g)) < gtol:
            status, message = 1, 'gtol satisfied'
            break
        A = Jw.T @ Jw
        d = np.diag(A).copy()
        d[d <= 0] = 1.0
        dtd = d if dtd is None else np.maximum(dtd, d)

        accepted = False
        n_reject = 0
        while not accepted and res.nfev < max_nfev:
            try:
                M = A + lam * np.diag(dtd)
                d1 = np.linalg.solve(M, -g)
            except np.linalg.LinAlgError:
                d1 = np.linalg.lstsq(A + lam * np.diag(dtd), -g, rcond=None)[0]

            step = d1
            if accel and np.linalg.norm(d1) > 0:
                # 2nd directional derivative, ONE extra (frozen-weight) eval
                h = h_accel
                r_h = np.asarray(fun(np.clip(x + h * d1, lo, hi)), dtype=float)
                res.nfev += 1
                fvv = 2.0 / h * ((r_h - r) / h - (J @ d1) / h) * w
                d2 = np.linalg.solve(M, -(Jw.T @ fvv))
                if 2.0 * np.linalg.norm(d2) <= alpha * np.linalg.norm(d1):
                    step = d1 + 0.5 * d2

            x_try = np.clip(x + step, lo, hi)
            r_try = np.asarray(fun(x_try), dtype=float)
            res.nfev += 1
            _, cost_try = _robust_weights(r_try, loss, f_scale)

            if cost_try < cost:
                dx = np.linalg.norm(x_try - x)
                dcost = cost - cost_try
                x, r = x_try, r_try
                w, cost = _robust_weights(r, loss, f_scale)
                lam = max(lam / lam_down, 1e-12)
                accepted = True
                res.history.append({'cost': cost, 'lam': lam,
                                    'accepted': True})
                if verbose:
                    print(f'  glm accept cost={cost:.6g} lam={lam:.2e}')
                if dcost <= ftol * max(cost, 1e-300):
                    status, message = 2, 'ftol satisfied'
                if dx <= xtol * (np.linalg.norm(x) + xtol):
                    status, message = 3, 'xtol satisfied'
            else:
                lam *= lam_up
                n_reject += 1
                res.history.append({'cost': cost_try, 'lam': lam,
                                    'accepted': False})
                if verbose > 1:
                    print(f'  glm reject cost={cost_try:.6g} lam={lam:.2e}')
                if n_reject >= max_reject or lam > 1e12:
                    status, message = 4, 'trust region collapsed (lambda cap)'
                    break
        if status in (2, 3, 4):
            break
        if not accepted:
            break
        J = np.asarray(jac(x), dtype=float)
        res.njev += 1

    res.x, res.fun, res.cost = x, r, cost
    res.jac = J
    res.status, res.message = status, message if status else 'max_nfev reached'
    res.success = status in (1, 2, 3)
    return res


# =====================================================================
# Sobol multistart
# =====================================================================

def sobol_starts(x0: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 n: int, rel_width: float = 0.25, seed: int = 0) -> np.ndarray:
    """n Sobol points in the per-parameter box. Where explicit finite bounds
    exist the box is the bounds; otherwise a +-rel_width multiplicative
    (or additive for near-zero values) box around x0."""
    from scipy.stats import qmc
    npar = x0.size
    lo_b = np.where(np.isfinite(lo), lo,
                    np.where(np.abs(x0) > 1e-8,
                             x0 - rel_width * np.abs(x0), x0 - rel_width))
    hi_b = np.where(np.isfinite(hi), hi,
                    np.where(np.abs(x0) > 1e-8,
                             x0 + rel_width * np.abs(x0), x0 + rel_width))
    eng = qmc.Sobol(d=npar, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n, 2))))
    pts = eng.random_base2(m=m)[:n]
    X = lo_b[None, :] + pts * (hi_b - lo_b)[None, :]
    X[0] = np.clip(x0, lo_b, hi_b)      # start 0 = the incumbent
    return X


def multistart_fit(prob, n_starts: int = 32, top_q: int = 5,
                   screen_nfev: int = 8, full_nfev: int = 60,
                   select: Optional[List[str]] = None, bounds=None,
                   loss: str = 'soft_l1', f_scale='auto',
                   rel_width: float = 0.25, seed: int = 0,
                   vary: Optional[List[str]] = None,
                   screen_cost_fn: Optional[Callable] = None,
                   method: str = 'geodesic', verbose: int = 1):
    """Sobol multistart: screen n_starts cheaply, short-LM the survivors,
    fully solve the top_q, cluster basins, adopt the best.

    vary : optional label substrings -- Sobol-scatter ONLY these dimensions
        (the multimodal ones); all other selected parameters start at their
        current values in every start (but are still fitted). Never scatter
        what you know.
    screen_cost_fn : optional (X (K, n) -> costs (K,)) population-batched
        evaluator (pyduo.population); serial evaluation otherwise.
    Returns list of basins sorted by cost: {'cost', 'x', 'result'};
    parameters are left at the BEST basin's solution.
    """
    params = prob.params if select is None else \
        [p for p in prob.params if any(s in p.label for s in select)]
    if not params:
        raise ValueError('no fit parameters selected')
    saved = prob.params
    prob.params = params
    try:
        x0 = np.array([p.get() for p in params])
        n = x0.size
        lo = np.full(n, -np.inf)
        hi = np.full(n, np.inf)
        if bounds:
            for k, p in enumerate(params):
                for key, (a, b) in bounds.items():
                    if key in p.label:
                        lo[k], hi[k] = a, b
        if vary is None:
            X = sobol_starts(x0, lo, hi, n_starts, rel_width, seed)
        else:
            vm = np.array([any(s in p.label for s in vary) for p in params])
            if not vm.any():
                raise ValueError('vary matched no selected parameters')
            Xv = sobol_starts(x0[vm], lo[vm], hi[vm], n_starts,
                              rel_width, seed)
            X = np.tile(x0, (n_starts, 1))
            X[:, vm] = Xv

        if f_scale == 'auto':
            r0 = prob.residuals(x0)
            f_scale = float(max(1e-6, 1.4826 * np.median(np.abs(r0))))

        # ---- stage 1: screen by cost (population-batched when possible) --
        t0 = time.time()
        if screen_cost_fn is None:
            try:
                if getattr(prob, '_jac_base_key', None) is None or \
                        getattr(prob, '_guarded', False):
                    prob.residuals(x0)     # valid centre for the pop solver
                screen_cost_fn, _pop = population_cost_fn(
                    prob, loss=loss, f_scale=f_scale)
            except Exception as e:
                if verbose:
                    print(f'multistart: population screen unavailable '
                          f'({e}); screening serially')
        if screen_cost_fn is not None:
            costs = np.asarray(screen_cost_fn(X), dtype=float)
        else:
            costs = np.empty(n_starts)
            for k in range(n_starts):
                r = prob.residuals(X[k])
                costs[k] = _robust_weights(r, loss, f_scale)[1]
        order = np.argsort(costs)
        keep = order[:max(2 * top_q, top_q + 1)]
        if verbose:
            print(f'multistart: screened {n_starts} starts in '
                  f'{time.time() - t0:.1f} s; best {costs[order[0]]:.4g}, '
                  f'median {np.median(costs):.4g}')

        # ---- stage 2: short LM on survivors ----
        short = []
        for k in keep:
            g = geodesic_lm(prob.residuals, prob.hf_jacobian, X[k],
                            bounds=(lo, hi), loss=loss, f_scale=f_scale,
                            max_nfev=screen_nfev)
            short.append((g.cost, g.x))
            if verbose:
                print(f'  screen start {k}: {costs[k]:.4g} -> {g.cost:.4g}')
        short.sort(key=lambda t: t[0])

        # ---- stage 3: full solves on top_q ----
        finals = []
        for cost_s, xs in short[:top_q]:
            g = geodesic_lm(prob.residuals, prob.hf_jacobian, xs,
                            bounds=(lo, hi), loss=loss, f_scale=f_scale,
                            max_nfev=full_nfev)
            finals.append(g)
            if verbose:
                print(f'  full solve: {cost_s:.4g} -> {g.cost:.4g} '
                      f'({g.message})')

        # ---- basin clustering ----
        basins = []
        for g in sorted(finals, key=lambda g: g.cost):
            xnorm = np.maximum(np.abs(g.x), 1e-8)
            if any(np.max(np.abs(g.x - b['x']) / xnorm) < 1e-3
                   for b in basins):
                continue
            basins.append({'cost': g.cost, 'x': g.x, 'result': g})
        if verbose:
            print(f'multistart: {len(basins)} distinct basin(s); '
                  f'best cost {basins[0]["cost"]:.6g}')
        for p, xv in zip(params, basins[0]['x']):
            p.set(xv)
        return basins
    finally:
        prob.params = saved


def population_cost_fn(prob, loss: str = 'linear', f_scale: float = 1.0,
                       **pop_kwargs):
    """Build a population-batched cost function X (B, n) -> (B,) over the
    GPU PopulationSolver (one generation = one batched solve; ~20 ms/member
    at J<=10 with eig='batch'). Requires a prior valid prob.residuals() at
    the centre. Returns (cost_fn, pop) -- call pop.set_center() after the
    centre moves (ICP outer loop)."""
    from pyduo.population import PopulationSolver
    pop_kwargs.setdefault('eig', 'batch')
    pop = PopulationSolver(prob, **pop_kwargs)

    def cost(X):
        R = pop.residuals(np.asarray(X, dtype=float))
        if loss == 'linear':
            return 0.5 * (R ** 2).sum(axis=1)
        return np.array([_robust_weights(r, loss, f_scale)[1] for r in R])

    return cost, pop


# =====================================================================
# CMA-ES over a batched cost function
# =====================================================================

def cma_search(cost_batch: Callable, x0: np.ndarray, sigma0=0.2,
               bounds=None, popsize: Optional[int] = None,
               max_gen: int = 60, target: Optional[float] = None,
               seed: int = 0, verbose: int = 1):
    """CMA-ES (pycma, ask/tell) with per-parameter affine scaling so mixed
    units (cm-1 / Angstrom / dimensionless) don't wreck the covariance.

    cost_batch : X (B, n) -> costs (B,). One generation = one call, so a
        population-batched GPU objective evaluates a whole generation in a
        single kernel pass.
    sigma0 : scalar relative scale, or per-parameter array of absolute
        scales. Returns (x_best, cost_best, es).
    """
    import cma
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    if np.isscalar(sigma0):
        scale = np.where(np.abs(x0) > 1e-8, sigma0 * np.abs(x0), sigma0)
    else:
        scale = np.asarray(sigma0, dtype=float)
    z0 = np.zeros(n)                      # z-space: x = x0 + scale * z
    opts = {'seed': seed, 'verbose': -1 if verbose < 2 else 1}
    if popsize:
        opts['popsize'] = int(popsize)
    if bounds is not None:
        lo, hi = bounds
        opts['bounds'] = [list((np.asarray(lo) - x0) / scale),
                          list((np.asarray(hi) - x0) / scale)]
    es = cma.CMAEvolutionStrategy(z0, 1.0, opts)
    best_x, best_c = x0.copy(), np.inf
    for gen in range(max_gen):
        Z = np.array(es.ask())
        X = x0[None, :] + Z * scale[None, :]
        costs = np.asarray(cost_batch(X), dtype=float)
        es.tell(list(Z), list(costs))
        k = int(np.argmin(costs))
        if costs[k] < best_c:
            best_c, best_x = float(costs[k]), X[k].copy()
        if verbose:
            print(f'  cma gen {gen + 1:3d}: best {costs[k]:.6g} '
                  f'(overall {best_c:.6g})')
        if target is not None and best_c <= target:
            break
        if es.stop():
            break
    return best_x, best_c, es


def cma_stage(prob, select: Optional[List[str]] = None, bounds=None,
              popsize: int = 32, max_gen: int = 150, sigma_frac: float = 0.2,
              recenter_every: int = 20, target: Optional[float] = None,
              pop_kwargs: Optional[dict] = None, verbose: int = 1):
    """CMA-ES over a parameter subset driven by the GPU population solver:
    one generation = ONE batched solve (~20-200 ms/member). The frozen
    assignment is re-centred every `recenter_every` generations (ICP
    structure): parameters move to the best-so-far, a serial evaluation
    re-matches, and pop.set_center() refreshes the vibrational span.

    sigma_frac : initial step as a fraction of each parameter's bound width
        (falls back to 0.25*|x0| or 0.25 where unbounded).
    Leaves the parameters at the best point found (serially re-evaluated)."""
    import cma as _cma
    params = prob.params if select is None else \
        [p for p in prob.params if any(s in p.label for s in select)]
    if not params:
        raise ValueError('no fit parameters selected')
    saved = prob.params
    prob.params = params
    try:
        x0 = np.array([p.get() for p in params])
        n = x0.size
        lo = np.full(n, -np.inf)
        hi = np.full(n, np.inf)
        if bounds:
            for k, p in enumerate(params):
                for key, (a, b) in bounds.items():
                    if key in p.label:
                        lo[k], hi[k] = a, b
        width = np.where(np.isfinite(hi - lo), hi - lo,
                         np.maximum(np.abs(x0), 1.0))
        scale = sigma_frac * width          # per-parameter affine scaling

        # ALWAYS refresh the centre at stage entry: the J scope may have
        # changed since the last evaluation, and the population solver's
        # frozen matching + structure factors must match the CURRENT scope
        prob.residuals(x0)
        if getattr(prob, '_guarded', False):
            raise RuntimeError('cma_stage centre evaluation is guarded -- '
                               'start from a valid model')
        cost_fn, pop = population_cost_fn(prob, loss='linear',
                                          **(pop_kwargs or {}))

        opts = {'seed': 7, 'verbose': -9, 'popsize': int(popsize)}
        z_lo = np.where(np.isfinite(lo), (lo - x0) / scale, None)
        z_hi = np.where(np.isfinite(hi), (hi - x0) / scale, None)
        opts['bounds'] = [list(z_lo), list(z_hi)]
        es = _cma.CMAEvolutionStrategy(np.zeros(n), 1.0, opts)
        best_x, best_c = x0.copy(), float(cost_fn(x0[None, :])[0])
        c_inc = best_c
        if verbose:
            print(f'  cma incumbent cost: {c_inc:.6g}')
        t0 = time.time()
        for gen in range(max_gen):
            Z = np.array(es.ask())
            X = x0[None, :] + Z * scale[None, :]
            costs = np.asarray(cost_fn(X), dtype=float)
            es.tell(list(Z), list(costs))
            k = int(np.argmin(costs))
            if costs[k] < best_c:
                best_c, best_x = float(costs[k]), X[k].copy()
            if verbose and (gen + 1) % 10 == 0:
                print(f'  cma gen {gen + 1:4d}: gen-best {costs[k]:.6g}  '
                      f'overall {best_c:.6g}  '
                      f'({(time.time() - t0) / (gen + 1):.1f} s/gen)')
            if target is not None and best_c <= target:
                break
            if es.stop():
                if verbose:
                    print(f'  cma stop: {es.stop()}')
                break
            if recenter_every and (gen + 1) % recenter_every == 0:
                for p, xv in zip(params, best_x):
                    p.set(xv)
                prob.residuals()             # serial re-match at incumbent
                if not getattr(prob, '_guarded', False):
                    pop.set_center()
        # adopt the best point only if it actually beats the incumbent
        if best_c >= c_inc:
            best_x, best_c = x0.copy(), c_inc
            if verbose:
                print('cma stage: no improvement over incumbent -- '
                      'parameters restored')
        for p, xv in zip(params, best_x):
            p.set(xv)
        prob.residuals()
        if verbose:
            print(f'cma stage: best cost {best_c:.6g} '
                  f'(incumbent {c_inc:.6g}) after '
                  f'{(time.time() - t0) / 60:.1f} min')
        return best_x, best_c
    finally:
        prob.params = saved


# =====================================================================
# staged-release cold-start scheduler
# =====================================================================

def _expansion_order(p):
    """Expansion order of a parameter: B3 -> 3; non-B (V0/RE/...) -> -1."""
    import re as _re
    m = _re.fullmatch(r'B(\d+)', p.name)
    return int(m.group(1)) if m else -1


def default_stages(prob):
    """Build the staged-release schedule from the problem's actual registry.

    Design (informed by the failed first gate run + literature practice):
    - stage what you RELEASE, never scatter what you know: V0/RE stay near
      their inputs (bounds), coupling morphs start where they are;
    - release B_k by expansion order (the probe proved a J-curriculum alone
      improves the fit but not identifiability);
    - coupling SIGNS are a discrete topology choice: swept exhaustively
      (sign_sweep) AFTER the potentials are roughly right, never explored
      by continuous descent from the wrong basin;
    - loss anneals linear -> robust: a robust loss far from the solution
      down-weights exactly the displaced-state residuals that carry the
      information (the 'entrenchment' failure mode).
    """
    pot = [p.label for p in prob.params if p.label.startswith('poten')]
    cpl = [p.label for p in prob.params if not p.label.startswith('poten')]
    by_label = {p.label: p for p in prob.params}
    pot_lead = [l for l in pot if _expansion_order(by_label[l]) <= 0]
    cpl_lead = [l for l in cpl if _expansion_order(by_label[l]) <= 0]
    b1 = [p.label for p in prob.params if _expansion_order(p) == 1]
    rest = [p.label for p in prob.params if _expansion_order(p) >= 2]
    stages = [
        dict(name='origins', select=pot_lead, j_filter=4.0, max_nfev=20,
             loss='linear'),
        dict(name='couplings', select=pot_lead + cpl_lead, j_filter=10.0,
             max_nfev=20, loss='linear'),
        # The potential-shape block (all B_k jointly, with V0/RE free to
        # compensate) is the multimodal step. Gate evidence: v2/v3 -- a
        # single descent lands in a compensating basin low-J data cannot
        # reject and high-J data cannot escape; v4 -- searching B1 alone
        # conditioned on wrong B2+ ranks basins under a wrong backdrop.
        # So: JOINT global search over the whole poten block, CMA-ES over
        # the GPU population solver, at J<=30 where the rotational ladder
        # discriminates shapes.
        dict(name='shape-cma', cma=dict(select=pot, popsize=32, max_gen=60,
                                        sigma_frac=0.1, recenter_every=15),
             j_filter=30.0, loss='linear'),
        dict(name='shape-polish', select=None, j_filter=30.0, max_nfev=30,
             loss='linear'),
        # The DISCRETE coupling-topology sweep is what actually unlocked the
        # v5b breakthrough (gate log evidence: shape-cma never beat its
        # incumbent; removing the sweep in v6 regressed 0.90 -> 21.3). But
        # v7's log showed the chicken-and-egg: with wrong shapes the sweep
        # patterns score within 3% of each other (coin flip), and with a
        # wrong topology the shape descent stalls. Standard answer:
        # ALTERNATE -- sweep, full-J descent, re-sweep with the better
        # model, until the sweep keeps its pattern.
        dict(name='topology-loop',
             loop=dict(sweep=([l for l in cpl
                               if l.lower().startswith(('lx', 'l+', 'lxly'))
                               and _expansion_order(by_label[l]) <= 0] +
                              [l for l in cpl
                               if _expansion_order(by_label[l]) >= 1])[:4],
                       rounds=3, sweep_j=30.0, sweep_nfev=8, fit_nfev=35),
             j_filter=None, loss='linear'),
        # Couplings become identifiable only once the potentials are right
        # (gate v5b: potentials converged, residual 0.9 cm-1 carried by the
        # coupling block -- wrong-sign SO33:B1 at a bound, collapsed lx12).
        # Joint CMA over ALL coupling parameters, signs included, at the
        # END; this subsumes the earlier discrete sign-sweep stage.
        dict(name='couplings-cma',
             cma=dict(select=cpl, popsize=24, max_gen=80, sigma_frac=0.2,
                      recenter_every=15),
             j_filter=30.0, loss='linear'),
        dict(name='couplings-polish', select=cpl, j_filter=None,
             max_nfev=20, loss='linear'),
        dict(name='polish', select=None, j_filter=None, max_nfev=40),
    ]
    return stages


def sign_sweep(prob, labels: Sequence[str], bounds=None, max_nfev: int = 8,
               use_hf_jac: bool = True, mag_floor: float = 0.5,
               verbose: int = 1):
    """Exhaustive discrete search over the 2^k sign patterns of the given
    parameters (coupling topology). Each pattern gets a short polish of
    JUST those parameters, scored by unweighted sum of squares; the best
    pattern's polished values are adopted. PGOPHER-style: exact and cheap
    at k <= ~5, exactly the decision continuous descent cannot make.

    mag_floor restores a workable magnitude before flipping: descent from a
    wrong-sign start typically COLLAPSES the magnitude toward zero, and
    flipping the sign of ~0 explores nothing (morphing factors are O(1))."""
    import itertools
    params = [p for p in prob.params if p.label in set(labels)]
    if not params:
        return None
    x_base = {p.label: p.get() for p in prob.params}
    best = None
    for signs in itertools.product((+1.0, -1.0), repeat=len(params)):
        for lab, val in x_base.items():          # reset ALL params
            next(p for p in prob.params if p.label == lab).set(val)
        for p, s in zip(params, signs):
            p.set(s * max(abs(x_base[p.label]), mag_floor))
        try:
            prob.fit(select=[p.label for p in params], bounds=bounds,
                     loss='linear', max_nfev=max_nfev,
                     use_hf_jac=use_hf_jac, refresh=False,
                     method='geodesic')
        except Exception as e:                   # a sign pattern may be wild
            if verbose:
                print(f'  sign {signs}: fit failed ({e})')
            continue
        r = prob.residuals()
        cost = float(r @ r)
        if verbose:
            print(f'  sign pattern {tuple(int(s) for s in signs)}: '
                  f'cost {cost:.6g}')
        state = {p.label: p.get() for p in prob.params}
        if best is None or cost < best[0]:
            best = (cost, signs, state)
    if best is None:
        return None
    for lab, val in best[2].items():
        next(p for p in prob.params if p.label == lab).set(val)
    if verbose:
        print(f'sign sweep: adopted {tuple(int(s) for s in best[1])} '
              f'(cost {best[0]:.6g})')
    return best


def fit_coldstart(prob, stages=None, bounds=None, loss: str = 'soft_l1',
                  f_scale='auto', use_hf_jac: bool = True,
                  method: str = 'geodesic', verbose: int = 1):
    """Run the staged-release schedule (default_stages(prob) unless given).

    Stage dicts: name, select (label/substring list or None=all), j_filter,
    max_nfev, and optional per-stage 'loss'/'f_scale'/'method' overrides or
    'sign_sweep': [labels] (discrete topology stage). Each stage warm-starts
    the next; f_scale='auto' re-anneals the robust scale per stage.
    Returns (stage_results, final FitResult)."""
    if stages is None:
        stages = default_stages(prob)
    saved_jf = prob.j_filter
    results = []
    t_all = time.time()
    try:
        for k, st in enumerate(stages):
            prob.j_filter = st.get('j_filter', saved_jf)
            st_loss = st.get('loss', loss)
            st_fscale = st.get('f_scale',
                               f_scale if st_loss != 'linear' else 1.0)
            t0 = time.time()
            if 'sign_sweep' in st:
                if verbose:
                    print(f"\n=== stage {k} '{st.get('name', k)}': sign "
                          f"sweep over {st['sign_sweep']} "
                          f"(j_filter={prob.j_filter}) ===")
                sign_sweep(prob, st['sign_sweep'], bounds=bounds,
                           max_nfev=st.get('max_nfev', 8),
                           use_hf_jac=use_hf_jac, verbose=verbose)
                results.append({'stage': st.get('name', k), 'result': None,
                                't': time.time() - t0})
                continue
            sel = st.get('select')
            if 'loop' in st and st['loop']:
                # alternate discrete topology sweep <-> continuous descent
                # until the sweep keeps its pattern (coordinate descent over
                # a discrete x continuous product space)
                cfg = st['loop']
                prev_pattern = None
                for rnd in range(cfg.get('rounds', 3)):
                    prob.j_filter = cfg.get('sweep_j', st.get('j_filter'))
                    if verbose:
                        print(f"\n=== stage {k} '{st.get('name', k)}' "
                              f"round {rnd + 1}: sign sweep over "
                              f"{cfg['sweep']} (j_filter={prob.j_filter}) ===")
                    best = sign_sweep(prob, cfg['sweep'], bounds=bounds,
                                      max_nfev=cfg.get('sweep_nfev', 8),
                                      use_hf_jac=use_hf_jac, verbose=verbose)
                    pattern = best[1] if best else None
                    prob.j_filter = st.get('j_filter', saved_jf)
                    if verbose:
                        print(f"=== stage {k} round {rnd + 1}: full fit "
                              f"(j_filter={prob.j_filter}) ===")
                    fr = prob.fit(select=None, bounds=bounds, loss=st_loss,
                                  f_scale=st_fscale, use_hf_jac=use_hf_jac,
                                  max_nfev=cfg.get('fit_nfev', 35),
                                  method=st.get('method', method))
                    results.append({'stage': f"{st.get('name', k)}-r{rnd + 1}",
                                    'result': fr, 't': time.time() - t0})
                    if pattern is not None and pattern == prev_pattern:
                        if verbose:
                            print('topology loop: pattern stable -- done')
                        break
                    prev_pattern = pattern
                continue
            if 'cma' in st and st['cma']:
                if verbose:
                    print(f"\n=== stage {k} '{st.get('name', k)}': CMA-ES "
                          f"(j_filter={prob.j_filter}) ===")
                cma_stage(prob, bounds=bounds, verbose=verbose, **st['cma'])
                results.append({'stage': st.get('name', k), 'result': None,
                                't': time.time() - t0})
                continue
            if 'multistart' in st and st['multistart']:
                if verbose:
                    print(f"\n=== stage {k} '{st.get('name', k)}': "
                          f"multistart (j_filter={prob.j_filter}, "
                          f"vary={st['multistart'].get('vary')}) ===")
                multistart_fit(prob, select=sel, bounds=bounds,
                               loss=st_loss, f_scale=st_fscale,
                               method=st.get('method', method),
                               verbose=verbose, **st['multistart'])
                results.append({'stage': st.get('name', k), 'result': None,
                                't': time.time() - t0})
                continue
            if verbose:
                nsel = len(prob.params) if sel is None else sum(
                    1 for p in prob.params
                    if any(s in p.label for s in sel))
                print(f"\n=== stage {k} '{st.get('name', k)}': "
                      f"{nsel} params, j_filter={prob.j_filter}, "
                      f"loss={st_loss}, max_nfev={st.get('max_nfev')} ===")
            fr = prob.fit(select=sel, bounds=bounds, loss=st_loss,
                          f_scale=st_fscale, use_hf_jac=use_hf_jac,
                          max_nfev=st.get('max_nfev'),
                          method=st.get('method', method))
            results.append({'stage': st.get('name', k), 'result': fr,
                            't': time.time() - t0})
        if verbose:
            print(f'\ncold-start schedule complete in '
                  f'{(time.time() - t_all) / 60:.1f} min')
        final = next((r['result'] for r in reversed(results)
                      if r['result'] is not None), None)
        return results, final
    finally:
        prob.j_filter = saved_jf
