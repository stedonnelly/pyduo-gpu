"""End-to-end: build a model in Python, make synthetic data, refit it.

No input file is written by hand and no observed line list is needed. The run
is a closed loop, which is what makes it a TEST as well as a demo: the data
come from the model itself, so the true parameter values are known and the
refinement either recovers them or it does not.

    1. build a 2-state toy model (X 1Sigma+ / A 1Pi + an L+ coupling) with
       pyduo.model.ModelBuilder and solve it on the GPU for J = 0..5
    2. export its levels as a synthetic observations CSV
    3. rebuild the same model with two potential parameters detuned by 3%
    4. build_problem() from the CSV and refit those two parameters
    5. assert they come back to better than 1e-4 relative
    6. leave the Duo-like output set behind (prob.output_dir)

Run:  python examples/build_and_fit.py      (~1-3 min on an RTX 5070)
"""
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

import numpy as np

from pyduo.model import (ModelBuilder, build_problem, load_observations_csv,
                         write_observations_csv)

OUTDIR = os.path.join(HERE, 'out_build_and_fit')
JMAX = 5.0
VMAX_OBS = 4            # keep the synthetic set to well-bound, safely
EMAX_OBS = 14000.0      # assigned levels: v <= 4 and E < 14000 cm-1

# --- the truth parameters --------------------------------------------------
# CaO-like but deliberately generic: a 1Sigma+ ground state, a 1Pi excited
# state 8700 cm-1 up, and the L-uncoupling operator between them.
X_B = [1.37032848858, -0.18690639745, -0.29961818103, -0.05039232666]
A_B = [1.13075086866, -0.16677793926, -0.01665160806, -0.07878184456]
LX_B0 = -1.05520806420

# ab initio grid the L+ coupling is morphed against: a smooth 9-point curve,
# in Angstrom, so it is a real (if invented) Duo-style abinitio section.
AB_R = np.linspace(1.30, 3.20, 9)
AB_V = 1.0 + 0.35 * np.tanh(2.0 * (AB_R - 1.95))


def make_model(name, x_b0_scale=1.0, a_b0_scale=1.0):
    """The toy model, optionally detuned in the two B0 coefficients."""
    b = ModelBuilder(name, workdir=OUTDIR,
                     title=f'toy CaO-like 2-state model [{name}]')
    b.set_masses(39.9625906, 15.99491463)
    b.set_grid(301, 1.2, 3.5, re=1.82)
    b.set_jrot(f'0.0 - {JMAX:.1f}')
    b.set_contraction(20)

    b.add_state(1, 'X1Sigma+', lambda_=0, spin=0.0, symmetry='+')
    b.add_state(2, 'A1Pi', lambda_=1, spin=0.0)

    xb = [X_B[0] * x_b0_scale] + X_B[1:]
    ab = [A_B[0] * a_b0_scale] + A_B[1:]
    # 4 B terms marked `fit` in each potential -> 8 registry parameters
    b.add_emo_potential(1, V0=0.0, RE=1.8221706442, DE=48359.4515,
                        NL=2, NR=3, B=xb, fit=True)
    b.add_emo_potential(2, V0=8690.5051598, RE=2.0743367757, DE=48359.4515,
                        NL=2, NR=3, B=ab, fit=True)

    # L+ (Lx) coupling: |dLambda| = 1, morphed onto the abinitio grid, B0 fit
    b.add_coupling('lx', 1, 2, factor='i', xlzy=(0, 'i'),
                   RE=1.8, P=2.0, B=[LX_B0, 0.0, 0.0], BINF=1.0,
                   fit=['B0'], abinitio=(AB_R, AB_V))
    return b


def banner(n, text):
    print(f'\n{"=" * 74}\n[{n}] {text}\n{"=" * 74}')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t_start = time.time()

    # ---------------------------------------------------------------- step 1
    banner(1, 'build the toy model in Python and solve it (the "truth")')
    truth = make_model('toy_truth')
    print(truth.to_input().split('\n')[0])
    print(f'  states     : {len(truth.states)} '
          f'({", ".join(s["name"] for s in truth.states.values())})')
    print(f'  potentials : EMO, 4 B terms each, all marked fit')
    print(f'  couplings  : Lx 1 2, BOBLEROY x morphing grid '
          f'({len(AB_R)} points)')
    print(f'  grid       : {truth.grid["npoints"]} points, '
          f'{truth.grid["rmin"]}-{truth.grid["rmax"]} Ang;  jrot {truth.jrot}')

    t0 = time.time()
    solver, truth_inp = truth.build(use_gpu=True)
    solver.run()
    print(f'\n  input written : {truth_inp}')
    print(f'  solved        : {len(solver.energy_levels)} levels for '
          f'J = {min(solver.j_list):g}..{max(solver.j_list):g} '
          f'in {time.time() - t0:.1f} s')

    # ---------------------------------------------------------------- step 2
    banner(2, 'export the computed levels as a synthetic observations CSV')
    # one datum per (J, parity, state, v, |lambda|, |sigma|, |omega|) key --
    # exactly the key FittingProblem.match uses, so a duplicate key would be
    # an ambiguous "observation" no fit could sharpen.
    seen, picked, dup = set(), [], 0
    for l in sorted(solver.energy_levels, key=lambda l: l.energy):
        if l.v > VMAX_OBS or l.energy > EMAX_OBS:
            continue
        key = (l.J, l.parity, l.state, l.v, abs(l.lambda_),
               round(2 * abs(l.sigma)), round(2 * abs(l.omega)))
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        picked.append(l)

    obs_csv = os.path.join(OUTDIR, 'obs_synthetic.csv')
    write_observations_csv(picked, obs_csv, weight=1.0)
    obs = load_observations_csv(obs_csv)
    per_state = {s: sum(1 for o in obs if o.state == s) for s in (1, 2)}
    print(f'  kept {len(picked)} levels (v <= {VMAX_OBS}, '
          f'E < {EMAX_OBS:g} cm-1), dropped {dup} duplicate-key levels')
    print(f'  per state: X={per_state[1]}  A={per_state[2]}')
    print(f'  energy range: {min(o.energy for o in obs):.2f} .. '
          f'{max(o.energy for o in obs):.2f} cm-1')
    print(f'  written  : {obs_csv}')
    print(f'  re-read  : {len(obs)} Observation objects '
          f'({"OK" if len(obs) == len(picked) else "MISMATCH"})')

    # a FITTING block round-trip: the same observations written INTO an input
    truth.set_observations(obs)
    fit_inp = truth.write(os.path.join(OUTDIR, 'toy_truth_with_fitting.inp'))
    from pyduo.fitting import FittingProblem
    reparsed = FittingProblem(fit_inp, solver=None, verbose=0)
    print(f'  FITTING block round-trip: {len(reparsed.obs)} observations '
          f'parsed back out of {len(obs)} written')

    # ---------------------------------------------------------------- step 3
    banner(3, 'detune two fitted parameters by 3% -> the starting model')
    start = make_model('toy_start', x_b0_scale=1.03, a_b0_scale=0.97)
    targets = {'poten1:B0': X_B[0], 'poten2:B0': A_B[0]}
    for lab, true_v in targets.items():
        scale = 1.03 if lab == 'poten1:B0' else 0.97
        print(f'  {lab:12s} true {true_v:.11f} -> start '
              f'{true_v * scale:.11f}  ({100 * (scale - 1):+.1f}%)')

    # ---------------------------------------------------------------- step 4
    banner(4, 'build_problem() from the CSV: solver + observations + registry')
    prob = build_problem(start, obs_csv, use_gpu=True, verbose=1)
    prob.j_filter = JMAX
    prob.output_dir = OUTDIR
    print(f'  registry ({len(prob.params)}): '
          f'{", ".join(p.label for p in prob.params)}')

    r0 = prob.residuals()
    rms0 = float(np.sqrt(np.mean(r0 ** 2)))
    print(f'  starting wRMS = {rms0:.4f} cm-1 over {len(r0)} residual rows, '
          f'{prob.n_unmatched} unmatched')

    # ---------------------------------------------------------------- step 5
    banner(5, 'refit the two detuned parameters (Hellmann-Feynman Jacobian)')
    fr = prob.fit(select=list(targets), use_hf_jac=True, loss='linear',
                  max_nfev=15)

    print('\n  recovery:')
    print(f'  {"parameter":12s} {"true":>17s} {"fitted":>17s} '
          f'{"rel. error":>12s}')
    worst = 0.0
    got = {p.label: p.get() for p in prob.params}
    for lab, true_v in targets.items():
        rel = abs(got[lab] - true_v) / abs(true_v)
        worst = max(worst, rel)
        print(f'  {lab:12s} {true_v:17.11f} {got[lab]:17.11f} {rel:12.3e}')
    r1 = prob.residuals()
    rms1 = float(np.sqrt(np.mean(r1 ** 2)))
    print(f'\n  wRMS {rms0:.4f} -> {rms1:.6f} cm-1  '
          f'(factor {rms0 / max(rms1, 1e-12):.3g} better)')
    print(f'  untouched parameters stayed put: ' +
          ', '.join(f'{p.label}={p.get():.6g}' for p in prob.params
                    if p.label not in targets))
    print('\n' + fr.pretty())

    # ---------------------------------------------------------------- step 6
    banner(6, 'model_final.inp is a restart: reload it and check the numbers')
    # closes the loop the other way: a builder-generated input, refined, is
    # still a valid input whose registry carries the fitted values.
    model_final = os.path.join(OUTDIR, 'model_final.inp')
    from pyduo import DuoSolver
    s2 = DuoSolver(use_gpu=True, verbose=0)
    s2.load_input(model_final)
    prob2 = FittingProblem(model_final, solver=s2, verbose=0, observations=obs)
    prob2.build_registry(s2.parser)
    back = {p.label: p.get() for p in prob2.params}
    drift = max(abs(back[k] - v) / max(abs(v), 1e-12) for k, v in got.items())
    print(f'  reloaded registry: {len(prob2.params)} parameters '
          f'(was {len(prob.params)})')
    print(f'  worst parameter drift through the file: {drift:.3e} relative')

    with open(os.path.join(OUTDIR, 'curves.csv')) as f:
        head = f.readline().strip().split(',')
        col = np.array([[float(x) for x in l.split(',')] for l in f])
    lx = col[:, head.index('lx12')]
    print(f'  curves.csv columns: {", ".join(head)}')
    print(f'  the morphed L+ curve is live: lx12 spans '
          f'{lx.min():.4f} .. {lx.max():.4f} over the grid')

    # ---------------------------------------------------------------- step 7
    banner(7, 'optional: the same builder emits spin-orbit sections too')
    # a 3Pi state plus <X 1Sigma+|LSY|a 3Pi> -- shown as a full solve so the
    # emitted spin-orbit-x section is proven to reach the Hamiltonian.
    so = make_model('toy_spinorbit')
    so.add_state(3, 'a3Pi', lambda_=1, spin=1.0)
    so.add_emo_potential(3, V0=8380.4242781, RE=2.0758370348, DE=33149.3885,
                         NL=2, NR=3,
                         B=[1.43075719712, -0.03717691461, -0.21473698519,
                            0.99022330747], fit=True)
    so.add_coupling('spin-orbit-x', 1, 3, sigma_i=0.0, sigma_j=1.0,
                    factor='i', xlzy=(0, 'i'), RE=1.8, P=2.0,
                    B=[0.67792215010, 0.0], BINF=1.0, fit=['B0'],
                    abinitio=(AB_R, 55.0 + 12.0 * np.tanh(AB_R - 2.0)))
    s3, so_inp = so.build(use_gpu=True)
    s3.run()
    sof = s3.parser.spinorbit[0]
    print(f'  parsed  : spin-orbit {sof.istate}-{sof.jstate}, '
          f'{sof.func_type}, morphing={sof.morphing}, '
          f'factor={sof.factor:g}, complex_f={sof.complex_f}')
    trip = [l for l in s3.energy_levels if l.state == 3]
    omegas = sorted({round(abs(l.omega), 1) for l in trip})
    print(f'  solved  : {len(s3.energy_levels)} levels, {len(trip)} of them '
          f'a3Pi with |omega| in {omegas}')
    print(f'  input   : {so_inp}')

    # ---------------------------------------------------------------- step 8
    banner(8, 'files produced')
    for fn in sorted(os.listdir(OUTDIR)):
        fp = os.path.join(OUTDIR, fn)
        print(f'  {fn:28s} {os.path.getsize(fp):9d} bytes')

    ok = worst < 1e-4 and drift < 1e-8
    print(f'\n{"PASS" if ok else "FAIL"}: worst relative recovery error '
          f'{worst:.3e} (tolerance 1e-4), restart drift {drift:.3e}, '
          f'total {time.time() - t_start:.1f} s')
    assert worst < 1e-4, f'parameter recovery failed: rel. error {worst:.3e}'
    assert drift < 1e-8, f'model_final.inp round-trip drift {drift:.3e}'
    return 0


if __name__ == '__main__':
    sys.exit(main())
