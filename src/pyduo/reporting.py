"""Notebook-first fit reporting for PyDuo.

Usage (in a notebook)::

    prob.fit(..., use_hf_jac=True)
    rep = prob.report                 # attached to the live problem
    rep.convergence()                 # matplotlib Figure, displays inline
    rep.residuals()                   # per-state facets, before vs after
    rep.residual_hist()
    rep.params_trajectory()
    rep.summary()                     # per-state stats (renders as a table)

    # or watch a fit that is running in ANOTHER process:
    from pyduo.reporting import FitReport
    rep = FitReport.from_monitor('tracking/fit_monitor_freq.jsonl')
    rep.watch(interval=10)            # live-refreshing dashboard cell

All figures are returned, never shown-and-closed, so they can be customized,
combined and saved. Styling follows the validated reference palette
(colorblind-safe order; scatter panels are faceted one-state-per-axes, so no
all-pairs palette limits apply).
"""
import json
import os
from typing import List, Optional

import numpy as np

__all__ = ['FitReport']

# validated reference palette (light mode), fixed assignment order
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
           '#008300', '#4a3aa7', '#e34948']
GRID = '#e4e3df'
INK2 = '#52514e'

STATE_NAMES = {1: 'X1Sigma+', 2: 'Ap1Pi', 3: 'a3Pi', 4: 'b3Sigma+', 5: 'A1Sigma+'}


def _style(ax):
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


class FitReport:
    """Render convergence / residual / parameter views of a fit.

    Data model: a list of snapshot dicts (eval, rms, wrms, unmatched, params,
    obs) -- the same records the monitor stream holds. Only the FIRST and the
    LATEST snapshots need per-obs data; the trajectory uses the summaries.
    """

    def __init__(self, snapshots: List[dict]):
        if not snapshots:
            raise ValueError('no fit snapshots yet -- run residuals()/fit() first')
        self.snapshots = snapshots

    # ---------- constructors ----------
    @classmethod
    def from_monitor(cls, path: str) -> 'FitReport':
        recs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return cls(recs)

    @classmethod
    def from_problem(cls, problem) -> 'FitReport':
        return cls(problem.snapshots)

    # ---------- data helpers ----------
    @property
    def first(self) -> dict:
        return self.snapshots[0]

    @property
    def last(self) -> dict:
        return self.snapshots[-1]

    def _residuals(self, snap) -> np.ndarray:
        return np.array([o['calc'] - o['obs'] for o in snap.get('obs', [])
                         if o.get('calc') is not None])

    def refresh(self, path: str) -> None:
        """Re-read a monitor stream in place (used by watch())."""
        self.snapshots = FitReport.from_monitor(path).snapshots

    # ---------- figures ----------
    def convergence(self, log: bool = False):
        """RMS and weighted RMS per objective evaluation."""
        import matplotlib.pyplot as plt
        xs = [s['eval'] for s in self.snapshots]
        rms = [s['rms'] for s in self.snapshots]
        wrms = [s['wrms'] for s in self.snapshots]
        fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=110)
        ax.plot(xs, rms, color=PALETTE[0], lw=2, marker='o', ms=3.5,
                label=f'RMS  (final {rms[-1]:.4f})')
        ax.plot(xs, wrms, color=PALETTE[1], lw=2, marker='o', ms=3.5,
                label=f'wRMS (final {wrms[-1]:.4f})')
        if log:
            ax.set_yscale('log')
        ax.set_xlabel('objective evaluation', fontsize=9, color=INK2)
        ax.set_ylabel('cm$^{-1}$', fontsize=9, color=INK2)
        ax.legend(frameon=False, fontsize=9)
        _style(ax)
        fig.tight_layout()
        return fig

    def residuals(self, before: bool = True, clip_percentile: float = None):
        """Per-electronic-state residual facets: after (colored) vs before
        (gray ghost). One state per axes -> no palette all-pairs limits."""
        import matplotlib.pyplot as plt
        obs0 = self.first.get('obs', [])
        obs1 = self.last.get('obs', [])
        states = sorted({o['st'] for o in obs1})
        ncol = 2
        nrow = (len(states) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(8.4, 2.4 * nrow),
                                 dpi=110, squeeze=False)
        for k, st in enumerate(states):
            ax = axes[k // ncol][k % ncol]
            p0 = [(o['obs'], o['calc'] - o['obs']) for o in obs0
                  if o['st'] == st and o.get('calc') is not None]
            p1 = [(o['obs'], o['calc'] - o['obs']) for o in obs1
                  if o['st'] == st and o.get('calc') is not None]
            if before and p0:
                ax.scatter(*zip(*p0), s=6, color=INK2, alpha=0.28, lw=0)
            if p1:
                ax.scatter(*zip(*p1), s=8, color=PALETTE[k % len(PALETTE)], lw=0)
                d = np.array([y for _, y in p1])
                rms = float(np.sqrt((d ** 2).mean()))
                ax.set_title(f'{STATE_NAMES.get(st, f"state {st}")}   '
                             f'n={len(p1)}  RMS={rms:.3f}',
                             fontsize=9, loc='left')
                if clip_percentile:
                    lim = np.percentile(np.abs(d), clip_percentile)
                    ax.set_ylim(-1.1 * lim, 1.1 * lim)
            ax.axhline(0, color=INK2, lw=0.8, alpha=0.6)
            _style(ax)
        for k in range(len(states), nrow * ncol):
            axes[k // ncol][k % ncol].set_visible(False)
        fig.supxlabel('observed (cm$^{-1}$)', fontsize=9, color=INK2)
        fig.supylabel('calc − obs (cm$^{-1}$)', fontsize=9, color=INK2)
        fig.tight_layout()
        return fig

    def residual_hist(self, bins: int = 61):
        """Residual distribution, before (dashed) vs after (solid)."""
        import matplotlib.pyplot as plt
        r0 = self._residuals(self.first)
        r1 = self._residuals(self.last)
        lim = max(float(np.percentile(np.abs(np.concatenate([r0, r1])), 99)),
                  1e-3)
        edges = np.linspace(-lim, lim, bins)
        fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=110)
        ax.hist(np.clip(r0, -lim, lim), bins=edges, histtype='step',
                color=PALETTE[1], lw=2, ls='--', label='before')
        ax.hist(np.clip(r1, -lim, lim), bins=edges, histtype='step',
                color=PALETTE[0], lw=2, label='after')
        ax.axvline(0, color=INK2, lw=0.8, alpha=0.6)
        ax.set_xlabel('calc − obs (cm$^{-1}$), clipped at p99',
                      fontsize=9, color=INK2)
        ax.legend(frameon=False, fontsize=9)
        _style(ax)
        fig.tight_layout()
        return fig

    def params_trajectory(self, top: int = 6):
        """Relative parameter motion per evaluation. The `top` largest movers
        are colored and labeled; the rest stay recessive gray."""
        import matplotlib.pyplot as plt
        names = list(self.last.get('params', {}).keys())
        if not names:
            raise ValueError('snapshots carry no parameter values')
        xs = [s['eval'] for s in self.snapshots]
        traj = {n: np.array([s['params'].get(n, np.nan)
                             for s in self.snapshots]) for n in names}
        rel = {}
        for n, v in traj.items():
            v0 = v[np.isfinite(v)][0]
            rel[n] = (v - v0) / (abs(v0) if abs(v0) > 1e-300 else 1.0)
        movers = sorted(rel, key=lambda n: -np.nanmax(np.abs(rel[n])))
        fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=110)
        for n in movers[top:]:
            ax.plot(xs, 100 * rel[n], color=INK2, alpha=0.25, lw=1)
        for k, n in enumerate(movers[:top]):
            ax.plot(xs, 100 * rel[n], color=PALETTE[k % len(PALETTE)], lw=2)
            ax.annotate(n, (xs[-1], 100 * rel[n][-1]), fontsize=8,
                        color=PALETTE[k % len(PALETTE)],
                        xytext=(4, 0), textcoords='offset points',
                        va='center')
        ax.set_xlabel('objective evaluation', fontsize=9, color=INK2)
        ax.set_ylabel('parameter change (%)', fontsize=9, color=INK2)
        _style(ax)
        fig.tight_layout()
        return fig

    # ---------- tables ----------
    def summary(self):
        """Per-state statistics of the latest snapshot. Returns a list of
        dicts; renders as an HTML table in notebooks."""
        rows = []
        obs1 = self.last.get('obs', [])
        for st in sorted({o['st'] for o in obs1}):
            d = np.array([o['calc'] - o['obs'] for o in obs1
                          if o['st'] == st and o.get('calc') is not None])
            if len(d) == 0:
                continue
            rows.append({'state': STATE_NAMES.get(st, str(st)), 'n': len(d),
                         'RMS': round(float(np.sqrt((d ** 2).mean())), 4),
                         'mean': round(float(d.mean()), 4),
                         'max|d|': round(float(np.abs(d).max()), 3)})
        return _Table(rows)

    def save_all(self, prefix: str):
        """Save every figure as PNG next to `prefix` (non-notebook use)."""
        figs = {'convergence': self.convergence(),
                'residuals': self.residuals(),
                'hist': self.residual_hist()}
        try:
            figs['params'] = self.params_trajectory()
        except ValueError:
            pass
        out = []
        for name, fig in figs.items():
            p = f'{prefix}_{name}.png'
            fig.savefig(p, bbox_inches='tight')
            out.append(p)
        return out


class _Table(list):
    """List of dicts that renders as an HTML table in notebooks."""

    def _repr_html_(self):
        if not self:
            return '<i>empty</i>'
        cols = list(self[0].keys())
        head = ''.join(f'<th style="text-align:left;padding:2px 12px 2px 0">{c}</th>'
                       for c in cols)
        rows = ''.join(
            '<tr>' + ''.join(f'<td style="padding:2px 12px 2px 0">{r[c]}</td>'
                             for c in cols) + '</tr>' for r in self)
        return f'<table><tr>{head}</tr>{rows}</table>'
