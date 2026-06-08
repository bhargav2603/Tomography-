"""
Publication-quality matplotlib figures for VQE and Shadow-VQE results.

All functions return (fig, axes) and optionally save PNG + PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from .utils import OptimizationResult, get_logger

_log = get_logger(__name__)

# Use non-interactive backend by default (safe in headless envs)
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_STYLE: dict = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "legend.framealpha": 0.8,
    "lines.linewidth": 2.0,
}

_COLORS = {
    "vqe": "#2196F3",       # blue
    "shadow_vqe": "#FF5722",# orange-red
    "exact": "#4CAF50",     # green
    "error": "#9C27B0",     # purple
    "histogram": "#607D8B", # blue-grey
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


def _save(fig: plt.Figure, path: str | Path, stem: str) -> None:
    """Save figure as PNG and PDF side by side."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fp = p / f"{stem}.{ext}"
        fig.savefig(fp, bbox_inches="tight")
        _log.info("Saved figure: %s", fp)


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_convergence(
    result: OptimizationResult,
    exact_energy: float | None = None,
    save_dir: str | Path | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Energy vs. iteration convergence plot for a single method.

    Parameters
    ----------
    result       : VQEResult or ShadowVQEResult.
    exact_energy : Reference line if provided.
    save_dir     : Directory to save PNG+PDF. Skipped if None.
    title        : Custom plot title.

    Returns
    -------
    (fig, ax)
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    history = result.energy_history()
    iters = list(range(1, len(history) + 1))
    color = _COLORS.get(result.method.lower().replace("-", "_"), _COLORS["vqe"])

    ax.plot(iters, history, color=color, label=result.method, alpha=0.9)

    if exact_energy is not None:
        ax.axhline(
            exact_energy,
            color=_COLORS["exact"],
            linestyle="--",
            linewidth=1.5,
            label=f"Exact ({exact_energy:.6f})",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(title or f"{result.method} Convergence")
    ax.legend()

    fig.tight_layout()
    if save_dir:
        stem = f"{result.method.lower().replace(' ', '_')}_convergence"
        _save(fig, save_dir, stem)
    return fig, ax


def plot_energy_error(
    results: list[OptimizationResult],
    exact_energy: float,
    save_dir: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    |E_iter − E_exact| (log scale) vs. iteration for one or more methods.

    Parameters
    ----------
    results      : list of VQEResult / ShadowVQEResult.
    exact_energy : Exact ground state energy.
    save_dir     : Where to save figures.

    Returns
    -------
    (fig, ax)
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for result in results:
        history = result.energy_history()
        errors = np.abs(np.array(history) - exact_energy)
        # Clip zeros to avoid log(0)
        errors = np.clip(errors, 1e-14, None)
        iters = list(range(1, len(errors) + 1))
        color = _COLORS.get(result.method.lower().replace("-", "_"), _COLORS["vqe"])
        ax.semilogy(iters, errors, color=color, label=result.method, alpha=0.9)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("|E − E_exact|  (Ha)")
    ax.set_title("Energy Error vs. Iteration")
    ax.legend()
    fig.tight_layout()

    if save_dir:
        _save(fig, save_dir, "energy_error")
    return fig, ax


def plot_shadow_histogram(
    shadow_energies: Sequence[float],
    true_energy: float | None = None,
    n_bins: int = 40,
    save_dir: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Histogram of per-snapshot shadow energy estimates.

    Parameters
    ----------
    shadow_energies : Sequence of single-snapshot energy estimates.
    true_energy     : Reference line (e.g., exact or VQE result).
    n_bins          : Histogram bins.
    save_dir        : Where to save.

    Returns
    -------
    (fig, ax)
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    arr = np.array(shadow_energies, dtype=float)
    ax.hist(
        arr,
        bins=n_bins,
        color=_COLORS["histogram"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
        label=f"Shadow estimates (N={len(arr)})",
    )

    mean_e = float(np.mean(arr))
    ax.axvline(mean_e, color=_COLORS["vqe"], linewidth=2, label=f"Mean = {mean_e:.5f}")

    if true_energy is not None:
        ax.axvline(
            true_energy,
            color=_COLORS["exact"],
            linestyle="--",
            linewidth=2,
            label=f"Reference = {true_energy:.5f}",
        )

    ax.set_xlabel("Shadow Energy Estimate (Ha)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Classical Shadow Energy Estimates")
    ax.legend()
    fig.tight_layout()

    if save_dir:
        _save(fig, save_dir, "shadow_histogram")
    return fig, ax


def plot_comparison(
    vqe_result: OptimizationResult,
    shadow_result: OptimizationResult,
    exact_energy: float | None = None,
    save_dir: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    2×2 publication-quality comparison dashboard.

    Panel layout:
        [A] VQE convergence  |  [B] Shadow-VQE convergence
        [C] Energy error (log) | [D] Shadow variance history (if available)

    Parameters
    ----------
    vqe_result    : Standard VQE result.
    shadow_result : Shadow-VQE result.
    exact_energy  : Reference energy line.
    save_dir      : Output directory.

    Returns
    -------
    (fig, axes_2x2)
    """
    _apply_style()
    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    vqe_hist = vqe_result.energy_history()
    shad_hist = shadow_result.energy_history()
    vqe_iters = list(range(1, len(vqe_hist) + 1))
    shad_iters = list(range(1, len(shad_hist) + 1))

    # Panel A — VQE convergence
    ax_a.plot(vqe_iters, vqe_hist, color=_COLORS["vqe"], label="VQE")
    if exact_energy is not None:
        ax_a.axhline(exact_energy, color=_COLORS["exact"], linestyle="--", linewidth=1.5,
                     label=f"Exact\n{exact_energy:.5f}")
    ax_a.set_title("(A) VQE Convergence")
    ax_a.set_xlabel("Iteration")
    ax_a.set_ylabel("Energy (Ha)")
    ax_a.legend(fontsize=9)

    # Panel B — Shadow-VQE convergence
    ax_b.plot(shad_iters, shad_hist, color=_COLORS["shadow_vqe"], label="Shadow-VQE")
    if exact_energy is not None:
        ax_b.axhline(exact_energy, color=_COLORS["exact"], linestyle="--", linewidth=1.5,
                     label=f"Exact\n{exact_energy:.5f}")
    ax_b.set_title("(B) Shadow-VQE Convergence")
    ax_b.set_xlabel("Iteration")
    ax_b.set_ylabel("Energy (Ha)")
    ax_b.legend(fontsize=9)

    # Panel C — Energy error log
    if exact_energy is not None:
        vqe_err = np.clip(np.abs(np.array(vqe_hist) - exact_energy), 1e-14, None)
        shad_err = np.clip(np.abs(np.array(shad_hist) - exact_energy), 1e-14, None)
        ax_c.semilogy(vqe_iters, vqe_err, color=_COLORS["vqe"], label="VQE")
        ax_c.semilogy(shad_iters, shad_err, color=_COLORS["shadow_vqe"], label="Shadow-VQE")
        ax_c.set_title("(C) |E − E_exact| (log scale)")
        ax_c.set_xlabel("Iteration")
        ax_c.set_ylabel("|Error| (Ha)")
        ax_c.legend(fontsize=9)
    else:
        ax_c.text(0.5, 0.5, "No exact energy provided", ha="center", va="center",
                  transform=ax_c.transAxes, color="grey")

    # Panel D — Shadow variance
    var_hist = shadow_result.variance_history()
    valid_vars = [(i + 1, v) for i, v in enumerate(var_hist) if v is not None]
    if valid_vars:
        v_iters, v_vals = zip(*valid_vars)
        ax_d.semilogy(v_iters, v_vals, color=_COLORS["error"], label="Shadow Variance")
        ax_d.set_title("(D) Shadow-VQE Variance History")
        ax_d.set_xlabel("Iteration")
        ax_d.set_ylabel("Variance (Ha²)")
        ax_d.legend(fontsize=9)
    else:
        ax_d.text(0.5, 0.5, "No variance data available", ha="center", va="center",
                  transform=ax_d.transAxes, color="grey")
        ax_d.set_title("(D) Shadow-VQE Variance History")

    fig.suptitle("VQE vs Shadow-VQE Comparison", fontsize=15, fontweight="bold", y=1.01)

    if save_dir:
        _save(fig, save_dir, "vqe_vs_shadow_comparison")
    return fig, np.array([[ax_a, ax_b], [ax_c, ax_d]])


def plot_all(
    vqe_result: OptimizationResult,
    shadow_result: OptimizationResult,
    exact_energy: float | None = None,
    shadow_energies: Sequence[float] | None = None,
    save_dir: str | Path = "figures",
) -> None:
    """
    Convenience function: generate all standard figures and save them.

    Parameters
    ----------
    vqe_result     : Standard VQE result.
    shadow_result  : Shadow-VQE result.
    exact_energy   : Reference exact energy.
    shadow_energies: Per-snapshot estimates for histogram.
    save_dir       : Output directory.
    """
    save_dir = Path(save_dir)
    plot_convergence(vqe_result, exact_energy, save_dir=save_dir)
    plot_convergence(shadow_result, exact_energy, save_dir=save_dir)
    plot_energy_error([vqe_result, shadow_result], exact_energy, save_dir=save_dir)
    plot_comparison(vqe_result, shadow_result, exact_energy, save_dir=save_dir)
    if shadow_energies is not None:
        plot_shadow_histogram(shadow_energies, exact_energy, save_dir=save_dir)
    _log.info("All figures saved to %s", save_dir)
