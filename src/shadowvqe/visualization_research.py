"""
Publication-quality research figures for shadowvqe benchmarks.

All figures follow academic publishing standards:
- Clear, descriptive titles and axis labels with units
- Professional color schemes (colorblind-friendly)
- Proper typography (sans-serif, readable sizes)
- High-quality output (150 DPI, both PNG + PDF)
- Minimal clutter (no redundant elements)
- Proper spacing and legend placement
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

matplotlib.use("Agg")

# Professional style for research papers
_STYLE = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
}

# Colorblind-friendly palette (Okabe-Ito)
_PALETTE = {
    "vqe": "#0173B2",           # blue
    "shadow": "#DE8F05",        # orange
    "fmo": "#CC78BC",           # purple
    "adaptive": "#CA9161",      # brown
    "random": "#56B4E9",        # light blue
    "exact": "#029E73",         # green
    "error": "#D55E00",         # red-orange
    "reference": "#999999",     # gray
}


def _setup():
    """Apply global style settings."""
    plt.rcParams.update(_STYLE)


def _save(fig: plt.Figure, savedir: Path | str, name: str) -> None:
    """Save figure as PNG and PDF at publication quality."""
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        path = savedir / f"{name}.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")


# ============================================================================
# Research Figure 1: Shadow Error Scaling (1/√N)
# ============================================================================

def plot_shadow_scaling_research(
    n_shadows_list: Sequence[int],
    mean_errors: Sequence[float],
    std_errors: Sequence[float],
    theory_slope: float = -0.5,
    savedir: Path | str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Classical shadow error vs. number of shadows (log-log plot).

    Shows 1/√N scaling (slope = -0.5 in log-log space).
    """
    _setup()
    fig, ax = plt.subplots(figsize=(7, 5))

    n_shadows_arr = np.array(n_shadows_list, dtype=float)
    mean_errors = np.array(mean_errors, dtype=float)
    std_errors = np.array(std_errors, dtype=float)

    # Main data
    ax.loglog(n_shadows_arr, mean_errors, "o-", color=_PALETTE["shadow"],
              linewidth=2, markersize=7, label="Observed error", zorder=3)
    ax.fill_between(n_shadows_arr, mean_errors - std_errors, mean_errors + std_errors,
                    alpha=0.2, color=_PALETTE["shadow"], zorder=2)

    # Theory line
    theory_line = mean_errors[0] * (n_shadows_arr / n_shadows_arr[0]) ** theory_slope
    ax.loglog(n_shadows_arr, theory_line, "--", color=_PALETTE["reference"],
              linewidth=1.5, label=f"Theory: N$^{{{theory_slope}}}$", zorder=2)

    ax.set_xlabel("Number of Shadows", fontsize=11, fontweight="bold")
    ax.set_ylabel("Absolute Error in Energy (Ha)", fontsize=11, fontweight="bold")
    ax.set_title("Classical Shadow Tomography: Error Scaling", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    if savedir:
        _save(fig, savedir, "research_shadow_scaling")
    return fig, ax


# ============================================================================
# Research Figure 2: VQE vs Shadow-VQE Direct Comparison
# ============================================================================

def plot_vqe_shadow_comparison_research(
    methods: list[str],
    energies: list[float],
    exact: float,
    errors: list[float],
    colors: list[str] | None = None,
    savedir: Path | str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Side-by-side: (A) Final energies, (B) Absolute errors vs exact.

    Shows accuracy and convergence in a clean 1x2 layout.
    """
    _setup()
    if colors is None:
        colors = [_PALETTE.get(m.lower(), _PALETTE["vqe"]) for m in methods]

    fig = plt.figure(figsize=(12, 4.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    x = np.arange(len(methods))
    w = 0.6

    # Panel A: Energies
    bars_a = ax1.bar(x, energies, w, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    ax1.axhline(exact, color=_PALETTE["exact"], linestyle="--", linewidth=2,
                label="Exact", zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylabel("Energy (Ha)", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Ground State Energy", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: Errors (log scale)
    bars_b = ax2.bar(x, errors, w, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.set_ylabel("|E − E_exact| (Ha)", fontsize=11, fontweight="bold")
    ax2.set_title("(B) Absolute Energy Error", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, which="both")

    # Annotate error values on bars
    for i, (bar, err) in enumerate(zip(bars_b, errors)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.3,
                f'{err:.2e}', ha='center', va='bottom', fontsize=8)

    fig.suptitle("VQE vs Shadow-VQE Accuracy", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    if savedir:
        _save(fig, savedir, "research_vqe_shadow_comparison")
    return fig, np.array([ax1, ax2])


# ============================================================================
# Research Figure 3: Shot Budget Fairness
# ============================================================================

def plot_shot_budget_research(
    budgets: Sequence[int],
    vqe_errors_mean: Sequence[float],
    vqe_errors_std: Sequence[float],
    shadow_errors_mean: Sequence[float],
    shadow_errors_std: Sequence[float],
    chem_accuracy: float = 1.594e-3,
    savedir: Path | str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Error vs. fixed shot budget, comparing VQE and Shadow-VQE.

    Clearly shows where shadows gain the advantage.
    """
    _setup()
    fig, ax = plt.subplots(figsize=(8, 5))

    budgets_arr = np.array(budgets, dtype=float)

    # Error bars
    ax.errorbar(budgets_arr, vqe_errors_mean, yerr=vqe_errors_std,
                fmt="o-", color=_PALETTE["vqe"], linewidth=2, markersize=8,
                label="Standard VQE", capsize=5, capthick=1.5, alpha=0.85)
    ax.errorbar(budgets_arr, shadow_errors_mean, yerr=shadow_errors_std,
                fmt="s--", color=_PALETTE["shadow"], linewidth=2, markersize=8,
                label="Shadow-VQE", capsize=5, capthick=1.5, alpha=0.85)

    # Chemical accuracy threshold
    ax.axhline(chem_accuracy, color=_PALETTE["reference"], linestyle=":",
               linewidth=1.8, label="Chemical Accuracy (1 kcal/mol)")

    # Shaded advantage region
    shadow_better = np.array(shadow_errors_mean) < np.array(vqe_errors_mean)
    if np.any(shadow_better):
        ax.fill_between(budgets_arr[shadow_better], ax.get_ylim()[0], ax.get_ylim()[1],
                        alpha=0.08, color=_PALETTE["shadow"], label="Shadow advantage zone")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total Shot Budget (Shots)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Absolute Energy Error (Ha)", fontsize=11, fontweight="bold")
    ax.set_title("Fixed Shot Budget Comparison: VQE vs Shadow-VQE",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.92)
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    if savedir:
        _save(fig, savedir, "research_shot_budget")
    return fig, ax


# ============================================================================
# Research Figure 4: Measurement Cost Scaling
# ============================================================================

def plot_measurement_cost_research(
    system_sizes: Sequence[int],
    vqe_settings: Sequence[int],
    shadow_settings: Sequence[int],
    labels: Sequence[str] | None = None,
    savedir: Path | str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Measurement setting count vs. system size.

    Shows how FMO+shadows scales better than direct VQE.
    """
    _setup()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(system_sizes))
    w = 0.35

    bars1 = ax.bar(x - w/2, vqe_settings, w, label="Standard VQE (grouped Pauli)",
                   color=_PALETTE["vqe"], alpha=0.85, edgecolor="black", linewidth=1)
    bars2 = ax.bar(x + w/2, shadow_settings, w, label="FMO + Shadow-VQE",
                   color=_PALETTE["shadow"], alpha=0.85, edgecolor="black", linewidth=1)

    # Annotate bar heights
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels else [f"System {i}" for i in system_sizes], fontsize=10)
    ax.set_ylabel("Number of Measurement Settings Required", fontsize=11, fontweight="bold")
    ax.set_xlabel("System Complexity", fontsize=11, fontweight="bold")
    ax.set_title("Resource Scaling: VQE vs FMO+Shadows",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if savedir:
        _save(fig, savedir, "research_measurement_cost")
    return fig, ax


# ============================================================================
# Research Figure 5: Variance Reduction (Adaptive Shadows)
# ============================================================================

def plot_variance_reduction_research(
    n_shadows: Sequence[int],
    random_variance: Sequence[float],
    adaptive_variance: Sequence[float],
    reduction_factor: Sequence[float] | None = None,
    savedir: Path | str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    2x1 layout:
    (A) Variance vs. shadows (both methods)
    (B) Variance reduction factor
    """
    _setup()
    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    n_shadows_arr = np.array(n_shadows, dtype=float)

    # Panel A: Variances
    ax1.loglog(n_shadows_arr, random_variance, "o-", color=_PALETTE["random"],
               linewidth=2, markersize=7, label="Random bases", alpha=0.85)
    ax1.loglog(n_shadows_arr, adaptive_variance, "s--", color=_PALETTE["adaptive"],
               linewidth=2, markersize=7, label="Hamiltonian-adaptive bases", alpha=0.85)
    ax1.set_xlabel("Number of Shadows", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Sample Variance (Ha²)", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Variance: Random vs Adaptive", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, which="both", alpha=0.2)

    # Panel B: Reduction factor
    if reduction_factor is not None:
        reduction_factor = np.array(reduction_factor, dtype=float)
        ax2.semilogx(n_shadows_arr, reduction_factor, "D-", color=_PALETTE["error"],
                     linewidth=2.5, markersize=8, alpha=0.85)
        ax2.axhline(1.0, color=_PALETTE["reference"], linestyle="--", linewidth=1.5)
        ax2.fill_between(n_shadows_arr, 1.0, reduction_factor,
                        where=(reduction_factor >= 1.0), alpha=0.15, color=_PALETTE["error"])
        ax2.set_xlabel("Number of Shadows", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Variance Reduction Factor", fontsize=11, fontweight="bold")
        ax2.set_title("(B) Adaptive Advantage", fontsize=12, fontweight="bold")
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.set_ylim(bottom=0.5)

    fig.suptitle("Variance Reduction via Hamiltonian-Adaptive Shadows",
                fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    if savedir:
        _save(fig, savedir, "research_variance_reduction")
    return fig, np.array([ax1, ax2])


# ============================================================================
# Research Figure 6: FMO Energy Assembly Scaling
# ============================================================================

def plot_fmo_scaling_research(
    system_names: Sequence[str],
    direct_vqe_qubits: Sequence[int],
    fmo_max_qubits: Sequence[int],
    fmo_n_subsystems: Sequence[int],
    direct_energies: Sequence[float] | None = None,
    fmo_energies: Sequence[float] | None = None,
    exact_energies: Sequence[float] | None = None,
    savedir: Path | str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    3-panel FMO comparison:
    (A) Qubit requirement scaling
    (B) Number of subsystems
    (C) Energy accuracy (if provided)
    """
    _setup()
    fig = plt.figure(figsize=(15, 4.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.3)

    x = np.arange(len(system_names))
    w = 0.35

    # Panel A: Qubit scaling
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - w/2, direct_vqe_qubits, w, label="Direct VQE",
           color=_PALETTE["vqe"], alpha=0.85, edgecolor="black", linewidth=1)
    ax1.bar(x + w/2, fmo_max_qubits, w, label="FMO (max subsystem)",
           color=_PALETTE["fmo"], alpha=0.85, edgecolor="black", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(system_names, fontsize=9, rotation=15)
    ax1.set_ylabel("Qubits Required", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Qubit Requirement", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: Number of subsystems
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, fmo_n_subsystems, w*2, color=_PALETTE["fmo"], alpha=0.85,
           edgecolor="black", linewidth=1)
    for i, (xi, n) in enumerate(zip(x, fmo_n_subsystems)):
        ax2.text(xi, n, f'{int(n)}', ha='center', va='bottom', fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(system_names, fontsize=9, rotation=15)
    ax2.set_ylabel("Number of Subsystems", fontsize=11, fontweight="bold")
    ax2.set_title("(B) Fragment Count", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Panel C: Energy accuracy (if available)
    ax3 = fig.add_subplot(gs[0, 2])
    if direct_energies is not None and fmo_energies is not None and exact_energies is not None:
        direct_err = np.abs(np.array(direct_energies) - np.array(exact_energies))
        fmo_err = np.abs(np.array(fmo_energies) - np.array(exact_energies))
        ax3.bar(x - w/2, direct_err, w, label="Direct VQE",
               color=_PALETTE["vqe"], alpha=0.85, edgecolor="black", linewidth=1)
        ax3.bar(x + w/2, fmo_err, w, label="FMO-VQE",
               color=_PALETTE["fmo"], alpha=0.85, edgecolor="black", linewidth=1)
        ax3.set_yscale("log")
        ax3.set_ylabel("|E − E_exact| (Ha)", fontsize=11, fontweight="bold")
        ax3.set_title("(C) Energy Error", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, "Energy data\nnot available", ha="center", va="center",
                transform=ax3.transAxes, fontsize=11, color="gray", style="italic")
        ax3.set_title("(C) Energy Error", fontsize=12, fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(system_names, fontsize=9, rotation=15)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle("Fragment Molecular Orbital Energy Assembly Scaling",
                fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    if savedir:
        _save(fig, savedir, "research_fmo_scaling")
    return fig, np.array([ax1, ax2, ax3])
