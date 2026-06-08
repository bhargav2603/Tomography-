"""
Study 7 — Variance Reduction via Hamiltonian-Adaptive Classical Shadows.

Standard classical shadows sample Pauli bases uniformly at random (X/Y/Z
with probability 1/3 each). For a specific target Hamiltonian H, this is
inefficient when some Pauli bases are much more common in H than others.

Hamiltonian-adaptive shadows bias the selection toward bases that appear
frequently in H, replacing the correction factor 3 with 1/q_i (where q_i
is the actual sampling probability). The estimator remains unbiased but
has lower variance — meaning fewer shots needed for the same accuracy.

This study:
  1. Visualises the basis probability distributions for each molecule.
  2. Benchmarks error vs. number of shadows for both methods.
  3. Computes the empirical variance reduction factor.
  4. Shows shot-efficiency: how many shadows each method needs for
     chemical accuracy (1 kcal/mol = 1.594e-3 Ha).

Hamiltonians tested (no PySCF required):
  - H2 (2 qubits)          — reference with balanced basis distribution
  - Heisenberg-4q (4 qubits) — Z-heavy, strong basis imbalance

Output:
    figures/study7_variance_reduction.png (.pdf)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from shadowvqe.hamiltonians import h2_hamiltonian, heisenberg_hamiltonian
from shadowvqe.ansatz import hardware_efficient_ansatz
from shadowvqe.vqe import VQE
from shadowvqe.derandomized_shadows import (
    HamiltonianAdaptedShadows,
    compute_hamiltonian_basis_probs,
    compare_estimators,
)
from shadowvqe.visualization_research import plot_vqe_shadow_comparison_research, plot_variance_reduction_research, plot_measurement_cost_research, plot_fmo_scaling_research
from shadowvqe.validation import exact_ground_state_energy

# ── Configuration ─────────────────────────────────────────────────────────
SEED = 42
CHEM_ACC = 1.594e-3
N_TRIALS = 20
N_SHADOWS_LIST = [100, 250, 500, 1000, 2000, 5000, 10000]
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def _get_optimal_circuit(ham, seed: int, reps: int = 1):
    """Return a fully bound circuit at the VQE optimum."""
    ansatz = hardware_efficient_ansatz(ham.num_qubits, reps=reps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VQE(ansatz, ham, optimizer="cobyla", max_iter=400, seed=seed).run()
    bound = ansatz.assign_parameters(
        dict(zip(sorted(ansatz.parameters, key=lambda p: p.name),
                 res.optimal_parameters))
    )
    return bound, res.ground_state_energy


def run_study() -> None:
    print("\n" + "=" * 70)
    print("  Study 7: Variance Reduction — Adaptive vs Random Shadows")
    print("=" * 70)

    # ── Hamiltonians ──────────────────────────────────────────────────────
    molecules = [
        ("H2",      h2_hamiltonian()),
        ("Heis-4q", heisenberg_hamiltonian(4, periodic=False)),
    ]

    all_results = []
    basis_prob_data = []

    for name, ham in molecules:
        exact = exact_ground_state_energy(ham)
        n_q = ham.num_qubits
        n_terms = len(ham)

        print(f"\n  [{name}]  qubits={n_q},  terms={n_terms},  "
              f"exact={exact:.5f} Ha")

        # ── Basis probabilities ───────────────────────────────────────────
        probs = compute_hamiltonian_basis_probs(ham)
        basis_prob_data.append({"name": name, "n_qubits": n_q, "probs": probs})

        print("  Adaptive basis probabilities per qubit:")
        for q in range(n_q):
            p = probs[q]
            print(f"    qubit {q}:  X={p['X']:.3f}  Y={p['Y']:.3f}  Z={p['Z']:.3f}  "
                  f"(uniform would be 0.333 each)")

        # ── Near-optimal circuit for evaluation ──────────────────────────
        bound_circ, _ = _get_optimal_circuit(ham, seed=SEED)

        # ── Compare estimators at N_SHADOWS_LIST ─────────────────────────
        print(f"\n  Running {N_TRIALS} trials per shadow count ...")
        results = compare_estimators(
            circuit=bound_circ,
            hamiltonian=ham,
            n_shadows_list=N_SHADOWS_LIST,
            n_trials=N_TRIALS,
            exact_energy=exact,
            seed=SEED,
        )
        results["name"] = name
        results["n_qubits"] = n_q
        results["exact_energy"] = exact
        results["basis_probs"] = probs
        all_results.append(results)

        # ── Print table ───────────────────────────────────────────────────
        print(f"\n  {'N':>7}  {'Rand err':>12}  {'Adap err':>12}  "
              f"{'Var red.':>10}  {'Speedup':>9}")
        print("  " + "-" * 58)
        for i, n in enumerate(N_SHADOWS_LIST):
            re = results["random"]["mean_errors"][i]
            ae = results["adapted"]["mean_errors"][i]
            vr = results["variance_reduction_factor"][i]
            speedup = vr   # same accuracy with 1/vr shots
            print(f"  {n:>7}  {re:>12.4e}  {ae:>12.4e}  "
                  f"{vr:>10.2f}x  {speedup:>8.2f}x")

    # ── Summary ───────────────────────────────────────────────────────────
    _print_summary(all_results)

    # ── Figures ───────────────────────────────────────────────────────────
    _plot(all_results, basis_prob_data)


def _print_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 75)
    print("  VARIANCE REDUCTION SUMMARY")
    print("=" * 75)
    for r in all_results:
        vr_list = r["variance_reduction_factor"]
        avg_vr = float(np.nanmean(vr_list))
        max_vr = float(np.nanmax(vr_list))
        # Find shadow count at which adaptive first reaches chem accuracy
        adapt_n = None
        random_n = None
        for i, n in enumerate(r["n_shadows_list"]):
            if adapt_n is None and r["adapted"]["mean_errors"][i] < CHEM_ACC:
                adapt_n = n
            if random_n is None and r["random"]["mean_errors"][i] < CHEM_ACC:
                random_n = n
        print(f"  {r['name']:<12}  avg VR={avg_vr:.2f}x  max VR={max_vr:.2f}x")
        if adapt_n and random_n:
            print(f"             Adaptive reaches chem. acc. at N={adapt_n}  "
                  f"(random needs N={random_n},  {random_n//adapt_n}x more shots)")
        elif adapt_n:
            print(f"             Adaptive reaches chem. acc. at N={adapt_n}  "
                  "(random did not reach it in tested range)")
        else:
            print("             Neither method reached chem. acc. in tested range")
    print("=" * 75 + "\n")


def _plot(all_results: list[dict], basis_prob_data: list[dict]) -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "figure.facecolor": "white",
    })

    PAL = {"random": "#2980b9", "adapt": "#e74c3c",
           "acc": "#27ae60", "vr": "#8e44ad"}
    lw, ms = 2.0, 6
    n_mol = len(all_results)

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, n_mol, hspace=0.42, wspace=0.32)

    for col, r in enumerate(all_results):
        N = r["n_shadows_list"]
        name = r["name"]

        # ── Row 0: Mean error vs N (log-log) ─────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        re = np.array(r["random"]["mean_errors"])
        ae = np.array(r["adapted"]["mean_errors"])
        rs = np.array(r["random"]["std_errors"])
        as_ = np.array(r["adapted"]["std_errors"])

        ax0.loglog(N, re, "o-", color=PAL["random"], lw=lw, ms=ms,
                   label="Random shadows")
        ax0.fill_between(N, np.maximum(re - rs, 1e-8), re + rs,
                         alpha=0.18, color=PAL["random"])
        ax0.loglog(N, ae, "s--", color=PAL["adapt"], lw=lw, ms=ms,
                   label="Adaptive shadows")
        ax0.fill_between(N, np.maximum(ae - as_, 1e-8), ae + as_,
                         alpha=0.18, color=PAL["adapt"])
        ax0.axhline(CHEM_ACC, color=PAL["acc"], ls="--", lw=1.6,
                    label="Chem. acc.")
        ax0.set_xlabel("Number of Shadows N", fontsize=11)
        ax0.set_ylabel("|E - E_exact| (Hartree)", fontsize=11)
        ax0.set_title(f"{chr(65+col)}   {name}: Error vs Shadows",
                      fontweight="bold", fontsize=12)
        ax0.legend(fontsize=8)

        # ── Row 1: Variance reduction factor ─────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        vr = np.array(r["variance_reduction_factor"])
        ax1.semilogx(N, vr, "D-", color=PAL["vr"], lw=lw, ms=ms,
                     label="Var. reduction factor")
        ax1.axhline(1.0, color="gray", ls="--", lw=1.2, label="No reduction (=1)")
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("Number of Shadows N", fontsize=11)
        ax1.set_ylabel("Variance Reduction Factor\n(random var / adaptive var)",
                       fontsize=11)
        ax1.set_title(f"{chr(65+n_mol+col)}   {name}: Variance Reduction",
                      fontweight="bold", fontsize=12)
        ax1.legend(fontsize=8)

        # Annotate mean reduction
        avg_vr = float(np.nanmean(vr))
        ax1.text(0.97, 0.08, f"Mean: {avg_vr:.2f}x",
                 transform=ax1.transAxes, ha="right", fontsize=10,
                 color=PAL["vr"], fontweight="bold")

        # ── Row 2: Basis probability heatmap ─────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        bp = basis_prob_data[col]["probs"]
        n_q = basis_prob_data[col]["n_qubits"]
        matrix = np.array(
            [[bp[q]["X"], bp[q]["Y"], bp[q]["Z"]] for q in range(n_q)]
        )
        im = ax2.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn",
                        aspect="auto")
        ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(["X", "Y", "Z"])
        ax2.set_yticks(range(n_q))
        ax2.set_yticklabels([f"q{i}" for i in range(n_q)])
        # Add text annotations
        for qi in range(n_q):
            for bi, b in enumerate(["X", "Y", "Z"]):
                val = matrix[qi, bi]
                ax2.text(bi, qi, f"{val:.2f}", ha="center", va="center",
                         fontsize=9, color="black" if val > 0.3 else "white")
        ax2.axvline(0.5, color="white", lw=1)
        ax2.axvline(1.5, color="white", lw=1)
        ax2.set_title(f"{chr(65+2*n_mol+col)}   {name}: Adaptive Basis Probs",
                      fontweight="bold", fontsize=12)
        ax2.set_xlabel("Pauli Basis", fontsize=11)
        ax2.set_ylabel("Qubit", fontsize=11)
        fig.colorbar(im, ax=ax2, fraction=0.04, label="Probability")
        # Mark uniform reference
        ax2.text(0.98, -0.15, "Uniform = 0.333 each",
                 transform=ax2.transAxes, ha="right", fontsize=8, color="gray")

    fig.suptitle(
        "Study 7 — Hamiltonian-Adaptive Shadows: Variance Reduction",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for fmt in ("png", "pdf"):
        fig.savefig(FIGURES / f"study7_variance_reduction.{fmt}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURES}/study7_variance_reduction.png  (.pdf)")


if __name__ == "__main__":
    run_study()
