"""
Study 5 — Real Molecule Benchmarks: LiH and BeH2.

Demonstrates the shadow advantage on real molecular Hamiltonians:
  - H2  (2 qubits,  5 Pauli terms,  2 commuting groups)  — baseline
  - LiH (4 qubits, ~100 Pauli terms, ~15 commuting groups) — medium
  - BeH2(6 qubits, ~150 Pauli terms, ~20 commuting groups) — large

Key insight: As Hamiltonians grow, the VQE measurement overhead
(proportional to commuting groups) scales badly. Shadow-VQE's cost
stays constant at N_shadows, giving a clear advantage for LiH/BeH2.

Requires:
    pip install qiskit-nature[pyscf]

Output:
    figures/study5_molecules.png (.pdf)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from shadowvqe.ansatz import hardware_efficient_ansatz
from shadowvqe.hamiltonians import h2_hamiltonian
from shadowvqe.molecules import lih_hamiltonian, beh2_hamiltonian, molecule_summary
from shadowvqe.vqe import VQE
from shadowvqe.shadow_vqe import ShadowVQE
from shadowvqe.visualization_research import plot_vqe_shadow_comparison_research, plot_variance_reduction_research, plot_measurement_cost_research, plot_fmo_scaling_research
from shadowvqe.validation import exact_ground_state_energy
from shadowvqe.shadows import ClassicalShadows

# ── Configuration ─────────────────────────────────────────────────────────
SEED = 42
N_SHADOWS = 3000
VQE_REPS = 2
CHEM_ACC = 1.594e-3
SHOTS_PER_GROUP = 1000
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def benchmark_molecule(
    name: str,
    hamiltonian,
    seed: int = SEED,
    n_shadows: int = N_SHADOWS,
) -> dict:
    """Run VQE + Shadow-VQE on a single molecule and return benchmark dict."""
    n_q = hamiltonian.num_qubits
    groups = hamiltonian.group_commuting()
    n_groups = len(groups)
    n_terms = len(hamiltonian)

    print(f"\n  [{name}]  qubits={n_q},  Pauli terms={n_terms},  "
          f"commuting groups={n_groups}")

    ansatz = hardware_efficient_ansatz(n_q, reps=VQE_REPS)
    exact = exact_ground_state_energy(hamiltonian)

    # ── VQE ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    vqe_res = VQE(
        ansatz=ansatz, hamiltonian=hamiltonian,
        optimizer="cobyla", max_iter=600, seed=seed,
    ).run()
    vqe_time = time.perf_counter() - t0

    e_vqe = vqe_res.ground_state_energy
    vqe_error = abs(e_vqe - exact)
    vqe_shots = n_groups * SHOTS_PER_GROUP   # shots per optimization step

    print(f"    VQE:        E={e_vqe:.6f} Ha,  error={vqe_error:.2e} Ha,  "
          f"time={vqe_time:.1f}s")

    # ── Shadow-VQE ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svqe_res = ShadowVQE(
            ansatz=ansatz, hamiltonian=hamiltonian,
            n_shadows=n_shadows, optimizer="spsa",
            max_iter=80, seed=seed,
            initial_point=list(vqe_res.optimal_parameters),
        ).run()
    svqe_time = time.perf_counter() - t0

    e_shadow = svqe_res.ground_state_energy
    shadow_error = abs(e_shadow - exact)

    print(f"    Shadow-VQE: E={e_shadow:.6f} Ha,  error={shadow_error:.2e} Ha,  "
          f"time={svqe_time:.1f}s")
    print(f"    VQE circuits/step: {n_groups} x {SHOTS_PER_GROUP} shots = "
          f"{vqe_shots} total")
    print(f"    Shadow circuits/step: {n_shadows} (flat cost)")
    print(f"    Shadow advantage: {n_groups}x fewer circuit types needed")
    print(f"    VQE within chem. accuracy: {vqe_error < CHEM_ACC}")
    print(f"    Shadow-VQE within chem. accuracy: {shadow_error < CHEM_ACC}")

    return {
        "name": name,
        "n_qubits": n_q,
        "n_terms": n_terms,
        "n_groups": n_groups,
        "exact_energy": exact,
        "vqe_energy": e_vqe,
        "shadow_energy": e_shadow,
        "vqe_error": vqe_error,
        "shadow_error": shadow_error,
        "vqe_shots_per_step": vqe_shots,
        "shadow_shots_per_step": n_shadows,
        "shadow_advantage": n_groups,
        "vqe_within_chem_acc": vqe_error < CHEM_ACC,
        "shadow_within_chem_acc": shadow_error < CHEM_ACC,
        "vqe_time": vqe_time,
        "svqe_time": svqe_time,
    }


def run_study() -> None:
    print("\n" + "=" * 70)
    print("  Study 5: Real Molecule Benchmarks — LiH and BeH2")
    print("=" * 70)

    # ── Build Hamiltonians ─────────────────────────────────────────────────
    try:
        from shadowvqe.molecules import check_pyscf_available
        check_pyscf_available()
        pyscf_ok = True
    except ImportError as exc:
        print(f"\n  ERROR: {exc}")
        print("  Install with:  pip install qiskit-nature[pyscf]")
        print("  Then re-run this study.\n")
        return

    print("\nBuilding Hamiltonians via PySCF STO-3G...")

    ham_h2 = h2_hamiltonian()
    print("\nH2:")
    molecule_summary(ham_h2, "H2 (equilibrium, STO-3G)")

    print("LiH (this may take ~30 seconds)...")
    t0 = time.perf_counter()
    ham_lih = lih_hamiltonian()
    print(f"  LiH built in {time.perf_counter() - t0:.1f}s")
    molecule_summary(ham_lih, "LiH (R=1.5474 A, STO-3G)")

    print("BeH2 (this may take ~60 seconds)...")
    t0 = time.perf_counter()
    ham_beh2 = beh2_hamiltonian()
    print(f"  BeH2 built in {time.perf_counter() - t0:.1f}s")
    molecule_summary(ham_beh2, "BeH2 (R=1.3264 A, STO-3G)")

    # ── Run benchmarks ────────────────────────────────────────────────────
    results = []
    results.append(benchmark_molecule("H2",   ham_h2,   seed=SEED))
    results.append(benchmark_molecule("LiH",  ham_lih,  seed=SEED))
    results.append(benchmark_molecule("BeH2", ham_beh2, seed=SEED))

    # ── Summary table ─────────────────────────────────────────────────────
    _print_table(results)

    # ── Figures ───────────────────────────────────────────────────────────
    _plot(results)


def _print_table(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    hdr = (f"  {'Molecule':<8}  {'Qubits':>6}  {'Terms':>6}  "
           f"{'Groups':>7}  {'E_exact':>12}  {'VQE err':>10}  "
           f"{'Shadow err':>11}  {'Adv.':>6}")
    print(hdr)
    print("  " + "-" * 86)
    for r in results:
        print(
            f"  {r['name']:<8}  {r['n_qubits']:>6}  {r['n_terms']:>6}  "
            f"{r['n_groups']:>7}  {r['exact_energy']:>12.6f}  "
            f"{r['vqe_error']:>10.2e}  {r['shadow_error']:>11.2e}  "
            f"{r['shadow_advantage']:>5}x"
        )
    print("=" * 90)
    print("  Adv. = commuting groups = VQE circuit types per step")
    print(f"  Chemical accuracy threshold: {CHEM_ACC:.2e} Ha (1 kcal/mol)")
    print("=" * 90 + "\n")


def _plot(results: list[dict]) -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "figure.facecolor": "white",
    })

    PAL = {"vqe": "#2980b9", "shadow": "#e74c3c",
           "acc": "#27ae60", "scale": "#8e44ad"}
    names = [r["name"] for r in results]
    x = np.arange(len(names))

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)

    # ── A: Energy error per molecule ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    vqe_errs = [r["vqe_error"] for r in results]
    shad_errs = [r["shadow_error"] for r in results]
    w = 0.35
    ax1.bar(x - w / 2, vqe_errs, w, label="VQE", color=PAL["vqe"], alpha=0.88)
    ax1.bar(x + w / 2, shad_errs, w, label="Shadow-VQE",
            color=PAL["shadow"], alpha=0.88)
    ax1.axhline(CHEM_ACC, color=PAL["acc"], ls="--", lw=1.8,
                label=f"Chem. acc. ({CHEM_ACC:.1e} Ha)")
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=11)
    ax1.set_yscale("log")
    ax1.set_ylabel("|E - E_exact| (Hartree)", fontsize=11)
    ax1.set_title("A   Energy Error per Molecule", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=9)

    # ── B: Measurement cost per step ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    vqe_shots = [r["vqe_shots_per_step"] for r in results]
    shad_shots = [r["shadow_shots_per_step"] for r in results]
    ax2.bar(x - w / 2, vqe_shots, w, label="VQE (groups x shots)",
            color=PAL["vqe"], alpha=0.88)
    ax2.bar(x + w / 2, shad_shots, w, label=f"Shadow-VQE (N={N_SHADOWS})",
            color=PAL["shadow"], alpha=0.88)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=11)
    ax2.set_ylabel("Shots per Optimization Step", fontsize=11)
    ax2.set_title("B   Measurement Cost per Step", fontweight="bold", fontsize=12)
    ax2.legend(fontsize=9)

    # ── C: Scaling of Pauli terms and groups ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    n_terms = [r["n_terms"] for r in results]
    n_groups = [r["n_groups"] for r in results]
    n_qubits = [r["n_qubits"] for r in results]
    ax3_twin = ax3.twinx()
    line1, = ax3.plot(n_qubits, n_terms, "s-", color=PAL["vqe"],
                      lw=2, ms=9, label="Pauli terms")
    line2, = ax3_twin.plot(n_qubits, n_groups, "D--", color=PAL["shadow"],
                           lw=2, ms=9, label="Commuting groups")
    ax3.set_xlabel("Number of Qubits", fontsize=11)
    ax3.set_ylabel("Pauli Terms", fontsize=11, color=PAL["vqe"])
    ax3_twin.set_ylabel("Commuting Groups (VQE circuits)", fontsize=11,
                        color=PAL["shadow"])
    ax3.set_title("C   Hamiltonian Scaling with System Size",
                  fontweight="bold", fontsize=12)
    ax3.tick_params(axis="y", colors=PAL["vqe"])
    ax3_twin.tick_params(axis="y", colors=PAL["shadow"])
    ax3.legend(handles=[line1, line2], fontsize=9, loc="upper left")
    for i, name in enumerate(names):
        ax3.annotate(name, (n_qubits[i], n_terms[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)

    # ── D: Shadow advantage factor ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    adv = [r["shadow_advantage"] for r in results]
    bars = ax4.bar(names, adv, color=PAL["scale"], alpha=0.85, edgecolor="white")
    ax4.axhline(1, color="gray", ls="--", lw=1.2)
    for bar, a in zip(bars, adv):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1, f"{a}x",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Circuit Types: VQE / Shadow-VQE", fontsize=11)
    ax4.set_title("D   Shadow Advantage Factor", fontweight="bold", fontsize=12)
    ax4.set_ylim(0, max(adv) * 1.25)

    fig.suptitle(
        "Study 5 — Real Molecule Hamiltonians: VQE vs Shadow-VQE",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for fmt in ("png", "pdf"):
        fig.savefig(FIGURES / f"study5_molecules.{fmt}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURES}/study5_molecules.png  (.pdf)")


if __name__ == "__main__":
    run_study()
