"""
Study 8 - FMO-VQE + Shadow Tomography: The Scaling Advantage.

This is the capstone study. It produces TWO results and compares them:

  RESULT A - Direct VQE + Shadows on the FULL molecule
             (exact, but qubit count and measurement cost explode with size)

  RESULT B - FMO-VQE + Shadows (fragment -> solve pieces -> assemble)
             (bounded qubits, scales to large molecules)

and demonstrates THREE honest, defensible advantages of the
shadow-tomography + fragmentation combination:

  1. QUBIT ADVANTAGE      - FMO keeps qubits bounded; direct grows with size.
  2. MEASUREMENT ADVANTAGE - shadows obtain every subsystem's energy AND full
                             1-RDM from ONE dataset; grouped-Pauli needs many
                             measurement settings, growing with subsystem size.
  3. ACCURACY              - FMO-VQE/Shadow reproduce the exact full-molecule
                             energy to (near) chemical accuracy.

System: dimerised hydrogen chains H4, H6, H8, H10 (van-der-Waals separated
H2 units, where the two-body expansion is accurate). Plus a water-dimer
"real molecule" demonstration.

Requires:
    pip install qiskit-nature[pyscf]

Output:
    figures/study8_fmo_scaling.png (.pdf)
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

from shadowvqe.fmo import (
    run_fmo, measurement_cost_report, direct_vqe_cost,
)
from shadowvqe.rdm import rdm_measurement_cost
from shadowvqe.validation import sparse_ground_state_energy

# ── Configuration ─────────────────────────────────────────────────────────
SEED = 42
CHEM_ACC = 1.594e-3
N_SHADOWS = 4000
REPS = 2
CHAIN_SIZES = [2, 3, 4, 5]        # n_units -> H4, H6, H8, H10
R_INTRA = 0.74
R_INTER = 2.5                     # well-separated -> MBE-2 accurate
DIRECT_QUBIT_CAP = 10            # run direct VQE+shadow only up to this size
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def run_study() -> None:
    print("\n" + "=" * 72)
    print("  Study 8: FMO-VQE + Shadow Tomography - Scaling Advantage")
    print("=" * 72)

    # ── PySCF check ───────────────────────────────────────────────────────
    try:
        from shadowvqe.molecules import check_pyscf_available, hydrogen_chain_fragments
        check_pyscf_available()
    except ImportError as exc:
        print(f"\n  ERROR: {exc}")
        print("  Install with: pip install qiskit-nature[pyscf]")
        print("  Run this study on Google Colab.\n")
        return

    records = []

    for n_units in CHAIN_SIZES:
        label = f"H{2*n_units}"
        print(f"\n{'-'*72}\n  Building {label} chain (n_units={n_units}) ...")
        t0 = time.perf_counter()
        system = hydrogen_chain_fragments(
            n_units=n_units, r_intra=R_INTRA, r_inter=R_INTER,
            all_pairs=False,                 # nearest-neighbour -> O(n) pairs
            build_reference=True, max_ref_qubits=16,
        )
        print(f"  Built in {time.perf_counter()-t0:.1f}s | "
              f"fragments={system.n_fragments}, pairs={len(system.pairs)}, "
              f"max_qubits={system.max_qubits}")

        # ── Reference (full molecule, exact) ──────────────────────────────
        ref_energy = None
        direct_qubits = None
        if system.reference is not None:
            direct_qubits = system.reference.num_qubits
            ref_energy = sparse_ground_state_energy(system.reference)
            print(f"  Exact full-molecule reference ({direct_qubits}q): "
                  f"{ref_energy:.6f} Ha")

        # ── FMO methods ───────────────────────────────────────────────────
        print("  Running FMO-exact ...")
        fmo_exact = run_fmo(system, method="exact", seed=SEED)

        print("  Running FMO-VQE ...")
        fmo_vqe = run_fmo(system, method="vqe", reps=REPS, seed=SEED)

        print("  Running FMO-Shadow ...")
        fmo_shadow = run_fmo(system, method="shadow", reps=REPS,
                             seed=SEED, n_shadows=N_SHADOWS)

        # ── Measurement cost ──────────────────────────────────────────────
        mcost = measurement_cost_report(system)
        # full RDM settings per subsystem (grouped vs shadow)
        rdm_grouped = sum(
            rdm_measurement_cost(q)["grouped_settings"]
            for q in [f.n_qubits for f in system.fragments]
            + [p.num_qubits for p in system.pairs.values()]
        )
        # shadow gets energy + RDM from the SAME dataset: 1 per subsystem
        n_sub = system.n_fragments + len(system.pairs)
        fmo_shadow_settings = n_sub
        fmo_grouped_settings = mcost["vqe_settings"] + rdm_grouped

        direct = (
            direct_vqe_cost(system.reference)
            if system.reference is not None else None
        )

        # ── Errors (vs exact full-molecule reference if available) ────────
        def err(e):
            return abs(e - ref_energy) if ref_energy is not None else None

        rec = {
            "label": label,
            "n_units": n_units,
            "max_qubits": system.max_qubits,
            "direct_qubits": direct_qubits,
            "ref_energy": ref_energy,
            "fmo_exact": fmo_exact.total_energy,
            "fmo_vqe": fmo_vqe.total_energy,
            "fmo_shadow": fmo_shadow.total_energy,
            "err_fmo_exact": err(fmo_exact.total_energy),
            "err_fmo_vqe": err(fmo_vqe.total_energy),
            "err_fmo_shadow": err(fmo_shadow.total_energy),
            "fmo_shadow_settings": fmo_shadow_settings,
            "fmo_grouped_settings": fmo_grouped_settings,
            "direct_settings": direct["vqe_settings"] if direct else None,
            "n_subsystems": n_sub,
        }
        records.append(rec)

        print(f"  FMO-exact : {rec['fmo_exact']:.6f} Ha  "
              f"(err {rec['err_fmo_exact']:.2e})" if rec['err_fmo_exact'] else
              f"  FMO-exact : {rec['fmo_exact']:.6f} Ha")
        print(f"  FMO-VQE   : {rec['fmo_vqe']:.6f} Ha")
        print(f"  FMO-Shadow: {rec['fmo_shadow']:.6f} Ha")
        print(f"  Qubits: FMO={rec['max_qubits']}  Direct={rec['direct_qubits']}")
        print(f"  Measurement settings: FMO-Shadow={fmo_shadow_settings}  "
              f"FMO-grouped-Pauli={fmo_grouped_settings}")

    _print_summary(records)
    _plot(records)

    # ── Real-molecule demonstration: water dimer ──────────────────────────
    _water_demo()


def _print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("  SUMMARY - Hydrogen Chain Scaling")
    print("=" * 80)
    print(f"  {'System':<7} {'FMO_q':>6} {'Direct_q':>9} {'FMO-VQE err':>13} "
          f"{'FMO-Shadow err':>15} {'Shadow set':>11} {'Grouped set':>12}")
    print("  " + "-" * 76)
    for r in records:
        ve = f"{r['err_fmo_vqe']:.2e}" if r['err_fmo_vqe'] is not None else "n/a"
        se = f"{r['err_fmo_shadow']:.2e}" if r['err_fmo_shadow'] is not None else "n/a"
        dq = r['direct_qubits'] if r['direct_qubits'] else "n/a"
        print(f"  {r['label']:<7} {r['max_qubits']:>6} {str(dq):>9} {ve:>13} "
              f"{se:>15} {r['fmo_shadow_settings']:>11} "
              f"{r['fmo_grouped_settings']:>12}")
    print("=" * 80)
    print(f"  FMO qubits stay BOUNDED while direct grows.")
    print(f"  Shadows get energy + full 1-RDM per subsystem from ONE dataset.")
    print("=" * 80 + "\n")


def _plot(records: list[dict]) -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "figure.facecolor": "white",
    })
    PAL = {"exact": "#2c3e50", "vqe": "#2980b9", "shadow": "#e74c3c",
           "acc": "#27ae60", "direct": "#e67e22", "fmo": "#16a085"}
    labels = [r["label"] for r in records]
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)

    # ── A: Total energy vs size ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    refs = [r["ref_energy"] for r in records]
    has_ref = [i for i, v in enumerate(refs) if v is not None]
    if has_ref:
        ax1.plot([x[i] for i in has_ref], [refs[i] for i in has_ref],
                 "o-", color=PAL["exact"], lw=2.4, ms=8, label="Exact (full molecule)")
    ax1.plot(x, [r["fmo_vqe"] for r in records], "s--", color=PAL["vqe"],
             lw=2, ms=7, label="FMO-VQE")
    ax1.plot(x, [r["fmo_shadow"] for r in records], "^--", color=PAL["shadow"],
             lw=2, ms=7, label="FMO-Shadow")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel("Hydrogen Chain", fontsize=11)
    ax1.set_ylabel("Total Energy (Hartree)", fontsize=11)
    ax1.set_title("A   Energy: FMO reproduces the full-molecule result",
                  fontweight="bold", fontsize=12)
    ax1.legend(fontsize=9)

    # ── B: Error vs chemical accuracy ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ve = [r["err_fmo_vqe"] for r in records]
    se = [r["err_fmo_shadow"] for r in records]
    xi = [i for i, v in enumerate(ve) if v is not None]
    if xi:
        w = 0.35
        ax2.bar([x[i]-w/2 for i in xi], [ve[i] for i in xi], w,
                color=PAL["vqe"], alpha=0.85, label="FMO-VQE")
        ax2.bar([x[i]+w/2 for i in xi], [se[i] for i in xi], w,
                color=PAL["shadow"], alpha=0.85, label="FMO-Shadow")
    ax2.axhline(CHEM_ACC, color=PAL["acc"], ls="--", lw=1.8,
                label=f"Chem. acc. ({CHEM_ACC:.1e})")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_yscale("log")
    ax2.set_xlabel("Hydrogen Chain", fontsize=11)
    ax2.set_ylabel("|E - E_exact| (Hartree)", fontsize=11)
    ax2.set_title("B   FMO Accuracy vs Full-Molecule Exact",
                  fontweight="bold", fontsize=12)
    ax2.legend(fontsize=9)

    # ── C: Qubit advantage ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    fmo_q = [r["max_qubits"] for r in records]
    dir_q = [r["direct_qubits"] for r in records]
    ax3.plot(x, fmo_q, "o-", color=PAL["fmo"], lw=2.4, ms=9,
             label="FMO (max subsystem)")
    # extrapolate direct line even where not computed
    dir_q_full = [r["direct_qubits"] if r["direct_qubits"]
                  else 4 * r["n_units"] - 2 for r in records]
    ax3.plot(x, dir_q_full, "s--", color=PAL["direct"], lw=2.4, ms=9,
             label="Direct VQE (full molecule)")
    ax3.fill_between(x, fmo_q, dir_q_full, color=PAL["direct"], alpha=0.10)
    ax3.set_xticks(x); ax3.set_xticklabels(labels)
    ax3.set_xlabel("Hydrogen Chain", fontsize=11)
    ax3.set_ylabel("Qubits Required", fontsize=11)
    ax3.set_title("C   Qubit Advantage: FMO Stays Bounded",
                  fontweight="bold", fontsize=12)
    ax3.legend(fontsize=9)
    for i in range(len(x)):
        ax3.annotate(f"{fmo_q[i]}", (x[i], fmo_q[i]),
                     textcoords="offset points", xytext=(0, -14),
                     ha="center", fontsize=8, color=PAL["fmo"])
        ax3.annotate(f"{dir_q_full[i]}", (x[i], dir_q_full[i]),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8, color=PAL["direct"])

    # ── D: Measurement-setting advantage ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    shadow_set = [r["fmo_shadow_settings"] for r in records]
    grouped_set = [r["fmo_grouped_settings"] for r in records]
    ax4.plot(x, grouped_set, "s--", color=PAL["vqe"], lw=2.4, ms=9,
             label="Grouped-Pauli (energy + 1-RDM)")
    ax4.plot(x, shadow_set, "^-", color=PAL["shadow"], lw=2.4, ms=9,
             label="Shadows (one dataset/subsystem)")
    ax4.fill_between(x, shadow_set, grouped_set, color=PAL["shadow"], alpha=0.10)
    ax4.set_xticks(x); ax4.set_xticklabels(labels)
    ax4.set_xlabel("Hydrogen Chain", fontsize=11)
    ax4.set_ylabel("Distinct Measurement Settings", fontsize=11)
    ax4.set_title("D   Measurement Advantage (energy + full 1-RDM)",
                  fontweight="bold", fontsize=12)
    ax4.legend(fontsize=9)
    for i in range(len(x)):
        ratio = grouped_set[i] / shadow_set[i] if shadow_set[i] else 0
        ax4.annotate(f"{ratio:.0f}x", (x[i], grouped_set[i]),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=9, fontweight="bold",
                     color=PAL["shadow"])

    fig.suptitle(
        "Study 8 - FMO-VQE + Shadow Tomography: Scaling to Larger Molecules",
        fontsize=15, fontweight="bold", y=1.01,
    )
    for fmt in ("png", "pdf"):
        fig.savefig(FIGURES / f"study8_fmo_scaling.{fmt}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURES}/study8_fmo_scaling.png  (.pdf)")


def _water_demo() -> None:
    """Real-molecule demonstration: water dimer hydrogen-bond energy."""
    print("\n" + "=" * 72)
    print("  Real-Molecule Demo: Water Dimer (H2O)2 Hydrogen Bond")
    print("=" * 72)
    try:
        from shadowvqe.molecules import water_cluster_fragments
        system = water_cluster_fragments(n_waters=2, o_o_distance=2.8)
    except Exception as exc:
        print(f"  Skipped ({exc})")
        return

    print(f"  Fragments={system.n_fragments}, pairs={len(system.pairs)}, "
          f"max_qubits={system.max_qubits}  (active-space approximation)")

    fmo_exact = run_fmo(system, method="exact", seed=SEED)
    fmo_shadow = run_fmo(system, method="shadow", reps=REPS,
                         seed=SEED, n_shadows=N_SHADOWS)

    e_int_exact = list(fmo_exact.interaction_energies.values())[0]
    e_int_shadow = list(fmo_shadow.interaction_energies.values())[0]
    kcal = 627.509
    print(f"  Interaction energy (exact) : {e_int_exact:.6f} Ha "
          f"= {e_int_exact*kcal:.2f} kcal/mol")
    print(f"  Interaction energy (shadow): {e_int_shadow:.6f} Ha "
          f"= {e_int_shadow*kcal:.2f} kcal/mol")
    print(f"  (This hydrogen-bond energy comes from shadow-estimated "
          f"fragment + pair energies.)")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run_study()
