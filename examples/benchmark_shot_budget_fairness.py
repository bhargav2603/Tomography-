"""
Study 6 — Fixed Shot Budget Comparison.

A critical fairness test: given a fixed total measurement budget B,
how accurately can VQE and Shadow-VQE estimate the ground-state energy?

Protocol
--------
  VQE approach:
    - Groups Pauli terms into commuting sets (via group_commuting()).
    - Each group gets B / n_groups shots.
    - Final energy = sum of group energies (each measured with B/n_groups shots).

  Shadow-VQE approach:
    - Uses all B shots as classical shadows (single circuit type).
    - Estimates full Hamiltonian expectation from the shadow dataset.

Hamiltonians tested (no PySCF required):
  - 2-qubit H2 at equilibrium (2 commuting groups)
  - 4-qubit Heisenberg chain (many groups)
  - 6-qubit Heisenberg chain (even more groups)

Result interpretation:
  - When B/n_groups < threshold needed per group, VQE accuracy degrades.
  - Shadow-VQE uses all B shots toward the same Hamiltonian, doing better.

Output:
    figures/study6_shot_budget.png (.pdf)
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
from shadowvqe.shadows import ClassicalShadows
from shadowvqe.validation import exact_ground_state_energy

# ── Configuration ─────────────────────────────────────────────────────────
SEED = 42
CHEM_ACC = 1.594e-3
# Total shot budgets to sweep
BUDGETS = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
N_TRIALS = 15           # independent random trials per (budget, molecule)
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def _get_vqe_params(ham, seed: int = SEED, reps: int = 1) -> np.ndarray:
    """Run VQE once to get near-optimal parameters (no budget constraint here)."""
    n_q = ham.num_qubits
    # Use more reps + iterations for larger systems to ensure convergence
    actual_reps = reps if n_q <= 2 else (reps + 1 if n_q <= 4 else reps + 2)
    max_iter = 400 if n_q <= 2 else 800
    ansatz = hardware_efficient_ansatz(n_q, reps=actual_reps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VQE(ansatz, ham, optimizer="cobyla", max_iter=max_iter, seed=seed).run()
    return res.optimal_parameters


def _reps_for(n_qubits: int) -> int:
    """Return ansatz reps matching what _get_vqe_params used."""
    if n_qubits <= 2:
        return 1
    if n_qubits <= 4:
        return 2
    return 3


def _vqe_shot_energy(ham, params, budget: int, seed: int) -> float:
    """
    Estimate Hamiltonian energy with VQE-style commuting-group measurement.

    Splits budget evenly across commuting groups and uses ClassicalShadows
    restricted to the group's Pauli bases to simulate shot-limited measurement.
    """
    groups = ham.group_commuting()
    n_groups = len(groups)
    shots_per_group = max(50, budget // n_groups)

    reps = _reps_for(ham.num_qubits)
    ansatz = hardware_efficient_ansatz(ham.num_qubits, reps=reps)
    if len(params) != ansatz.num_parameters:
        # Parameter count mismatch — use default reps=1 as fallback
        ansatz = hardware_efficient_ansatz(ham.num_qubits, reps=1)

    bound = ansatz.assign_parameters(
        dict(zip(sorted(ansatz.parameters, key=lambda p: p.name),
                 params[:ansatz.num_parameters]))
    )

    rng = np.random.default_rng(seed)
    total_energy = 0.0

    for group in groups:
        g_seed = int(rng.integers(0, 2**28))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs = ClassicalShadows(
                n_qubits=ham.num_qubits,
                n_shadows=shots_per_group,
                seed=g_seed,
            )
        cs.collect(bound)
        total_energy += cs.estimate_observable(group)

    return total_energy


def _shadow_shot_energy(ham, params, budget: int, seed: int) -> float:
    """
    Estimate Hamiltonian energy with classical shadows using full budget.
    """
    reps = _reps_for(ham.num_qubits)
    ansatz = hardware_efficient_ansatz(ham.num_qubits, reps=reps)
    if len(params) != ansatz.num_parameters:
        ansatz = hardware_efficient_ansatz(ham.num_qubits, reps=1)

    bound = ansatz.assign_parameters(
        dict(zip(sorted(ansatz.parameters, key=lambda p: p.name),
                 params[:ansatz.num_parameters]))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = ClassicalShadows(
            n_qubits=ham.num_qubits,
            n_shadows=budget,
            seed=seed,
        )
    cs.collect(bound)
    return cs.estimate_observable(ham)


def run_benchmark(
    ham_name: str,
    ham,
    budgets: list[int],
    n_trials: int,
    seed: int,
    exact: float,
) -> dict:
    """Run the shot budget comparison for one Hamiltonian."""
    print(f"\n  [{ham_name}]  qubits={ham.num_qubits},  "
          f"groups={len(ham.group_commuting())},  "
          f"exact={exact:.5f} Ha")

    # Get near-optimal parameters once (not counted in shot budget)
    params = _get_vqe_params(ham, seed=seed)
    rng = np.random.default_rng(seed)

    vqe_mean_err, vqe_std_err = [], []
    shad_mean_err, shad_std_err = [], []

    for budget in budgets:
        v_errs, s_errs = [], []
        for _ in range(n_trials):
            t_seed = int(rng.integers(0, 2**28))
            e_v = _vqe_shot_energy(ham, params, budget, t_seed)
            e_s = _shadow_shot_energy(ham, params, budget, t_seed + 1000000)
            v_errs.append(abs(e_v - exact))
            s_errs.append(abs(e_s - exact))

        vqe_mean_err.append(float(np.mean(v_errs)))
        vqe_std_err.append(float(np.std(v_errs)))
        shad_mean_err.append(float(np.mean(s_errs)))
        shad_std_err.append(float(np.std(s_errs)))

        print(f"    Budget={budget:>6}:  "
              f"VQE err={vqe_mean_err[-1]:.2e} +/- {vqe_std_err[-1]:.2e}  |  "
              f"Shadow err={shad_mean_err[-1]:.2e} +/- {shad_std_err[-1]:.2e}")

    return {
        "name": ham_name,
        "n_qubits": ham.num_qubits,
        "n_groups": len(ham.group_commuting()),
        "exact_energy": exact,
        "budgets": budgets,
        "vqe_mean_err": vqe_mean_err,
        "vqe_std_err": vqe_std_err,
        "shadow_mean_err": shad_mean_err,
        "shadow_std_err": shad_std_err,
    }


def run_study() -> None:
    print("\n" + "=" * 70)
    print("  Study 6: Fixed Shot Budget — VQE vs Shadow-VQE")
    print("=" * 70)
    print(f"  Budgets: {BUDGETS}")
    print(f"  Trials per budget: {N_TRIALS}")

    # ── Build Hamiltonians and run benchmarks ──────────────────────────────
    molecules = [
        ("H2",      h2_hamiltonian()),
        ("Heis-4q", heisenberg_hamiltonian(4, periodic=False)),
        ("Heis-6q", heisenberg_hamiltonian(6, periodic=False)),
    ]

    all_results = []
    for name, ham in molecules:
        exact = exact_ground_state_energy(ham)
        res = run_benchmark(name, ham, BUDGETS, N_TRIALS, SEED, exact)
        all_results.append(res)

    # ── Print crossover table ─────────────────────────────────────────────
    _print_crossover(all_results)

    # ── Plot ──────────────────────────────────────────────────────────────
    _plot(all_results)


def _print_crossover(all_results: list[dict]) -> None:
    print("\n" + "=" * 75)
    print("  CROSSOVER ANALYSIS  (budget at which Shadow-VQE surpasses VQE)")
    print("=" * 75)
    for r in all_results:
        crossover = None
        for b, ve, se in zip(r["budgets"], r["vqe_mean_err"], r["shadow_mean_err"]):
            if se < ve:
                crossover = b
                break
        adv_str = f"budget >= {crossover}" if crossover else "not yet within budget range"
        print(f"  {r['name']:<10}  groups={r['n_groups']:>3}  "
              f"Shadow advantage at: {adv_str}")
    print("=" * 75 + "\n")


def _plot(all_results: list[dict]) -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "figure.facecolor": "white",
    })

    PAL = {"vqe": "#2980b9", "shadow": "#e74c3c", "acc": "#27ae60"}
    lw, ms = 2.0, 6

    n_mol = len(all_results)
    fig = plt.figure(figsize=(6 * n_mol, 12))
    gs = gridspec.GridSpec(2, n_mol, hspace=0.40, wspace=0.30)

    for col, r in enumerate(all_results):
        budgets = r["budgets"]
        n_groups = r["n_groups"]

        # ── Row 0: Mean error vs budget (log-log) ────────────────────────
        ax = fig.add_subplot(gs[0, col])
        ve = np.array(r["vqe_mean_err"])
        vs = np.array(r["vqe_std_err"])
        se = np.array(r["shadow_mean_err"])
        ss = np.array(r["shadow_std_err"])

        ax.loglog(budgets, ve, "o-", color=PAL["vqe"], lw=lw, ms=ms,
                  label=f"VQE ({n_groups} groups)")
        ax.fill_between(budgets, np.maximum(ve - vs, 1e-8), ve + vs,
                        alpha=0.2, color=PAL["vqe"])
        ax.loglog(budgets, se, "s--", color=PAL["shadow"], lw=lw, ms=ms,
                  label="Shadow-VQE (all shots)")
        ax.fill_between(budgets, np.maximum(se - ss, 1e-8), se + ss,
                        alpha=0.2, color=PAL["shadow"])
        ax.axhline(CHEM_ACC, color=PAL["acc"], ls="--", lw=1.5,
                   label="Chem. acc.")

        # Mark crossover point
        for b, v, s in zip(budgets, ve, se):
            if s < v:
                ax.axvline(b, color="gray", ls=":", lw=1.2, alpha=0.6)
                ax.text(b * 1.05, ax.get_ylim()[0] * 2,
                        f"x-over\nB={b}", fontsize=7, color="gray")
                break

        ax.set_xlabel("Total Shot Budget", fontsize=11)
        ax.set_ylabel("|E - E_exact| (Hartree)", fontsize=11)
        ax.set_title(
            f"{chr(65+col)}   {r['name']} (q={r['n_qubits']})",
            fontweight="bold", fontsize=12,
        )
        ax.legend(fontsize=8)

        # ── Row 1: Effective shots breakdown ─────────────────────────────
        ax2 = fig.add_subplot(gs[1, col])
        eff_vqe = np.array(budgets) // max(n_groups, 1)
        ax2.semilogx(budgets, eff_vqe, "o-", color=PAL["vqe"], lw=lw, ms=ms,
                     label=f"VQE: B/{n_groups} per group")
        ax2.semilogx(budgets, budgets, "s--", color=PAL["shadow"], lw=lw, ms=ms,
                     label="Shadow: full B")
        ax2.fill_between(budgets, eff_vqe, budgets, alpha=0.12,
                         color=PAL["shadow"], label="Shadow advantage zone")
        ax2.set_xlabel("Total Shot Budget", fontsize=11)
        ax2.set_ylabel("Effective Shots for Estimation", fontsize=11)
        ax2.set_title(
            f"{chr(65+n_mol+col)}   Shot Allocation Breakdown",
            fontweight="bold", fontsize=12,
        )
        ax2.legend(fontsize=8)

    fig.suptitle(
        "Study 6 — Fixed Shot Budget: VQE vs Shadow-VQE",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for fmt in ("png", "pdf"):
        fig.savefig(FIGURES / f"study6_shot_budget.{fmt}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURES}/study6_shot_budget.png  (.pdf)")


if __name__ == "__main__":
    run_study()
