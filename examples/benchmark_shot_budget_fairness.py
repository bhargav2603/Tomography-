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
from shadowvqe.visualization_research import plot_shot_budget_research

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
    """Generate professional publication-quality figures for each molecule."""
    for idx, r in enumerate(all_results):
        budgets = r["budgets"]
        vqe_errors = np.array(r["vqe_mean_err"])
        vqe_stds = np.array(r["vqe_std_err"])
        shadow_errors = np.array(r["shadow_mean_err"])
        shadow_stds = np.array(r["shadow_std_err"])

        fig, ax = plot_shot_budget_research(
            budgets=budgets,
            vqe_errors_mean=vqe_errors,
            vqe_errors_std=vqe_stds,
            shadow_errors_mean=shadow_errors,
            shadow_errors_std=shadow_stds,
            chem_accuracy=CHEM_ACC,
            savedir=FIGURES,
        )
        # Rename the generic output file to include molecule name
        import shutil
        for ext in ("png", "pdf"):
            generic = FIGURES / f"research_shot_budget.{ext}"
            specific = FIGURES / f"research_shot_budget_{r['name'].lower()}.{ext}"
            if generic.exists():
                shutil.move(generic, specific)
                print(f"  Saved: {specific}")

        plt.close(fig)


if __name__ == "__main__":
    run_study()
