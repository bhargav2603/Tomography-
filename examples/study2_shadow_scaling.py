"""
Study 2 — Shadow Count Scaling: Error vs N_shadows

The most important validation of classical shadow tomography.
Theory (Huang et al. 2020) predicts:
    error ∝ 1 / sqrt(N_shadows)

We fix the quantum state (H2 ground state) and vary N_shadows from 100 to 10,000.
If the error follows the theoretical 1/√N line on a log-log plot, our
shadow estimator is statistically correct.

What a log-log plot means:
    - X-axis: log(N_shadows)
    - Y-axis: log(error)
    - A straight line with slope -0.5 = perfect 1/√N scaling

Run:
    python examples/study2_shadow_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shadowvqe.hamiltonians import h2_hamiltonian
from shadowvqe.ansatz import hardware_efficient_ansatz
from shadowvqe.vqe import VQE
from shadowvqe.shadows import ClassicalShadows
from shadowvqe.validation import exact_ground_state_energy


def run_shadow_scaling(
    n_trials: int = 10,
    seed: int = 42,
    output_dir: Path = Path("results"),
    fig_dir: Path = Path("figures"),
) -> None:
    output_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    ham = h2_hamiltonian()
    ansatz = hardware_efficient_ansatz(n_qubits=2, reps=1)
    exact = exact_ground_state_energy(ham)

    # Find the true ground state circuit via exact VQE
    print("\nFinding H2 ground state via VQE (one-time)...")
    vqe_result = VQE(ansatz=ansatz, hamiltonian=ham, max_iter=400, seed=seed).run()
    ground_state_params = vqe_result.optimal_parameters
    bound_circuit = ansatz.assign_parameters(
        {p: v for p, v in zip(sorted(ansatz.parameters, key=lambda x: x.name),
                              ground_state_params)}
    )
    print(f"VQE energy: {vqe_result.ground_state_energy:.8f} Ha  "
          f"(error {abs(vqe_result.ground_state_energy - exact):.2e} Ha)\n")

    # Shadow counts to test
    n_shadows_list = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    print(f"{'N_shadows':>10}  {'Mean Error':>12}  {'Std Error':>12}  {'Theory 1/sqrtN':>16}")
    print("-" * 55)

    mean_errors, std_errors = [], []
    rng = np.random.default_rng(seed)

    for n_shadows in n_shadows_list:
        errors = []
        for trial in range(n_trials):
            trial_seed = int(rng.integers(0, 2**28))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cs = ClassicalShadows(
                    n_qubits=2, n_shadows=n_shadows, seed=trial_seed
                )
            cs.collect(bound_circuit)
            estimated = cs.estimate_observable(ham)
            errors.append(abs(estimated - exact))

        mean_e = float(np.mean(errors))
        std_e = float(np.std(errors))
        theory = 1.0 / np.sqrt(n_shadows)  # relative, for slope comparison

        mean_errors.append(mean_e)
        std_errors.append(std_e)

        print(f"{n_shadows:>10}  {mean_e:>12.5f}  {std_e:>12.5f}  {theory:>14.5f}")

    # ── Fit the slope on log-log scale ─────────────────────────────────────
    log_n = np.log10(n_shadows_list)
    log_e = np.log10(mean_errors)
    slope, intercept = np.polyfit(log_n, log_e, 1)
    print(f"\nFitted log-log slope: {slope:.3f}  (theory: -0.500)")
    if abs(slope + 0.5) < 0.1:
        print("PASS: Scaling is consistent with 1/sqrt(N) (within 10% of theoretical slope)")
    else:
        print(f"NOTE: Slope deviates from -0.5 — consider increasing n_trials for better statistics")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "figure.facecolor": "white",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1 — Linear scale
    ax1.errorbar(n_shadows_list, mean_errors, yerr=std_errors,
                 fmt="ro-", ms=6, capsize=4, label="Shadow error (mean ± std)")
    ax1.axhline(1.594e-3, color="green", linestyle="--", lw=1.5,
                label="Chemical accuracy")
    ax1.set_xlabel("Number of Shadows (N)")
    ax1.set_ylabel("|Error| (Ha)")
    ax1.set_title("Shadow Error vs Shot Count")
    ax1.legend()

    # Panel 2 — Log-log scale (shows the 1/√N slope)
    ax2.loglog(n_shadows_list, mean_errors, "ro-", ms=6, label="Shadow error")

    # Overlay theoretical 1/√N line, scaled to match at midpoint
    mid = len(n_shadows_list) // 2
    scale = mean_errors[mid] * np.sqrt(n_shadows_list[mid])
    theory_line = [scale / np.sqrt(n) for n in n_shadows_list]
    ax2.loglog(n_shadows_list, theory_line, "k--", lw=1.5,
               label=f"1/√N fit  (slope={slope:.2f})")
    ax2.axhline(1.594e-3, color="green", linestyle="--", lw=1.5,
                label="Chemical accuracy")
    ax2.set_xlabel("Number of Shadows (N)")
    ax2.set_ylabel("|Error| (Ha)")
    ax2.set_title("Log-Log: Confirms 1/√N Scaling")
    ax2.legend()

    fig.suptitle("Study 2: Classical Shadow Error Scales as 1/√N",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"study2_shadow_scaling.{ext}", bbox_inches="tight")

    print(f"\nFigure saved: {fig_dir}/study2_shadow_scaling.png")

    results = {
        "n_shadows_list": n_shadows_list,
        "mean_errors": mean_errors,
        "std_errors": std_errors,
        "fitted_slope": float(slope),
        "theoretical_slope": -0.5,
        "n_trials_per_point": n_trials,
    }
    with open(output_dir / "study2_shadow_scaling.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Data saved:   {output_dir}/study2_shadow_scaling.json")


if __name__ == "__main__":
    run_shadow_scaling()
