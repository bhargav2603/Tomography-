"""
Study 4 — Pauli Term Count Scaling: The Core Shadow Advantage

This study directly tests the theoretical claim:
    Standard VQE: measurement cost grows with number of Pauli terms (O(K))
    Shadow-VQE:   measurement cost is CONSTANT regardless of Pauli terms (O(1))

We use 4-qubit random Hamiltonians and vary the number of Pauli terms from 5 to 80.
We measure the OBJECTIVE EVALUATION TIME per optimizer step for both methods.

Expected outcome:
    - VQE evaluation time per step: roughly constant (statevector is always exact)
      [Note: in real hardware VQE, this would grow — statevector simulator is O(2^n)]
    - Shadow-VQE evaluation time per step: roughly constant regardless of K
    - Shadow-VQE TOTAL SHADOWS stays the same regardless of K

The key metric this study reveals:
    "For K Pauli terms, standard VQE requires K distinct circuits per step.
     Shadow-VQE requires the SAME N shadow shots regardless of K."
     This is the measurement complexity advantage.

Run:
    python examples/study4_pauli_term_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import json
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shadowvqe.hamiltonians import random_hamiltonian
from shadowvqe.ansatz import hardware_efficient_ansatz
from shadowvqe.vqe import VQE
from shadowvqe.shadow_vqe import ShadowVQE
from shadowvqe.validation import exact_ground_state_energy


def run_pauli_scaling(
    n_qubits: int = 4,
    pauli_counts: list[int] | None = None,
    n_shadows: int = 1000,
    max_iter: int = 80,
    seed: int = 42,
    output_dir: Path = Path("results"),
    fig_dir: Path = Path("figures"),
) -> None:
    if pauli_counts is None:
        # 4 qubits has max 4^4 - 1 = 255 non-identity Pauli terms
        pauli_counts = [5, 10, 15, 20, 30, 40, 55, 70]

    output_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    results = []

    print("\n" + "=" * 70)
    print(f"  Study 4: Pauli Term Count Scaling ({n_qubits} qubits)")
    print("=" * 70)
    print(f"\n{'K (terms)':>10}  {'VQE err':>10}  {'Shad err':>10}  "
          f"{'VQE (s)':>9}  {'Shad (s)':>9}  {'Speedup':>9}")
    print("-" * 65)

    for k in pauli_counts:
        ham = random_hamiltonian(
            n_qubits=n_qubits, n_terms=k,
            max_weight=min(3, n_qubits),  # sparse: max 3-body terms
            seed=seed,
        )
        actual_k = len(ham)
        ansatz = hardware_efficient_ansatz(n_qubits=n_qubits, reps=2)
        exact = exact_ground_state_energy(ham)

        # Standard VQE
        t0 = time.perf_counter()
        vqe_res = VQE(
            ansatz=ansatz, hamiltonian=ham,
            optimizer="cobyla", max_iter=max_iter, seed=seed,
        ).run()
        vqe_time = time.perf_counter() - t0

        # Shadow-VQE
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svqe_res = ShadowVQE(
                ansatz=ansatz, hamiltonian=ham,
                n_shadows=n_shadows,
                optimizer="cobyla", max_iter=max_iter, seed=seed,
            ).run()
        shad_time = time.perf_counter() - t0

        vqe_err = abs(vqe_res.ground_state_energy - exact)
        shad_err = abs(svqe_res.ground_state_energy - exact)
        speedup = vqe_time / shad_time if shad_time > 0 else float("nan")

        results.append({
            "requested_k": k,
            "actual_k": actual_k,
            "n_qubits": n_qubits,
            "exact_energy": float(exact),
            "vqe_energy": vqe_res.ground_state_energy,
            "shadow_energy": svqe_res.ground_state_energy,
            "vqe_error": vqe_err,
            "shadow_error": shad_err,
            "vqe_time_s": vqe_time,
            "shadow_time_s": shad_time,
            "speedup": speedup,
            "total_shadows": svqe_res.total_shadows,
        })

        print(
            f"{actual_k:>10}  {vqe_err:>10.3e}  {shad_err:>10.3e}  "
            f"{vqe_time:>9.2f}  {shad_time:>9.2f}  {speedup:>9.2f}x"
        )

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "figure.facecolor": "white",
    })

    ks = [r["actual_k"] for r in results]
    vqe_times = [r["vqe_time_s"] for r in results]
    shad_times = [r["shadow_time_s"] for r in results]
    vqe_errs = [r["vqe_error"] for r in results]
    shad_errs = [r["shadow_error"] for r in results]
    speedups = [r["speedup"] for r in results]
    total_shadows = [r["total_shadows"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A — Runtime vs Pauli terms
    axes[0, 0].plot(ks, vqe_times, "b-o", ms=7, label="VQE")
    axes[0, 0].plot(ks, shad_times, "r-s", ms=7, label="Shadow-VQE")
    axes[0, 0].set_xlabel("Number of Pauli Terms (K)")
    axes[0, 0].set_ylabel("Total Runtime (s)")
    axes[0, 0].set_title("(A) Runtime vs Pauli Term Count")
    axes[0, 0].legend()

    # Panel B — Speedup vs Pauli terms
    axes[0, 1].plot(ks, speedups, "g-^", ms=7)
    axes[0, 1].axhline(1.0, color="grey", linestyle="--", lw=1.5, label="Break-even")
    axes[0, 1].set_xlabel("Number of Pauli Terms (K)")
    axes[0, 1].set_ylabel("VQE Time / Shadow-VQE Time")
    axes[0, 1].set_title("(B) Speedup Factor (>1 = Shadow-VQE wins)")
    axes[0, 1].legend()

    # Panel C — Error comparison
    axes[1, 0].semilogy(ks, vqe_errs, "b-o", ms=7, label="VQE error")
    axes[1, 0].semilogy(ks, shad_errs, "r-s", ms=7, label="Shadow-VQE error")
    axes[1, 0].axhline(1.594e-3, color="green", linestyle="--",
                       lw=1.5, label="Chemical accuracy")
    axes[1, 0].set_xlabel("Number of Pauli Terms (K)")
    axes[1, 0].set_ylabel("|Error| (Ha)")
    axes[1, 0].set_title("(C) Accuracy vs Pauli Term Count")
    axes[1, 0].legend()

    # Panel D — Total shadows used (should be roughly constant)
    axes[1, 1].plot(ks, total_shadows, "m-D", ms=7)
    axes[1, 1].set_xlabel("Number of Pauli Terms (K)")
    axes[1, 1].set_ylabel("Total Shadow Snapshots Used")
    axes[1, 1].set_title("(D) Shadow Budget is Constant w.r.t. K\n(this is the key advantage)")

    fig.suptitle(
        f"Study 4: Shadow-VQE Measurement Cost is O(1) in Pauli Terms\n"
        f"({n_qubits} qubits, {n_shadows} shadows/step)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"study4_pauli_scaling.{ext}", bbox_inches="tight")

    print(f"\nFigure saved: {fig_dir}/study4_pauli_scaling.png")

    with open(output_dir / "study4_pauli_scaling.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Data saved:   {output_dir}/study4_pauli_scaling.json")

    # Print the key point
    print("\n-- Key Takeaway --")
    print(f"Total shadows used per run (should be roughly constant): "
          f"{min(total_shadows):,} – {max(total_shadows):,}")
    print(f"This is the core advantage: shadow measurement cost does NOT")
    print(f"grow with the number of Pauli terms in the Hamiltonian.")


if __name__ == "__main__":
    run_pauli_scaling()
