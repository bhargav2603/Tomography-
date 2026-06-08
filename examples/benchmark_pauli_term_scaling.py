"""
Benchmark: Pauli Term Scaling

As quantum systems grow, the number of terms in the Hamiltonian grows.
This benchmark shows how the number of Pauli terms scales with system size.

Reference: Standard quantum chemistry scaling
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shadowvqe.hamiltonians import h2_hamiltonian, heisenberg_hamiltonian

def run():
    print("\n" + "=" * 60)
    print("  Benchmark: Pauli Term Scaling")
    print("=" * 60)

    systems = [
        ("H2 (2q)", h2_hamiltonian()),
        ("Heisenberg-3q", heisenberg_hamiltonian(3)),
        ("Heisenberg-4q", heisenberg_hamiltonian(4)),
        ("Heisenberg-5q", heisenberg_hamiltonian(5)),
        ("Heisenberg-6q", heisenberg_hamiltonian(6)),
    ]

    n_qubits = []
    n_terms = []
    n_groups = []

    print(f"\n{'System':<15}  {'Qubits':<8}  {'Pauli Terms':<12}  {'Commuting Groups':<16}")
    print("-" * 55)

    for name, ham in systems:
        q = ham.num_qubits
        t = len(ham)
        g = len(ham.group_commuting())
        n_qubits.append(q)
        n_terms.append(t)
        n_groups.append(g)
        print(f"{name:<15}  {q:<8}  {t:<12}  {g:<16}")

    # Professional publication-quality figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Pauli term growth
    ax1.plot(n_qubits, n_terms, "o-", color="#0173B2", linewidth=2.5, markersize=8,
            label="Pauli terms", alpha=0.85)
    ax1.set_xlabel("Number of Qubits", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Pauli Terms", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Hamiltonian Complexity Growth", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Panel B: Commuting group scaling
    ax2.plot(n_qubits, n_groups, "s--", color="#DE8F05", linewidth=2.5, markersize=8,
            label="Commuting groups (VQE circuits)", alpha=0.85)
    ax2.set_xlabel("Number of Qubits", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of Commuting Groups", fontsize=11, fontweight="bold")
    ax2.set_title("(B) Measurement Setting Cost", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    fig.suptitle("Pauli Term and Measurement Scaling", fontsize=13, fontweight="bold")
    fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/research_pauli_scaling.{ext}", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: figures/research_pauli_scaling.png (.pdf)")

if __name__ == "__main__":
    run()
