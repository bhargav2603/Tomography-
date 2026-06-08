"""
Benchmark: Pauli Term Scaling
Shows how Hamiltonian complexity grows with system size.
Tests H2, H4, H6 hydrogen chains.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib.pyplot as plt
from shadowvqe.hamiltonians import h2_hamiltonian
from shadowvqe.molecules import h4_hamiltonian, h6_hamiltonian

# (name, hamiltonian builder) - built lazily inside run()
MOLECULES = [
    ("H2", h2_hamiltonian),
    ("H4", h4_hamiltonian),
    ("H6", h6_hamiltonian),
]

def run():
    print("\n" + "="*70)
    print("  Benchmark: Pauli Term Scaling in Hydrogen Chains")
    print("="*70)

    Path("figures").mkdir(exist_ok=True)

    print(f"\n{'Molecule':<10} {'Qubits':<8} {'Pauli Terms':<15} {'Commuting Groups':<18}")
    print("-" * 70)

    names = []
    qubits = []
    n_terms = []
    n_groups = []

    for mol_name, builder in MOLECULES:
        ham = builder()
        names.append(mol_name)
        q = ham.num_qubits
        t = len(ham)
        g = len(ham.group_commuting())

        qubits.append(q)
        n_terms.append(t)
        n_groups.append(g)

        print(f"{mol_name:<10} {q:<8} {t:<15} {g:<18}")

    # Professional figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Pauli term growth
    ax1.plot(qubits, n_terms, "o-", color="#0173B2", linewidth=2.5, markersize=8, label="Pauli terms", alpha=0.85)
    ax1.set_xlabel("Number of Qubits", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Pauli Terms", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Hamiltonian Complexity Growth", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    for i, mol in enumerate(names):
        ax1.annotate(f"{mol}\n({n_terms[i]} terms)", xy=(qubits[i], n_terms[i]),
                    xytext=(10, 10), textcoords="offset points", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

    # Panel B: Commuting group scaling
    ax2.plot(qubits, n_groups, "s--", color="#DE8F05", linewidth=2.5, markersize=8, label="Commuting groups", alpha=0.85)
    ax2.set_xlabel("Number of Qubits", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of Commuting Groups", fontsize=11, fontweight="bold")
    ax2.set_title("(B) VQE Measurement Setting Cost", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    for i, mol in enumerate(names):
        ax2.annotate(f"{mol}\n({n_groups[i]} groups)", xy=(qubits[i], n_groups[i]),
                    xytext=(10, 10), textcoords="offset points", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))

    fig.suptitle("Hamiltonian Complexity Scaling", fontsize=13, fontweight="bold")
    fig.tight_layout()

    fig.savefig("figures/research_pauli_scaling.png", dpi=150, bbox_inches="tight")
    fig.savefig("figures/research_pauli_scaling.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "="*70)
    print("  ANALYSIS: Why Larger Molecules Need Shadow Tomography")
    print("="*70)
    print("\n  Key Finding: Complexity grows RAPIDLY with system size\n")
    for name, q, t, g in zip(names, qubits, n_terms, n_groups):
        print(f"  {name} ({q}q):  {t:>3} Pauli terms,  {g:>2} commuting groups")
    print("""
  Why this is critical:
  - VQE must MEASURE each commuting group separately
  - Each group = separate circuit measurement pass
  - More groups = more measurement overhead

  Shadow advantage:
  - Single randomized measurement protocol
  - Works for ALL Pauli terms simultaneously
  - Cost is INDEPENDENT of number of groups

  Practical implication:
  For H6+, shadow tomography becomes not just faster, but NECESSARY
  for practical quantum advantage.
    """)
    print("="*70 + "\n")

if __name__ == "__main__":
    run()
