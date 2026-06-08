"""
Study 3 — Heisenberg Chain Scaling: When do Shadows win?

This script demonstrates the crossover point where Shadow-VQE requires
fewer circuit executions per optimization step than standard VQE.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Assuming you have a Heisenberg builder in your hamiltonians.py
from shadowvqe.hamiltonians import heisenberg_1d_hamiltonian

def run_heisenberg_scaling():
    qubit_sizes = [2, 3, 4, 5, 6, 7, 8]
    
    vqe_circuits_per_step = []
    shadow_circuits_per_step = []
    
    shots_per_vqe_group = 1000
    fixed_shadows = 2000

    print(f"{'Qubits':<8} | {'Pauli Terms':<12} | {'VQE Circuits':<15} | {'Shadow Circuits'}")
    print("-" * 60)

    for n in qubit_sizes:
        ham = heisenberg_1d_hamiltonian(n_qubits=n)
        
        # Standard VQE needs to measure mutually commuting groups.
        # Qiskit's SparsePauliOp.group_commuting() gives the minimal groups.
        groups = ham.group_commuting()
        n_groups = len(groups)
        
        vqe_cost = n_groups * shots_per_vqe_group
        vqe_circuits_per_step.append(vqe_cost)
        
        # Shadows always cost the fixed shadow count, regardless of Hamiltonian size
        shadow_circuits_per_step.append(fixed_shadows)
        
        print(f"{n:<8} | {len(ham):<12} | {vqe_cost:<15} | {fixed_shadows}")

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    plt.plot(qubit_sizes, vqe_circuits_per_step, 'bo-', label=f"Standard VQE ({shots_per_vqe_group} shots/group)")
    plt.plot(qubit_sizes, shadow_circuits_per_step, 'ro-', label=f"Shadow-VQE ({fixed_shadows} fixed shadows)")
    
    plt.fill_between(qubit_sizes, vqe_circuits_per_step, shadow_circuits_per_step, 
                     where=[v > fixed_shadows for v in vqe_circuits_per_step], 
                     color='red', alpha=0.1, label="Shadow Advantage Zone")
                     
    plt.title("Study 3: Measurement Cost Scaling (Heisenberg Chain)")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Circuit Executions per VQE Step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "study3_heisenberg_scaling.png", bbox_inches="tight")
    print(f"\nSaved plot to {out_dir}/study3_heisenberg_scaling.png")

if __name__ == "__main__":
    # Note: If `heisenberg_1d_hamiltonian` doesn't exist yet, you will need to add it 
    # to `src/shadowvqe/hamiltonians.py` using Qiskit's SparsePauliOp!
    run_heisenberg_scaling()