"""
Quickstart: minimal usage of the shadowvqe library.

Run:
    python examples/quickstart.py

Shadow-VQE note: SPSA is the recommended optimizer for shadow-based objectives
because it handles stochastic energy estimates gracefully. COBYLA assumes a
smooth deterministic landscape and can stall early on noisy inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import shadowvqe as svq
from shadowvqe.validation import exact_ground_state_energy

ham = svq.h2_hamiltonian()
circ = svq.hardware_efficient_ansatz(n_qubits=ham.num_qubits, reps=1)
exact = exact_ground_state_energy(ham)
print(f"Exact ground state:  {exact:.6f} Ha\n")

# Standard VQE — exact statevector oracle, converges reliably
result = svq.VQE(ansatz=circ, hamiltonian=ham, seed=42).run()
print(f"VQE energy:          {result.ground_state_energy:.6f} Ha  "
      f"(error {abs(result.ground_state_energy - exact):.2e} Ha)")

# Shadow-VQE — replaces energy oracle with classical shadows
# Use SPSA (handles noisy objectives) and warm-start from VQE solution
shadow_result = svq.ShadowVQE(
    ansatz=circ,
    hamiltonian=ham,
    n_shadows=2000,
    optimizer="spsa",
    max_iter=80,
    seed=42,
    initial_point=result.optimal_parameters,   # warm-start from VQE
).run()
print(f"Shadow-VQE energy:   {shadow_result.ground_state_energy:.6f} Ha  "
      f"(error {abs(shadow_result.ground_state_energy - exact):.2e} Ha)")
print(f"Total shadows used:  {shadow_result.total_shadows:,}")
print("\nNote: Shadow-VQE is designed for scale (many Pauli terms). On H2 (5 terms),")
print("VQE is more efficient; shadows shine on large molecules (100+ Pauli terms).")
