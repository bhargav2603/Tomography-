"""
H2 Benchmark: Exact vs VQE vs Shadow-VQE.

Run from the project root:
    python examples/run_h2_benchmark.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import shadowvqe as svq
from shadowvqe.validation import exact_ground_state_energy, BenchmarkResult
from shadowvqe.visualization import plot_all

logging.basicConfig(level=logging.INFO)


def main() -> None:
    print("\n" + "=" * 60)
    print("  H2 Molecule: Exact vs VQE vs Shadow-VQE")
    print("=" * 60 + "\n")

    # --- Hamiltonian + Ansatz -----------------------------------------
    ham = svq.h2_hamiltonian()
    ansatz = svq.hardware_efficient_ansatz(n_qubits=ham.num_qubits, reps=1)
    print(f"Hamiltonian: {ham.num_qubits} qubits, {len(ham)} Pauli terms")
    print(f"Ansatz:      {ansatz.num_parameters} parameters\n")

    exact_energy = exact_ground_state_energy(ham)
    print(f"Exact ground state: {exact_energy:.8f} Ha\n")

    # --- Standard VQE -------------------------------------------------
    print("Running Standard VQE (COBYLA, exact oracle)...")
    vqe = svq.VQE(
        ansatz=ansatz,
        hamiltonian=ham,
        optimizer="cobyla",
        max_iter=500,
        seed=42,
    )
    vqe_result = vqe.run()
    print(f"  VQE energy:     {vqe_result.ground_state_energy:.8f} Ha")
    print(f"  VQE error:      {abs(vqe_result.ground_state_energy - exact_energy):.2e} Ha")
    print(f"  Iterations:     {vqe_result.n_iterations}")
    print(f"  Time:           {vqe_result.total_runtime_s:.2f} s\n")

    # --- Shadow-VQE (SPSA, warm-started from VQE) --------------------
    # SPSA is the recommended optimizer for stochastic energy estimates.
    # Warm-starting from VQE's optimal point gives a fair comparison of
    # the shadow estimation accuracy rather than optimiser trajectory.
    print("Running Shadow-VQE (SPSA, warm-started from VQE, 2000 shadows/step)...")
    svqe = svq.ShadowVQE(
        ansatz=ansatz,
        hamiltonian=ham,
        n_shadows=2000,
        optimizer="spsa",
        max_iter=80,
        seed=42,
        initial_point=vqe_result.optimal_parameters,
    )
    svqe_result = svqe.run()
    print(f"  ShadowVQE energy:  {svqe_result.ground_state_energy:.8f} Ha")
    print(f"  ShadowVQE error:   {abs(svqe_result.ground_state_energy - exact_energy):.2e} Ha")
    print(f"  Total shadows:     {svqe_result.total_shadows:,}")
    print(f"  Iterations:        {svqe_result.n_iterations}")
    print(f"  Time:              {svqe_result.total_runtime_s:.2f} s\n")

    # --- Figures -------------------------------------------------------
    print("Generating figures...")
    out_dir = Path("results")
    fig_dir = Path("figures")
    plot_all(
        vqe_result=vqe_result,
        shadow_result=svqe_result,
        exact_energy=exact_energy,
        save_dir=fig_dir,
    )

    # --- Save JSON results -------------------------------------------
    out_dir.mkdir(exist_ok=True)
    vqe_result.save_json(out_dir / "vqe_result.json")
    svqe_result.save_json(out_dir / "shadow_vqe_result.json")

    # --- Summary table -----------------------------------------------
    benchmark = BenchmarkResult(exact_energy=exact_energy)
    benchmark.add(vqe_result, exact_energy)
    benchmark.add(svqe_result, exact_energy)
    print("Comparison Table:")
    benchmark.print_table()
    benchmark.save_json(out_dir / "benchmark.json")
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
