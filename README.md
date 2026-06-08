# shadowvqe

A research-grade Python library for **Classical Shadow Tomography** and **Shadow-assisted VQE** built on Qiskit 1.x / 2.x.

---

## Features

| Module | What it does |
|---|---|
| `hamiltonians` | H₂, Heisenberg, random SparsePauliOp builders |
| `ansatz` | Hardware-efficient (EfficientSU2) and TwoLocal ansätze |
| `shadows` | Classical shadow tomography (Huang et al. 2020) |
| `estimators` | Exact statevector + shadow-based expectation estimators |
| `optimizers` | COBYLA and SPSA with convergence tracking |
| `vqe` | Standard VQE (exact oracle) |
| `shadow_vqe` | Shadow-assisted VQE |
| `validation` | Exact diagonalisation + comparison metrics table |
| `visualization` | Publication-quality matplotlib figures (PNG + PDF) |

---

## Installation

```bash
pip install -e ".[aer,dev]"
```

Requires Python ≥ 3.10, Qiskit ≥ 1.0, qiskit-aer ≥ 0.13.

---

## Quick Start

```python
import shadowvqe as svq

ham  = svq.h2_hamiltonian()
circ = svq.hardware_efficient_ansatz(n_qubits=2, reps=1)

# Standard VQE
result = svq.VQE(ansatz=circ, hamiltonian=ham, seed=42).run()
print(f"VQE:        {result.ground_state_energy:.6f} Ha")

# Shadow-VQE
sr = svq.ShadowVQE(ansatz=circ, hamiltonian=ham, n_shadows=2000, seed=42).run()
print(f"Shadow-VQE: {sr.ground_state_energy:.6f} Ha")
print(f"Total shadows used: {sr.total_shadows:,}")
```

---

## Run the H₂ Benchmark

```bash
python examples/run_h2_benchmark.py
```

Outputs:
- Console comparison table (Exact / VQE / Shadow-VQE)
- `results/benchmark.json`
- `figures/vqe_vs_shadow_comparison.{png,pdf}`

---

## Run Tests

```bash
pytest
```

---

## How Classical Shadows Reduce VQE Measurement Cost

Standard VQE estimates `⟨H⟩` by grouping Pauli terms and measuring each group separately, requiring **O(N_groups)** distinct circuits per optimizer step, which scales poorly for large molecules.

**Classical shadows** (Huang, Kueng, Preskill, *Nature Physics* 2020) instead:

1. Apply **N random single-qubit basis rotations** (X, Y, or Z chosen uniformly per qubit)
2. Measure in the computational basis → store N (basis, bitstring) pairs
3. Estimate `⟨P⟩` for **any** Pauli P from the same snapshot bank using:

```
⟨P⟩ ≈ (1/N) Σ_k  Π_i  factor(b_i^k, m_i^k, P_i)

where factor(b, m, P) =
    1             if P == I
    3 (-1)^m      if b == P   (basis matched)
    0             if b ≠ P   (mismatch → term skipped)
```

The factor of **3** compensates for the 1/3 probability of any single basis matching, making the estimator unbiased.

**Key cost advantage:** A single shadow bank of size N estimates *all* Pauli expectations simultaneously. For a Hamiltonian with K Pauli terms:
- Standard VQE: O(K) measurement circuits per step
- Shadow-VQE: O(1) shadow-collection pass per step, N shots total

For the H₂ molecule (5 Pauli terms) the gain is modest, but for realistic molecules (hundreds to thousands of terms) shadows cut the per-step circuit count dramatically.

---

## Project Structure

```
shadowvqe/
├── src/shadowvqe/
│   ├── __init__.py
│   ├── utils.py          # logging, seeding, dataclasses
│   ├── hamiltonians.py   # H2, Heisenberg, random
│   ├── ansatz.py         # EfficientSU2, TwoLocal
│   ├── shadows.py        # ClassicalShadows engine
│   ├── estimators.py     # StatevectorEstimator, ShadowEstimator
│   ├── optimizers.py     # COBYLA, SPSA
│   ├── vqe.py            # Standard VQE
│   ├── shadow_vqe.py     # Shadow-VQE
│   ├── validation.py     # Exact diag + BenchmarkResult
│   └── visualization.py  # All matplotlib figures
├── tests/
│   ├── test_hamiltonians.py
│   ├── test_shadows.py
│   ├── test_vqe.py
│   └── test_estimators.py
├── examples/
│   ├── quickstart.py
│   └── run_h2_benchmark.py
├── pyproject.toml
└── README.md
```

---

## Citation

If you use this library in a publication, please cite:

> Huang, H.-Y., Kueng, R., & Preskill, J. (2020).
> Predicting many properties of a quantum system from very few measurements.
> *Nature Physics*, 16, 1050–1057.
