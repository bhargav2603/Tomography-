# ShadowVQE

Classical Shadow Tomography integrated with the Variational Quantum Eigensolver (VQE) for quantum chemistry on real molecules.

## What it does

Compares standard VQE against shadow-assisted VQE across H2, H4, and H6 hydrogen chains, demonstrating where classical shadows reduce measurement overhead as system size grows.

Five benchmarks ship out of the box:

| Benchmark | What it shows |
|---|---|
| `benchmark_shadow_error_scaling.py` | Shadow error follows 1/sqrt(N) universally across all system sizes |
| `benchmark_shot_budget_fairness.py` | Shadows win over VQE at a lower budget threshold for larger molecules |
| `benchmark_heisenberg_measurement_cost.py` | VQE measurement groups grow with system size; shadows stay constant at 1 |
| `benchmark_pauli_term_scaling.py` | Pauli term count vs qubit count across H2/H4/H6 |
| `benchmark_adaptive_shadow_variance.py` | Hamiltonian-adapted shadows reduce variance more for larger systems |

All benchmarks produce publication-quality figures (PNG + PDF) in `figures/`.

## Installation

```bash
pip install -e ".[aer,dev]"
```

Molecular Hamiltonians (H4, H6) require PySCF:

```bash
pip install pyscf
```

H2 works without PySCF via a built-in Qiskit-Nature Hamiltonian.

## Running on Google Colab

The recommended way to run (H4/H6 require more than 8 GB RAM):

```python
# Cell 1 — install
!pip install qiskit qiskit-aer pyscf matplotlib scipy -q
!git clone https://github.com/<your-repo>/shadowvqe && pip install -e shadowvqe/

# Cell 2 — run a benchmark
%cd shadowvqe
%run examples/benchmark_shadow_error_scaling.py
```

See `COLAB_RUN_H2_H4_H6.md` for the full cell sequence.

## Quick local test (H2 only, no PySCF)

```python
from shadowvqe.hamiltonians import h2_hamiltonian
from shadowvqe.shadows import ClassicalShadows
from shadowvqe.ansatz import hardware_efficient_ansatz
from shadowvqe.vqe import VQE

ham    = h2_hamiltonian()
ansatz = hardware_efficient_ansatz(ham.num_qubits, reps=1)
result = VQE(ansatz, ham, max_iter=200, seed=42).run()

cs = ClassicalShadows(ham.num_qubits, n_shadows=1000, seed=0)
cs.collect(ansatz.assign_parameters(
    {p: v for p, v in zip(sorted(ansatz.parameters, key=lambda x: x.name),
                          result.optimal_parameters)}
))
print(f"Shadow estimate: {cs.estimate_observable(ham):.6f} Ha")
```

## Tests

```bash
pytest        # 51 tests
```

## Project layout

```
src/shadowvqe/
    shadows.py              # ClassicalShadows — random Pauli measurements
    derandomized_shadows.py # HamiltonianAdaptedShadows — weighted basis
    vqe.py                  # VQE with COBYLA/SPSA
    shadow_vqe.py           # Shadow-assisted VQE loop
    hamiltonians.py         # H2 (built-in)
    molecules.py            # H4, H6 via PySCF
    fmo.py                  # Fragment Molecular Orbital energy assembly
    visualization_research.py  # Publication-quality figures
examples/                   # Five runnable benchmarks
tests/                      # 51 pytest tests
```

## Key references

- Huang, H.-Y., Kueng, R., & Preskill, J. (2020). Predicting many properties of a quantum system from very few measurements. *Nature Physics*, 16, 1050-1057.
- Peruzzo, A. et al. (2014). A variational eigenvalue solver on a photonic chip. *Nature Communications*, 5, 4213.

## License

MIT — see `pyproject.toml`.
