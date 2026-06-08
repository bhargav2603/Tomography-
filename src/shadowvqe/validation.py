"""
Validation and benchmarking suite.

Provides exact-diagonalisation reference energies and a comparison table
between Exact, VQE, and Shadow-VQE results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .utils import OptimizationResult, get_logger

if TYPE_CHECKING:
    pass

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exact diagonalisation
# ---------------------------------------------------------------------------

def exact_ground_state_energy(hamiltonian: SparsePauliOp) -> float:
    """
    Return the exact ground-state energy via full diagonalisation.

    Only feasible for small systems (≤ 20 qubits).

    Parameters
    ----------
    hamiltonian : SparsePauliOp

    Returns
    -------
    float
        Minimum eigenvalue (real; imaginary part asserted < 1e-10).
    """
    if hamiltonian.num_qubits > 20:
        raise ValueError(
            f"exact_ground_state_energy: system too large "
            f"({hamiltonian.num_qubits} qubits, matrix dimension "
            f"{2**hamiltonian.num_qubits}).  Limit is 20 qubits."
        )
    matrix = hamiltonian.to_matrix()
    eigenvalues = np.linalg.eigvalsh(matrix)
    e0 = float(eigenvalues[0].real)
    _log.debug("Exact ground state energy: %.10f", e0)
    return e0


def sparse_ground_state_energy(hamiltonian: SparsePauliOp, max_qubits: int = 18) -> float:
    """
    Exact ground-state energy via sparse Lanczos (scipy eigsh).

    Feasible for moderately larger systems than dense diagonalisation
    (up to ~18 qubits) because it never forms the dense matrix.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
    max_qubits  : Hard ceiling to avoid runaway memory.

    Returns
    -------
    float : smallest eigenvalue.
    """
    n = hamiltonian.num_qubits
    if n > max_qubits:
        raise ValueError(
            f"sparse_ground_state_energy: {n} qubits exceeds limit {max_qubits}."
        )
    if n <= 10:
        return exact_ground_state_energy(hamiltonian)
    from scipy.sparse.linalg import eigsh
    sp = hamiltonian.to_matrix(sparse=True).tocsr()
    vals = eigsh(sp, k=1, which="SA", return_eigenvectors=False, maxiter=5000)
    e0 = float(np.real(vals[0]))
    _log.debug("Sparse ground state energy (%d qubits): %.10f", n, e0)
    return e0


def exact_ground_state(hamiltonian: SparsePauliOp) -> tuple[float, np.ndarray]:
    """
    Return the exact ground-state energy and eigenvector.

    Returns
    -------
    (energy, state_vector)
    """
    if hamiltonian.num_qubits > 20:
        raise ValueError("System too large for exact diagonalisation (> 20 qubits).")
    matrix = hamiltonian.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return float(eigenvalues[0].real), eigenvectors[:, 0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class MethodMetrics:
    method: str
    energy: float
    error: float                 # |energy - exact|
    relative_error_pct: float    # |error / exact| * 100
    converged: bool
    n_evals: int
    runtime_s: float


@dataclass
class BenchmarkResult:
    exact_energy: float
    methods: list[MethodMetrics] = field(default_factory=list)

    def add(self, result: OptimizationResult, exact_energy: float) -> None:
        error = abs(result.ground_state_energy - exact_energy)
        rel = (error / abs(exact_energy)) * 100 if abs(exact_energy) > 1e-12 else float("inf")
        self.methods.append(
            MethodMetrics(
                method=result.method,
                energy=result.ground_state_energy,
                error=error,
                relative_error_pct=rel,
                converged=result.converged,
                n_evals=result.n_function_evals,
                runtime_s=result.total_runtime_s,
            )
        )

    def print_table(self) -> None:
        """Print a formatted comparison table to stdout."""
        sep = "-" * 75
        lines = [
            sep,
            f"{'Method':<15} {'Energy':>12} {'Error':>12} {'Rel.Err%':>10} "
            f"{'Converged':>10} {'N_evals':>8} {'Time(s)':>9}",
            sep,
            f"{'Exact':<15} {self.exact_energy:>12.8f} {'N/A':>12} {'N/A':>10} "
            f"{'N/A':>10} {'N/A':>8} {'N/A':>9}",
        ]
        for m in self.methods:
            lines.append(
                f"{m.method:<15} {m.energy:>12.8f} {m.error:>12.2e} "
                f"{m.relative_error_pct:>10.4f} {str(m.converged):>10} "
                f"{m.n_evals:>8} {m.runtime_s:>9.2f}"
            )
        lines.append(sep)
        # Use sys.stdout with UTF-8 to avoid Windows cp1252 issues
        import sys
        out = "\n".join(lines) + "\n"
        try:
            print(out, end="")
        except UnicodeEncodeError:
            sys.stdout.buffer.write(out.encode("ascii", errors="replace") + b"\n")

    def to_dict(self) -> dict:
        return {
            "exact_energy": self.exact_energy,
            "methods": [
                {
                    "method": m.method,
                    "energy": m.energy,
                    "error": m.error,
                    "relative_error_pct": m.relative_error_pct,
                    "converged": m.converged,
                    "n_evals": m.n_evals,
                    "runtime_s": m.runtime_s,
                }
                for m in self.methods
            ],
        }

    def save_json(self, path: str | Path) -> None:
        import numpy as np

        class _NpEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, np.integer): return int(o)
                if isinstance(o, np.floating): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                return super().default(o)

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, cls=_NpEncoder)
        _log.info("Saved benchmark results to %s", p)


# ---------------------------------------------------------------------------
# High-level validation suite
# ---------------------------------------------------------------------------

class ValidationSuite:
    """
    Runs Exact, VQE, and Shadow-VQE on a given Hamiltonian and compares.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
    n_shadows   : shadows per Shadow-VQE step.
    max_iter    : optimizer iteration limit for both VQE methods.
    seed        : global seed.
    """

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        n_shadows: int = 2000,
        max_iter: int = 300,
        seed: int = 42,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.n_shadows = n_shadows
        self.max_iter = max_iter
        self.seed = seed

    def run(
        self,
        ansatz=None,
        output_dir: str | Path = "results",
        save: bool = True,
    ) -> BenchmarkResult:
        """
        Execute all three methods and return a BenchmarkResult.

        Parameters
        ----------
        ansatz     : Parameterized QuantumCircuit. Auto-built if None.
        output_dir : Where to save JSON results.
        save       : Whether to write results to disk.

        Returns
        -------
        BenchmarkResult
        """
        from .ansatz import hardware_efficient_ansatz
        from .vqe import VQE
        from .shadow_vqe import ShadowVQE

        n_qubits = self.hamiltonian.num_qubits
        if ansatz is None:
            ansatz = hardware_efficient_ansatz(n_qubits=n_qubits, reps=1)

        _log.info("=== Validation Suite: %d qubits ===", n_qubits)

        # Step 1: Exact diagonalisation
        exact_energy = exact_ground_state_energy(self.hamiltonian)
        _log.info("Exact ground state energy: %.8f Ha", exact_energy)

        benchmark = BenchmarkResult(exact_energy=exact_energy)

        # Step 2: Standard VQE
        _log.info("Running VQE...")
        vqe = VQE(
            ansatz=ansatz,
            hamiltonian=self.hamiltonian,
            optimizer="cobyla",
            max_iter=self.max_iter,
            seed=self.seed,
        )
        vqe_result = vqe.run()
        benchmark.add(vqe_result, exact_energy)

        # Step 3: Shadow-VQE
        _log.info("Running Shadow-VQE (%d shadows/step)...", self.n_shadows)
        svqe = ShadowVQE(
            ansatz=ansatz,
            hamiltonian=self.hamiltonian,
            n_shadows=self.n_shadows,
            optimizer="cobyla",
            max_iter=self.max_iter,
            seed=self.seed,
        )
        svqe_result = svqe.run()
        benchmark.add(svqe_result, exact_energy)

        benchmark.print_table()

        if save:
            out = Path(output_dir)
            benchmark.save_json(out / "benchmark_results.json")
            vqe_result.save_json(out / "vqe_result.json")
            svqe_result.save_json(out / "shadow_vqe_result.json")

        return benchmark, vqe_result, svqe_result
