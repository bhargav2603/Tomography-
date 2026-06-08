"""
Shadow-assisted VQE (Shadow-VQE).

Replaces the statevector energy oracle with classical-shadow estimates,
reducing the number of distinct measurement circuits needed from O(N_Pauli)
to O(N_shadows) per optimizer step — a key cost advantage for large
Hamiltonians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .estimators import ShadowEstimator
from .optimizers import COBYLA, SPSA
from .utils import (
    IterationRecord,
    OptimizationResult,
    Timer,
    get_logger,
    seed_everything,
    validate_positive_int,
    validate_seed,
)

_log = get_logger(__name__)

OptimizerName = Literal["cobyla", "spsa"]


@dataclass
class ShadowVQEResult(OptimizationResult):
    """Shadow-VQE specific result container."""
    n_shadows_per_step: int = 0
    total_shadows: int = 0


class ShadowVQE:
    """
    Variational Quantum Eigensolver with Classical Shadow expectation estimation.

    At each optimizer step, a fresh set of ``n_shadows`` snapshots is collected
    from the current ansatz state and used to estimate ⟨H⟩.  This trades
    shot-count precision for measurement diversity.

    Parameters
    ----------
    ansatz         : Parameterized QuantumCircuit.
    hamiltonian    : SparsePauliOp Hamiltonian.
    n_shadows      : Shadows per optimizer call (default 1000).
    optimizer      : 'cobyla' or 'spsa' (default 'cobyla').
    max_iter       : Maximum optimizer iterations.
    tol            : Convergence tolerance on energy change.
    seed           : Global random seed.
    initial_point  : Optional fixed initial parameters.

    Examples
    --------
    >>> from shadowvqe import h2_hamiltonian, hardware_efficient_ansatz, ShadowVQE
    >>> ham = h2_hamiltonian()
    >>> circ = hardware_efficient_ansatz(n_qubits=2, reps=1)
    >>> svqe = ShadowVQE(ansatz=circ, hamiltonian=ham, n_shadows=2000, seed=42)
    >>> result = svqe.run()
    >>> abs(result.ground_state_energy + 1.857) < 0.1  # rough convergence
    True
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        n_shadows: int = 1000,
        optimizer: OptimizerName = "cobyla",
        max_iter: int = 300,
        tol: float = 1e-5,
        seed: int = 0,
        initial_point: list[float] | np.ndarray | None = None,
    ) -> None:
        self._validate_inputs(ansatz, hamiltonian, optimizer)
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.n_shadows = validate_positive_int(n_shadows, "n_shadows")
        self.optimizer_name = optimizer.lower()
        self.max_iter = validate_positive_int(max_iter, "max_iter")
        self.tol = float(tol)
        self.seed = validate_seed(seed)
        self.initial_point = (
            np.array(initial_point, dtype=float)
            if initial_point is not None
            else None
        )
        self._estimator = ShadowEstimator(n_shadows=n_shadows, seed=seed)
        self._history: list[IterationRecord] = []
        self._iter_count = 0
        self._total_shadows = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ShadowVQEResult:
        """
        Run the Shadow-VQE optimisation.

        Returns
        -------
        ShadowVQEResult
        """
        rng = seed_everything(self.seed)
        n_params = self.ansatz.num_parameters
        if n_params == 0:
            raise ValueError("ansatz has no free parameters.")

        x0 = (
            self.initial_point
            if self.initial_point is not None
            else rng.uniform(0, 2 * np.pi, n_params)
        )
        if len(x0) != n_params:
            raise ValueError(
                f"initial_point length {len(x0)} != ansatz parameters {n_params}."
            )

        self._history.clear()
        self._iter_count = 0
        self._total_shadows = 0
        self._estimator.reset()

        optimizer = self._build_optimizer()

        _log.info(
            "ShadowVQE starting: %d qubits, %d params, %d shadows/step, optimizer=%s",
            self.ansatz.num_qubits,
            n_params,
            self.n_shadows,
            self.optimizer_name,
        )

        with Timer() as t:
            opt_params, _, converged, n_evals = optimizer.minimize(
                self._objective, x0
            )

        # Final evaluation with a large shadow set for accuracy
        final_estimator = ShadowEstimator(
            n_shadows=max(self.n_shadows, 5000),
            seed=self.seed + 99999,
        )
        final_energy, final_variance = final_estimator.estimate(
            self._bind(opt_params), self.hamiltonian
        )

        _log.info(
            "ShadowVQE finished in %.2f s — E = %.6f ± %.2e (converged=%s)",
            t.elapsed,
            final_energy,
            (final_variance ** 0.5) if final_variance is not None else float("nan"),
            converged,
        )

        return ShadowVQEResult(
            method="ShadowVQE",
            ground_state_energy=final_energy,
            optimal_parameters=opt_params.tolist(),
            n_iterations=self._iter_count,
            n_function_evals=n_evals,
            converged=converged,
            history=list(self._history),
            total_runtime_s=t.elapsed,
            n_shadows_per_step=self.n_shadows,
            total_shadows=self._total_shadows,
            extra={"final_variance": final_variance},
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _objective(self, params: np.ndarray) -> float:
        import time
        t0 = time.perf_counter()
        bound = self._bind(params)
        energy, variance = self._estimator.estimate(bound, self.hamiltonian)
        elapsed = time.perf_counter() - t0

        self._iter_count += 1
        self._total_shadows += self.n_shadows
        self._history.append(
            IterationRecord(
                iteration=self._iter_count,
                energy=energy,
                variance=variance,
                n_shots=self.n_shadows,
                runtime_s=elapsed,
            )
        )
        _log.debug(
            "ShadowVQE iter %d: E = %.6f, var = %.2e",
            self._iter_count,
            energy,
            variance if variance is not None else 0.0,
        )
        return energy

    def _bind(self, params: np.ndarray) -> QuantumCircuit:
        sorted_params = sorted(self.ansatz.parameters, key=lambda p: p.name)
        binding = {p: float(v) for p, v in zip(sorted_params, params)}
        return self.ansatz.assign_parameters(binding)

    def _build_optimizer(self) -> COBYLA | SPSA:
        if self.optimizer_name == "cobyla":
            return COBYLA(max_iter=self.max_iter, tol=self.tol)
        return SPSA(max_iter=self.max_iter, tol=self.tol, seed=self.seed)

    @staticmethod
    def _validate_inputs(
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        optimizer: str,
    ) -> None:
        if not isinstance(ansatz, QuantumCircuit):
            raise TypeError(f"ansatz must be QuantumCircuit, got {type(ansatz)}")
        if not isinstance(hamiltonian, SparsePauliOp):
            raise TypeError(f"hamiltonian must be SparsePauliOp, got {type(hamiltonian)}")
        if ansatz.num_qubits != hamiltonian.num_qubits:
            raise ValueError(
                f"ansatz has {ansatz.num_qubits} qubits but "
                f"hamiltonian has {hamiltonian.num_qubits} qubits."
            )
        valid = {"cobyla", "spsa"}
        if optimizer.lower() not in valid:
            raise ValueError(f"optimizer must be in {valid}, got '{optimizer}'.")
