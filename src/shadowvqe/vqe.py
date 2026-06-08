"""
Standard Variational Quantum Eigensolver (VQE).

Uses exact statevector expectation values as the energy oracle, so it
provides the ground-truth VQE baseline for comparison with Shadow-VQE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .estimators import StatevectorEstimator
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
class VQEResult(OptimizationResult):
    """VQE-specific result container (method always 'VQE')."""
    pass


class VQE:
    """
    Standard VQE with exact statevector energy evaluation.

    Parameters
    ----------
    ansatz       : Parameterized QuantumCircuit (parameters named θ[i]).
    hamiltonian  : SparsePauliOp Hamiltonian.
    optimizer    : 'cobyla' (default) or 'spsa'.
    max_iter     : Maximum optimizer iterations.
    tol          : Convergence tolerance on energy change.
    seed         : Global random seed (init parameters + optimizer noise).
    initial_point: Optional fixed initial parameters; random if None.

    Examples
    --------
    >>> from shadowvqe import h2_hamiltonian, hardware_efficient_ansatz, VQE
    >>> ham = h2_hamiltonian()
    >>> circ = hardware_efficient_ansatz(n_qubits=2, reps=1)
    >>> vqe = VQE(ansatz=circ, hamiltonian=ham, seed=42)
    >>> result = vqe.run()
    >>> result.ground_state_energy  # should be close to -1.8572
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        optimizer: OptimizerName = "cobyla",
        max_iter: int = 500,
        tol: float = 1e-6,
        seed: int = 0,
        initial_point: list[float] | np.ndarray | None = None,
    ) -> None:
        self._validate_inputs(ansatz, hamiltonian, optimizer)
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.optimizer_name = optimizer.lower()
        self.max_iter = validate_positive_int(max_iter, "max_iter")
        self.tol = float(tol)
        self.seed = validate_seed(seed)
        self.initial_point = (
            np.array(initial_point, dtype=float)
            if initial_point is not None
            else None
        )
        self._estimator = StatevectorEstimator()
        self._history: list[IterationRecord] = []
        self._iter_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> VQEResult:
        """
        Execute the VQE optimisation loop.

        Returns
        -------
        VQEResult
        """
        rng = seed_everything(self.seed)
        n_params = self.ansatz.num_parameters
        if n_params == 0:
            raise ValueError("ansatz has no free parameters — cannot optimise.")

        x0 = (
            self.initial_point
            if self.initial_point is not None
            else rng.uniform(0, 2 * np.pi, n_params)
        )
        if len(x0) != n_params:
            raise ValueError(
                f"initial_point has length {len(x0)} but ansatz has {n_params} parameters."
            )

        self._history.clear()
        self._iter_count = 0

        optimizer = self._build_optimizer()

        _log.info(
            "VQE starting: %d qubits, %d params, optimizer=%s",
            self.ansatz.num_qubits,
            n_params,
            self.optimizer_name,
        )

        with Timer() as t:
            opt_params, _, converged, n_evals = optimizer.minimize(
                self._objective, x0
            )

        final_energy, _ = self._estimator.estimate(
            self._bind(opt_params), self.hamiltonian
        )

        _log.info(
            "VQE finished in %.2f s — E = %.8f (converged=%s)",
            t.elapsed,
            final_energy,
            converged,
        )

        return VQEResult(
            method="VQE",
            ground_state_energy=final_energy,
            optimal_parameters=opt_params.tolist(),
            n_iterations=self._iter_count,
            n_function_evals=n_evals,
            converged=converged,
            history=list(self._history),
            total_runtime_s=t.elapsed,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _objective(self, params: np.ndarray) -> float:
        bound = self._bind(params)
        energy, _ = self._estimator.estimate(bound, self.hamiltonian)
        self._iter_count += 1
        self._history.append(
            IterationRecord(
                iteration=self._iter_count,
                energy=energy,
            )
        )
        return energy

    def _bind(self, params: np.ndarray) -> QuantumCircuit:
        """Bind parameter array to the ansatz circuit."""
        sorted_params = sorted(self.ansatz.parameters, key=lambda p: p.name)
        if len(params) != len(sorted_params):
            raise ValueError(
                f"Parameter count mismatch: circuit has {len(sorted_params)}, "
                f"got array of length {len(params)}."
            )
        binding = {p: float(v) for p, v in zip(sorted_params, params)}
        return self.ansatz.assign_parameters(binding)

    def _build_optimizer(self) -> COBYLA | SPSA:
        if self.optimizer_name == "cobyla":
            return COBYLA(max_iter=self.max_iter, tol=self.tol)
        elif self.optimizer_name == "spsa":
            return SPSA(max_iter=self.max_iter, tol=self.tol, seed=self.seed)
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer_name}'")

    @staticmethod
    def _validate_inputs(
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        optimizer: str,
    ) -> None:
        if not isinstance(ansatz, QuantumCircuit):
            raise TypeError(f"ansatz must be a QuantumCircuit, got {type(ansatz)}")
        if not isinstance(hamiltonian, SparsePauliOp):
            raise TypeError(f"hamiltonian must be SparsePauliOp, got {type(hamiltonian)}")
        if ansatz.num_qubits != hamiltonian.num_qubits:
            raise ValueError(
                f"ansatz has {ansatz.num_qubits} qubits but "
                f"hamiltonian has {hamiltonian.num_qubits} qubits."
            )
        valid_optimizers = {"cobyla", "spsa"}
        if optimizer.lower() not in valid_optimizers:
            raise ValueError(
                f"optimizer must be one of {valid_optimizers}, got '{optimizer}'."
            )
