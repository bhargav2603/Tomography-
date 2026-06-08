"""
Expectation-value estimators.

Two back-ends
-------------
StatevectorEstimator  — exact ⟨ψ|H|ψ⟩ via Qiskit Statevector (no shots).
ShadowEstimator       — classical-shadow estimate using ClassicalShadows.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .shadows import ClassicalShadows
from .utils import get_logger, validate_positive_int

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol for duck-typing
# ---------------------------------------------------------------------------

class Estimator(Protocol):
    """Common interface for all expectation estimators."""

    def estimate(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
    ) -> tuple[float, float | None]:
        """
        Returns
        -------
        (mean, variance)
            variance may be None for exact estimators.
        """
        ...


# ---------------------------------------------------------------------------
# Exact statevector estimator
# ---------------------------------------------------------------------------

class StatevectorEstimator:
    """
    Exact ⟨H⟩ via Qiskit's ``Statevector.expectation_value``.

    No shots, no noise — ideal for validation and as VQE oracle.
    """

    def estimate(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
    ) -> tuple[float, None]:
        """
        Parameters
        ----------
        circuit    : bound (parameter-free) QuantumCircuit.
        observable : SparsePauliOp Hamiltonian.

        Returns
        -------
        (energy, None)
        """
        if circuit.num_parameters > 0:
            raise ValueError(
                "StatevectorEstimator requires a fully bound circuit (no free Parameters)."
            )
        if circuit.num_qubits != observable.num_qubits:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} qubits but "
                f"observable has {observable.num_qubits} qubits."
            )
        sv = Statevector(circuit)
        energy = float(sv.expectation_value(observable).real)
        _log.debug("StatevectorEstimator: energy = %.8f", energy)
        return energy, None


# ---------------------------------------------------------------------------
# Classical shadow estimator
# ---------------------------------------------------------------------------

class ShadowEstimator:
    """
    ⟨H⟩ estimated via classical shadows (Huang et al. 2020).

    A fresh set of shadows is collected at every call to :meth:`estimate`,
    matching the paper's protocol (no shadow reuse across optimizer steps).

    Parameters
    ----------
    n_shadows : int
        Number of snapshots per estimation call.
    seed : int
        Base seed — incremented per call to avoid identical shadow sets
        across optimizer iterations.
    """

    def __init__(self, n_shadows: int = 1000, seed: int = 0) -> None:
        self.n_shadows = validate_positive_int(n_shadows, "n_shadows")
        self.seed = seed
        self._call_count = 0

    def estimate(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
    ) -> tuple[float, float]:
        """
        Parameters
        ----------
        circuit    : bound QuantumCircuit.
        observable : SparsePauliOp.

        Returns
        -------
        (energy, variance)
        """
        if circuit.num_parameters > 0:
            raise ValueError(
                "ShadowEstimator requires a fully bound circuit (no free Parameters)."
            )
        if circuit.num_qubits != observable.num_qubits:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} qubits but "
                f"observable has {observable.num_qubits} qubits."
            )

        # Deterministic but different seed each call
        call_seed = (self.seed + self._call_count) % (2**31)
        self._call_count += 1

        shadows = ClassicalShadows(
            n_qubits=circuit.num_qubits,
            n_shadows=self.n_shadows,
            seed=call_seed,
        )
        shadows.collect(circuit)

        energy = shadows.estimate_observable(observable)
        variance = shadows.estimate_observable_variance(observable)

        _log.debug(
            "ShadowEstimator call %d: energy = %.6f, var = %.6e",
            self._call_count,
            energy,
            variance,
        )
        return energy, variance

    def reset(self) -> None:
        """Reset the internal call counter."""
        self._call_count = 0
