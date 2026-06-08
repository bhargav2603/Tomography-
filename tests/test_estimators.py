"""Unit tests for expectation-value estimators."""

import numpy as np
import pytest
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from shadowvqe.estimators import StatevectorEstimator, ShadowEstimator
from shadowvqe.hamiltonians import h2_hamiltonian
from shadowvqe.validation import exact_ground_state_energy


def _ground_state_circuit() -> QuantumCircuit:
    """Approximate H2 ground state via a hard-coded rotation."""
    qc = QuantumCircuit(2)
    qc.ry(np.pi * 0.98, 0)
    qc.cx(0, 1)
    return qc


HAM = h2_hamiltonian()


class TestStatevectorEstimator:
    def test_returns_float(self):
        est = StatevectorEstimator()
        qc = QuantumCircuit(2)  # |00>
        energy, var = est.estimate(qc, HAM)
        assert isinstance(energy, float)
        assert var is None

    def test_known_state_energy_range(self):
        est = StatevectorEstimator()
        qc = QuantumCircuit(2)  # |00> is not ground state but should give a valid energy
        energy, _ = est.estimate(qc, HAM)
        assert -3 < energy < 0  # physically meaningful range

    def test_unbound_circuit_raises(self):
        from qiskit.circuit import ParameterVector
        qc = QuantumCircuit(2)
        theta = ParameterVector("θ", 1)
        qc.ry(theta[0], 0)
        est = StatevectorEstimator()
        with pytest.raises(ValueError, match="bound"):
            est.estimate(qc, HAM)

    def test_qubit_mismatch_raises(self):
        from qiskit.circuit import QuantumCircuit
        qc_3 = QuantumCircuit(3)
        est = StatevectorEstimator()
        with pytest.raises(ValueError, match="qubits"):
            est.estimate(qc_3, HAM)


class TestShadowEstimator:
    def test_returns_float_and_variance(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = ShadowEstimator(n_shadows=200, seed=0)
        qc = QuantumCircuit(2)
        energy, var = est.estimate(qc, HAM)
        assert isinstance(energy, float)
        assert isinstance(var, float)
        assert var >= 0

    def test_reset_call_count(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = ShadowEstimator(n_shadows=100, seed=0)
        qc = QuantumCircuit(2)
        est.estimate(qc, HAM)
        est.estimate(qc, HAM)
        assert est._call_count == 2
        est.reset()
        assert est._call_count == 0

    def test_energy_in_range(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = ShadowEstimator(n_shadows=500, seed=42)
        qc = QuantumCircuit(2)
        energy, _ = est.estimate(qc, HAM)
        assert -5 < energy < 5  # loose range for shadow estimate
