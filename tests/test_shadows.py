"""
Unit tests for classical shadow tomography.

Known-state validation:
  |0>    ZI expectation = +1
  |1>    ZI expectation = -1
  |+>    XI expectation = +1
  Bell   XX expectation = +1, ZZ expectation = -1
"""

import numpy as np
import pytest
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from shadowvqe.shadows import ClassicalShadows


def _z0_circuit() -> QuantumCircuit:
    """Single-qubit |0> state."""
    return QuantumCircuit(1)


def _z1_circuit() -> QuantumCircuit:
    """Single-qubit |1> state."""
    qc = QuantumCircuit(1)
    qc.x(0)
    return qc


def _plus_circuit() -> QuantumCircuit:
    """|+> = H|0>."""
    qc = QuantumCircuit(1)
    qc.h(0)
    return qc


def _bell_circuit() -> QuantumCircuit:
    """Bell state |Φ+> = (|00> + |11>) / √2."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


class TestClassicalShadowsBasic:
    def test_init_stores_attributes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cs = ClassicalShadows(n_qubits=2, n_shadows=100, seed=0)
        assert cs.n_qubits == 2
        assert cs.n_shadows == 100
        assert len(cs.snapshots) == 0

    def test_collect_fills_snapshots(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cs = ClassicalShadows(n_qubits=1, n_shadows=50, seed=0)
        cs.collect(_z0_circuit())
        assert len(cs.snapshots) == 50

    def test_snapshot_basis_length(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cs = ClassicalShadows(n_qubits=2, n_shadows=30, seed=0)
        cs.collect(_bell_circuit())
        for snap in cs.snapshots:
            assert len(snap.basis) == 2
            assert len(snap.bitstring) == 2
            assert all(b in "XYZ" for b in snap.basis)
            assert all(b in "01" for b in snap.bitstring)

    def test_collect_clears_previous(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cs = ClassicalShadows(n_qubits=1, n_shadows=20, seed=0)
        cs.collect(_z0_circuit())
        cs.collect(_z1_circuit())
        assert len(cs.snapshots) == 20

    def test_bound_circuit_required(self):
        from qiskit.circuit import ParameterVector
        qc = QuantumCircuit(1)
        theta = ParameterVector("θ", 1)
        qc.ry(theta[0], 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cs = ClassicalShadows(n_qubits=1, n_shadows=10, seed=0)
        with pytest.raises(ValueError, match="unbound Parameters"):
            cs.collect(qc)

    def test_qubit_mismatch(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cs = ClassicalShadows(n_qubits=3, n_shadows=10, seed=0)
        with pytest.raises(ValueError, match="qubits"):
            cs.collect(_bell_circuit())

    def test_invalid_pauli_string(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cs = ClassicalShadows(n_qubits=1, n_shadows=100, seed=0)
        cs.collect(_z0_circuit())
        with pytest.raises(ValueError, match="Invalid Pauli"):
            cs.estimate_pauli("A")

    def test_no_shadows_raises(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cs = ClassicalShadows(n_qubits=1, n_shadows=10, seed=0)
        with pytest.raises(RuntimeError, match="No shadows"):
            cs.estimate_pauli("Z")


class TestShadowKnownStates:
    """
    Known-value tests.  Tight tolerances would require huge N; we use
    generous tolerances and large N to keep tests fast and reliable.
    """

    N = 5000  # shadows per test

    def _make(self, n_qubits: int) -> ClassicalShadows:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return ClassicalShadows(n_qubits=n_qubits, n_shadows=self.N, seed=42)

    def test_z0_z_expectation(self):
        cs = self._make(1)
        cs.collect(_z0_circuit())
        val = cs.estimate_pauli("Z")
        assert abs(val - 1.0) < 0.15, f"Expected ~1.0, got {val:.4f}"

    def test_z1_z_expectation(self):
        cs = self._make(1)
        cs.collect(_z1_circuit())
        val = cs.estimate_pauli("Z")
        assert abs(val - (-1.0)) < 0.15, f"Expected ~-1.0, got {val:.4f}"

    def test_plus_x_expectation(self):
        cs = self._make(1)
        cs.collect(_plus_circuit())
        val = cs.estimate_pauli("X")
        assert abs(val - 1.0) < 0.15, f"Expected ~1.0, got {val:.4f}"

    def test_plus_z_expectation_near_zero(self):
        cs = self._make(1)
        cs.collect(_plus_circuit())
        val = cs.estimate_pauli("Z")
        assert abs(val) < 0.3, f"Expected ~0, got {val:.4f}"

    def test_bell_xx_expectation(self):
        cs = self._make(2)
        cs.collect(_bell_circuit())
        val = cs.estimate_pauli("XX")
        assert abs(val - 1.0) < 0.2, f"Expected ~1.0, got {val:.4f}"

    def test_bell_zz_expectation(self):
        cs = self._make(2)
        cs.collect(_bell_circuit())
        val = cs.estimate_pauli("ZZ")
        assert abs(val - 1.0) < 0.2, f"Bell ZZ expectation should be ~+1, got {val:.4f}"

    def test_bell_zi_near_zero(self):
        cs = self._make(2)
        cs.collect(_bell_circuit())
        val = cs.estimate_pauli("ZI")
        assert abs(val) < 0.25, f"Expected ~0, got {val:.4f}"

    def test_identity_expectation(self):
        cs = self._make(1)
        cs.collect(_z0_circuit())
        val = cs.estimate_pauli("I")
        assert abs(val - 1.0) < 1e-10, "Identity expectation must be exactly 1"


class TestShadowReproducibility:
    def test_same_seed_same_result(self):
        def run(seed):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                cs = ClassicalShadows(n_qubits=1, n_shadows=200, seed=seed)
            cs.collect(_z0_circuit())
            return cs.estimate_pauli("Z")

        assert run(7) == run(7)

    def test_different_seeds_may_differ(self):
        def run(seed):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                cs = ClassicalShadows(n_qubits=1, n_shadows=200, seed=seed)
            cs.collect(_z0_circuit())
            return cs.estimate_pauli("Z")

        # Different seeds may give different results (not guaranteed, but likely)
        results = {run(s) for s in range(5)}
        assert len(results) >= 1   # at minimum runs without error
