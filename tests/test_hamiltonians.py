"""Unit tests for hamiltonians module."""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from shadowvqe.hamiltonians import h2_hamiltonian, heisenberg_hamiltonian, random_hamiltonian
from shadowvqe.validation import exact_ground_state_energy


class TestH2Hamiltonian:
    def test_type(self):
        ham = h2_hamiltonian()
        assert isinstance(ham, SparsePauliOp)

    def test_qubits(self):
        assert h2_hamiltonian().num_qubits == 2

    def test_terms(self):
        assert len(h2_hamiltonian()) == 5

    def test_hermitian(self):
        mat = h2_hamiltonian().to_matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_ground_state_energy(self):
        # Known value for equilibrium H2 (JW, STO-3G): −1.8572 Ha
        e0 = exact_ground_state_energy(h2_hamiltonian())
        assert abs(e0 - (-1.8572750302023773)) < 1e-6


class TestHeisenbergHamiltonian:
    def test_type(self):
        ham = heisenberg_hamiltonian(n_qubits=4)
        assert isinstance(ham, SparsePauliOp)

    def test_qubits(self):
        for n in (2, 3, 4, 6):
            assert heisenberg_hamiltonian(n_qubits=n).num_qubits == n

    def test_hermitian(self):
        mat = heisenberg_hamiltonian(n_qubits=3).to_matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_too_few_qubits(self):
        with pytest.raises(ValueError, match="at least 2"):
            heisenberg_hamiltonian(n_qubits=1)


class TestRandomHamiltonian:
    def test_type(self):
        ham = random_hamiltonian(n_qubits=3, n_terms=5, seed=0)
        assert isinstance(ham, SparsePauliOp)

    def test_reproducible(self):
        h1 = random_hamiltonian(n_qubits=3, n_terms=5, seed=7)
        h2 = random_hamiltonian(n_qubits=3, n_terms=5, seed=7)
        assert np.allclose(
            sorted(h1.coeffs.real),
            sorted(h2.coeffs.real),
        )

    def test_different_seeds(self):
        h1 = random_hamiltonian(n_qubits=3, n_terms=8, seed=0)
        h2 = random_hamiltonian(n_qubits=3, n_terms=8, seed=1)
        # With different seeds, coefficients should differ (with overwhelming probability)
        assert not np.allclose(sorted(h1.coeffs.real), sorted(h2.coeffs.real))

    def test_invalid_max_weight(self):
        with pytest.raises(ValueError):
            random_hamiltonian(n_qubits=3, n_terms=5, max_weight=5)
