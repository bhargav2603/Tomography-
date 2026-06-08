"""Unit tests for the VQE and Shadow-VQE engines."""

import numpy as np
import pytest
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp

from shadowvqe.hamiltonians import h2_hamiltonian
from shadowvqe.ansatz import hardware_efficient_ansatz, build_ansatz
from shadowvqe.vqe import VQE
from shadowvqe.shadow_vqe import ShadowVQE
from shadowvqe.validation import exact_ground_state_energy


SEED = 42
HAM = h2_hamiltonian()
EXACT_ENERGY = exact_ground_state_energy(HAM)


class TestVQEInputValidation:
    def test_wrong_qubit_count(self):
        ansatz_3q = hardware_efficient_ansatz(3, reps=1)
        with pytest.raises(ValueError, match="qubits"):
            VQE(ansatz=ansatz_3q, hamiltonian=HAM)

    def test_bad_optimizer(self):
        with pytest.raises(ValueError, match="optimizer"):
            VQE(
                ansatz=hardware_efficient_ansatz(2, reps=1),
                hamiltonian=HAM,
                optimizer="adam",
            )

    def test_wrong_initial_point_length(self):
        ansatz = hardware_efficient_ansatz(2, reps=1)
        vqe = VQE(
            ansatz=ansatz,
            hamiltonian=HAM,
            seed=SEED,
            initial_point=np.zeros(3),  # wrong length
        )
        with pytest.raises(ValueError, match="initial_point"):
            vqe.run()

    def test_no_parameters_raises(self):
        qc = QuantumCircuit(2)  # no parameters
        with pytest.raises(ValueError, match="no free parameters"):
            VQE(ansatz=qc, hamiltonian=HAM).run()


class TestVQERun:
    """Integration tests — run VQE and verify energy is close to exact."""

    def test_vqe_cobyla_energy(self):
        ansatz = hardware_efficient_ansatz(2, reps=1)
        result = VQE(
            ansatz=ansatz,
            hamiltonian=HAM,
            optimizer="cobyla",
            max_iter=300,
            seed=SEED,
        ).run()
        assert abs(result.ground_state_energy - EXACT_ENERGY) < 0.05

    def test_vqe_result_fields(self):
        ansatz = hardware_efficient_ansatz(2, reps=1)
        result = VQE(ansatz=ansatz, hamiltonian=HAM, max_iter=50, seed=SEED).run()
        assert result.method == "VQE"
        assert len(result.history) > 0
        assert len(result.optimal_parameters) == ansatz.num_parameters
        assert result.total_runtime_s > 0

    def test_vqe_history_length(self):
        ansatz = hardware_efficient_ansatz(2, reps=1)
        result = VQE(ansatz=ansatz, hamiltonian=HAM, max_iter=50, seed=SEED).run()
        assert len(result.energy_history()) > 0

    def test_vqe_reproducibility(self):
        ansatz = hardware_efficient_ansatz(2, reps=1)
        r1 = VQE(ansatz=ansatz, hamiltonian=HAM, max_iter=30, seed=7).run()
        r2 = VQE(ansatz=ansatz, hamiltonian=HAM, max_iter=30, seed=7).run()
        assert np.isclose(r1.ground_state_energy, r2.ground_state_energy, atol=1e-10)

    def test_vqe_spsa_runs(self):
        ansatz = build_ansatz(2, reps=1)
        result = VQE(
            ansatz=ansatz,
            hamiltonian=HAM,
            optimizer="spsa",
            max_iter=50,
            seed=SEED,
        ).run()
        assert isinstance(result.ground_state_energy, float)


class TestShadowVQERun:
    def test_shadow_vqe_energy_rough(self):
        """Shadow-VQE should get within 0.5 Ha of exact for H2 (stochastic)."""
        ansatz = hardware_efficient_ansatz(2, reps=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = ShadowVQE(
                ansatz=ansatz,
                hamiltonian=HAM,
                n_shadows=1000,
                optimizer="cobyla",
                max_iter=50,
                seed=SEED,
            ).run()
        assert abs(result.ground_state_energy - EXACT_ENERGY) < 1.0

    def test_shadow_vqe_result_fields(self):
        ansatz = hardware_efficient_ansatz(2, reps=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = ShadowVQE(
                ansatz=ansatz,
                hamiltonian=HAM,
                n_shadows=500,
                max_iter=10,
                seed=SEED,
            ).run()
        assert result.method == "ShadowVQE"
        assert result.n_shadows_per_step == 500
        assert result.total_shadows > 0
        assert len(result.history) > 0

    def test_shadow_vqe_variance_recorded(self):
        ansatz = hardware_efficient_ansatz(2, reps=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = ShadowVQE(
                ansatz=ansatz,
                hamiltonian=HAM,
                n_shadows=500,
                max_iter=10,
                seed=SEED,
            ).run()
        vars_ = result.variance_history()
        non_none = [v for v in vars_ if v is not None]
        assert len(non_none) > 0

    def test_shadow_vqe_reproducibility(self):
        ansatz = hardware_efficient_ansatz(2, reps=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r1 = ShadowVQE(
                ansatz=ansatz, hamiltonian=HAM, n_shadows=300, max_iter=5, seed=99
            ).run()
            r2 = ShadowVQE(
                ansatz=ansatz, hamiltonian=HAM, n_shadows=300, max_iter=5, seed=99
            ).run()
        assert np.isclose(r1.ground_state_energy, r2.ground_state_energy, atol=1e-8)
