"""
Hamiltonian-Adaptive Classical Shadows — Variance Reduction.

Standard classical shadows draw Pauli bases uniformly at random (X/Y/Z with
probability 1/3 each). For a specific target Hamiltonian H, this is
suboptimal: Pauli terms whose bases are rare in H are estimated noisily.

This module implements a Hamiltonian-weighted basis selection strategy:

    q_i(b) ∝ max( Σ_{j: P_j[i]=b} |c_j| , ε )

where the sum runs over all Pauli terms P_j of H with basis b on qubit i,
weighted by coefficient magnitude |c_j|. ε is a small floor probability
to keep all bases reachable.

Correction for unbiasedness
---------------------------
With biased sampling q_i(b) ≠ 1/3, the standard factor of 3 × (−1)^m
is replaced by (1/q_i(P_i)) × (−1)^m so the estimator remains unbiased:

    E[contribution] = E[ ∏_i (1/q_i(P_i)) (−1)^{m_i} I(b_i = P_i) ]
                    = ∏_i <P_i> = <P>   (unbiased)

Variance reduction
------------------
When q_i(P_i) > 1/3 (P_i is a common basis in H), the correction factor
1/q_i is smaller than 3, reducing variance for that term. The overall
variance reduction depends on Hamiltonian structure; typical values for
molecular Hamiltonians are 2–10×.

Reference: You & Kim (2021) "Efficient learning of quantum states with
single-copy measurements", arXiv:2101.10020
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .shadows import _basis_rotation_circuit, ShadowSnapshot
from .utils import get_logger, validate_positive_int, seed_everything

_log = get_logger(__name__)

_SHOT_WARN_THRESHOLD = 500
_FLOOR_PROB = 0.05          # minimum probability for any basis on any qubit


# ---------------------------------------------------------------------------
# Basis probability computation
# ---------------------------------------------------------------------------

def compute_hamiltonian_basis_probs(
    hamiltonian: SparsePauliOp,
    floor: float = _FLOOR_PROB,
) -> dict[int, dict[str, float]]:
    """
    Compute per-qubit Pauli basis sampling probabilities from a Hamiltonian.

    For qubit i, the probability of choosing basis b is:

        q_i(b) ∝ max( Σ_{j: P_j[i]=b} |c_j| , floor )

    Parameters
    ----------
    hamiltonian : SparsePauliOp
    floor       : Minimum probability for any basis (prevents zero-prob bases).

    Returns
    -------
    dict[qubit_index → dict[basis → probability]]
    """
    n = hamiltonian.num_qubits

    # Accumulate |coeff| weight for each qubit × basis
    weights: dict[int, dict[str, float]] = {
        q: {"X": 0.0, "Y": 0.0, "Z": 0.0} for q in range(n)
    }

    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = pauli.to_label()           # e.g. "XZIY"
        w = abs(float(coeff.real))
        for q in range(n):
            # Qiskit convention: label[n-1-q] = Pauli acting on qubit q
            char = label[n - 1 - q]
            if char in ("X", "Y", "Z"):
                weights[q][char] += w

    # Normalize with floor
    probs: dict[int, dict[str, float]] = {}
    for q in range(n):
        raw = {b: max(weights[q][b], floor) for b in ("X", "Y", "Z")}
        total = sum(raw.values())
        probs[q] = {b: raw[b] / total for b in ("X", "Y", "Z")}

    return probs


# ---------------------------------------------------------------------------
# Adaptive shadow data container
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveShadowSnapshot:
    """Single classical shadow snapshot with per-qubit sampling probabilities."""
    basis: str                          # Qiskit convention (qubit n-1 first)
    bitstring: str
    q_probs: list[float]                # q_i(b_i) for each qubit position


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class HamiltonianAdaptedShadows:
    """
    Classical shadow tomography with Hamiltonian-weighted basis selection.

    Reduces variance of energy estimates compared to uniform random shadows
    by biasing measurements toward Pauli bases that appear frequently in H.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
        Target Hamiltonian — used to compute basis probabilities.
    n_shadows   : int
        Number of shadow snapshots per collection.
    seed        : int
        Random seed for reproducibility.
    floor_prob  : float
        Minimum sampling probability for any basis (default 0.05).

    Examples
    --------
    >>> from shadowvqe import h2_hamiltonian, hardware_efficient_ansatz, VQE
    >>> ham = h2_hamiltonian()
    >>> circ = hardware_efficient_ansatz(2, reps=1)
    >>> result = VQE(circ, ham, seed=42).run()
    >>> bound = circ.assign_parameters(dict(zip(
    ...     sorted(circ.parameters, key=lambda p: p.name),
    ...     result.optimal_parameters)))
    >>> ads = HamiltonianAdaptedShadows(ham, n_shadows=1000, seed=42)
    >>> ads.collect(bound)
    >>> ads.estimate_observable(ham)  # doctest: +SKIP
    """

    hamiltonian: SparsePauliOp
    n_shadows: int
    seed: int = 0
    floor_prob: float = _FLOOR_PROB
    snapshots: list[AdaptiveShadowSnapshot] = field(
        default_factory=list, init=False, repr=False
    )
    _rng: np.random.Generator = field(init=False, repr=False)
    _basis_probs: dict[int, dict[str, float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.n_shadows = validate_positive_int(self.n_shadows, "n_shadows")
        if self.n_shadows < _SHOT_WARN_THRESHOLD:
            warnings.warn(
                f"n_shadows={self.n_shadows} is low. Recommend >= {_SHOT_WARN_THRESHOLD}.",
                UserWarning,
                stacklevel=2,
            )
        self._rng = seed_everything(self.seed)
        self._basis_probs = compute_hamiltonian_basis_probs(
            self.hamiltonian, floor=self.floor_prob
        )
        _log.debug(
            "Adaptive basis probs computed for %d qubits.", self.hamiltonian.num_qubits
        )

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(self, circuit: QuantumCircuit) -> "HamiltonianAdaptedShadows":
        """
        Collect adaptive classical shadows from a bound QuantumCircuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            Fully parameterized (no free Parameters).

        Returns
        -------
        self  (for chaining)
        """
        if circuit.num_parameters > 0:
            raise ValueError("Circuit must be fully bound before collecting shadows.")
        n = self.hamiltonian.num_qubits
        if circuit.num_qubits != n:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} qubits; "
                f"Hamiltonian has {n} qubits."
            )

        self.snapshots.clear()
        state = Statevector(circuit)
        basis_keys = ["X", "Y", "Z"]

        for _ in range(self.n_shadows):
            # Hamiltonian-weighted basis selection per qubit
            bases: list[str] = []
            q_probs_used: list[float] = []
            for q in range(n):
                p = [self._basis_probs[q][b] for b in basis_keys]
                chosen_idx = int(self._rng.choice(3, p=p))
                b = basis_keys[chosen_idx]
                bases.append(b)
                q_probs_used.append(p[chosen_idx])

            # Rotate and sample
            rot_circ = _basis_rotation_circuit(n, bases)
            probs = state.evolve(rot_circ).probabilities()
            idx = int(self._rng.choice(len(probs), p=probs))
            bitstring = format(idx, f"0{n}b")

            # Store in reversed (Qiskit big-endian) convention
            # q_probs_used[q] = prob used for qubit q (little-endian order)
            self.snapshots.append(
                AdaptiveShadowSnapshot(
                    basis="".join(reversed(bases)),   # qubit n-1 first
                    bitstring=bitstring,
                    q_probs=list(reversed(q_probs_used)),  # qubit n-1 first
                )
            )

        _log.debug("Collected %d adaptive shadow snapshots.", len(self.snapshots))
        return self

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate_pauli(self, pauli_string: str) -> float:
        """
        Estimate ⟨P⟩ with Hamiltonian-adaptive correction factors.

        The correction (1/q_i) replaces the standard factor of 3,
        maintaining unbiasedness while reducing variance.

        Parameters
        ----------
        pauli_string : str
            Length n_qubits, Qiskit convention (qubit n-1 first).

        Returns
        -------
        float
        """
        self._validate_pauli(pauli_string)
        if not self.snapshots:
            raise RuntimeError("No snapshots collected. Call collect() first.")
        estimates = [
            self._snapshot_contribution(snap, pauli_string)
            for snap in self.snapshots
        ]
        return float(np.mean(estimates))

    def estimate_observable(self, observable: SparsePauliOp) -> float:
        """Estimate ⟨H⟩ by linearity over all Pauli terms."""
        if not self.snapshots:
            raise RuntimeError("No snapshots collected. Call collect() first.")
        total = 0.0
        for pauli, coeff in zip(observable.paulis, observable.coeffs):
            c = float(coeff.real)
            if abs(c) < 1e-12:
                continue
            total += c * self.estimate_pauli(pauli.to_label())
        return total

    def estimate_variance(self, pauli_string: str) -> float:
        """Sample variance of per-snapshot Pauli estimates."""
        self._validate_pauli(pauli_string)
        estimates = np.array(
            [self._snapshot_contribution(snap, pauli_string) for snap in self.snapshots],
            dtype=float,
        )
        return float(np.var(estimates, ddof=1) / len(estimates))

    def estimate_observable_variance(self, observable: SparsePauliOp) -> float:
        """Variance of the observable estimate (independent-term approximation)."""
        total_var = 0.0
        for pauli, coeff in zip(observable.paulis, observable.coeffs):
            c = float(coeff.real)
            if abs(c) < 1e-12:
                continue
            total_var += c**2 * self.estimate_variance(pauli.to_label())
        return total_var

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_contribution(
        self, snap: AdaptiveShadowSnapshot, pauli_string: str
    ) -> float:
        """
        Adaptive single-snapshot estimate for Pauli P.

        Correction factor: (1/q_i(P_i)) × (−1)^m_i  for matching qubits.
        """
        product = 1.0
        for i, (p_char, b_char, m_char, q_i) in enumerate(
            zip(pauli_string, snap.basis, snap.bitstring, snap.q_probs)
        ):
            if p_char == "I":
                continue
            if b_char != p_char:
                return 0.0
            # Unbiased correction: replace standard "3" with "1/q_i"
            correction = 1.0 / q_i
            bit = int(m_char)
            product *= correction * ((-1) ** bit)
        return product

    def _validate_pauli(self, pauli_string: str) -> None:
        n = self.hamiltonian.num_qubits
        if len(pauli_string) != n:
            raise ValueError(
                f"Pauli string length {len(pauli_string)} != n_qubits {n}."
            )
        invalid = set(pauli_string) - {"I", "X", "Y", "Z"}
        if invalid:
            raise ValueError(f"Invalid Pauli characters: {invalid}")

    def basis_distribution_summary(self) -> dict[int, dict[str, float]]:
        """Return the sampling probabilities used for each qubit."""
        return dict(self._basis_probs)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __repr__(self) -> str:
        return (
            f"HamiltonianAdaptedShadows("
            f"n_qubits={self.hamiltonian.num_qubits}, "
            f"n_shadows={self.n_shadows}, "
            f"collected={len(self.snapshots)})"
        )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_estimators(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    n_shadows_list: Sequence[int],
    n_trials: int = 20,
    exact_energy: float | None = None,
    seed: int = 42,
) -> dict:
    """
    Compare random vs Hamiltonian-adapted shadow estimators.

    For each N in n_shadows_list and each trial, collects shadows and records
    the energy estimate. Returns statistics for both methods.

    Parameters
    ----------
    circuit        : Fully bound QuantumCircuit (the state to estimate).
    hamiltonian    : Target SparsePauliOp.
    n_shadows_list : Shot counts to benchmark.
    n_trials       : Independent trials per (method, N) for variance estimation.
    exact_energy   : If provided, computes absolute error.
    seed           : Base random seed.

    Returns
    -------
    dict with keys 'random' and 'adapted', each containing:
        - mean_errors   : list[float]  (mean |estimate − exact| over trials)
        - std_errors    : list[float]  (std of |estimate − exact|)
        - mean_variance : list[float]  (mean sample variance over trials)
        - variance_reduction : list[float]  (random_var / adapted_var per N)
    """
    from .shadows import ClassicalShadows

    rng_master = np.random.default_rng(seed)
    random_mean_errors, random_std_errors, random_variances = [], [], []
    adapted_mean_errors, adapted_std_errors, adapted_variances = [], [], []

    for n_shadows in n_shadows_list:
        r_errors, r_vars = [], []
        a_errors, a_vars = [], []

        for trial in range(n_trials):
            t_seed = int(rng_master.integers(0, 2**28))

            # Random shadows
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cs = ClassicalShadows(
                    n_qubits=hamiltonian.num_qubits,
                    n_shadows=n_shadows,
                    seed=t_seed,
                )
            cs.collect(circuit)
            r_est = cs.estimate_observable(hamiltonian)
            r_var = cs.estimate_observable_variance(hamiltonian)
            r_errors.append(abs(r_est - exact_energy) if exact_energy is not None else 0.0)
            r_vars.append(r_var)

            # Adaptive shadows
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ads = HamiltonianAdaptedShadows(
                    hamiltonian=hamiltonian,
                    n_shadows=n_shadows,
                    seed=t_seed + 100000,
                )
            ads.collect(circuit)
            a_est = ads.estimate_observable(hamiltonian)
            a_var = ads.estimate_observable_variance(hamiltonian)
            a_errors.append(abs(a_est - exact_energy) if exact_energy is not None else 0.0)
            a_vars.append(a_var)

        random_mean_errors.append(float(np.mean(r_errors)))
        random_std_errors.append(float(np.std(r_errors)))
        random_variances.append(float(np.mean(r_vars)))
        adapted_mean_errors.append(float(np.mean(a_errors)))
        adapted_std_errors.append(float(np.std(a_errors)))
        adapted_variances.append(float(np.mean(a_vars)))

    # Variance reduction factor: > 1 means adaptive is better
    var_reduction = [
        rv / av if av > 0 else float("nan")
        for rv, av in zip(random_variances, adapted_variances)
    ]

    return {
        "n_shadows_list": list(n_shadows_list),
        "random": {
            "mean_errors": random_mean_errors,
            "std_errors": random_std_errors,
            "mean_variance": random_variances,
        },
        "adapted": {
            "mean_errors": adapted_mean_errors,
            "std_errors": adapted_std_errors,
            "mean_variance": adapted_variances,
        },
        "variance_reduction_factor": var_reduction,
    }
