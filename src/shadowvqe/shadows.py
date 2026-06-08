"""
Classical Shadow Tomography (Huang, Kueng, Preskill — Nature Physics 2020).

Protocol (single-qubit Pauli-basis variant)
--------------------------------------------
1. For each shadow snapshot:
   a. Sample a random single-qubit basis b_i ∈ {X, Y, Z} for each qubit i.
   b. Apply the corresponding rotation gate to map that basis to Z.
   c. Measure in the computational basis → bitstring.
   d. Store (basis_string, bitstring).

2. Expectation of a Pauli string P is estimated as:
       <P> ≈ (1/N) Σ_k  Π_i shadow_channel_inverse(b_i^(k), m_i^(k), P_i)

   where for qubit i:
     - If P_i == I  : factor = 1
     - If b_i == P_i (basis matches Pauli): factor = 3 * (-1)^m_i
                       (the 3 comes from inverting the depolarising channel)
     - If b_i != P_i (basis mismatch)     : factor = 0  → whole term = 0
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .utils import get_logger, validate_positive_int, seed_everything

_log = get_logger(__name__)

# Minimum recommended shadows to suppress variance warnings
_SHOT_WARN_THRESHOLD = 500


# ---------------------------------------------------------------------------
# Rotation gates used to measure in X / Y / Z basis
# ---------------------------------------------------------------------------

def _basis_rotation_circuit(n_qubits: int, bases: list[str]) -> QuantumCircuit:
    """
    Return a shallow circuit that rotates each qubit into its measurement basis.

    Basis 'X': apply H before measurement  (H |±> = |0/1>)
    Basis 'Y': apply S†H before measurement
    Basis 'Z': identity
    """
    assert len(bases) == n_qubits
    qc = QuantumCircuit(n_qubits)
    for i, b in enumerate(bases):
        if b == "X":
            qc.h(i)
        elif b == "Y":
            qc.sdg(i)
            qc.h(i)
        elif b == "Z":
            pass  # no rotation needed
        else:
            raise ValueError(f"Unknown basis '{b}'. Must be X, Y, or Z.")
    return qc


# ---------------------------------------------------------------------------
# Core shadow data container
# ---------------------------------------------------------------------------

@dataclass
class ShadowSnapshot:
    """A single classical shadow snapshot."""
    basis: str          # e.g. "XYZ" — one char per qubit (qubit 0 = rightmost)
    bitstring: str      # e.g. "010"  — measurement result in computational basis


@dataclass
class ClassicalShadows:
    """
    Classical shadow tomography engine.

    Parameters
    ----------
    n_qubits : int
        System size.
    n_shadows : int
        Number of shadow snapshots to collect.
    seed : int
        Global random seed for reproducibility.

    Attributes
    ----------
    snapshots : list[ShadowSnapshot]
        Populated after calling :meth:`collect`.

    Examples
    --------
    >>> from qiskit.quantum_info import Statevector
    >>> from qiskit.circuit import QuantumCircuit
    >>> qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)   # Bell state
    >>> shadows = ClassicalShadows(n_qubits=2, n_shadows=1000, seed=42)
    >>> shadows.collect(qc)
    >>> exp = shadows.estimate_pauli("ZZ")
    >>> abs(exp)  < 0.15   # should be ~0 for Bell state XX
    True
    """

    n_qubits: int
    n_shadows: int
    seed: int = 0
    snapshots: list[ShadowSnapshot] = field(default_factory=list, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.n_qubits = validate_positive_int(self.n_qubits, "n_qubits")
        self.n_shadows = validate_positive_int(self.n_shadows, "n_shadows")
        if self.n_shadows < _SHOT_WARN_THRESHOLD:
            warnings.warn(
                f"n_shadows={self.n_shadows} is low; variance may be large. "
                f"Recommend >= {_SHOT_WARN_THRESHOLD}.",
                UserWarning,
                stacklevel=2,
            )
        self._rng = seed_everything(self.seed)

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(self, circuit: QuantumCircuit) -> "ClassicalShadows":
        """
        Collect classical shadows by simulating ``circuit`` under random
        single-qubit Pauli bases.

        Uses Qiskit's exact statevector simulator — no real hardware noise.
        Each snapshot is stored in :attr:`snapshots`.

        Parameters
        ----------
        circuit : QuantumCircuit
            Unparameterized state-preparation circuit (no free Parameters).

        Returns
        -------
        self
            Enables chaining: ``shadows.collect(circ).estimate_pauli("ZZ")``.
        """
        if circuit.num_parameters > 0:
            raise ValueError(
                "circuit has unbound Parameters. Bind all parameters before collecting shadows."
            )
        if circuit.num_qubits != self.n_qubits:
            raise ValueError(
                f"circuit has {circuit.num_qubits} qubits but ClassicalShadows "
                f"was initialised with n_qubits={self.n_qubits}."
            )

        self.snapshots.clear()
        state = Statevector(circuit)
        basis_choices = ["X", "Y", "Z"]
        basis_chars = np.array(basis_choices)
        n = self.n_qubits

        # Fast vectorized generation of all random measurement bases
        rand_ints = self._rng.integers(0, 3, size=(self.n_shadows, n))
        all_bases = ["".join(row) for row in basis_chars[rand_ints]]
        basis_counts = Counter(all_bases)

        for b_str, count in basis_counts.items():
            rot_circ = _basis_rotation_circuit(n, list(b_str))
            rotated_state = state.evolve(rot_circ)
            probs = rotated_state.probabilities()
            
            # Vectorized sampling for this specific measurement basis
            outcomes = self._rng.choice(len(probs), size=count, p=probs)
            rev_basis = "".join(reversed(b_str))
            
            for outcome_idx in outcomes:
                self.snapshots.append(
                    ShadowSnapshot(
                        basis=rev_basis,
                        bitstring=format(outcome_idx, f"0{n}b"),
                    )
                )

        # Shuffle to restore random snapshot order across batches
        self._rng.shuffle(self.snapshots)

        _log.debug("Collected %d shadow snapshots.", len(self.snapshots))
        return self

    def rng_int(self, high: int) -> int:
        """Draw a single integer in [0, high)."""
        return int(self._rng.integers(0, high))

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate_pauli(self, pauli_string: str) -> float:
        """
        Estimate the expectation value ⟨P⟩ of a Pauli string.

        Algorithm (Huang et al. 2020, single-qubit Clifford shadows):
            <P> ≈ mean over snapshots of  Π_i  factor_i(b_i, m_i, P_i)

        where factor_i is:
            1              if P_i == 'I'
            3 * (-1)^m_i   if b_i == P_i   (basis matched)
            0              if b_i != P_i   (basis mismatch → term dropped)

        Parameters
        ----------
        pauli_string : str
            Pauli string of length n_qubits (e.g. "XZI").
            Uses Qiskit little-endian convention: rightmost char = qubit 0.

        Returns
        -------
        float
            Estimated expectation value.
        """
        self._validate_pauli(pauli_string)
        if not self.snapshots:
            raise RuntimeError("No shadows collected yet. Call collect() first.")

        estimates = np.array(
            [self._snapshot_contribution(snap, pauli_string) for snap in self.snapshots],
            dtype=float,
        )
        return float(np.mean(estimates))

    def estimate_observable(self, observable: SparsePauliOp) -> float:
        """
        Estimate ⟨H⟩ for a SparsePauliOp by linearity:
            ⟨H⟩ = Σ_j c_j ⟨P_j⟩

        Parameters
        ----------
        observable : SparsePauliOp

        Returns
        -------
        float
            Estimated expectation value.
        """
        if observable.num_qubits != self.n_qubits:
            raise ValueError(
                f"Observable has {observable.num_qubits} qubits, "
                f"expected {self.n_qubits}."
            )
        if not self.snapshots:
            raise RuntimeError("No shadows collected yet. Call collect() first.")

        # Build per-snapshot contributions for every Pauli term
        total = 0.0
        for pauli, coeff in zip(observable.paulis, observable.coeffs):
            pauli_str = pauli.to_label()
            coeff_r = float(coeff.real)
            if abs(coeff_r) < 1e-12:
                continue
            pauli_est = self.estimate_pauli(pauli_str)
            total += coeff_r * pauli_est
        return total

    def estimate_variance(self, pauli_string: str) -> float:
        """
        Return the sample variance of the per-snapshot Pauli estimates.

        Var[<P>] ≈ Var[f_k] / N
        where f_k is the snapshot contribution for snapshot k.
        """
        self._validate_pauli(pauli_string)
        if not self.snapshots:
            raise RuntimeError("No shadows collected yet.")
        estimates = np.array(
            [self._snapshot_contribution(snap, pauli_string) for snap in self.snapshots],
            dtype=float,
        )
        return float(np.var(estimates, ddof=1) / len(estimates))

    def estimate_observable_variance(self, observable: SparsePauliOp) -> float:
        """
        Variance of the observable estimate via error propagation.
        Assumes covariances between Pauli terms are zero (independent bases).
        """
        if observable.num_qubits != self.n_qubits:
            raise ValueError("Observable qubit count mismatch.")
        if not self.snapshots:
            raise RuntimeError("No shadows collected yet.")

        paulis = [
            (p.to_label(), float(c.real))
            for p, c in zip(observable.paulis, observable.coeffs)
            if abs(float(c.real)) >= 1e-12
        ]
        
        estimates = np.array([
            sum(c * self._snapshot_contribution(snap, p_str) for p_str, c in paulis)
            for snap in self.snapshots
        ], dtype=float)
        
        return float(np.var(estimates, ddof=1) / len(estimates))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_contribution(self, snap: ShadowSnapshot, pauli_string: str) -> float:
        """
        Single-snapshot estimate contribution for a given Pauli string.

        Returns 0.0 immediately if any non-identity Pauli qubit has a
        basis mismatch (a key efficiency property of the shadow protocol).
        """
        product = 1.0
        for i, (p_char, b_char, m_char) in enumerate(
            zip(pauli_string, snap.basis, snap.bitstring)
        ):
            if p_char == "I":
                continue  # factor = 1
            if b_char != p_char:
                return 0.0  # mismatch → whole term is 0
            # Matching basis: factor = 3 * (-1)^bit
            bit = int(m_char)
            product *= 3.0 * ((-1) ** bit)
        return product

    def _validate_pauli(self, pauli_string: str) -> None:
        if len(pauli_string) != self.n_qubits:
            raise ValueError(
                f"Pauli string length {len(pauli_string)} != n_qubits {self.n_qubits}."
            )
        invalid = set(pauli_string) - {"I", "X", "Y", "Z"}
        if invalid:
            raise ValueError(f"Invalid Pauli characters: {invalid}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def basis_frequencies(self) -> dict[str, float]:
        """Return observed measurement frequencies for each single-qubit basis."""
        counts: dict[str, int] = {"X": 0, "Y": 0, "Z": 0}
        total = len(self.snapshots) * self.n_qubits
        for snap in self.snapshots:
            for b in snap.basis:
                counts[b] += 1
        return {b: c / total for b, c in counts.items()}

    def __len__(self) -> int:
        return len(self.snapshots)

    def __repr__(self) -> str:
        return (
            f"ClassicalShadows(n_qubits={self.n_qubits}, "
            f"n_shadows={self.n_shadows}, "
            f"collected={len(self.snapshots)})"
        )
