"""
Reduced Density Matrix (RDM) reconstruction from Classical Shadows.

This module is the concrete demonstration of *the* shadow-tomography
advantage that matters for fragment methods: from a SINGLE shadow dataset,
reconstruct the entire one-particle reduced density matrix (1-RDM), every
element of which would otherwise require its own measurement setting.

The fermionic 1-RDM is

    D_pq = ⟨a†_p a_q⟩

Under the Jordan–Wigner (JW) mapping each element becomes a small sum of
Pauli operators:

    Diagonal   : D_pp = (1 − ⟨Z_p⟩) / 2
    Off-diag   : Re D_pq = ½ ( ⟨X_p Z…Z X_q⟩ + ⟨Y_p Z…Z Y_q⟩ )   (p < q)

For a real-valued ground state (the usual VQE case) the imaginary part
vanishes, so the real symmetric 1-RDM is the physical object. Its trace
equals the particle number, which is a free correctness check.

Why this is the advantage
-------------------------
A 1-RDM on n spin-orbitals has O(n²) independent elements. Measuring them
by grouped-Pauli tomography needs many distinct measurement bases. Classical
shadows estimate all O(n²) elements from one randomized-measurement dataset
— exactly the "many observables from few measurements" guarantee of
Huang, Kueng & Preskill (2020).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .shadows import ClassicalShadows
from .utils import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pauli operators for individual 1-RDM elements (Jordan-Wigner)
# ---------------------------------------------------------------------------

def _diagonal_op(p: int, n: int) -> SparsePauliOp:
    """D_pp = (I - Z_p) / 2  =  number operator ⟨n_p⟩."""
    return SparsePauliOp.from_sparse_list(
        [("", [], 0.5), ("Z", [p], -0.5)], num_qubits=n
    )


def _offdiagonal_real_op(p: int, q: int, n: int) -> SparsePauliOp:
    """
    Real symmetric part of D_pq for p < q under Jordan-Wigner:

        ½ ( X_p Z…Z X_q + Y_p Z…Z Y_q )

    with the JW Z-string spanning qubits p+1 … q-1.
    """
    if p >= q:
        raise ValueError("require p < q")
    mid = list(range(p + 1, q))
    qubits = [p] + mid + [q]
    xx_label = "X" + "Z" * len(mid) + "X"
    yy_label = "Y" + "Z" * len(mid) + "Y"
    return SparsePauliOp.from_sparse_list(
        [(xx_label, qubits, 0.5), (yy_label, qubits, 0.5)], num_qubits=n
    )


def one_rdm_pauli_terms(n: int) -> SparsePauliOp:
    """
    All Pauli strings appearing in the full real 1-RDM, as one operator.

    Used only to count measurement settings (grouped-Pauli cost). The numeric
    coefficients are irrelevant for the count, so unit coefficients are used.
    """
    ops: list[SparsePauliOp] = []
    for p in range(n):
        ops.append(_diagonal_op(p, n))
    for p in range(n):
        for q in range(p + 1, n):
            ops.append(_offdiagonal_real_op(p, q, n))
    total = ops[0]
    for op in ops[1:]:
        total = total + op
    return total.simplify()


# ---------------------------------------------------------------------------
# Reconstruction from a shadow dataset
# ---------------------------------------------------------------------------

@dataclass
class RDMResult:
    """Reconstructed 1-RDM plus diagnostics."""
    matrix: np.ndarray            # real symmetric (n x n)
    particle_number: float        # trace
    n_orbitals: int

    def occupations(self) -> np.ndarray:
        """Natural-orbital occupation numbers (eigenvalues, descending)."""
        vals = np.linalg.eigvalsh(self.matrix)
        return np.sort(vals)[::-1]


def one_rdm_from_shadows(shadows: ClassicalShadows) -> RDMResult:
    """
    Reconstruct the real symmetric fermionic 1-RDM from a shadow dataset.

    Parameters
    ----------
    shadows : a ClassicalShadows instance that has already called collect().

    Returns
    -------
    RDMResult
    """
    if not shadows.snapshots:
        raise RuntimeError("Shadow dataset is empty. Call collect() first.")
    n = shadows.n_qubits
    D = np.zeros((n, n), dtype=float)

    # Diagonal
    for p in range(n):
        D[p, p] = shadows.estimate_observable(_diagonal_op(p, n))

    # Off-diagonal (symmetric)
    for p in range(n):
        for q in range(p + 1, n):
            val = shadows.estimate_observable(_offdiagonal_real_op(p, q, n))
            D[p, q] = val
            D[q, p] = val

    trace = float(np.trace(D))
    _log.debug("1-RDM reconstructed: n=%d, trace(N)=%.4f", n, trace)
    return RDMResult(matrix=D, particle_number=trace, n_orbitals=n)


def exact_one_rdm(state, n: int) -> np.ndarray:
    """
    Exact real symmetric 1-RDM of a Statevector, for validation.

    Parameters
    ----------
    state : qiskit Statevector (or any object with expectation_value()).
    n     : number of qubits / spin-orbitals.
    """
    D = np.zeros((n, n), dtype=float)
    for p in range(n):
        D[p, p] = float(np.real(state.expectation_value(_diagonal_op(p, n))))
    for p in range(n):
        for q in range(p + 1, n):
            val = float(np.real(state.expectation_value(_offdiagonal_real_op(p, q, n))))
            D[p, q] = val
            D[q, p] = val
    return D


# ---------------------------------------------------------------------------
# Measurement-cost comparison (the advantage metric)
# ---------------------------------------------------------------------------

def rdm_measurement_cost(n: int) -> dict:
    """
    Compare measurement settings to obtain the full 1-RDM.

    grouped-Pauli : number of qubit-wise-commuting groups over all 1-RDM
                    Pauli strings (each group = one measurement basis).
    shadow        : 1 (a single randomized-measurement dataset).

    Returns
    -------
    dict with n_orbitals, n_rdm_elements, grouped_settings, shadow_settings,
    advantage_ratio.
    """
    terms = one_rdm_pauli_terms(n)
    try:
        groups = terms.group_commuting(qubit_wise=True)
    except TypeError:
        groups = terms.group_commuting()
    grouped = len(groups)
    n_elements = n + n * (n - 1) // 2     # diagonal + upper-triangle
    return {
        "n_orbitals": n,
        "n_rdm_elements": n_elements,
        "n_pauli_terms": len(terms),
        "grouped_settings": grouped,
        "shadow_settings": 1,
        "advantage_ratio": float(grouped),
    }
