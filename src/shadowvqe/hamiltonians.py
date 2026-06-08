"""
Hamiltonian factory functions.

All functions return a qiskit.quantum_info.SparsePauliOp so the rest of the
library only ever deals with one Hamiltonian type.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .utils import get_logger, validate_positive_int

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

def h2_hamiltonian(bond_distance: float = 0.735) -> SparsePauliOp:
    """
    Two-qubit Jordan-Wigner mapped H₂ Hamiltonian.

    The coefficients are the standard STO-3G, Jordan-Wigner reduced-to-2-qubit
    form used in virtually every VQE tutorial.  The ground-state energy at
    equilibrium geometry (bond_distance ≈ 0.735 Å) is ≈ −1.8572 Ha.

    Parameters
    ----------
    bond_distance:
        H–H bond distance in Ångströms.  Only the equilibrium case has
        pre-tabulated coefficients here; other distances raise a warning.

    Returns
    -------
    SparsePauliOp
        2-qubit Hamiltonian with 5 terms.
    """
    if not np.isclose(bond_distance, 0.735, atol=0.001):
        _log.warning(
            "h2_hamiltonian: only equilibrium coefficients are tabulated "
            "(bond_distance=0.735 Å).  The returned Hamiltonian is for "
            "equilibrium geometry regardless of the supplied value."
        )

    # Standard 2-qubit JW-mapped H2 at equilibrium geometry (STO-3G)
    paulis = ["II", "ZI", "IZ", "ZZ", "XX"]
    coeffs = [
        -1.0523732,
         0.3979374,
        -0.3979374,
        -0.0112801,
         0.1809312,
    ]
    hamiltonian = SparsePauliOp.from_list(list(zip(paulis, coeffs)))
    _log.debug("H2 Hamiltonian created: %d terms, %d qubits", len(paulis), hamiltonian.num_qubits)
    return hamiltonian


def heisenberg_hamiltonian(
    n_qubits: int,
    j_coupling: float = 1.0,
    h_field: float = 0.5,
    periodic: bool = True,
) -> SparsePauliOp:
    """
    1-D transverse-field Heisenberg (XXX) model.

    H = J Σ (XX + YY + ZZ) + h Σ Z

    Parameters
    ----------
    n_qubits    : number of sites / qubits.
    j_coupling  : spin-spin coupling constant.
    h_field     : transverse magnetic field strength.
    periodic    : whether to include the boundary term (site n−1 → 0).
    """
    n_qubits = validate_positive_int(n_qubits, "n_qubits")
    if n_qubits < 2:
        raise ValueError("Heisenberg chain requires at least 2 qubits.")

    terms: list[tuple[str, complex]] = []
    n = n_qubits

    def _two_body(op: str, i: int, j: int) -> str:
        """Build n-qubit Pauli string with op on sites i and j."""
        pauli = ["I"] * n
        pauli[i] = op
        pauli[j] = op
        # Qiskit uses little-endian: qubit 0 is the rightmost character
        return "".join(reversed(pauli))

    def _one_body(op: str, i: int) -> str:
        pauli = ["I"] * n
        pauli[i] = op
        return "".join(reversed(pauli))

    # Two-body interactions
    pairs = list(range(n - 1))
    # A 2-qubit periodic chain is identical to an open chain; avoid doubling the terms
    if periodic and n > 2:
        pairs.append(n - 1)

    for i in pairs:
        j = (i + 1) % n
        for op in ("X", "Y", "Z"):
            terms.append((_two_body(op, i, j), j_coupling))

    # Longitudinal field
    for i in range(n):
        terms.append((_one_body("Z", i), h_field))

    hamiltonian = SparsePauliOp.from_list(terms).simplify()
    _log.debug(
        "Heisenberg Hamiltonian: %d qubits, %d terms", n_qubits, len(hamiltonian)
    )
    return hamiltonian


def random_hamiltonian(
    n_qubits: int,
    n_terms: int,
    max_weight: int | None = None,
    seed: int = 0,
) -> SparsePauliOp:
    """
    Random Hermitian Hamiltonian as a sum of random Pauli strings.

    Useful for benchmarking — coefficients are drawn from N(0,1).

    Parameters
    ----------
    n_qubits  : system size.
    n_terms   : number of distinct Pauli terms.
    max_weight: maximum Pauli weight (number of non-identity sites).
                Defaults to n_qubits (fully dense).
    seed      : reproducibility seed.
    """
    n_qubits = validate_positive_int(n_qubits, "n_qubits")
    n_terms = validate_positive_int(n_terms, "n_terms")
    max_weight = n_qubits if max_weight is None else int(max_weight)
    if not (1 <= max_weight <= n_qubits):
        raise ValueError(f"max_weight must be in [1, n_qubits={n_qubits}], got {max_weight}")

    rng = np.random.default_rng(seed)
    pauli_chars = ["I", "X", "Y", "Z"]
    terms: list[tuple[str, complex]] = []
    seen: set[str] = set()

    attempts = 0
    max_attempts = n_terms * 100
    while len(terms) < n_terms and attempts < max_attempts:
        attempts += 1
        # choose how many non-I sites for this term
        weight = rng.integers(1, max_weight + 1)
        sites = rng.choice(n_qubits, size=weight, replace=False)
        pauli = ["I"] * n_qubits
        for s in sites:
            pauli[s] = pauli_chars[rng.integers(1, 4)]
        pauli_str = "".join(reversed(pauli))  # little-endian
        if pauli_str in seen:
            continue
        seen.add(pauli_str)
        coeff = rng.standard_normal()
        terms.append((pauli_str, coeff))

    if len(terms) < n_terms:
        _log.warning(
            "Could only generate %d unique Pauli terms (requested %d). "
            "Consider reducing n_terms or increasing n_qubits.",
            len(terms),
            n_terms,
        )

    hamiltonian = SparsePauliOp.from_list(terms).simplify()
    _log.debug("Random Hamiltonian: %d qubits, %d terms", n_qubits, len(hamiltonian))
    return hamiltonian


# Alias so both names work
heisenberg_1d_hamiltonian = heisenberg_hamiltonian


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def hamiltonian_info(hamiltonian: SparsePauliOp) -> dict:
    """Return a summary dict for logging / display."""
    import numpy as np
    coeffs = hamiltonian.coeffs.real
    return {
        "n_qubits": hamiltonian.num_qubits,
        "n_terms": len(hamiltonian),
        "norm": float(np.linalg.norm(coeffs)),
        "max_coeff": float(np.max(np.abs(coeffs))),
        "pauli_strings": [str(p) for p in hamiltonian.paulis],
    }
