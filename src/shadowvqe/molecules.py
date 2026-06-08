"""
Molecule Hamiltonian Builders.

Computes second-quantized Hamiltonians for real molecules using PySCF
and maps them to qubit operators via qiskit-nature's Jordan-Wigner /
Parity mappers.

Installation (one time):
    pip install qiskit-nature[pyscf]

Molecules supported
-------------------
H2   — 2 qubits (Parity mapped, STO-3G)
LiH  — 4 qubits (Parity mapped, frozen core, STO-3G)
BeH2 — 6 qubits (Parity mapped, frozen core, STO-3G)

Each function returns a SparsePauliOp ready for VQE / ShadowVQE.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Sequence

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .utils import get_logger

_log = get_logger(__name__)

_INSTALL_MSG = (
    "\nMolecule Hamiltonians require PySCF and qiskit-nature.\n"
    "Install with:\n"
    "    pip install qiskit-nature[pyscf]\n"
    "On Google Colab:\n"
    "    !pip install qiskit-nature[pyscf]\n"
    "Then restart the Python kernel / runtime."
)


# ---------------------------------------------------------------------------
# Core PySCF → SparsePauliOp builder
# ---------------------------------------------------------------------------

def _build_hamiltonian(
    geometry: str,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    freeze_core: bool = True,
    remove_orbitals: list[int] | None = None,
) -> SparsePauliOp:
    """
    Compute a molecular Hamiltonian using PySCF + qiskit-nature.

    Parameters
    ----------
    geometry        : PySCF atom string, e.g. "H 0 0 0; H 0 0 0.735".
                      Coordinates in Ångströms.
    basis           : Gaussian basis set name (default 'sto-3g').
    charge          : Total molecular charge.
    spin            : Spin multiplicity − 1  (0 = singlet).
    freeze_core     : Whether to freeze core orbitals.
    remove_orbitals : Additional virtual orbital indices to remove
                      (reduces qubit count for larger molecules).

    Returns
    -------
    SparsePauliOp
    """
    try:
        return _nature_build(geometry, basis, charge, spin, freeze_core, remove_orbitals)
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def _nature_build(
    geometry: str,
    basis: str,
    charge: int,
    spin: int,
    freeze_core: bool,
    remove_orbitals: list[int] | None,
) -> SparsePauliOp:
    """qiskit-nature + PySCF backend."""
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import ParityMapper
        from qiskit_nature.second_q.transformers import FreezeCoreTransformer, ActiveSpaceTransformer
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    # Handle DistanceUnit gracefully across qiskit-nature versions
    try:
        from qiskit_nature.units import DistanceUnit
        driver_kw: dict = dict(
            atom=geometry, basis=basis, charge=charge, spin=spin,
            unit=DistanceUnit.ANGSTROM,
        )
    except ImportError:
        driver_kw = dict(atom=geometry, basis=basis, charge=charge, spin=spin)

    driver = PySCFDriver(**driver_kw)
    problem = driver.run()

    if freeze_core:
        problem = FreezeCoreTransformer(freeze_core=True).transform(problem)

    if remove_orbitals:
        # Further reduce active space by removing high-energy virtuals
        n_active = problem.num_spatial_orbitals - len(remove_orbitals)
        n_e = problem.num_particles
        problem = ActiveSpaceTransformer(
            num_electrons=n_e, num_spatial_orbitals=n_active
        ).transform(problem)

    mapper = ParityMapper(num_particles=problem.num_particles)
    op = mapper.map(problem.hamiltonian.second_q_op())

    hamiltonian = op.simplify(atol=1e-10)
    _log.info(
        "Built Hamiltonian: %d qubits, %d Pauli terms",
        hamiltonian.num_qubits,
        len(hamiltonian),
    )
    return hamiltonian


# ---------------------------------------------------------------------------
# Public molecule functions
# ---------------------------------------------------------------------------

def h2_pes(
    distances: Sequence[float] | None = None,
    basis: str = "sto-3g",
) -> dict[float, SparsePauliOp]:
    """
    Compute H2 potential energy surface at multiple bond distances.

    Parameters
    ----------
    distances : H-H bond lengths in Ångströms.
                Defaults to 15 points from 0.3 → 3.0 Å.
    basis     : Gaussian basis set (default 'sto-3g').

    Returns
    -------
    dict mapping distance → SparsePauliOp
    """
    if distances is None:
        distances = [0.30, 0.40, 0.50, 0.60, 0.70, 0.735,
                     0.80, 0.90, 1.00, 1.20, 1.50, 1.80,
                     2.00, 2.50, 3.00]

    result: dict[float, SparsePauliOp] = {}
    for r in distances:
        geom = f"H 0 0 0; H 0 0 {r:.4f}"
        _log.info("H2 PES: computing R = %.4f Å", r)
        result[r] = _build_hamiltonian(
            geometry=geom, basis=basis, charge=0, spin=0, freeze_core=False
        )
    return result


def lih_hamiltonian(
    bond_distance: float = 1.5474,
    basis: str = "sto-3g",
) -> SparsePauliOp:
    """
    LiH Hamiltonian (lithium hydride) — 4 qubits after Parity mapping + freeze core.

    Parameters
    ----------
    bond_distance : Li-H bond length in Ångströms (default = equilibrium 1.5474 Å).
    basis         : Gaussian basis (default 'sto-3g').

    Returns
    -------
    SparsePauliOp
        Typically 4 qubits, ~100+ Pauli terms in STO-3G.
    """
    geom = f"Li 0 0 0; H 0 0 {bond_distance:.4f}"
    _log.info("LiH: computing at R = %.4f Å", bond_distance)
    return _build_hamiltonian(
        geometry=geom, basis=basis, charge=0, spin=0, freeze_core=True
    )


def beh2_hamiltonian(
    bond_distance: float = 1.3264,
    basis: str = "sto-3g",
) -> SparsePauliOp:
    """
    BeH2 Hamiltonian (beryllium dihydride, linear geometry) — 6 qubits.

    Parameters
    ----------
    bond_distance : Be-H bond length in Ångströms (default = equilibrium 1.3264 Å).
    basis         : Gaussian basis (default 'sto-3g').

    Returns
    -------
    SparsePauliOp
        Typically 6 qubits, ~150+ Pauli terms in STO-3G.
    """
    geom = f"Be 0 0 0; H 0 0 {bond_distance:.4f}; H 0 0 -{bond_distance:.4f}"
    _log.info("BeH2: computing at R = %.4f Å", bond_distance)
    return _build_hamiltonian(
        geometry=geom, basis=basis, charge=0, spin=0, freeze_core=True
    )


def lih_pes(
    distances: Sequence[float] | None = None,
    basis: str = "sto-3g",
) -> dict[float, SparsePauliOp]:
    """
    LiH potential energy surface.

    Parameters
    ----------
    distances : Li-H bond lengths in Ångströms.
    basis     : Gaussian basis set.
    """
    if distances is None:
        distances = [1.0, 1.2, 1.4, 1.5474, 1.7, 2.0, 2.5, 3.0]
    return {r: lih_hamiltonian(r, basis) for r in distances}


# ---------------------------------------------------------------------------
# Utility: check if PySCF is available
# ---------------------------------------------------------------------------

def check_pyscf_available() -> bool:
    """
    Return True if qiskit-nature + pyscf are importable.
    Raises ImportError with installation instructions if not.
    """
    try:
        import pyscf  # noqa: F401
        from qiskit_nature.second_q.drivers import PySCFDriver  # noqa: F401
        return True
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def molecule_summary(hamiltonian: SparsePauliOp, name: str = "Molecule") -> None:
    """Print a summary of a molecular Hamiltonian."""
    coeffs = np.abs(hamiltonian.coeffs.real)
    groups = hamiltonian.group_commuting()
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Qubits:          {hamiltonian.num_qubits}")
    print(f"  Pauli terms:     {len(hamiltonian)}")
    print(f"  Commuting groups:{len(groups)}  (VQE circuits/step)")
    print(f"  Norm:            {float(np.linalg.norm(coeffs)):.4f}")
    print(f"  Largest |coeff|: {float(np.max(coeffs)):.6f}")
    print(f"{'─'*50}\n")
