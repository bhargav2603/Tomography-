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
    print(f"\n{'-'*50}")
    print(f"  {name}")
    print(f"{'-'*50}")
    print(f"  Qubits:          {hamiltonian.num_qubits}")
    print(f"  Pauli terms:     {len(hamiltonian)}")
    print(f"  Commuting groups:{len(groups)}  (VQE circuits/step)")
    print(f"  Norm:            {float(np.linalg.norm(coeffs)):.4f}")
    print(f"  Largest |coeff|: {float(np.max(coeffs)):.6f}")
    print(f"{'-'*50}\n")


# ---------------------------------------------------------------------------
# Fragment builders for FMO / many-body-expansion studies
# ---------------------------------------------------------------------------

def _active_space_build(
    geometry: str,
    basis: str,
    charge: int,
    spin: int,
    n_electrons: int,
    n_orbitals: int,
) -> SparsePauliOp:
    """Build a reduced active-space Hamiltonian (for keeping qubit count small)."""
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import ParityMapper
        from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    try:
        from qiskit_nature.units import DistanceUnit
        driver_kw: dict = dict(atom=geometry, basis=basis, charge=charge,
                               spin=spin, unit=DistanceUnit.ANGSTROM)
    except ImportError:
        driver_kw = dict(atom=geometry, basis=basis, charge=charge, spin=spin)

    problem = PySCFDriver(**driver_kw).run()
    problem = ActiveSpaceTransformer(
        num_electrons=n_electrons, num_spatial_orbitals=n_orbitals
    ).transform(problem)
    mapper = ParityMapper(num_particles=problem.num_particles)
    return mapper.map(problem.hamiltonian.second_q_op()).simplify(atol=1e-10)


def hydrogen_chain_fragments(
    n_units: int,
    r_intra: float = 0.74,
    r_inter: float = 2.5,
    basis: str = "sto-3g",
    all_pairs: bool = True,
    build_reference: bool = True,
    max_ref_qubits: int = 16,
):
    """
    Build a dimerised hydrogen chain as a FragmentSystem of H2 units.

    Geometry: ``n_units`` H2 molecules placed along z. Each unit has
    intra-bond ``r_intra`` (≈ equilibrium 0.74 A); units are separated by
    ``r_inter`` (set large so the chain is a van-der-Waals assembly where
    the two-body expansion is accurate).

    Fragments  : single H2 units  (2 qubits each, parity-mapped)
    Pairs      : H2-H2 dimers      (6 qubits each)
    Reference  : full H_{2n} chain (only built if <= ``max_ref_qubits``)

    Returns
    -------
    FragmentSystem  (from shadowvqe.fmo)
    """
    from .fmo import Fragment, FragmentSystem
    if n_units < 2:
        raise ValueError("n_units must be >= 2")

    # Atomic z-positions for each unit
    unit_geoms: list[str] = []
    z = 0.0
    unit_z: list[tuple[float, float]] = []
    for _ in range(n_units):
        z0, z1 = z, z + r_intra
        unit_z.append((z0, z1))
        z = z1 + r_inter

    def _h2_geom(z0: float, z1: float) -> str:
        return f"H 0 0 {z0:.4f}; H 0 0 {z1:.4f}"

    def _pair_geom(a: int, b: int) -> str:
        za0, za1 = unit_z[a]
        zb0, zb1 = unit_z[b]
        return (f"H 0 0 {za0:.4f}; H 0 0 {za1:.4f}; "
                f"H 0 0 {zb0:.4f}; H 0 0 {zb1:.4f}")

    # Fragments
    fragments = []
    for k, (z0, z1) in enumerate(unit_z):
        ham = _build_hamiltonian(_h2_geom(z0, z1), basis=basis, freeze_core=False)
        fragments.append(Fragment(f"H2_{k}", ham))
        _log.info("Hydrogen-chain fragment %d: %d qubits", k, ham.num_qubits)

    # Pairs
    pair_indices = (
        [(i, j) for i in range(n_units) for j in range(i + 1, n_units)]
        if all_pairs else
        [(i, i + 1) for i in range(n_units - 1)]
    )
    pairs = {}
    for (i, j) in pair_indices:
        ham = _build_hamiltonian(_pair_geom(i, j), basis=basis, freeze_core=False)
        pairs[(i, j)] = ham
        _log.info("Hydrogen-chain pair (%d,%d): %d qubits", i, j, ham.num_qubits)

    # Full-chain reference
    reference = None
    if build_reference:
        full_geom = "; ".join(
            f"H 0 0 {zz:.4f}" for pair in unit_z for zz in pair
        )
        ref = _build_hamiltonian(full_geom, basis=basis, freeze_core=False)
        if ref.num_qubits <= max_ref_qubits:
            reference = ref
            _log.info("Hydrogen-chain reference: %d qubits", ref.num_qubits)
        else:
            _log.info("Reference skipped (%d > %d qubits)",
                      ref.num_qubits, max_ref_qubits)

    return FragmentSystem(f"H{2*n_units} chain", fragments, pairs, reference)


def water_cluster_fragments(
    n_waters: int = 2,
    o_o_distance: float = 2.8,
    basis: str = "sto-3g",
    mono_active: tuple[int, int] = (2, 2),
    pair_active: tuple[int, int] = (4, 4),
    build_reference: bool = False,
):
    """
    Build a water cluster as a FragmentSystem (active-space approximation).

    Each water is placed along x separated by ``o_o_distance`` (O-O distance).
    To keep qubit counts tractable, monomers and dimers use reduced active
    spaces (documented approximation):

        monomer : (n_e, n_o) = ``mono_active``  -> e.g. (2,2) = 2 qubits
        dimer   : (n_e, n_o) = ``pair_active``  -> e.g. (4,4) = 6 qubits

    Returns
    -------
    FragmentSystem  (from shadowvqe.fmo)
    """
    from .fmo import Fragment, FragmentSystem
    if n_waters < 2:
        raise ValueError("n_waters must be >= 2")

    # Standard water internal geometry (Angstrom)
    def _water(ox: float) -> str:
        # O at (ox,0,0); two H in the xy-plane (bond 0.9572 A, angle 104.5 deg)
        return (f"O {ox:.4f} 0.0000 0.0000; "
                f"H {ox+0.7570:.4f} 0.5860 0.0000; "
                f"H {ox+0.7570:.4f} -0.5860 0.0000")

    centers = [i * o_o_distance for i in range(n_waters)]

    fragments = []
    for k, ox in enumerate(centers):
        ham = _active_space_build(_water(ox), basis, 0, 0, *mono_active)
        fragments.append(Fragment(f"H2O_{k}", ham))
        _log.info("Water fragment %d: %d qubits", k, ham.num_qubits)

    pair_indices = [(i, j) for i in range(n_waters) for j in range(i + 1, n_waters)]
    pairs = {}
    for (i, j) in pair_indices:
        geom = _water(centers[i]) + "; " + _water(centers[j])
        ham = _active_space_build(geom, basis, 0, 0, *pair_active)
        pairs[(i, j)] = ham
        _log.info("Water pair (%d,%d): %d qubits", i, j, ham.num_qubits)

    reference = None
    if build_reference and n_waters == 2:
        geom = _water(centers[0]) + "; " + _water(centers[1])
        reference = _active_space_build(geom, basis, 0, 0, *pair_active)

    return FragmentSystem(f"(H2O){n_waters}", fragments, pairs, reference)
