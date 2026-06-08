"""
Fragment-based energy assembly for VQE + Classical Shadows.

This module implements the two-body Many-Body Expansion (MBE-2), the
computational core of the Fragment Molecular Orbital (FMO) method:

    E_total = Σ_I  E_I  +  Σ_{I<J} ( E_IJ − E_I − E_J )
              └────────┘    └─────────────────────────┘
              monomers           dimer corrections
                                 (interaction energies)

Each fragment energy E_I and pair energy E_IJ is obtained from a small
VQE problem. The fragmentation keeps every subsystem small, so the qubit
count stays BOUNDED by the largest pair — regardless of total molecule
size. That is the resource advantage FMO provides for quantum hardware.

Why combine with Classical Shadows
----------------------------------
Standard VQE measures only the ENERGY of each subsystem. FMO embedding and
property analysis additionally need reduced density matrices (RDMs) and
many observables. Classical shadows estimate ALL of these from a SINGLE
measurement dataset per subsystem, whereas grouped-Pauli measurement needs
a separate measurement setting per commuting group. The measurement-setting
advantage grows with subsystem size (see `measurement_cost_report`).

Note
----
This is the MBE-2 / FMO2 expansion *without* the electrostatic embedding
potential. It is exact for non-covalent assemblies in the limit of a
complete basis and converges systematically with expansion order. It is the
standard, defensible starting point for demonstrating quantum-resource
scaling; embedding can be layered on later.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .ansatz import hardware_efficient_ansatz
from .vqe import VQE
from .shadows import ClassicalShadows
from .validation import exact_ground_state_energy
from .utils import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Fragment:
    """A single molecular fragment (monomer)."""
    name: str
    hamiltonian: SparsePauliOp

    @property
    def n_qubits(self) -> int:
        return self.hamiltonian.num_qubits


@dataclass
class FragmentSystem:
    """
    A molecule decomposed into fragments plus their pair Hamiltonians.

    Parameters
    ----------
    name        : Label for the whole system (e.g. "H8 chain").
    fragments   : List of Fragment monomers.
    pairs       : Dict {(i, j): SparsePauliOp} of pair (dimer) Hamiltonians.
                  Keys are fragment-index tuples with i < j.
    reference   : Optional exact full-molecule SparsePauliOp for validation.
    """
    name: str
    fragments: list[Fragment]
    pairs: dict[tuple[int, int], SparsePauliOp]
    reference: SparsePauliOp | None = None

    @property
    def n_fragments(self) -> int:
        return len(self.fragments)

    @property
    def max_qubits(self) -> int:
        """Largest subsystem size — the qubit requirement on hardware."""
        frag_q = max(f.n_qubits for f in self.fragments)
        pair_q = max((p.num_qubits for p in self.pairs.values()), default=0)
        return max(frag_q, pair_q)

    def __post_init__(self) -> None:
        # Validate pair keys
        for (i, j) in self.pairs:
            if not (0 <= i < j < self.n_fragments):
                raise ValueError(
                    f"Invalid pair key ({i}, {j}); must satisfy 0 <= i < j < "
                    f"{self.n_fragments}."
                )


@dataclass
class FMOResult:
    """Outcome of an FMO-VQE(+shadow) run."""
    name: str
    method: str                       # 'exact', 'vqe', or 'shadow'
    total_energy: float
    monomer_energies: dict[int, float]
    pair_energies: dict[tuple[int, int], float]
    interaction_energies: dict[tuple[int, int], float]
    max_qubits: int
    reference_energy: float | None = None

    @property
    def error(self) -> float | None:
        if self.reference_energy is None:
            return None
        return abs(self.total_energy - self.reference_energy)


# ---------------------------------------------------------------------------
# Energy assembly (MBE-2 / FMO2 formula)
# ---------------------------------------------------------------------------

def assemble_fmo2_energy(
    monomer_energies: dict[int, float],
    pair_energies: dict[tuple[int, int], float],
) -> tuple[float, dict[tuple[int, int], float]]:
    """
    Assemble the total energy from monomer and pair energies (MBE-2).

    Returns
    -------
    total_energy : float
    interaction_energies : dict {(i, j): E_IJ - E_I - E_J}
    """
    total = sum(monomer_energies.values())
    interactions: dict[tuple[int, int], float] = {}
    for (i, j), e_ij in pair_energies.items():
        delta = e_ij - monomer_energies[i] - monomer_energies[j]
        interactions[(i, j)] = delta
        total += delta
    return total, interactions


# ---------------------------------------------------------------------------
# Single-subsystem solvers
# ---------------------------------------------------------------------------

def _solve_subsystem_exact(ham: SparsePauliOp) -> float:
    return exact_ground_state_energy(ham)


def _solve_subsystem_vqe(
    ham: SparsePauliOp,
    reps: int,
    optimizer: str,
    max_iter: int,
    seed: int,
) -> tuple[float, list[float]]:
    """Return (energy, optimal_parameters)."""
    ansatz = hardware_efficient_ansatz(ham.num_qubits, reps=reps)
    res = VQE(
        ansatz=ansatz, hamiltonian=ham,
        optimizer=optimizer, max_iter=max_iter, seed=seed,
    ).run()
    return res.ground_state_energy, list(res.optimal_parameters)


def _solve_subsystem_shadow(
    ham: SparsePauliOp,
    reps: int,
    optimizer: str,
    max_iter: int,
    seed: int,
    n_shadows: int,
) -> float:
    """VQE to find the optimal circuit, then estimate energy via shadows."""
    ansatz = hardware_efficient_ansatz(ham.num_qubits, reps=reps)
    res = VQE(
        ansatz=ansatz, hamiltonian=ham,
        optimizer=optimizer, max_iter=max_iter, seed=seed,
    ).run()
    bound = ansatz.assign_parameters(
        dict(zip(sorted(ansatz.parameters, key=lambda p: p.name),
                 res.optimal_parameters))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = ClassicalShadows(
            n_qubits=ham.num_qubits, n_shadows=n_shadows, seed=seed,
        )
    cs.collect(bound)
    return cs.estimate_observable(ham)


# ---------------------------------------------------------------------------
# Top-level FMO driver
# ---------------------------------------------------------------------------

def run_fmo(
    system: FragmentSystem,
    method: Literal["exact", "vqe", "shadow"] = "shadow",
    reps: int = 2,
    optimizer: str = "cobyla",
    max_iter: int = 600,
    seed: int = 42,
    n_shadows: int = 4000,
) -> FMOResult:
    """
    Run the full FMO-2 energy assembly on a FragmentSystem.

    Parameters
    ----------
    system    : FragmentSystem to solve.
    method    : 'exact'  — exact diagonalisation per subsystem (reference)
                'vqe'    — VQE energy per subsystem
                'shadow' — VQE circuit + classical-shadow energy estimate
    reps      : Ansatz repetition depth (for vqe / shadow).
    optimizer : 'cobyla' or 'spsa'.
    max_iter  : VQE optimizer iterations.
    seed      : Base random seed.
    n_shadows : Shadows per subsystem (shadow method only).

    Returns
    -------
    FMOResult
    """
    _log.info("FMO run: system=%s, method=%s, fragments=%d, pairs=%d",
              system.name, method, system.n_fragments, len(system.pairs))

    monomer_E: dict[int, float] = {}
    pair_E: dict[tuple[int, int], float] = {}

    # ── Monomers ──────────────────────────────────────────────────────────
    for idx, frag in enumerate(system.fragments):
        s = seed + idx
        if method == "exact":
            monomer_E[idx] = _solve_subsystem_exact(frag.hamiltonian)
        elif method == "vqe":
            monomer_E[idx], _ = _solve_subsystem_vqe(
                frag.hamiltonian, reps, optimizer, max_iter, s)
        else:  # shadow
            monomer_E[idx] = _solve_subsystem_shadow(
                frag.hamiltonian, reps, optimizer, max_iter, s, n_shadows)

    # ── Pairs ─────────────────────────────────────────────────────────────
    for k, ((i, j), ham_ij) in enumerate(system.pairs.items()):
        s = seed + 1000 + k
        if method == "exact":
            pair_E[(i, j)] = _solve_subsystem_exact(ham_ij)
        elif method == "vqe":
            pair_E[(i, j)], _ = _solve_subsystem_vqe(
                ham_ij, reps, optimizer, max_iter, s)
        else:  # shadow
            pair_E[(i, j)] = _solve_subsystem_shadow(
                ham_ij, reps, optimizer, max_iter, s, n_shadows)

    # ── Assemble ──────────────────────────────────────────────────────────
    total, interactions = assemble_fmo2_energy(monomer_E, pair_E)

    ref = (
        exact_ground_state_energy(system.reference)
        if system.reference is not None else None
    )

    return FMOResult(
        name=system.name,
        method=method,
        total_energy=total,
        monomer_energies=monomer_E,
        pair_energies=pair_E,
        interaction_energies=interactions,
        max_qubits=system.max_qubits,
        reference_energy=ref,
    )


# ---------------------------------------------------------------------------
# Measurement-cost accounting (the advantage metric)
# ---------------------------------------------------------------------------

def _qwc_groups(ham: SparsePauliOp) -> int:
    """Number of qubit-wise commuting groups (distinct measurement bases)."""
    try:
        groups = ham.group_commuting(qubit_wise=True)
    except TypeError:
        groups = ham.group_commuting()
    return len(groups)


def measurement_cost_report(system: FragmentSystem) -> dict:
    """
    Compare measurement SETTINGS for grouped-Pauli VQE vs classical shadows.

    For each subsystem, grouped-Pauli VQE needs one measurement basis per
    qubit-wise-commuting group. Classical shadows need a single randomized
    protocol per subsystem that estimates every observable in it.

    Returns
    -------
    dict with:
        vqe_settings    : total commuting groups across all subsystems
        shadow_settings : number of subsystems (one shadow protocol each)
        per_subsystem   : list of (label, n_qubits, n_terms, n_groups)
        advantage_ratio : vqe_settings / shadow_settings
    """
    per_sub = []
    vqe_settings = 0

    for idx, frag in enumerate(system.fragments):
        g = _qwc_groups(frag.hamiltonian)
        vqe_settings += g
        per_sub.append((f"F{idx}", frag.n_qubits, len(frag.hamiltonian), g))

    for (i, j), ham in system.pairs.items():
        g = _qwc_groups(ham)
        vqe_settings += g
        per_sub.append((f"P{i}{j}", ham.num_qubits, len(ham), g))

    shadow_settings = system.n_fragments + len(system.pairs)
    ratio = vqe_settings / shadow_settings if shadow_settings else float("nan")

    return {
        "system": system.name,
        "vqe_settings": vqe_settings,
        "shadow_settings": shadow_settings,
        "advantage_ratio": ratio,
        "per_subsystem": per_sub,
        "max_qubits": system.max_qubits,
    }


def direct_vqe_cost(reference: SparsePauliOp) -> dict:
    """Measurement settings + qubit count for solving the FULL molecule directly."""
    return {
        "n_qubits": reference.num_qubits,
        "n_terms": len(reference),
        "vqe_settings": _qwc_groups(reference),
    }
