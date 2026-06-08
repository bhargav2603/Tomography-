"""
shadowvqe — Classical Shadow Tomography + VQE on Qiskit 1.x / 2.x.

Modules
-------
utils                : logging, seeding, result dataclasses
hamiltonians         : SparsePauliOp builders (H2, Heisenberg, custom)
molecules            : PySCF-backed molecular Hamiltonians (H2 PES, LiH, BeH2)
ansatz               : hardware-efficient and custom ansatz factories
shadows              : classical shadow tomography (Huang et al. 2020)
derandomized_shadows : Hamiltonian-adaptive shadows for variance reduction
estimators           : Pauli expectation via statevector or classical shadows
optimizers           : COBYLA / SPSA wrappers with convergence tracking
vqe                  : standard VQE engine
shadow_vqe           : shadow-assisted VQE engine
validation           : exact diagonalisation reference + metrics table
visualization        : publication-quality matplotlib figures
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shadowvqe")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

from .hamiltonians import h2_hamiltonian, heisenberg_hamiltonian, random_hamiltonian
from .molecules import hydrogen_chain, h4_hamiltonian, h6_hamiltonian
from .ansatz import hardware_efficient_ansatz, build_ansatz
from .shadows import ClassicalShadows
from .derandomized_shadows import (
    HamiltonianAdaptedShadows,
    compute_hamiltonian_basis_probs,
    compare_estimators,
)
from .vqe import VQE, VQEResult
from .shadow_vqe import ShadowVQE, ShadowVQEResult
from .fmo import (
    Fragment,
    FragmentSystem,
    FMOResult,
    run_fmo,
    assemble_fmo2_energy,
    measurement_cost_report,
    direct_vqe_cost,
)
from .rdm import (
    one_rdm_from_shadows,
    exact_one_rdm,
    rdm_measurement_cost,
    RDMResult,
)
from .validation import ValidationSuite
from .visualization import plot_convergence, plot_comparison, plot_shadow_histogram
from . import visualization_research

__all__ = [
    "__version__",
    # hamiltonians
    "h2_hamiltonian",
    "heisenberg_hamiltonian",
    "random_hamiltonian",
    # molecules
    "hydrogen_chain",
    "h4_hamiltonian",
    "h6_hamiltonian",
    # ansatz
    "hardware_efficient_ansatz",
    "build_ansatz",
    # shadows
    "ClassicalShadows",
    # adaptive shadows
    "HamiltonianAdaptedShadows",
    "compute_hamiltonian_basis_probs",
    "compare_estimators",
    # vqe
    "VQE",
    "VQEResult",
    # shadow vqe
    "ShadowVQE",
    "ShadowVQEResult",
    # fmo / fragment methods
    "Fragment",
    "FragmentSystem",
    "FMOResult",
    "run_fmo",
    "assemble_fmo2_energy",
    "measurement_cost_report",
    "direct_vqe_cost",
    # reduced density matrices
    "one_rdm_from_shadows",
    "exact_one_rdm",
    "rdm_measurement_cost",
    "RDMResult",
    # validation
    "ValidationSuite",
    # visualization
    "plot_convergence",
    "plot_comparison",
    "plot_shadow_histogram",
    "visualization_research",
]
