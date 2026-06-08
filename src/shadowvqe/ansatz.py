"""
Ansatz circuit factories.

Both return a plain QuantumCircuit with named Parameters so they plug
directly into the VQE / ShadowVQE engines.
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import EfficientSU2, TwoLocal

from .utils import get_logger, validate_positive_int

_log = get_logger(__name__)


def hardware_efficient_ansatz(
    n_qubits: int,
    reps: int = 2,
    entanglement: str = "linear",
    rotation_blocks: list[str] | None = None,
    insert_barriers: bool = False,
) -> QuantumCircuit:
    """
    Hardware-efficient ansatz via Qiskit's EfficientSU2.

    Parameters
    ----------
    n_qubits        : number of qubits.
    reps            : number of SU(2) repetition layers.
    entanglement    : 'linear', 'circular', or 'full'.
    rotation_blocks : list of single-qubit gate names, e.g. ['ry', 'rz'].
                      Defaults to ['ry', 'rz'].
    insert_barriers : add barriers between layers (useful for visualisation).

    Returns
    -------
    QuantumCircuit
        Parameterized circuit; parameters named θ[0], θ[1], …
    """
    n_qubits = validate_positive_int(n_qubits, "n_qubits")
    reps = validate_positive_int(reps, "reps")
    valid_entanglements = {"linear", "circular", "full", "sca"}
    if entanglement not in valid_entanglements:
        raise ValueError(
            f"entanglement must be one of {valid_entanglements}, got '{entanglement}'"
        )
    rotation_blocks = rotation_blocks or ["ry", "rz"]

    circ = EfficientSU2(
        num_qubits=n_qubits,
        reps=reps,
        entanglement=entanglement,
        su2_gates=rotation_blocks,
        insert_barriers=insert_barriers,
    )
    # Rename parameters to θ[i] for readability
    old_params = sorted(circ.parameters, key=lambda p: p.name)
    theta = ParameterVector("θ", len(old_params))
    param_map = {old: theta[i] for i, old in enumerate(old_params)}
    circ = circ.assign_parameters(param_map)

    _log.debug(
        "HEA ansatz: %d qubits, %d reps, %d parameters",
        n_qubits, reps, circ.num_parameters,
    )
    return circ


def build_ansatz(
    n_qubits: int,
    reps: int = 2,
    rotation_blocks: list[str] | None = None,
    entanglement_blocks: str = "cx",
    entanglement: str = "linear",
) -> QuantumCircuit:
    """
    General TwoLocal ansatz — more flexible than hardware_efficient_ansatz.

    Parameters
    ----------
    n_qubits            : number of qubits.
    reps                : repetition depth.
    rotation_blocks     : single-qubit gates (default ['ry']).
    entanglement_blocks : two-qubit gate name (default 'cx').
    entanglement        : connectivity pattern.

    Returns
    -------
    QuantumCircuit
        Parameterized circuit.
    """
    n_qubits = validate_positive_int(n_qubits, "n_qubits")
    reps = validate_positive_int(reps, "reps")
    rotation_blocks = rotation_blocks or ["ry"]

    circ = TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=rotation_blocks,
        entanglement_blocks=entanglement_blocks,
        entanglement=entanglement,
        reps=reps,
    )
    old_params = sorted(circ.parameters, key=lambda p: p.name)
    theta = ParameterVector("θ", len(old_params))
    param_map = {old: theta[i] for i, old in enumerate(old_params)}
    circ = circ.assign_parameters(param_map)

    _log.debug(
        "TwoLocal ansatz: %d qubits, %d reps, %d parameters",
        n_qubits, reps, circ.num_parameters,
    )
    return circ


def num_parameters(circuit: QuantumCircuit) -> int:
    """Return the number of free parameters in a circuit."""
    return circuit.num_parameters
